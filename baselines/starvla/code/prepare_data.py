"""Lossless frame cache and audited full-coverage plans for the published data."""
import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import av
from PIL import Image
import torch
from common import ROOT, DATA, RUNTIME

CACHE = ROOT / 'cache'
PROCESSOR = None

def paths(ep):
    stem=f'chunk-{ep//1000:03d}/episode_{ep:06d}'
    return DATA/f'data/{stem}.parquet', DATA/f'videos/chunk-{ep//1000:03d}/observation.images.image_0/episode_{ep:06d}.mp4', CACHE/stem

def init():
    global PROCESSOR
    torch.set_num_threads(1)
    sys.path.insert(0,str(RUNTIME))
    from deployment.model_server.policy_norm_processor import PolicyNormProcessor
    PROCESSOR=PolicyNormProcessor(str(ROOT/'baseline/checkpoints/steps_45000_pytorch_model.pt'),'oxe_bridge')

def atomic_array(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        np.save(f,value,allow_pickle=False)
    tmp.replace(path)

def prepare_episode(item):
    ep=int(item['episode_index'])
    parquet,video,base=paths(ep)
    ready=base.with_suffix('.ready.json')
    if ready.exists():
        return json.loads(ready.read_text())
    frame_data=pd.read_parquet(parquet)
    assert len(frame_data)==item['length'], ep
    assert (frame_data.episode_index.to_numpy()==ep).all(), ep
    timestamps=frame_data.timestamp.to_numpy()
    frames=[]
    pts=[]
    with av.open(str(video)) as container:
        container.streams.video[0].thread_count=1
        for frame in container.decode(video=0):
            pts.append(float(frame.pts*frame.time_base))
            frames.append(frame.to_ndarray(format='rgb24'))
    assert len(frames)>0, ep
    # Same closest-timestamp selection as the upstream PyAV/torchvision reader.
    indices=np.abs(np.asarray(pts)[:,None]-timestamps[None,:]).argmin(axis=0)
    assert np.abs(np.asarray(pts)[indices]-timestamps).max()<0.11, ep
    images=np.stack([np.asarray(Image.fromarray(frames[i]).resize((224,224))) for i in indices])
    raw=np.stack(frame_data.action).astype(np.float64)
    assert raw.shape==(len(frame_data),7)
    values={key:raw[:,i:i+1] for i,key in enumerate(PROCESSOR.action_keys)}
    transformed=PROCESSOR.transform(values)
    action=np.concatenate([transformed[key].numpy() for key in PROCESSOR.action_keys],axis=1).astype(np.float16)
    assert np.isfinite(action).all()
    tasks=frame_data.task_index.to_numpy(dtype=np.int32)
    atomic_array(base.with_suffix('.images.npy'),images)
    atomic_array(base.with_suffix('.actions.npy'),action)
    atomic_array(base.with_suffix('.tasks.npy'),tasks)
    result={'episode_index':ep,'length':len(frame_data),'unique_task_indices':np.unique(tasks).tolist()}
    ready.write_text(json.dumps(result))
    return result

def build_plans(episodes,epochs):
    tasks=[json.loads(x) for x in (DATA/'meta/tasks.jsonl').read_text().splitlines()]
    texts={int(x['task_index']):x['task'] for x in tasks}
    # Exact text groups only. No unverified semantic merging across objects/directions.
    text_ids={x:i for i,x in enumerate(sorted(set(texts.values())))}
    lengths=np.array([x['length'] for x in episodes],dtype=np.int64)
    offsets=np.concatenate(([0],np.cumsum(lengths)))
    total=int(offsets[-1])
    assert total==json.loads((DATA/'meta/info.json').read_text())['total_frames']
    action=np.empty((total,7),dtype=np.float16)
    prompts=np.empty(total,dtype=np.int32)
    for i,ep in enumerate(episodes):
        _,_,base=paths(ep['episode_index'])
        action[offsets[i]:offsets[i+1]]=np.load(base.with_suffix('.actions.npy'))
        tid=np.load(base.with_suffix('.tasks.npy'))
        prompts[offsets[i]:offsets[i+1]]=[text_ids[texts[int(t)]] for t in tid]
    atomic_array(CACHE/'actions.npy',action)
    atomic_array(CACHE/'offsets.npy',offsets)
    atomic_array(CACHE/'prompt_ids.npy',prompts)
    (CACHE/'prompts.json').write_text(json.dumps(sorted(text_ids),ensure_ascii=False))
    (CACHE/'episodes.json').write_text(json.dumps(episodes))
    group_order=np.argsort(prompts,kind='stable')
    splits=np.flatnonzero(np.diff(prompts[group_order]))+1
    prompt_groups=np.split(group_order,splits)
    for epoch in range(epochs):
        rng=np.random.default_rng(7+epoch)
        cohorts=[]
        leftovers=[]
        for ids in prompt_groups:
            episode_ids=np.searchsorted(offsets,ids,side='right')-1
            chunks=np.split(ids,np.flatnonzero(np.diff(episode_ids))+1)
            rng.shuffle(chunks)
            for chunk in chunks:
                rng.shuffle(chunk)
            # Round-robin across trajectories prevents adjacent windows dominating a cohort.
            interleaved=np.array([chunk[k] for k in range(max(map(len,chunks))) for chunk in chunks if k<len(chunk)],dtype=np.int64)
            n=len(interleaved)//20*20
            if n:
                cohorts.append(interleaved[:n].reshape(-1,20))
            leftovers.extend(interleaved[n:])
        cohort=np.concatenate(cohorts)
        rng.shuffle(cohort)
        remainder=np.asarray(leftovers,dtype=np.int64)
        rng.shuffle(remainder)
        pad=(-len(remainder))%20
        tail=np.pad(remainder,(0,pad),constant_values=0).reshape(-1,20)
        order=np.concatenate((cohort,tail))
        loss_mask=np.ones_like(order,dtype=np.uint8)
        if pad:
            loss_mask[-1,-pad:]=0
        csdr=np.zeros_like(order,dtype=np.uint8)
        csdr[:len(cohort)]=1
        ep_idx=np.searchsorted(offsets,order,side='right')-1
        # CSDR compares only shared real prefixes of at least four steps.
        csdr *= (order+4<=offsets[ep_idx+1])
        shuffle=rng.permutation(len(order))
        order,loss_mask,csdr=order[shuffle],loss_mask[shuffle],csdr[shuffle]
        actual=np.sort(order[loss_mask.astype(bool)])
        assert np.array_equal(actual,np.arange(total)), 'Coverage or duplication failure'
        plan=ROOT/f'plans/epoch_{epoch+1}.npz'
        plan.parent.mkdir(exist_ok=True)
        np.savez_compressed(plan,global_index=order.reshape(-1,10,2),
                            loss_mask=loss_mask.reshape(-1,10,2),csdr_mask=csdr.reshape(-1,10,2))
        summary={'epoch':epoch+1,'seed':7+epoch,'world_size':10,'micro_batch_per_gpu':2,
                 'grad_accum':8,'effective_global_batch':160,'micro_batches':len(order),
                 'optimizer_steps':(len(order)+7)//8,'real_samples':total,'padding_slots':pad,
                 'csdr_samples':int(csdr.sum()),'missing':0,'duplicates':0,
                 'exact_prompt_count':len(text_ids),'tail_actions':'repeat last; native default absolute=True'}
        plan.with_suffix('.json').write_text(json.dumps(summary,indent=2))
        print('PLAN_READY',json.dumps(summary),flush=True)
    table, audits = {}, {}
    all_ids = np.arange(total)
    true_remaining = offsets[np.searchsorted(offsets, all_ids, side='right')] - all_ids
    table, audits = {}, {}
    for h in range(16, 3, -1):
        rng = np.random.default_rng(42 if h == 16 else 42 + h)
        selected = rng.choice(np.flatnonzero(true_remaining >= h),
                              size=min(20000, int((true_remaining >= h).sum())), replace=False)
        chunks = action[selected[:, None] + np.arange(h)].astype(np.float32)
        pairs = rng.integers(0, len(chunks), (200000, 2))
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        scales = {}
        for name, part in [('action_translation', slice(0, 3)), ('action_rotation', slice(3, 6)),
                           ('action_gripper', slice(6, 7))]:
            d = np.sqrt(np.mean((chunks[pairs[:, 0], :, part] - chunks[pairs[:, 1], :, part]) ** 2, axis=(1, 2)))
            scales[name] = max(float(np.median(d)), 1e-6)
        table[str(h)] = scales
        audits[str(h)] = {'samples': len(selected), 'pairs': len(pairs), 'seed': 42 if h == 16 else 42 + h}
        print('SCALES', h, scales, flush=True)
    (ROOT/'control_scales.json').write_text(json.dumps({'scales':table['16'],'by_length':table,'calibration':audits},indent=2))
    (ROOT/'data_prepared.ready').write_text(str(total))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--plans-only',action='store_true')
    args=p.parse_args()
    episodes=[json.loads(x) for x in (DATA/'meta/episodes.jsonl').read_text().splitlines()]
    episodes.sort(key=lambda x:x['episode_index'])
    if args.plans_only:
        assert (ROOT/'data_prepared.ready').exists()
        build_plans(episodes,args.epochs)
        return
    if args.limit:
        episodes=episodes[:args.limit]
    pending={x['episode_index']:x for x in episodes}
    # Each worker owns its transform object and decoder. No CUDA access.
    import multiprocessing as mp
    with cf.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=init) as pool:
        while pending:
            available=[]
            for ep,item in pending.items():
                parquet,video,base=paths(ep)
                if base.with_suffix('.ready.json').exists() or (parquet.exists() and video.exists()):
                    available.append(item)
                if len(available)>=256:
                    break
            if not available:
                print('WAIT_DOWNLOAD',len(pending),flush=True)
                time.sleep(15)
                continue
            for result in pool.map(prepare_episode,available):
                pending.pop(result['episode_index'])
            status={'episodes_prepared':len(episodes)-len(pending),'episodes_total':len(episodes)}
            (ROOT/'data_prepare_status.json').write_text(json.dumps(status))
            print(json.dumps(status),flush=True)
    if not args.limit:
        build_plans(episodes,args.epochs)

if __name__=='__main__':
    main()
