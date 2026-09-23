"""Exact raw-observation cache; native normalization and training batches are unchanged."""
import argparse, json, os, time, hashlib
from pathlib import Path
import numpy as np
from common import *
CACHE=ROOT/'raw_window_cache'

class CachedEpisode:
    def __init__(self, folder):
        self.images=np.load(folder/'images.npy',mmap_mode='r')
        self.wrist=np.load(folder/'wrist.npy',mmap_mode='r')
        with np.load(folder/'controls.npz') as f:
            self.actions=f['actions'];self.states=f['states']
        self.lang=json.loads((folder/'language.json').read_text())
    def __len__(self):return len(self.actions)
    def __getitem__(self,t):
        return {'observation':{'image':self.images[t], 'wrist_image':self.wrist[t], 'state':self.states[t]},
                'action':self.actions[t], 'language_instruction':self.lang[t]}

def records():return [json.loads(x) for x in (ROOT/'trajectories.jsonl').read_text().splitlines()]

def prepare(shard, shards):
    import tensorflow_datasets as tfds
    from cohort_planning.planned_dataset import EpisodeDecoder
    items=records();decoders={};done=0;t0=time.monotonic()
    CACHE.mkdir(exist_ok=True)
    for i in range(shard,len(items),shards):
        record=items[i];folder=CACHE/f'episode_{i:05d}'
        if (folder/'complete.json').exists():done+=1;continue
        name=record['dataset']
        if name not in decoders:decoders[name]=EpisodeDecoder(DATA_ROOT/name/'1.0.0',OLD_PLAN/(name+'_tfrecord_index.json'))
        steps=list(tfds.as_numpy(decoders[name].decode(record['trajectory_id'])['steps']))
        assert len(steps)==record['raw_step_count']
        folder.mkdir(exist_ok=True)
        for dest,key in [('images','image'),('wrist','wrist_image')]:
            a=np.stack([s['observation'][key] for s in steps]);assert a.dtype==np.uint8 and a.shape[1:]==(256,256,3)
            tmp=folder/(dest+'.tmp.npy');np.save(tmp,a);tmp.replace(folder/(dest+'.npy'))
            # Verify the entire persisted image array, not only sampled frames.
            np.testing.assert_array_equal(np.load(folder/(dest+'.npy'),mmap_mode='r'),a)
        actions=np.stack([s['action'] for s in steps]);states=np.stack([s['observation']['state'] for s in steps])
        np.savez(folder/'controls.npz',actions=actions,states=states)
        def lang(v):
            if isinstance(v,bytes):return v.decode('utf-8')
            if isinstance(v,np.ndarray) and v.dtype.type is np.bytes_:return v.item().decode('utf-8')
            return str(v)
        atomic_json(folder/'language.json',[lang(s['language_instruction']) for s in steps])
        atomic_json(folder/'complete.json',{'trajectory_id':record['trajectory_id'],'length':len(steps),'images_exact':True})
        done+=1
        if done%25==0:print(json.dumps({'shard':shard,'done':done,'seconds':time.monotonic()-t0}),flush=True)
    for x in decoders.values():x.close()
    atomic_json(CACHE/f'shard_{shard}.json',{'complete':True,'episodes':done,'seconds':time.monotonic()-t0})

def cached_dataset(native,epoch,rank,start_micro=0):
    from data import dataset
    ds=dataset(native,epoch,rank,start_micro)
    def load(i):
        folder=CACHE/f'episode_{i:05d}'
        if not (folder/'complete.json').exists():raise RuntimeError(f'Missing verified raw cache: {folder}')
        return i,CachedEpisode(folder)
    ds._decode_episode=load
    # Handles uncached entry reached via the one-at-a-time fallback as well.
    def episode(i):
        found=ds._episode_cache.pop(i,None)
        if found is None:found=load(i)[1]
        ds._episode_cache[i]=found
        while len(ds._episode_cache)>64:ds._episode_cache.popitem(last=False)
        return found
    ds._episode=episode
    return ds

def verify():
    import torch
    from data import native_dataset,Window,dataset,collate
    from cohort_planning.planned_dataset import EpisodeDecoder
    torch.set_num_threads(2)
    items=records();assert all((CACHE/f'episode_{i:05d}/complete.json').exists() for i in range(len(items)))
    native=native_dataset();checks=[]
    for suite in SUITES:
        for i in [k for k,r in enumerate(items) if r['dataset']==suite+'_no_noops'][:2]:
            r=items[i];decoder=EpisodeDecoder(DATA_ROOT/r['dataset']/'1.0.0',OLD_PLAN/(r['dataset']+'_tfrecord_index.json'))
            original=Window.prepare_episode(decoder.decode(r['trajectory_id']),r);cached=CachedEpisode(CACHE/f'episode_{i:05d}')
            for t in sorted(set([0, len(original)//2]+[max(0,len(original)-h) for h in (1,3,4,8,12)])):
                a=native._build_step_sample(original,t,len(original));b=native._build_step_sample(cached,t,len(cached))
                assert a[1]==b[1]
                for view in range(2):
                    for key in a[0][view]:torch.testing.assert_close(a[0][view][key],b[0][view][key],rtol=0,atol=0)
                for field in (2,3,4):torch.testing.assert_close(a[field],b[field],rtol=0,atol=0)
                checks.append([i,t])
            decoder.close()
    timing={};fingerprints={}
    for name,factory in [('original',dataset),('cached',cached_dataset)]:
        ds=factory(native,1,0,27);it=iter(ds);digest=hashlib.sha256();t0=time.monotonic()
        for _ in range(80):
            item=next(it);x=item['native']
            digest.update(x[1].encode())
            for view in x[0]:
                for k in sorted(view):digest.update(view[k].numpy().tobytes())
            for value in x[2:]:digest.update(value.numpy().tobytes())
            digest.update(str([item[k] for k in ('sample_loss_mask','csdr_sample_mask','canonical_prompt_id')]).encode())
        timing[name]=time.monotonic()-t0;fingerprints[name]=digest.hexdigest();it.close();ds.close()
    assert fingerprints['original']==fingerprints['cached']
    atomic_json(ROOT/'cache_speed_validation.json',{'passed':True,'episodes':len(items),'native_comparisons':len(checks),
        'all_images_verified':True,'batch_fingerprints':fingerprints,'data_seconds_80_windows':timing,
        'input_exact':True,'batch_unchanged':True,'action_masks_unchanged':True})
    print(json.dumps({'checks':len(checks),'timing':timing,'passed':True}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4);p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:verify()
    else:prepare(a.shard,a.shards)
