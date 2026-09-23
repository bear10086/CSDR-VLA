"""Full-coverage, resumable 10-GPU StarVLA + CSDR continuation."""
import argparse
import faulthandler
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from common import ROOT, register_framework
from planned_data import PlannedDataset,collate
from training_model import TrainingModel,FP32MasterAdamW
from csdr_loss_tail import CSDRConfig,spatialvla_csdr_loss
from speed_validation import model_digest,tree_digest
from checkpoint_control import synchronize_control,requested_save

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}

def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])

def save_checkpoint(output,native,optimizer,scheduler,step,epoch,next_micro,args):
    rank=dist.get_rank()
    folder=output/f'step_{step:06d}'
    folder.mkdir(parents=True,exist_ok=True)
    state={'optimizer':optimizer.optim.state_dict(),'scheduler':scheduler.state_dict(),
           'outer_param_groups':[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups],
           'rng':rng_state(),'step':step,'epoch':epoch,'next_micro':next_micro,
           'world_size':dist.get_world_size(),'args':vars(args)}
    tmp=folder/f'rank_{rank}.pt.tmp'
    torch.save(state,tmp)
    tmp.replace(folder/f'rank_{rank}.pt')
    if rank==0:
        (folder/'checkpoints').mkdir(exist_ok=True)
        weights=folder/'checkpoints'/f'steps_{45000+step}_pytorch_model.pt'
        tmp=weights.with_suffix('.pt.tmp')
        torch.save({k:v.detach().cpu() for k,v in native.state_dict().items()},tmp)
        tmp.replace(weights)
        OmegaConf.save(native.config,folder/'config.yaml')
        shutil.copy2(ROOT/'baseline/dataset_statistics.json',folder/'dataset_statistics.json')
    dist.barrier()
    if rank==0:
        (folder/'complete.json').write_text(json.dumps({'step':step,'epoch':epoch,'next_micro':next_micro,
             'world_size':dist.get_world_size(),'checkpoint':str(weights)},indent=2))
        tmp=output/'latest.json.tmp'
        tmp.write_text(json.dumps({'directory':str(folder)}))
        tmp.replace(output/'latest.json')
    dist.barrier()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--output',type=Path,default=ROOT/'train')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke-steps',type=int,default=0)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--csdr-ratio',type=float,default=0.005)
    parser.add_argument('--speed-mode',choices=['no_checkpoint'],default='no_checkpoint')
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--benchmark-steps',type=int,default=0)
    args=parser.parse_args()
    faulthandler.register(signal.SIGUSR1,all_threads=True)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    rank=dist.get_rank(); world=dist.get_world_size(); local=int(os.environ['LOCAL_RANK'])
    assert world==10,'Plans are audited for 10 ranks'
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    random.seed(42+rank); np.random.seed(42+rank); torch.manual_seed(42+rank)
    args.output.mkdir(parents=True,exist_ok=True)
    summaries=[json.loads((ROOT/f'plans/epoch_{e+1}.json').read_text()) for e in range(args.epochs)]
    max_steps=sum(x['optimizer_steps'] for x in summaries)
    assert all(x['missing']==x['duplicates']==0 for x in summaries)
    resume_dir=None
    if args.resume_from or (args.resume and (args.output/'latest.json').exists()):
        resume_dir=args.resume_from or Path(json.loads((args.output/'latest.json').read_text())['directory'])
        meta=json.loads((resume_dir/'complete.json').read_text())
        assert meta['world_size']==world
        ckpt=Path(meta['checkpoint'])
    else:
        ckpt=ROOT/'baseline/checkpoints/steps_45000_pytorch_model.pt'
    print(f'RANK {rank} LOADING_MODEL',flush=True)
    native=register_framework().from_pretrained(str(ckpt)).to(device=local,dtype=torch.bfloat16)
    print(f'RANK {rank} MODEL_LOADED',flush=True)
    assert native.config.framework.action_model.repeated_diffusion_steps==4
    assert not native.config.datasets.vla_data.include_state
    native.qwen_vl_interface.model.config.use_cache=False
    if args.speed_mode in ('no_checkpoint','buckets_no_checkpoint'):
        native.qwen_vl_interface.model.gradient_checkpointing_disable()
    else:
        native.qwen_vl_interface.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    # These branches are unused by the released action-only, no-proprio forward.
    for name,p in native.named_parameters():
        if name.startswith('qwen_vl_interface.model.lm_head.') or name.startswith('action_model.state_encoder.'):
            p.requires_grad_(False)
    model=DDP(TrainingModel(native),device_ids=[local],find_unused_parameters=False,
              gradient_as_bucket_view=True,broadcast_buffers=False)
    print(f'RANK {rank} DDP_READY',flush=True)
    groups=[{'params':[p for n,p in native.named_parameters() if p.requires_grad and n.startswith('qwen_vl_interface.')],
             'lr':1e-6,'name':'backbone'},
            {'params':[p for n,p in native.named_parameters() if p.requires_grad and n.startswith('action_model.')],
             'lr':1e-5,'name':'action_head'}]
    optimizer=ZeroRedundancyOptimizer(groups,optimizer_class=FP32MasterAdamW,
                                      parameters_as_bucket_view=args.speed_mode in ('buckets','buckets_no_checkpoint'),
                                      lr=1e-5,betas=(0.9,0.95),eps=1e-8,weight_decay=1e-8)
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:
        min(1.,(s+1)/300.)*(0.1+0.9*0.5*(1+math.cos(math.pi*max(0,s-300)/max(1,max_steps-300)))))
    scales=json.loads((ROOT/'control_scales.json').read_text())['scales']
    cfg=CSDRConfig(action_translation_scale=scales['action_translation'],
                   action_rotation_scale=scales['action_rotation'],action_gripper_scale=scales['action_gripper'])
    by_length=json.loads((ROOT/'control_scales.json').read_text())['by_length']
    cfg.prefix_scales=tuple(tuple(by_length[str(max(h,4))][key] for key in
        ('action_translation','action_rotation','action_gripper')) for h in range(1,17))
    step=0; first_epoch=1; start_micro=0; pending_rng=None
    if resume_dir:
        state=torch.load(resume_dir/f'rank_{rank}.pt',map_location='cpu',weights_only=False)
        assert state['args']['epochs']==args.epochs,'Resume must keep the schedule horizon'
        optimizer.optim.load_state_dict(state['optimizer'])
        for group,saved in zip(optimizer.param_groups,state['outer_param_groups']):
            group.update(saved)
        schedule.load_state_dict(state['scheduler'])
        step=state['step']; first_epoch=state['epoch']; start_micro=state['next_micro']; pending_rng=state['rng']
    if rank==0:
        (args.output/'run_config.json').write_text(json.dumps({**vars(args),'output':str(args.output),
             'max_steps':max_steps,'micro_batch':2,'grad_accum':8,'world_size':world,
             'effective_batch':160,'regularizer_final_decay':False},indent=2,default=str))
        import wandb
        wb=wandb.init(project='vla-repr-optimization',name='starvla-CSDR',dir=str(args.output),
                       mode=os.environ.get('WANDB_MODE','offline'),config={'epochs':args.epochs,'max_steps':max_steps,'csdr_ratio':args.csdr_ratio})
    model.train()
    began=time.monotonic()
    initial_step=step
    step_times=[]
    first_gradient_digest=None
    for epoch in range(first_epoch,args.epochs+1):
        micro_begin=start_micro if epoch==first_epoch else 0
        total_micro=summaries[epoch-1]['micro_batches']
        if micro_begin>=total_micro:
            continue
        dataset=PlannedDataset(epoch,rank,micro_begin)
        loader=DataLoader(dataset,batch_size=2,shuffle=False,collate_fn=collate,num_workers=args.workers,
                          persistent_workers=args.workers>0,prefetch_factor=4 if args.workers else None)
        iterator=iter(loader)
        if pending_rng:
            restore_rng(pending_rng); pending_rng=None
        saves={math.ceil(summaries[epoch-1]['optimizer_steps']*q/4) for q in (1,2,3,4)}
        epoch_step=micro_begin//8
        for group_start in range(micro_begin,total_micro,8):
            torch.cuda.synchronize()
            step_started=time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            group_end=min(group_start+8,total_micro)
            group_valid=int(dataset.group_counts[group_start:group_end].sum())
            metrics_acc={}
            metric_parts=[]
            collect_metrics=(args.speed_mode=='reference' or args.benchmark_steps or args.smoke_steps
                             or step+1<=10 or (step+1)%10==0)
            for micro in range(group_start,group_end):
                batch=next(iterator)
                valid=torch.tensor([x['loss_mask'] for x in batch],device=local,dtype=torch.float32)
                csdr_valid=torch.tensor([x['csdr_mask'] for x in batch],device=local,dtype=torch.bool)
                context=torch.tensor([[x['prompt_id']] for x in batch],device=local,dtype=torch.long)
                count=int(dataset.group_counts[micro])
                with model.no_sync() if micro+1<group_end else nullcontext():
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        per_sample,hidden,actions,error=model(batch)
                        local_sum=(per_sample*valid).sum()
                        task=local_sum*world/max(count,1)
                        global_task=local_sum.detach().clone()
                        dist.all_reduce(global_task)
                        global_task/=max(count,1)
                        reg=spatialvla_csdr_loss(hidden,actions,context,error,cfg,csdr_valid,
                                              compute_metrics=bool(collect_metrics),
                                              action_valid_lengths=torch.tensor([x['action_valid_length'] for x in batch],device=local))
                        raw=reg['csdr_order_loss']
                        ramp=min(1.,(step+1)/300.)
                        scale=(global_task*args.csdr_ratio*ramp/reg['csdr_budget_reference'].clamp_min(1e-8)).clamp(max=1.)
                        regularizer=raw*scale
                        loss=(task+regularizer)*count/max(group_valid,1)
                    loss.backward()
                weight=count/max(group_valid,1)
                values={'task_loss':global_task,'regularizer':regularizer.detach(),**reg}
                if args.speed_mode=='reference':
                    for key,value in values.items():
                        metrics_acc[key]=metrics_acc.get(key,0.)+float(value.detach())*weight
                elif rank==0 and collect_metrics:
                    metric_parts.append((weight,{key:value.detach() for key,value in values.items()}))
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            finite,save_now,pause_now=synchronize_control(grad_norm,rank,local,enabled=not args.benchmark_steps)
            if not finite:
                raise FloatingPointError(f'Non-finite gradient at continuation step {step+1}')
            if args.benchmark_steps and step==initial_step and rank==0:
                first_gradient_digest=model_digest(native,gradients=True)
            optimizer.step(); schedule.step(); step+=1; epoch_step+=1
            if metric_parts:
                keys=list(metric_parts[0][1])
                # One GPU-to-CPU transfer, then the original Python accumulation order.
                packed=torch.stack([torch.stack([values[k] for k in keys])
                                    for _,values in metric_parts]).cpu().tolist()
                for (weight,_),values in zip(metric_parts,packed):
                    for key,value in zip(keys,values):
                        metrics_acc[key]=metrics_acc.get(key,0.)+value*weight
            torch.cuda.synchronize()
            elapsed=torch.tensor(time.monotonic()-step_started,device=local,dtype=torch.float32)
            dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
            step_times.append(float(elapsed))
            metrics={**metrics_acc,'step':step,'released_step':45000+step,'epoch':epoch,
                     'epoch_fraction':(group_end/total_micro),'gradient_norm':float(grad_norm),
                     'seconds_per_step':(time.monotonic()-began)/max(1,step-initial_step),
                     'current_step_seconds':float(elapsed),
                     'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9,
                     'backbone_lr':optimizer.param_groups[0]['lr']}
            if rank==0 and (step<=10 or step%10==0 or args.benchmark_steps):
                with (args.output/'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(metrics)+'\n')
                wb.log(metrics,step=step)
                print(json.dumps(metrics),flush=True)
            if args.benchmark_steps and step-initial_step>=args.benchmark_steps:
                # Hash all model/gradient tensors on rank 0 and every optimizer
                # shard, plus RNG. No approximate comparison is auto-promoted.
                fingerprints={'optimizer':tree_digest(optimizer.optim.state_dict()),
                              'rng':tree_digest(rng_state()),'scheduler':tree_digest(schedule.state_dict())}
                if rank==0:
                    fingerprints.update(model=model_digest(native),
                                        first_gradient=first_gradient_digest,
                                        last_gradient=model_digest(native,gradients=True))
                result={'rank':rank,'mode':args.speed_mode,'fingerprints':fingerprints,
                        'step_times':step_times,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
                        'start_step':initial_step,'end_step':step,'benchmark_steps':args.benchmark_steps}
                (args.output/f'benchmark_rank_{rank}.json').write_text(json.dumps(result,indent=2))
                dist.barrier()
                if rank==0:wb.finish()
                dist.destroy_process_group()
                return
            if args.smoke_steps and step>=args.smoke_steps:
                save_started=time.monotonic()
                save_checkpoint(args.output,native,optimizer,schedule,step,epoch,group_end,args)
                if rank==0:
                    metrics['checkpoint_save_seconds']=time.monotonic()-save_started
                    stable=step_times[3:]
                    metrics['benchmark_step_seconds']=float(np.mean(stable)) if stable else float(np.mean(step_times))
                    metrics['benchmark_measured_steps']=len(stable)
                    metrics['epoch_optimizer_steps']=summaries[0]['optimizer_steps']
                    metrics['estimated_epoch_seconds']=metrics['benchmark_step_seconds']*summaries[0]['optimizer_steps']+4*metrics['checkpoint_save_seconds']
                    (args.output/'smoke_passed.json').write_text(json.dumps(metrics,indent=2))
                dist.destroy_process_group()
                return
            if epoch_step in saves:
                save_checkpoint(args.output,native,optimizer,schedule,step,epoch,group_end,args)
            if save_now:
                requested_save(save_checkpoint,args.output,native,optimizer,schedule,step,epoch,group_end,args,pause_now)
                if pause_now:
                    if rank==0:wb.finish()
                    dist.destroy_process_group()
                    return
        del iterator,loader
    if rank==0:
        (args.output/'training_complete.json').write_text(json.dumps({'steps':step,'epochs':args.epochs}))
        wb.finish()
    dist.barrier(); dist.destroy_process_group()

if __name__=='__main__':
    main()
