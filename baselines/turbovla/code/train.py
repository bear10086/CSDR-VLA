"""Native TurboVLA continuation with true-prefix CSDR and fully resumable epoch checkpoints."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from common import *
from data import native_dataset, dataset, collate, plan_action_counts
from csdr_loss_tail import CSDRConfig, spatialvla_csdr_loss
from checkpoint_utils import fingerprint_files
from turbovla.models.configuration import TurboVLAConfig
from turbovla.models.turbovla import TurboVLA


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch']); torch.cuda.set_rng_state(value['cuda'])


def save(output, model, optimizer, scheduler, step, epoch, next_micro, max_steps, plan_identity):
    folder = output / f'step_{step:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    tmp = folder / f'rank_{rank}.pt.tmp'
    torch.save({'rng': rng_state(), 'step': step, 'epoch': epoch, 'next_micro': next_micro,
                'max_steps': max_steps, 'world_size': dist.get_world_size(), 'plan_identity': plan_identity}, tmp)
    tmp.replace(folder / f'rank_{rank}.pt')
    if rank == 0:
        checkpoint = folder / f'turbovla_csdr_step_{step}.pth'
        tmp = checkpoint.with_suffix('.tmp')
        torch.save({'global_step': step, 'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
            'model_config': model.module.config.to_dict(), 'base_checkpoint': str(BASE)}, tmp)
        tmp.replace(checkpoint)
    dist.barrier()
    if rank == 0:
        atomic_json(folder / 'complete.json', {'step': step, 'epoch': epoch, 'next_micro': next_micro,
            'max_steps': max_steps, 'world_size': dist.get_world_size(), 'checkpoint': str(checkpoint), 'plan_identity': plan_identity})
        atomic_json(output / 'latest.json', {'directory': str(folder)})
    dist.barrier()
    return folder


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT / 'train')
    p.add_argument('--resume-from', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int, default=0)
    p.add_argument('--raw-cache', action='store_true', help='Use verified lossless raw images; native processing is unchanged.')
    args = p.parse_args()
    if args.raw_cache:
        assert json.loads((ROOT / 'cache_speed_validation.json').read_text())['passed']
        from cache_speed import cached_dataset
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ['LOCAL_RANK'])
    assert world == 10
    torch.cuda.set_device(local); torch.set_num_threads(2)
    random.seed(7 + rank); np.random.seed(7 + rank); torch.manual_seed(7 + rank)
    torch.set_float32_matmul_precision('high')
    plans = [json.loads((ROOT / f'plans/epoch_seed{s}.summary.json').read_text()) for s in (7,8,9)]
    max_steps = sum(x['optimizer_steps'] for x in plans)
    plan_identity = fingerprint_files([ROOT/'trajectories.jsonl', *[ROOT/f'plans/epoch_seed{s}.npz' for s in (7,8,9)]])
    args.output.mkdir(parents=True, exist_ok=True)
    resume = args.resume_from
    if args.resume and (args.output / 'latest.json').exists():
        resume = Path(json.loads((args.output / 'latest.json').read_text())['directory'])
    meta = json.loads((resume / 'complete.json').read_text()) if resume else None
    if meta:
        assert meta['world_size'] == world and meta['max_steps'] == max_steps
        if meta.get('plan_identity') != plan_identity:
            raise ValueError('Resume requires a checkpoint with matching data-plan identity; legacy or changed plans cannot be silently resumed.')
    state = torch.load(meta['checkpoint'] if meta else BASE, map_location='cpu', weights_only=False)
    config = TurboVLAConfig.from_mapping(state['model_config'])
    assert config.action.horizon == 12
    config.text.model_name_or_path = BERT; config.text.local_files_only = True
    config.vision.model_name_or_path = DINO; config.vision.local_files_only = True
    native = TurboVLA(config)
    native.load_state_dict(state['model_state_dict'], strict=True)
    model = DDP(native.to(local), device_ids=[local], find_unused_parameters=False, gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=1e-5, weight_decay=1e-10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        min(1., (s+1)/300) * (.1 + .9*.5*(1+math.cos(math.pi*max(0,s-300)/max(1,max_steps-300)))))
    step, first_epoch, next_micro = 0, 1, 0
    pending_rng = None
    if meta:
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        recovery = torch.load(resume / f'rank_{rank}.pt', map_location='cpu', weights_only=False)
        assert all(recovery[k] == meta[k] for k in ('step','epoch','next_micro','max_steps','world_size','plan_identity'))
        pending_rng = recovery['rng']
        step, first_epoch, next_micro = meta['step'], meta['epoch'], meta['next_micro']
    del state
    table = json.loads((ROOT / 'control_scales.json').read_text())['by_length']
    cfg = CSDRConfig(prefix_scales=tuple(tuple(table[str(max(4,h))][k] for k in
        ('action_translation','action_rotation','action_gripper')) for h in range(1,13)))
    source = native_dataset(rank, world)
    run_config = {'max_steps': max_steps, 'epochs': 3, 'micro_batch': 4, 'grad_accum': 3,
        'world_size': world, 'csdr_ratio_cap': .005, 'regularizer_warmup': 300, 'final_decay': False,
        'base_checkpoint': str(BASE), 'resume_from': str(resume) if resume else None,
        'raw_cache': args.raw_cache}
    wb = None
    if rank == 0:
        atomic_json(args.output / 'run_config.json', run_config)
        try:
            import wandb
            wb = wandb.init(project='vla-repr-optimization', name='turbovla-CSDR',
                            dir=str(args.output), mode=os.environ.get('WANDB_MODE','disabled'), config=run_config)
        except ImportError:
            print('wandb unavailable; metrics.jsonl remains authoritative', flush=True)
    model.train()
    step_times = []
    for epoch in range(first_epoch, 4):
        begin = next_micro if epoch == first_epoch else 0
        counts = plan_action_counts(epoch)
        if not 0 <= begin <= len(counts) or (begin != len(counts) and begin % 3):
            raise ValueError('Resume cursor must be an optimizer-step boundary in the saved epoch.')
        if begin >= len(counts):
            continue
        ds = (cached_dataset if args.raw_cache else dataset)(source, epoch, rank, begin)
        iterator = iter(DataLoader(ds, batch_size=4, collate_fn=collate, num_workers=0))
        if pending_rng:
            restore_rng(pending_rng); pending_rng = None
        for start in range(begin, len(counts), 3):
            started = time.monotonic()
            end = min(start + 3, len(counts))
            group_actions = int(counts[start:end].sum())
            optimizer.zero_grad(set_to_none=True)
            metrics = {}
            data_wait_seconds = 0.0
            for micro in range(start, end):
                data_started = time.monotonic()
                b = next(iterator)
                data_wait_seconds += time.monotonic() - data_started
                samples = {k: v.to(local, non_blocking=True) for k, v in b['samples'].items()}
                actions, states = b['actions'].to(local), b['states'].to(local)
                masks = b['action_masks'].to(local) * b['sample_loss_mask'].to(local)[:, None]
                lengths = b['lengths'].to(local)
                with model.no_sync() if micro + 1 < end else nullcontext():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        predictions, hidden = model(b['instructions'], samples, states, return_action_hidden=True)
                        error = (predictions.float() - actions.float()).abs()
                        local_sum = (error * masks[:, :, None]).sum()
                        per_sample = (error * masks[:, :, None]).sum((1,2)) / (masks.sum(1)*7).clamp_min(1)
                        # Exact native mean over all valid action elements in the distributed accumulated batch.
                        task = local_sum * world / max(group_actions * 7, 1)
                        total = local_sum.detach().clone()
                        dist.all_reduce(total)
                        global_task = total / max(int(counts[micro]) * 7, 1)
                        reg = spatialvla_csdr_loss(hidden, actions, b['prompt_ids'].to(local), per_sample.detach(), cfg,
                            b['csdr_sample_mask'].to(local) & b['sample_loss_mask'].to(local).bool(),
                            action_valid_lengths=lengths, compute_metrics=True)
                        scale = (global_task * .005 * min(1.,(step+1)/300) /
                                 reg['csdr_budget_reference'].clamp_min(1e-8)).clamp(max=1)
                        contribution = reg['csdr_order_loss'] * scale
                        weight = int(counts[micro]) / max(group_actions,1)
                        loss = task + contribution * weight
                    loss.backward()
                for k, value in {'task_loss': global_task, 'regularizer': contribution, **reg}.items():
                    metrics[k] = metrics.get(k, 0) + value.detach() * weight
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            finite = torch.isfinite(norm).to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(f'Nonfinite gradient at {step+1}')
            optimizer.step(); scheduler.step(); step += 1
            torch.cuda.synchronize()
            elapsed = torch.tensor(time.monotonic() - started, device=local)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            step_times.append(float(elapsed))
            if rank == 0 and (step <= 10 or step % 10 == 0 or args.stop_after):
                keys = list(metrics)
                values = torch.stack([metrics[k] for k in keys]).cpu().tolist()
                row = dict(zip(keys, values))
                row.update(step=step, epoch=epoch, epoch_fraction=end/len(counts),
                    max_steps=max_steps, current_step_seconds=float(elapsed), gradient_norm=float(norm),
                    data_wait_seconds=data_wait_seconds,
                    peak_memory_gb=torch.cuda.max_memory_allocated()/1e9, lr=optimizer.param_groups[0]['lr'])
                with (args.output / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                if wb: wb.log(row, step=step)
                print(json.dumps(row), flush=True)
            stop = args.stop_after and step >= args.stop_after
            pause = torch.tensor(int((args.output / 'pause_requested').exists()) if rank == 0 else 0, device=local)
            dist.broadcast(pause, src=0)
            pause = bool(pause.item())
            if end == len(counts) or stop or pause:
                folder = save(args.output, model, optimizer, scheduler, step, epoch, end, max_steps, plan_identity)
                if stop or pause:
                    if rank == 0:
                        atomic_json(args.output / ('training_paused.json' if pause else 'preflight_passed.json'), {'passed': True, 'step': step,
                            'checkpoint': str(folder), 'resume_exercised': bool(meta),
                            'mean_step_seconds': float(np.mean(step_times[2:] or step_times))})
                    ds.close()
                    if wb: wb.finish()
                    dist.destroy_process_group()
                    return
        ds.close()
        del iterator, ds
    if rank == 0:
        atomic_json(args.output / 'training_complete.json', {'steps': step, 'epochs': 3})
        if wb: wb.finish()
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
