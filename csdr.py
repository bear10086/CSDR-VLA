#!/usr/bin/env python3
"""Portable launch recipes for the four final CSDR implementations (Linux)."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from csdr_paths import PROJECT, path, python, code_dir, vendor_dir, run_dir, environment

MODELS = ('openvla_oft', 'turbovla', 'spatialvla', 'starvla')
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')

def execute(command, model, dry=False, env=None, cwd=None):
    command = list(map(str, command))
    print(shlex.join(command), flush=True)
    if not dry:
        subprocess.run(command, check=True, env=env or environment(model), cwd=cwd or PROJECT)

def pairs(options):
    return [str(v) for k, value in options.items() for v in ('--'+k, value)]

def initialize(model):
    root = run_dir(model)
    root.mkdir(parents=True, exist_ok=True)
    if model in ('openvla_oft', 'turbovla'):
        config = path('work_dir')/'libero_config'
        config.mkdir(parents=True, exist_ok=True)
        lib = path('libero_repo')/'libero/libero'
        # JSON is also valid YAML. Avoid LIBERO's interactive first-run prompt.
        (config/'config.yaml').write_text(json.dumps(dict(
            benchmark_root=str(lib), bddl_files=str(lib/'bddl_files'),
            init_states=str(lib/'init_files'), datasets=str(path('libero_rlds')),
            assets=str(lib/'assets')), indent=2))
    if model == 'starvla':
        baseline = root/'baseline'
        baseline.mkdir(exist_ok=True)
        ckpt = path('starvla_weights')/'checkpoints/steps_45000_pytorch_model.pt'
        if not ckpt.is_file(): raise FileNotFoundError(ckpt)
        (baseline/'checkpoints').mkdir(exist_ok=True)
        link = baseline/'checkpoints'/ckpt.name
        if not link.exists(): link.symlink_to(ckpt)
        source = PROJECT/'baselines/starvla/config'
        # Only paths change; architecture and action normalization remain intact.
        text = (source/'config.yaml').read_text()
        import re
        text = re.sub(r'(?m)^(\s*base_vlm:) .+$', lambda m:m[1]+' '+json.dumps(str(path('starvla_backbone'))), text)
        text = re.sub(r'(?m)^(\s*data_root_dir:) .+$', lambda m:m[1]+' '+json.dumps(str(path('bridge_lerobot').parent)), text)
        (baseline/'config.yaml').write_text(text)
        shutil.copyfile(source/'dataset_statistics.json', baseline/'dataset_statistics.json')

def prepare_rlds(model, dry):
    libero = model in ('openvla_oft','turbovla')
    out = path('work_dir')/'plans'/('libero' if libero else 'bridge')
    scripts = PROJECT/'common/cohort_planning'
    data = {s+'_no_noops':path('libero_rlds')/(s+'_no_noops')/'1.0.0' for s in SUITES} if libero else {'bridge_oxe':path('bridge_rlds_root')/'bridge_oxe/0.1.0'}
    if not dry: (out/'plans').mkdir(parents=True, exist_ok=True)
    for name, folder in data.items():
        execute([python(model), scripts/'build_tfrecord_index.py', '--dataset-dir',folder,'--output',out/(name+'_tfrecord_index.json')],model,dry)
    command = [python(model),scripts/'build_rlds_manifest.py','--output-dir',out]
    for name, folder in data.items(): command += ['--dataset',f'{name}={folder}','--trim-steps-dataset',f'{name}={7 if libero else 2}']
    execute(command,model,dry)
    if not libero:
        execute([python(model),scripts/'cluster_prompt_catalog.py','--catalog',out/'prompt_catalog.json','--output-dir',out,'--cosine-threshold','0.94'],model,dry)
    for seed in ((7,8,9) if libero else (7,)):
        command = [python(model),scripts/'build_batch_plan.py','--manifest',out/'trajectories.jsonl','--output',out/f'plans/epoch_seed{seed}.npz','--world-size','10','--per-device-batch-size',str(4 if libero else 8),'--cohort-count','5','--seed',str(seed)]
        if not libero: command += ['--canonical-map',out/'canonical_prompt_map.json']
        execute(command,model,dry)

def prepare(model,dry):
    if not dry: initialize(model)
    if model != 'starvla': prepare_rlds(model,dry)
    if model == 'starvla':
        execute([python(model),code_dir(model)/'prepare_data.py','--epochs','1'],model,dry)
    elif model == 'turbovla':
        execute([python(model),code_dir(model)/'prepare.py'],model,dry)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i:execute([python(model),code_dir(model)/'cache_speed.py','--shard',i,'--shards',4],model,dry),range(4)))
        execute([python(model),code_dir(model)/'cache_speed.py','--verify'],model,dry)

def train(model,dry,resume=None):
    if not dry: initialize(model)
    root, code = run_dir(model), code_dir(model)
    prefix = [python(model),'-m','torch.distributed.run','--standalone','--nproc_per_node=10']
    if model == 'openvla_oft':
        base=path(model+'_weights')
        opts=dict(vla_path=base,data_root_dir=path('libero_rlds'),dataset_name='libero_4_task_suites_no_noops',
            planned_cohort_data_dir=path('work_dir')/'plans/libero',planned_epoch_count=3,planned_episode_cache_size=16,
            run_root_dir=root/'checkpoints',run_id_override='csdr',num_images_in_input=2,use_proprio=True,
            batch_size=4,grad_accumulation_steps=3,learning_rate=5e-5,num_steps_before_decay=1000000,
            max_steps=6543,save_freq=2181,save_latest_checkpoint_only=False,image_aug=True,lora_rank=32,
            skip_checkpoint_sync=True,warm_start_components=True,component_checkpoint_dir=base,
            component_checkpoint_step=300000,log_step_offset=300000,representation_layer=24,seed=7,
            csdr_order_weight=1.,csdr_order_to_task_ratio=.005,csdr_warmup_steps=300,
            csdr_fixed_scale_path=PROJECT/'baselines/openvla_oft/config/control_scales.json',
            csdr_synchronized_prompt_batches=False,wandb_project='CSDR',wandb_run_name='OpenVLA-OFT')
        if resume:
            step=int(resume.name.split('--')[-1].split('_')[0])
            opts.update(resume_training_state=True,training_state_path=resume/f'training_state--{step}_checkpoint.pt',resume_lora_adapter_dir=resume/'lora_adapter',vla_path=base,component_checkpoint_dir=resume,component_checkpoint_step=step)
        execute(prefix+[code/'finetune.py']+pairs(opts),model,dry,cwd=vendor_dir(model))
    elif model == 'spatialvla':
        opts=dict(model_name_or_path=path('spatialvla_weights'),data_root_dir=path('bridge_rlds_root'),data_mix='bridge_oxe_csdr',
            planned_cohort_data_dir=path('work_dir')/'plans/bridge',planned_epoch_count=1,planned_episode_cache_size=16,
            planned_prefetch_size=64,planned_decode_workers=4,output_dir=root/'checkpoints',do_train=True,csdr=True,
            csdr_order_to_task_ratio=.005,csdr_warmup_steps=300,csdr_action_translation_scale=.502,
            csdr_action_rotation_scale=.517,csdr_action_gripper_scale=.676,action_forward_steps=3,obs_backward_steps=0,
            use_raw_dataloader=True,dataloader_num_workers=0,per_device_train_batch_size=8,gradient_accumulation_steps=4,
            max_steps=2542,learning_rate=5e-5,lr_scheduler_type='constant',warmup_steps=0,optim='adamw_torch_fused',
            weight_decay=0.,lora=32,lora_alpha=32,lora_target='linear',grad_checkpoint=False,flash_attn=True,bf16=True,tf32=True,
            logging_steps=10,save_strategy='steps',save_steps=2542,save_total_limit=3,save_safetensors=True,
            remove_unused_columns=False,ddp_find_unused_parameters=True,report_to='tensorboard',run_name='SpatialVLA-CSDR')
        if resume:
            state=json.loads((resume/'trainer_state.json').read_text())
            if int(state['global_step']) >= opts['max_steps']:
                raise ValueError('SpatialVLA has reached the 2542-update budget; use evaluate instead of resume.')
            opts['resume_from_checkpoint']=resume
        env=environment(model);env['LAUNCHER']='pytorch'
        execute(prefix+[vendor_dir(model)/'train/spatialvla_finetune.py']+pairs(opts),model,dry,env,cwd=vendor_dir(model))
    else:
        command=prefix+[code/('train_speed.py' if model=='starvla' else 'train.py')]
        command += ['--epochs','1','--speed-mode','no_checkpoint'] if model=='starvla' else ['--raw-cache']
        if resume: command+=['--resume-from',resume]
        execute(command,model,dry)

def checkpoints(model):
    root=run_dir(model)
    if model=='openvla_oft': return sorted((root/'checkpoints').glob('csdr--*_chkpt'))
    if model=='spatialvla':
        checkpoint=root/'checkpoints/checkpoint-2542'
        return [checkpoint] if checkpoint.is_dir() else []
    return [Path(json.loads(p.read_text())['checkpoint']) for p in sorted((root/'train').glob('step_*/complete.json'))]

def evaluate(model,ckpt,dry):
    if not dry: initialize(model)
    if not ckpt: ckpt=path(model+'_weights')
    if model=='starvla' and ckpt.is_dir():
        if ckpt.resolve()==path('starvla_weights').resolve():
            ckpt=run_dir(model)/'baseline/checkpoints/steps_45000_pytorch_model.pt'
        elif (ckpt/'complete.json').is_file():
            ckpt=Path(json.loads((ckpt/'complete.json').read_text())['checkpoint'])
        else:
            raise ValueError('For StarVLA, supply the checkpoint .pt file or a completed step directory')
    out=run_dir(model)/'eval'/ckpt.name
    execute([python(model),PROJECT/'evaluate.py',model,'--checkpoint',ckpt,'--output',out],model,dry)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','train','evaluate','pipeline','infer'))
    p.add_argument('model',choices=MODELS)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--dry-run',action='store_true')
    args, extra=p.parse_known_args()
    if extra and args.action!='infer':p.error('Unknown arguments: '+' '.join(extra))
    if args.action=='prepare':prepare(args.model,args.dry_run)
    elif args.action in ('train','pipeline'):
        train(args.model,args.dry_run,args.resume)
        if args.action=='pipeline':
            if args.dry_run:
                print('After successful training: evaluate checkpoint-2542 only.' if args.model=='spatialvla'
                      else 'After successful training: evaluate each completed checkpoint.')
            else:
                found=checkpoints(args.model)
                if not found:raise RuntimeError('No completed checkpoint found')
                for ckpt in found:evaluate(args.model,ckpt,False)
    elif args.action=='evaluate':evaluate(args.model,args.checkpoint,args.dry_run)
    else:
        if not args.dry_run:initialize(args.model)
        command=[python('spatialvla_eval' if args.model=='spatialvla' else args.model),PROJECT/'infer.py',args.model]
        if args.checkpoint:command+=['--checkpoint',args.checkpoint]
        execute(command+extra,args.model,args.dry_run)

if __name__=='__main__':main()
