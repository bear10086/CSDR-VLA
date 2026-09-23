"""Multi-GPU scheduling; rollout logic remains in each baseline's evaluator."""
import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
from pathlib import Path
from csdr_paths import *
from checkpoint_utils import checkpoint_identity, cached_merge, fingerprint_files

def launch(command, model, gpu, log, cwd=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    env=environment(model,str(gpu))
    env['VK_ICD_FILENAMES']=env.get('VK_ICD_FILENAMES','/etc/vulkan/icd.d/nvidia_icd.json')
    env['HF_MODULES_CACHE']=str(log.parent/(log.stem+'_hf_modules'))
    with log.open('w') as f:
        subprocess.run(list(map(str,command)),check=True,env=env,cwd=cwd or PROJECT,stdout=f,stderr=subprocess.STDOUT)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',choices=('openvla_oft','spatialvla','starvla','turbovla'))
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7,8,9')
    args=p.parse_args();model=args.model;ckpt=args.checkpoint.resolve();out=args.output.resolve()
    gpus=[int(x) for x in args.gpus.split(',')]
    out.mkdir(parents=True,exist_ok=True)
    marker=out/'evaluation_config.json'
    config={'model':model,'checkpoint':str(ckpt),'seeds':[7] if model in ('turbovla','openvla_oft') else [7,8,9,10]}
    merge_identity = None
    if model == 'spatialvla':
        config['checkpoint_identity'] = checkpoint_identity(ckpt)
        if (ckpt/'adapter_config.json').exists():
            merge_identity = dict(adapter=config['checkpoint_identity'],
                base=checkpoint_identity(path('spatialvla_weights')),
                merge_code=fingerprint_files([code_dir(model)/'merge_lora_checkpoint.py']))
            config['merge_identity'] = merge_identity
    if marker.exists() and json.loads(marker.read_text())!=config:raise ValueError('Output directory belongs to another evaluation')
    marker.write_text(json.dumps(config,indent=2))
    sys.path.insert(0,str(code_dir(model)));sys.path.insert(0,str(vendor_dir(model)))
    if model=='starvla':
        import importlib.util
        spec=importlib.util.spec_from_file_location('native_evaluator',code_dir(model)/'evaluate.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.evaluate(ckpt,out,gpus);return
    if model=='turbovla':
        if gpus!=list(range(10)):raise ValueError('Final TurboVLA scheduler uses GPU IDs 0..9')
        import importlib.util
        spec=importlib.util.spec_from_file_location('native_evaluator',code_dir(model)/'evaluate.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.evaluate(ckpt,out);return
    if model=='spatialvla' and (ckpt/'adapter_config.json').exists():
        def merge(merged):
            subprocess.run([python(model),str(code_dir(model)/'merge_lora_checkpoint.py'),'--base-model',str(path('spatialvla_weights')),'--adapter',str(ckpt),'--output',str(merged)],check=True,env=environment(model))
            if checkpoint_identity(ckpt) != merge_identity['adapter'] or checkpoint_identity(path('spatialvla_weights')) != merge_identity['base']:
                raise RuntimeError('Merge inputs changed during merging; refusing publication')
        ckpt=cached_merge(run_dir(model)/'merged', merge_identity, merge)
    config['loaded_checkpoint'] = str(ckpt)
    # Keep the request marker stable; save actual model provenance separately.
    (out/'loaded_model.json').write_text(json.dumps(config,indent=2))
    if model=='openvla_oft':
        jobs=[(suite,task) for suite in ('libero_spatial','libero_object','libero_goal','libero_10') for task in range(10)]
    else:jobs=[(task,seed) for seed in (7,8,9,10) for task in ('carrot','spoon','stack','eggplant')]
    def worker(slot):
        results=[]
        for first,second in jobs[slot::len(gpus)]:
            result=out/f'{first}_{second}.json'
            if not result.exists():
                if model=='openvla_oft':
                    command=[python(model),vendor_dir(model)/'scripts/eval_libero_single_task.py',
                        '--pretrained_checkpoint',ckpt,'--task_suite_name',first,'--task_id',second,
                        '--num_trials_per_task',50,'--seed',7,'--local_log_dir',out/'logs',
                        '--summary_path',result,'--run_id_note','CSDR']
                else:
                    command=[python('spatialvla_eval'),PROJECT/'spatial_eval_worker.py',
                        '--checkpoint',ckpt,'--task',first,'--seed',second,'--output',result]
                launch(command,model,gpus[slot],result.with_suffix('.log'))
            value=json.loads(result.read_text())
            assert value['episodes']==(50 if model=='openvla_oft' else 24)
            results.append(dict(group=first,**value))
        return results
    with cf.ThreadPoolExecutor(max_workers=len(gpus)) as pool: rows=sum(pool.map(worker,range(len(gpus))),[])
    total=sum(r['episodes'] for r in rows);success=sum(r['successes'] for r in rows)
    summary=dict(checkpoint=str(ckpt),episodes=total,successes=success,success_rate=success/total,runs=rows)
    summary['model_provenance'] = config
    summary['per_group']={name:sum(x['successes'] for x in rows if x['group']==name)/sum(x['episodes'] for x in rows if x['group']==name) for name in sorted({x['group'] for x in rows})}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
