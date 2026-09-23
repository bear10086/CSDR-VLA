"""Single-observation inference through the same native adapters as evaluation.

LIBERO images must already have the same orientation as the evaluator's rotated
camera images. No robot connection or hardware command is issued by this script.
"""
import argparse
import json
import sys
import subprocess
from pathlib import Path
from csdr_paths import *

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',choices=('openvla_oft','turbovla','spatialvla','starvla'))
    p.add_argument('--checkpoint',type=Path);p.add_argument('--image',type=Path,required=True)
    p.add_argument('--wrist',type=Path);p.add_argument('--state',type=Path,help='JSON array: xyz, axis-angle (3), two gripper positions')
    p.add_argument('--instruction',required=True);p.add_argument('--suite',default='libero_spatial')
    p.add_argument('--output',type=Path,default=Path('actions.json'))
    a=p.parse_args()
    import numpy as np
    from PIL import Image
    ckpt=a.checkpoint or path(a.model+'_weights')
    sys.path.insert(0,str(vendor_dir(a.model)));sys.path.insert(0,str(code_dir(a.model)))
    image=np.asarray(Image.open(a.image).convert('RGB'))
    if a.model in ('openvla_oft','turbovla'):
        if not a.wrist or not a.state:p.error('LIBERO policies require --wrist and --state')
        wrist=np.asarray(Image.open(a.wrist).convert('RGB'));state=np.asarray(json.loads(a.state.read_text()),dtype=np.float32)
        if state.shape!=(8,):p.error('--state must contain 8 values')
    if a.model=='turbovla':
        from turbovla.evaluation.suite_policy import TurboVLAPolicy
        model=TurboVLAPolicy(str(ckpt),dinov3_path=str(path('dinov3')),bert_path=str(path('bert')),stats_path=vendor_dir(a.model)/'experiments/libero/configs/libero_all4_stats.json',stats_key='libero_all4_no_noops')
        value=model.predict_env_action_chunk(image,wrist,a.instruction,state).tolist()
    elif a.model=='openvla_oft':
        from experiments.robot.libero import run_libero_eval as lib
        import experiments.robot.openvla_utils as util
        util.model_is_on_hf_hub=lambda _:False;util.update_auto_map=lambda _:None;util.check_model_logic_mismatch=lambda _:None
        cfg=lib.GenerateConfig(pretrained_checkpoint=str(ckpt),use_l1_regression=True,use_diffusion=False,use_discrete_diffusion=False,num_images_in_input=2,use_proprio=True,center_crop=True,task_suite_name=a.suite,use_wandb=False)
        model,head,proprio,noisy,processor=lib.initialize_model(cfg)
        size=lib.get_image_resize_size(cfg)
        obs=dict(full_image=lib.resize_image_for_policy(image,size),wrist_image=lib.resize_image_for_policy(wrist,size),state=state)
        actions=lib.get_action(cfg,model,obs,a.instruction,processor,head,proprio,noisy)
        value=[lib.process_action(np.asarray(row),cfg.model_family).tolist() for row in actions]
    elif a.model=='spatialvla':
        if (ckpt/'adapter_config.json').is_file():
            merged=run_dir(a.model)/'merged'/ckpt.name
            if not (merged/'merge_complete.json').exists():
                subprocess.run([python(a.model),str(code_dir(a.model)/'merge_lora_checkpoint.py'),
                    '--base-model',str(path('spatialvla_weights')),'--adapter',str(ckpt),
                    '--output',str(merged)],check=True,env=environment(a.model))
                (merged/'merge_complete.json').write_text(json.dumps({'adapter':str(ckpt.resolve())}))
            ckpt=merged
        from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference
        model=SpatialVLAInference(saved_model_path=str(ckpt),policy_setup='widowx_bridge')
        raw,action=model.step(image,a.instruction)
        value={k:np.asarray(v).tolist() for k,v in action.items()}
    else:
        from common import register_framework
        register_framework()
        if ckpt.is_dir():
            if ckpt.resolve()==path('starvla_weights').resolve():
                ckpt=run_dir('starvla')/'baseline/checkpoints/steps_45000_pytorch_model.pt'
            elif (ckpt/'complete.json').is_file():
                ckpt=Path(json.loads((ckpt/'complete.json').read_text())['checkpoint'])
            else:
                raise ValueError('Supply the StarVLA .pt file or a completed step directory')
        from deployment.model_server.policy_wrapper import PolicyServerWrapper
        model=PolicyServerWrapper(str(ckpt),device='cuda',use_bf16=True,unnorm_key='oxe_bridge')
        value=model.predict_action([dict(image=[Image.fromarray(image)],lang=a.instruction)],unnorm_key='oxe_bridge')['actions'].tolist()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'model':a.model,'actions':value},indent=2))
    print(a.output)

if __name__=='__main__':main()
