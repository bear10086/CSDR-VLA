"""Official WidowX environment arguments and native SpatialVLA action adapter."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--task',choices=('carrot','spoon','stack','eggplant'),required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    import torch
    import tensorflow as tf
    from csdr_paths import PROJECT
    from simpler_env.evaluation.argparse import get_args
    from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
    from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference
    os.environ['DISPLAY']='';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);tf.random.set_seed(a.seed)
    gpus=tf.config.list_physical_devices('GPU')
    if gpus:tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
    names=dict(carrot='PutCarrotOnPlateInScene-v0',spoon='PutSpoonOnTableClothInScene-v0',stack='StackGreenCubeOnYellowCubeBakedTexInScene-v0',eggplant='PutEggplantInBasketScene-v0')
    egg=a.task=='eggplant';x,y=('0.127','0.06') if egg else ('0.147','0.028')
    overlay=PROJECT/'common/simpler/ManiSkill2_real2sim/data/real_inpainting'/('bridge_sink.png' if egg else 'bridge_real_eval_1.png')
    sys.argv=[sys.argv[0],'--policy-model','spatialvla','--ckpt-path',a.checkpoint,
        '--action-ensemble-temp','-0.8','--logging-dir',str(a.output.with_suffix('')),
        '--robot','widowx_sink_camera_setup' if egg else 'widowx','--policy-setup','widowx_bridge',
        '--control-freq','5','--sim-freq','500','--max-episode-steps','120' if egg else '60',
        '--env-name',names[a.task],'--scene-name','bridge_table_1_v2' if egg else 'bridge_table_1_v1',
        '--rgb-overlay-path',str(overlay),'--robot-init-x',x,x,'1','--robot-init-y',y,y,'1',
        '--obj-variation-mode','episode','--obj-episode-range','0','24',
        '--robot-init-rot-quat-center','0','0','0','1',
        '--robot-init-rot-rpy-range','0','0','1','0','0','1','0','0','1']
    args=get_args()
    model=SpatialVLAInference(saved_model_path=a.checkpoint,policy_setup=args.policy_setup,action_scale=args.action_scale,action_ensemble_temp=-.8)
    success=[bool(x) for x in maniskill2_evaluator(model,args)]
    assert len(success)==24
    a.output.write_text(json.dumps(dict(checkpoint=a.checkpoint,task=a.task,seed=a.seed,episodes=len(success),successes=sum(success),success_rate=float(np.mean(success)),episode_success=success),indent=2))

if __name__=='__main__':main()
