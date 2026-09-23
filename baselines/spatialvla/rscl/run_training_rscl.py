"""Original Trainer pathway, with progress and hardware measurements only."""
import json
import os
from pathlib import Path
import sys
import time

# SpatialVLA's original CSDR module is imported during model setup.
# RSCL is implemented in a separate loss path, so keep the original module in a valid compatibility mode.
os.environ['CSDR_LOSS_CONTROL'] = 'full'
os.environ['ACTION_INFONCE_TEMPERATURE'] = '0.2'

RSCL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RSCL_ROOT.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'baselines' / 'spatialvla' / 'vendor'))
sys.path.insert(0, str(PROJECT_ROOT / 'common'))
sys.path.insert(0, str(RSCL_ROOT))
from train import spatialvla_finetune as native
from transformers import Trainer, TrainerCallback
from rscl_checkpoint import AuxiliaryStateCallback

class Progress(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.previous = (time.monotonic(), state.global_step)
    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or state.global_step % 10:
            return
        import torch
        now = time.monotonic()
        value = dict(variant='rscl_trainable', step=state.global_step,
                     max_steps=state.max_steps, wall_time=time.time(),
                     recent_seconds_per_step=(now-self.previous[0])/max(state.global_step-self.previous[1],1),
                     peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
                     peak_reserved_gb=torch.cuda.max_memory_reserved()/2**30)
        path = Path(args.output_dir)/'progress.json'
        tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)
        self.previous = (now, state.global_step)

original_init = Trainer.__init__
def trainer_init(self, *args, **kwargs):
    kwargs['callbacks'] = list(kwargs.get('callbacks') or []) + [Progress(), AuxiliaryStateCallback()]
    original_init(self, *args, **kwargs)
Trainer.__init__ = trainer_init
native.main()
