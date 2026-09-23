"""Portable paths for the final StarVLA Bridge model."""
from pathlib import Path
import sys
from csdr_paths import PROJECT, path, run_dir, code_dir, vendor_dir, python, environment as _env

ROOT = run_dir('starvla')
CODE = code_dir('starvla')
RUNTIME = vendor_dir('starvla')
BASE = path('starvla_weights')
BACKBONE = path('starvla_backbone')
PYTHON = python('starvla')
SIM_PYTHON = python('starvla_eval')
SIM_ROOT = PROJECT / 'common/simpler'
DATA = path('bridge_lerobot')

def register_framework():
    import importlib
    sys.path.insert(0, str(RUNTIME))
    base = importlib.import_module('starVLA.model.framework.base_framework')
    importlib.import_module('starVLA.model.framework.VLM4A.QwenGR00T')
    base._FRAMEWORKS_IMPORTED = True
    return base.baseframework

def environment(gpu=None):
    env = _env('starvla', gpu)
    env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
               XLA_PYTHON_CLIENT_PREALLOCATE='false')
    sim_lib = str(Path(SIM_PYTHON).parent.parent / 'lib')
    env['LD_LIBRARY_PATH'] = sim_lib + ':' + env.get('LD_LIBRARY_PATH','')
    return env
