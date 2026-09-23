"""Portable paths for final TurboVLA training and native LIBERO evaluation."""
import os
from pathlib import Path
from csdr_paths import PROJECT, path, run_dir, code_dir, vendor_dir, python, environment as _env

ROOT = run_dir('turbovla')
CODE = code_dir('turbovla')
REPO = vendor_dir('turbovla')
OLD_PLAN = path('work_dir') / 'plans/libero'
DATA_ROOT = path('libero_rlds')
PYTHON = python('turbovla')
BASE = path('turbovla_weights')
DINO = str(path('dinov3'))
BERT = str(path('bert'))
STATS = REPO / 'experiments/libero/configs/libero_all4_stats.json'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
LIBERO = str(path('libero_repo'))

def environment(gpus=''):
    env = _env('turbovla', gpus)
    env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    return env

def atomic_json(path, data):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)
