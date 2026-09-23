"""All configured paths are relative to this release, not the current shell."""
from pathlib import Path
import json, os

PROJECT = Path(__file__).resolve().parent
CONFIG = Path(os.environ.get('CSDR_PATHS', PROJECT/'config/paths.json')).resolve()
SETTINGS = json.loads(CONFIG.read_text())

def path(key):
    value = Path(os.path.expandvars(SETTINGS[key])).expanduser()
    # Preserve virtual-environment interpreter symlinks: resolving bin/python
    # into the system executable would silently select the wrong environment.
    return Path(os.path.abspath(value if value.is_absolute() else PROJECT/value))

def run_dir(model): return path('work_dir')/model
def code_dir(model): return PROJECT/'baselines'/model/'code'
def vendor_dir(model): return PROJECT/'baselines'/model/'vendor'
def python(model): return str(path('python_'+model))

def environment(model, gpus=None):
    env = os.environ.copy()
    paths = [PROJECT, PROJECT/'common', PROJECT/'common/cohort_planning', code_dir(model), vendor_dir(model), path('libero_repo')]
    if model == 'turbovla': paths += [vendor_dir(model)/'third_party/vla_adapter']
    if model in ('starvla','spatialvla'): paths += [PROJECT/'common/simpler', PROJECT/'common/simpler/ManiSkill2_real2sim']
    env.update(PYTHONPATH=os.pathsep.join(map(str, paths)), PYTHONUNBUFFERED='1',
        TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled', OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', TF_NUM_INTRAOP_THREADS='2',
        TF_NUM_INTEROP_THREADS='1', TF_CPP_MIN_LOG_LEVEL='2',
        NCCL_P2P_LEVEL=env.get('NCCL_P2P_LEVEL','NVL'), TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
    env['LIBERO_CONFIG_PATH'] = str(path('work_dir')/'libero_config')
    env['PATH'] = str(Path(python(model)).parent)+os.pathsep+env.get('PATH','')
    if gpus is not None: env['CUDA_VISIBLE_DEVICES']=str(gpus)
    return env
