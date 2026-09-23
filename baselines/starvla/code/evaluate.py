"""Parallel official StarVLA SimplerEnv runs; each run keeps its own RNG stream."""
import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from common import ROOT, CODE, RUNTIME, PYTHON, SIM_PYTHON, SIM_ROOT, environment

TASKS = {
    'carrot': 'PutCarrotOnPlateInScene-v0',
    'eggplant': 'PutEggplantInBasketScene-v0',
    'spoon': 'PutSpoonOnTableClothInScene-v0',
    'stack': 'StackGreenCubeOnYellowCubeBakedTexInScene-v0',
}

def memory_free(gpu):
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    return {int(x.split(',')[0]): int(x.split(',')[1]) for x in lines}[gpu]

def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

def run_job(ckpt, dest, task, seed, gpu, episodes=24):
    stem = f'{task}_seed{seed}'
    output = dest / f'{stem}.json'
    if output.exists():
        old = json.loads(output.read_text())
        assert old['num_episodes'] == episodes and old['checkpoint'] == str(ckpt)
        return old
    while memory_free(gpu) < 13000:
        print(f'WAIT_GPU {gpu} {stem}', flush=True)
        time.sleep(30)
    # Reserve an unused port. Never kill an existing service to free a port.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = environment(gpu)
    env['NO_PROXY'] = '127.0.0.1,localhost,0.0.0.0'
    env['no_proxy'] = env['NO_PROXY']
    server = simulation = None
    with (dest / f'{stem}_server.log').open('w') as server_log, (dest / f'{stem}_eval.log').open('w') as eval_log:
        try:
            server = subprocess.Popen([PYTHON, '-u', str(CODE/'serve.py'), '--ckpt_path', str(ckpt),
                                       '--port', str(port), '--use_bf16', '--seed', str(seed)],
                                      cwd=RUNTIME, env=env, stdout=server_log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            start = time.time()
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'{stem}: policy server exited: {server.returncode}')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=1):
                        break
                except OSError:
                    if time.time()-start > 600:
                        raise TimeoutError(f'{stem}: server startup timed out')
                    time.sleep(3)
            eggplant = task == 'eggplant'
            x, y = ('0.127', '0.06') if eggplant else ('0.147', '0.028')
            cmd = [SIM_PYTHON, '-u', str(RUNTIME/'examples/simBenchmarks/SimplerEnv/eval_files/start_simpler_env.py'),
                   '--ckpt-path', str(ckpt), '--port', str(port), '--seed', str(seed),
                   '--results-file', str(output), '--logging-dir', str(dest/'videos'/stem),
                   '--robot', 'widowx_sink_camera_setup' if eggplant else 'widowx',
                   '--policy-setup', 'widowx_bridge', '--control-freq', '5', '--sim-freq', '500',
                   '--max-episode-steps', '120', '--env-name', TASKS[task],
                   '--scene-name', 'bridge_table_1_v2' if eggplant else 'bridge_table_1_v1',
                   '--rgb-overlay-path', str(SIM_ROOT/'ManiSkill2_real2sim/data/real_inpainting'/
                                             ('bridge_sink.png' if eggplant else 'bridge_real_eval_1.png')),
                   '--robot-init-x', x, x, '1', '--robot-init-y', y, y, '1',
                   '--obj-variation-mode', 'episode', '--obj-episode-range', '0', str(episodes),
                   '--robot-init-rot-quat-center', '0', '0', '0', '1',
                   '--robot-init-rot-rpy-range', '0', '0', '1', '0', '0', '1', '0', '0', '1']
            (dest/f'{stem}_command.json').write_text(json.dumps(cmd, indent=2))
            print('EVAL_START', stem, 'gpu', gpu, flush=True)
            simulation = subprocess.Popen(cmd, cwd=RUNTIME, env=env, stdout=eval_log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            if simulation.wait() != 0:
                raise RuntimeError(f'{stem}: evaluator exited: {simulation.returncode}')
            result = json.loads(output.read_text())
            assert result['num_episodes'] == episodes
            assert result['checkpoint'] == str(ckpt)
            print('EVAL_DONE', stem, result['success_rate'], flush=True)
            return result
        finally:
            stop(simulation)
            stop(server)

def evaluate(ckpt, dest, gpus, smoke=False):
    dest.mkdir(parents=True, exist_ok=True)
    jobs = [('carrot', 7)] if smoke else [(task, seed) for seed in (7,8,9,10) for task in TASKS]
    def worker(slot):
        values=[]
        for task, seed in jobs[slot::len(gpus)]:
            values.append(run_job(ckpt, dest, task, seed, gpus[slot], 1 if smoke else 24))
        return values
    with cf.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        runs = sum(list(pool.map(worker, range(len(gpus)))), [])
    assert len(runs) == len(jobs)
    summary = {'checkpoint': str(ckpt), 'protocol': 'StarVLA official v0; 120 steps; 24 episodes x 4 runs per task',
               'seeds': [7,8,9,10], 'smoke': smoke, 'runs': runs,
               'episodes': sum(x['num_episodes'] for x in runs),
               'successes': sum(x['num_successes'] for x in runs)}
    summary['success_rate'] = summary['successes']/summary['episodes']
    summary['per_task'] = {task: sum(x['num_successes'] for x in runs if x['task']==name)/
                          max(1,sum(x['num_episodes'] for x in runs if x['task']==name)) for task,name in TASKS.items()}
    tmp = dest/'summary.json.tmp'
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(dest/'summary.json')
    return summary

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=ROOT/'baseline/checkpoints/steps_45000_pytorch_model.pt')
    p.add_argument('--output',type=Path,default=ROOT/'eval/baseline')
    p.add_argument('--gpus',default='6,7,8,9')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    evaluate(args.checkpoint,args.output,[int(x) for x in args.gpus.split(',')],args.smoke)
