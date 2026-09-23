"""Ten independent task workers using unchanged native LIBERO evaluation arguments."""
import concurrent.futures
import json
import subprocess
from common import *

def validate(value, checkpoint, suite, task):
    assert value['ckpt_path'] == str(checkpoint)
    assert value['task_suite_name'] == suite and value['requested_task_ids'] == [task]
    assert value['num_trials_per_task'] == value['total_episodes'] == 50
    assert value['seed'] == 7 and value['precision'] == 'bf16' and (value['num_open_loop_steps'] == 12)
    assert len(value['tasks']) == 1 and value['tasks'][0]['task_id'] == task

def summarize(root, checkpoint):
    rows = []
    for suite in SUITES:
        for task in range(10):
            value = json.loads((root / suite / f'task_{task}.json').read_text())
            validate(value, checkpoint, suite, task)
            rows.append({'suite': suite, 'task': task, 'episodes': 50, 'successes': value['total_successes']})
    result = {'checkpoint': str(checkpoint), 'episodes': sum((v['episodes'] for v in rows)), 'successes': sum((v['successes'] for v in rows)), 'tasks': rows, 'protocol': 'native TurboVLA: seed7, bf16, 12-step open loop, 50 trials/task, all four suites'}
    result['success_rate'] = result['successes'] / result['episodes']
    result['per_suite'] = {s: sum((x['successes'] for x in rows if x['suite'] == s)) / 500 for s in SUITES}
    return result

def evaluate(checkpoint, output):
    jobs = [(suite, task) for suite in SUITES for task in range(10)]

    def worker(gpu):
        for suite, task in jobs[gpu::10]:
            folder = output / suite
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'task_{task}.json'
            if path.exists():
                validate(json.loads(path.read_text()), checkpoint, suite, task)
                continue
            command = [PYTHON, str(REPO / 'experiments/libero/evaluate.py'), '--ckpt_path', str(checkpoint), '--dinov3_path', DINO, '--bert_path', BERT, '--stats_path', str(STATS), '--stats_key', 'libero_all4_no_noops', '--task_suite_name', suite, '--task_ids', str(task), '--num_trials_per_task', '50', '--seed', '7', '--precision', 'bf16', '--num_open_loop_steps', '12', '--chunk_size', '12', '--save_video', 'false', '--mujoco_gl', 'egl', '--pyopengl_platform', 'egl', '--result_json_path', str(path), '--log_path', str(folder / f'task_{task}.log')]
            atomic_json(folder / f'task_{task}.command.json', command)
            with (folder / f'task_{task}.stdout').open('a') as f:
                child = subprocess.Popen(command, cwd=REPO, env=environment(str(gpu)), stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                atomic_json(folder / f'task_{task}.worker.json', {'pid': child.pid, 'gpu': gpu})
                if child.wait():
                    raise RuntimeError(f'Evaluation failed: {path}')
            validate(json.loads(path.read_text()), checkpoint, suite, task)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(worker, range(10)))
    result = summarize(output, checkpoint)
    atomic_json(output / 'summary.json', result)
    return result
