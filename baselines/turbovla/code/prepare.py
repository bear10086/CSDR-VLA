"""CPU-only full-start coverage plans and native-normalized true-prefix calibration."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import sys
import numpy as np
from common import *
sys.path.insert(0, str(PROJECT / 'common/cohort_planning'))
from build_batch_plan import load_manifest, save_compact_plan
from planner import build_full_coverage_plan, validate_full_coverage_plan


def plans():
    original = [json.loads(x) for x in (OLD_PLAN / 'trajectories.jsonl').read_text().splitlines()]
    records = [{**x, 'window_count': x['raw_step_count']} for x in original]
    (ROOT / 'trajectories.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in records))
    trajectories = load_manifest(ROOT / 'trajectories.jsonl', None)
    summaries = []
    for seed in (7, 8, 9):
        plan = build_full_coverage_plan(trajectories, world_size=10, per_device_batch_size=4,
                                        cohort_count=5, seed=seed)
        audit = validate_full_coverage_plan(plan, trajectories)
        assert audit['missing'] == audit['duplicates'] == audit['extras'] == 0
        path = ROOT / f'plans/epoch_seed{seed}.npz'
        save_compact_plan(path, plan)
        summary = {k: v for k, v in plan.items() if k != 'batches'}
        summary.update(validation=audit, batch_count=len(plan['batches']), optimizer_steps=(len(plan['batches']) + 2) // 3)
        atomic_json(path.with_suffix('.summary.json'), summary)
        summaries.append(summary)
        print('PLAN', seed, summary['total_windows'], summary['optimizer_steps'], flush=True)
    return records, summaries


def actions_cache(records):
    import tensorflow_datasets as tfds
    import torch
    from cohort_planning.planned_dataset import EpisodeDecoder
    from turbovla.data.mixed_suite import LiberoMixedRLDSDataset
    torch.set_num_threads(1)
    lengths = np.array([x['raw_step_count'] for x in records])
    offsets = np.r_[0, lengths.cumsum()]
    # Native normalizer, with no image processor/model construction needed.
    normalizer = object.__new__(LiberoMixedRLDSDataset)
    stats = json.loads(STATS.read_text())['libero_all4_no_noops']
    normalizer.action_min = torch.tensor(stats['action']['min'], dtype=torch.float32)
    normalizer.action_max = torch.tensor(stats['action']['max'], dtype=torch.float32)
    # Match mixed_suite's automatic binary-gripper normalization decision.
    lo, hi = float(normalizer.action_min[6]), float(normalizer.action_max[6])
    normalizer._normalize_binary_gripper = lo >= -1e-6 and hi <= 1.0 + 1e-6
    decoders = {name: EpisodeDecoder(DATA_ROOT / name / '1.0.0',
                       OLD_PLAN / (name + '_tfrecord_index.json')) for name in sorted({x['dataset'] for x in records})}
    out = np.empty((int(offsets[-1]), 7), dtype=np.float32)
    try:
        for i, record in enumerate(records):
            steps = list(tfds.as_numpy(decoders[record['dataset']].decode(record['trajectory_id'])['steps']))
            raw = np.stack([step['action'] for step in steps]).astype(np.float32)
            assert len(raw) == record['raw_step_count']
            out[offsets[i]:offsets[i+1]] = normalizer._normalize_action_chunk(torch.from_numpy(raw)).numpy()
            if i % 100 == 0:
                print('ACTION_SCAN', i, len(records), flush=True)
    finally:
        for decoder in decoders.values():
            decoder.close()
    assert np.isfinite(out).all()
    np.save(ROOT / 'actions.npy', out)
    np.save(ROOT / 'offsets.npy', offsets)
    return out, offsets


def scales(actions, offsets):
    ids = np.arange(len(actions))
    remaining = offsets[np.searchsorted(offsets, ids, side='right')] - ids
    table, info = {}, {}
    for h in range(4, 13):
        rng = np.random.default_rng(7 + h)
        selected = rng.choice(np.flatnonzero(remaining >= h), size=20000, replace=False)
        chunks = actions[selected[:, None] + np.arange(h)]
        pair = rng.integers(0, len(chunks), (200000, 2))
        pair = pair[pair[:, 0] != pair[:, 1]]
        table[str(h)] = {}
        for name, sl in [('action_translation', slice(0, 3)), ('action_rotation', slice(3, 6)), ('action_gripper', slice(6, 7))]:
            d = np.sqrt(np.mean((chunks[pair[:, 0], :, sl] - chunks[pair[:, 1], :, sl])**2, axis=(1,2)))
            # Preserve Turbo's former MEAN estimator; fix normalization and exclude padding.
            table[str(h)][name] = max(float(d.mean()), 1e-6)
        info[str(h)] = {'samples': len(selected), 'pairs': len(pair), 'seed': 7 + h, 'padded_samples': 0}
    atomic_json(ROOT / 'control_scales.json', {'scales': table['12'], 'by_length': table,
        'sampling': info, 'estimator': 'mean RMS', 'normalization': 'exact native mixed_suite normalizer including gripper'})


def main():
    records, summaries = plans()
    actions, offsets = actions_cache(records)
    scales(actions, offsets)
    report = {'passed': True, 'raw_starts': len(actions), 'trajectories': len(records),
              'restored_task_starts': sum(x['raw_step_count'] for x in records) - 261614,
              'epochs': summaries, 'optimizer_steps': sum(s['optimizer_steps'] for s in summaries)}
    atomic_json(ROOT / 'preparation.json', report)
    print('PREPARATION_COMPLETE', report['optimizer_steps'], flush=True)


if __name__ == '__main__':
    main()
