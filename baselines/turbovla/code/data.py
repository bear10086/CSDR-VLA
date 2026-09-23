"""Keep native Turbo inputs/masks, and expose true action lengths for CSDR."""
import json
import numpy as np
import torch
import tensorflow_datasets as tfds
from common import *
from cohort_planning.planned_dataset import PlannedRLDSDataset
from turbovla.data.mixed_suite import LiberoMixedRLDSDataset
from turbovla.data.libero_rlds import vla_collate_fn


class Window:
    def __init__(self, native):
        self.native = native

    @staticmethod
    def prepare_episode(episode, trajectory):
        return list(tfds.as_numpy(episode['steps']))

    def __call__(self, steps, trajectory, timestep):
        return {'native': self.native._build_step_sample(steps, timestep, len(steps))}


def collate(instances):
    samples, instructions, states, actions, masks = vla_collate_fn([x['native'] for x in instances])
    lengths = masks.sum(1).long()
    assert torch.equal(masks.bool(), torch.arange(12)[None, :] < lengths[:, None])
    return dict(samples=samples, instructions=instructions, states=states, actions=actions,
        action_masks=masks, lengths=lengths,
        sample_loss_mask=torch.tensor([x['sample_loss_mask'] for x in instances], dtype=torch.float32),
        csdr_sample_mask=torch.tensor([x['csdr_sample_mask'] for x in instances], dtype=torch.bool),
        prompt_ids=torch.tensor([[x['canonical_prompt_id']] for x in instances]))


def native_dataset(rank=0, world=10):
    dirs = [str(DATA_ROOT / (suite + '_no_noops') / '1.0.0') for suite in SUITES]
    return LiberoMixedRLDSDataset(dataset_dir=dirs[0], dataset_dirs=dirs,
        stats_path=str(STATS), stats_key='libero_all4_no_noops', LOCAL_DINOV3_PATH=DINO,
        rank=rank, world_size=world, chunk_size=12, split='train', shuffle_buffer=1,
        shuffle_steps_within_episode=False, step_mix_buffer_size=0, seed=7,
        local_files_only=True, expected_image_size=256)


def dataset(native, epoch, rank, start_micro=0):
    sources = {s + '_no_noops': (DATA_ROOT / (s + '_no_noops') / '1.0.0',
               OLD_PLAN / (s + '_no_noops_tfrecord_index.json')) for s in SUITES}
    window = Window(native)
    result = PlannedRLDSDataset(manifest_path=ROOT / 'trajectories.jsonl',
        plan_paths=[ROOT / f'plans/epoch_seed{6+epoch}.npz'], dataset_sources=sources, rank=rank,
        window_transform=window, episode_transform=window.prepare_episode,
        episode_cache_size=64, prefetch_size=16, decode_workers=4)
    result.start_micro = start_micro
    return result


def plan_action_counts(epoch):
    records = [json.loads(x) for x in (ROOT / 'trajectories.jsonl').read_text().splitlines()]
    lengths = np.array([x['raw_step_count'] for x in records])
    with np.load(ROOT / f'plans/epoch_seed{6+epoch}.npz') as p:
        count = np.minimum(12, lengths[p['trajectory_index']] - p['timestep'])
        return (count * p['loss_mask']).sum((1, 2)).astype(np.int64)
