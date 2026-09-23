"""Deterministic prompt-cohort batching for Bridge."""

from collections import deque

import torch.distributed as dist
from torch.utils.data import IterableDataset


def _prompt_key(sample):
    tokens = sample["input_ids"][sample["labels"] < 0]
    return tuple(int(token) for token in tokens.tolist())

def _layout(world_size, cohort_count, rank):
    cohort_count = min(max(int(cohort_count), 1), world_size)
    sizes = [
        world_size // cohort_count + int(index < world_size % cohort_count)
        for index in range(cohort_count)
    ]
    start = 0
    for cohort_index, size in enumerate(sizes):
        if start <= rank < start + size:
            return cohort_index, rank - start, size
        start += size
    raise RuntimeError(f"Rank {rank} is outside world size {world_size}.")


class SynchronizedPromptCohortDataset(IterableDataset):
    """Split one same-prompt sample pool across the ranks in each cohort."""

    def __init__(self, dataset, batch_size, cohorts=2, ready_prompt_count=12):
        if batch_size < 2:
            raise ValueError("Prompt cohorts require per-device batch size >= 2.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.cohorts = int(cohorts)
        self.ready_prompt_count = max(int(ready_prompt_count), 1)
        self.use_raw_dataloader = True
        self.dataset_statistics = getattr(dataset, "dataset_statistics", None)
        self.ds_stats_pc = getattr(dataset, "ds_stats_pc", None)

    def __len__(self):
        return len(self.dataset)

    def _restart_source(self):
        reset_iterator = getattr(self.dataset, "reset_rlds_iterator", None)
        if callable(reset_iterator):
            reset_iterator()
        return iter(self.dataset)

    def __iter__(self):
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            raise RuntimeError("Synchronized prompt cohorts require distributed training.")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        cohort_index, local_rank, cohort_size = _layout(world_size, self.cohorts, rank)
        group_size = self.batch_size * cohort_size
        cohort_layouts = []
        start = 0
        for index in range(min(max(self.cohorts, 1), world_size)):
            size = world_size // min(max(self.cohorts, 1), world_size)
            size += int(index < world_size % min(max(self.cohorts, 1), world_size))
            ranks = list(range(start, start + size))
            cohort_layouts.append((ranks, dist.new_group(ranks=ranks)))
            start += size
        cohort_ranks, cohort_group = cohort_layouts[cohort_index]
        leader = cohort_ranks[0]
        source = self._restart_source() if rank == leader else None
        buffers = {} if rank == leader else None
        source_pass = 0
        if rank == 0:
            print(
                f"CSDR leader-broadcast prompt cohorts: world_size={world_size}, "
                f"cohorts={self.cohorts}, samples_per_prompt={group_size}",
                flush=True,
            )

        while True:
            payload = [None]
            if rank == leader:
                ready_key = None
                while ready_key is None:
                    try:
                        sample = next(source)
                    except StopIteration:
                        source_pass += 1
                        # Do not combine an incomplete group from the previous
                        # pass with repeated transitions from the next pass.
                        buffers.clear()
                        source = self._restart_source()
                        try:
                            sample = next(source)
                        except StopIteration as error:
                            raise RuntimeError("The wrapped RLDS dataset is empty.") from error
                        print(
                            f"CSDR cohort {cohort_index} starts RLDS pass {source_pass + 1}.",
                            flush=True,
                        )
                    key = _prompt_key(sample)
                    buffers.setdefault(key, deque()).append(sample)
                    if len(buffers[key]) >= group_size:
                        ready_key = key
                payload[0] = [buffers[ready_key].popleft() for _ in range(group_size)]

            # Only the cohort leader reads RLDS. Broadcasting its complete pool
            # makes all local slices disjoint and guarantees an exact prompt match.
            dist.broadcast_object_list(payload, src=leader, group=cohort_group)
            group = payload[0]
            start = local_rank * self.batch_size
            for sample in group[start : start + self.batch_size]:
                yield sample
