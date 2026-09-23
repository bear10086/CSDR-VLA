"""Deterministic, full-coverage cohort planning over trajectory windows."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from hashlib import blake2b
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class TrajectoryRecord:
    dataset: str
    trajectory_id: str
    prompt: str
    canonical_prompt_id: str
    window_count: int


@dataclass(frozen=True)
class WindowRef:
    trajectory_index: int
    timestep: int


def stable_prompt_id(dataset: str, normalized_prompt: str) -> str:
    payload = f"{dataset}\0{normalized_prompt}".encode("utf-8")
    return blake2b(payload, digest_size=10).hexdigest()


def normalize_prompt(prompt: str) -> str:
    return " ".join(str(prompt).strip().lower().split())


def _diverse_chunks(
    trajectory_windows: dict[int, list[int]], group_size: int, rng: np.random.Generator
) -> tuple[list[list[WindowRef]], list[WindowRef]]:
    """Make fixed-size groups, preferring one window per trajectory per round."""
    queues: dict[int, deque[int]] = {}
    for trajectory_index, timesteps in trajectory_windows.items():
        shuffled = list(timesteps)
        rng.shuffle(shuffled)
        queues[trajectory_index] = deque(shuffled)

    groups: list[list[WindowRef]] = []
    while sum(len(queue) for queue in queues.values()) >= group_size:
        group: list[WindowRef] = []
        used_trajectories: set[int] = set()
        while len(group) < group_size:
            candidates = [
                index
                for index, queue in queues.items()
                if queue and index not in used_trajectories
            ]
            if not candidates:
                used_trajectories.clear()
                candidates = [index for index, queue in queues.items() if queue]
            if not candidates:
                break
            max_length = max(len(queues[index]) for index in candidates)
            candidates = [index for index in candidates if len(queues[index]) == max_length]
            trajectory_index = int(rng.choice(candidates))
            group.append(WindowRef(trajectory_index, queues[trajectory_index].popleft()))
            used_trajectories.add(trajectory_index)
        if len(group) != group_size:
            raise RuntimeError("Planner failed to fill a cohort despite sufficient windows.")
        groups.append(group)

    leftovers = [
        WindowRef(trajectory_index, timestep)
        for trajectory_index, queue in queues.items()
        for timestep in queue
    ]
    rng.shuffle(leftovers)
    return groups, leftovers


def build_full_coverage_plan(
    trajectories: Sequence[TrajectoryRecord],
    *,
    world_size: int,
    per_device_batch_size: int,
    cohort_count: int,
    seed: int,
) -> dict:
    """Build an epoch plan in which every real window occurs exactly once.

    Cohort batches contain one canonical prompt per rank cohort. Remaining
    windows are emitted in task-only batches. Padding repeats are masked and do
    not count toward either task loss or CSDR.
    """
    if world_size < 1 or per_device_batch_size < 1:
        raise ValueError("world_size and per_device_batch_size must be positive.")
    if cohort_count < 1 or world_size % cohort_count:
        raise ValueError("cohort_count must divide world_size.")

    ranks_per_cohort = world_size // cohort_count
    cohort_size = ranks_per_cohort * per_device_batch_size
    global_batch_size = world_size * per_device_batch_size
    rng = np.random.default_rng(seed)

    by_prompt: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    total_windows = 0
    first_window: WindowRef | None = None
    for trajectory_index, trajectory in enumerate(trajectories):
        if trajectory.window_count < 0:
            raise ValueError(f"Negative window count for {trajectory.trajectory_id}.")
        for timestep in range(trajectory.window_count):
            ref = WindowRef(trajectory_index, timestep)
            by_prompt[trajectory.canonical_prompt_id][trajectory_index].append(timestep)
            total_windows += 1
            if first_window is None:
                first_window = ref

    prompt_groups: dict[str, deque[list[WindowRef]]] = {}
    leftovers: list[WindowRef] = []
    cohort_window_count = 0
    for prompt_id in sorted(by_prompt):
        groups, prompt_leftovers = _diverse_chunks(by_prompt[prompt_id], cohort_size, rng)
        rng.shuffle(groups)
        if groups:
            prompt_groups[prompt_id] = deque(groups)
            cohort_window_count += len(groups) * cohort_size
        leftovers.extend(prompt_leftovers)

    rng.shuffle(leftovers)
    leftover_queue = deque(leftovers)
    cohort_batches: list[dict] = []
    padding_slots = 0
    while prompt_groups:
        prompt_ids = sorted(prompt_groups)
        rng.shuffle(prompt_ids)
        selected: list[str] = []
        # Use distinct prompts first, then reuse a prompt only when fewer than
        # cohort_count prompt queues remain near the end of an epoch.
        for prompt_id in prompt_ids:
            if len(selected) == cohort_count:
                break
            selected.append(prompt_id)
        while len(selected) < cohort_count and prompt_groups:
            candidates = [
                key
                for key, groups in prompt_groups.items()
                if len(groups) > selected.count(key)
            ]
            if not candidates:
                break
            selected.append(
                max(candidates, key=lambda key: len(prompt_groups[key]) - selected.count(key))
            )
        rank_samples: list[list[dict]] = [[] for _ in range(world_size)]
        for cohort_index, prompt_id in enumerate(selected):
            group = prompt_groups[prompt_id].popleft()
            for offset, ref in enumerate(group):
                local_rank = offset // per_device_batch_size
                rank = cohort_index * ranks_per_cohort + local_rank
                rank_samples[rank].append(
                    {**asdict(ref), "loss_mask": 1, "csdr_mask": 1, "prompt_id": prompt_id}
                )
            if not prompt_groups[prompt_id]:
                del prompt_groups[prompt_id]
        for cohort_index in range(len(selected), cohort_count):
            task_slot = []
            while leftover_queue and len(task_slot) < cohort_size:
                task_slot.append((leftover_queue.popleft(), 1))
            if task_slot and len(task_slot) < cohort_size:
                while len(task_slot) < cohort_size:
                    task_slot.append((task_slot[len(task_slot) % len(task_slot)][0], 0))
                    padding_slots += 1
            elif not task_slot:
                if first_window is None:
                    break
                task_slot = [(first_window, 0) for _ in range(cohort_size)]
                padding_slots += cohort_size
            for offset, (ref, loss_mask) in enumerate(task_slot):
                local_rank = offset // per_device_batch_size
                rank = cohort_index * ranks_per_cohort + local_rank
                rank_samples[rank].append(
                    {
                        **asdict(ref),
                        "loss_mask": loss_mask,
                        "csdr_mask": 0,
                        "prompt_id": trajectories[ref.trajectory_index].canonical_prompt_id,
                    }
                )
        cohort_batches.append(
            {"kind": "cohort" if len(selected) == cohort_count else "mixed", "ranks": rank_samples}
        )

    task_batches: list[dict] = []
    remaining_leftovers = list(leftover_queue)
    for start in range(0, len(remaining_leftovers), global_batch_size):
        real = remaining_leftovers[start : start + global_batch_size]
        if not real:
            continue
        padded: list[tuple[WindowRef, int]] = [(ref, 1) for ref in real]
        while len(padded) < global_batch_size:
            padded.append((real[(len(padded) - len(real)) % len(real)], 0))
            padding_slots += 1
        rank_samples = [[] for _ in range(world_size)]
        for offset, (ref, loss_mask) in enumerate(padded):
            rank = offset // per_device_batch_size
            rank_samples[rank].append(
                {
                    **asdict(ref),
                    "loss_mask": loss_mask,
                    "csdr_mask": 0,
                    "prompt_id": trajectories[ref.trajectory_index].canonical_prompt_id,
                }
            )
        task_batches.append({"kind": "task_only", "ranks": rank_samples})

    batches = cohort_batches + task_batches
    rng.shuffle(batches)
    task_only_windows = total_windows - cohort_window_count
    return {
        "version": 1,
        "seed": seed,
        "world_size": world_size,
        "per_device_batch_size": per_device_batch_size,
        "cohort_count": cohort_count,
        "ranks_per_cohort": ranks_per_cohort,
        "cohort_size": cohort_size,
        "global_batch_size": global_batch_size,
        "total_windows": total_windows,
        "cohort_windows": cohort_window_count,
        "task_only_windows": task_only_windows,
        "padding_slots": padding_slots,
        "batches": batches,
    }


def validate_full_coverage_plan(plan: dict, trajectories: Sequence[TrajectoryRecord]) -> dict:
    expected = {
        (trajectory_index, timestep)
        for trajectory_index, trajectory in enumerate(trajectories)
        for timestep in range(trajectory.window_count)
    }
    observed: list[tuple[int, int]] = []
    bad_cohort_batches = 0
    for batch in plan["batches"]:
        cohort_prompt_by_rank = []
        for rank in batch["ranks"]:
            real = [item for item in rank if item["loss_mask"]]
            observed.extend((item["trajectory_index"], item["timestep"]) for item in real)
            cohort_prompt_by_rank.append({item["prompt_id"] for item in rank if item["csdr_mask"]})
        if batch["kind"] in {"cohort", "mixed"}:
            ranks_per_cohort = plan["ranks_per_cohort"]
            for start in range(0, plan["world_size"], ranks_per_cohort):
                prompts = set().union(*cohort_prompt_by_rank[start : start + ranks_per_cohort])
                if prompts and len(prompts) != 1:
                    bad_cohort_batches += 1

    observed_set = set(observed)
    duplicates = len(observed) - len(observed_set)
    missing = expected - observed_set
    extras = observed_set - expected
    result = {
        "expected_windows": len(expected),
        "observed_windows": len(observed),
        "duplicates": duplicates,
        "missing": len(missing),
        "extras": len(extras),
        "bad_cohort_batches": bad_cohort_batches,
    }
    if duplicates or missing or extras or bad_cohort_batches:
        raise ValueError(f"Invalid full-coverage plan: {result}")
    return result
