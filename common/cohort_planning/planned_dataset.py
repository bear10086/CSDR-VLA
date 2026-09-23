"""Plan-driven PyTorch dataset over random-access TFDS episodes."""

from __future__ import annotations

import json
import os
import queue
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .tfrecord_store import RandomAccessTFRecordStore


_PREFETCH_END = object()


def _prefetch_in_order(source, buffer_size: int):
    """Overlap sample preparation with training while preserving source order."""
    if buffer_size <= 0:
        yield from source
        return

    buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
    cancelled = threading.Event()

    def put_unless_cancelled(value) -> bool:
        while not cancelled.is_set():
            try:
                buffer.put(value, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for item in source:
                if not put_unless_cancelled((item, None)):
                    return
        except BaseException as error:
            put_unless_cancelled((None, error))
        finally:
            put_unless_cancelled((_PREFETCH_END, None))

    producer = threading.Thread(
        target=produce,
        name="planned-rlds-prefetch",
        daemon=True,
    )
    producer.start()
    try:
        while True:
            item, error = buffer.get()
            if error is not None:
                raise error
            if item is _PREFETCH_END:
                return
            yield item
    finally:
        cancelled.set()
        producer.join(timeout=30)


class EpisodeDecoder:
    def __init__(self, tfds_dir: Path, index_path: Path) -> None:
        self.builder = tfds.builder_from_directory(str(tfds_dir))
        self.store = RandomAccessTFRecordStore(index_path)

    def decode(self, trajectory_id: str):
        serialized = self.store.serialized_episode(trajectory_id)
        return self.builder.info.features.deserialize_example(tf.convert_to_tensor(serialized))

    def close(self) -> None:
        self.store.close()


class PlannedRLDSDataset(IterableDataset):
    """Yield model-ready windows in a precomputed per-rank batch order."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        plan_paths: list[Path],
        dataset_sources: dict[str, tuple[Path, Path]],
        rank: int,
        window_transform: Callable,
        episode_transform: Callable | None = None,
        episode_cache_size: int = 16,
        prefetch_size: int = 0,
        decode_workers: int = 1,
    ) -> None:
        self.trajectories = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.plan_paths = [Path(path) for path in plan_paths]
        self.start_micro = 0
        self.consumed_micro_batches = 0
        self.manifest_path = Path(manifest_path)
        if not self.plan_paths:
            raise ValueError("At least one epoch plan is required.")
        self.rank = int(rank)
        self.window_transform = window_transform
        self.episode_transform = episode_transform
        self.episode_cache_size = max(int(episode_cache_size), 1)
        self.prefetch_size = max(int(prefetch_size), 0)
        self.decode_workers = max(int(decode_workers), 1)
        self.use_raw_dataloader = True
        self.decoders = {
            name: EpisodeDecoder(Path(tfds_dir), Path(index_path))
            for name, (tfds_dir, index_path) in dataset_sources.items()
        }
        self._episode_cache: OrderedDict[int, object] = OrderedDict()
        self.dataset_statistics = getattr(window_transform, "dataset_statistics", None)
        self.dataset_length = 0
        self.micro_batches_per_plan: list[int] = []
        self.world_size = None
        self.per_device_batch_size = None
        for path in self.plan_paths:
            with np.load(path, allow_pickle=False) as plan:
                metadata = json.loads(str(plan["metadata"]))
                plan_world_size = int(metadata["world_size"])
                plan_batch_size = int(metadata["per_device_batch_size"])
                if self.world_size not in (None, plan_world_size):
                    raise ValueError("All epoch plans must use the same world size.")
                if self.per_device_batch_size not in (None, plan_batch_size):
                    raise ValueError("All epoch plans must use the same per-device batch size.")
                self.world_size = plan_world_size
                self.per_device_batch_size = plan_batch_size
                plan_micro_batches = int(plan["trajectory_index"].shape[0])
                self.micro_batches_per_plan.append(plan_micro_batches)
                self.dataset_length += plan_micro_batches * plan_batch_size
        if not 0 <= self.rank < int(self.world_size):
            raise ValueError(f"Rank {self.rank} is outside plan world size {self.world_size}.")

    def __len__(self) -> int:
        self._validate_start_micro()
        return self.dataset_length - int(self.start_micro) * self.per_device_batch_size

    def _validate_start_micro(self):
        if not isinstance(self.start_micro, (int, np.integer)) or not 0 <= self.start_micro <= sum(self.micro_batches_per_plan):
            raise ValueError("start_micro must identify a valid next batch in the plan")

    def _episode(self, trajectory_index: int):
        episode = self._episode_cache.pop(trajectory_index, None)
        if episode is None:
            trajectory = self.trajectories[trajectory_index]
            decoder = self.decoders[trajectory["dataset"]]
            episode = decoder.decode(trajectory["trajectory_id"])
            if self.episode_transform is not None:
                episode = self.episode_transform(episode, trajectory)
        self._episode_cache[trajectory_index] = episode
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return episode

    def _decode_episode(self, trajectory_index: int):
        trajectory = self.trajectories[trajectory_index]
        decoder = self.decoders[trajectory["dataset"]]
        episode = decoder.decode(trajectory["trajectory_id"])
        if self.episode_transform is not None:
            episode = self.episode_transform(episode, trajectory)
        return trajectory_index, episode

    def _prime_episode_cache(self, trajectory_indices, executor: ThreadPoolExecutor) -> None:
        # Decoding and normalization are deterministic. Image augmentation and
        # processor calls remain sequential below, preserving their RNG order.
        missing = []
        seen = set()
        for value in trajectory_indices:
            trajectory_index = int(value)
            if trajectory_index in self._episode_cache:
                self._episode_cache.move_to_end(trajectory_index)
            elif trajectory_index not in seen:
                missing.append(trajectory_index)
                seen.add(trajectory_index)
        if not missing:
            return
        for trajectory_index, episode in executor.map(self._decode_episode, missing):
            self._episode_cache[trajectory_index] = episode
            while len(self._episode_cache) > self.episode_cache_size:
                self._episode_cache.popitem(last=False)

    def _iter_samples(self):
        self._validate_start_micro()
        remaining_skip = int(self.start_micro)
        if get_worker_info() is not None:
            raise RuntimeError(
                "PlannedRLDSDataset requires DataLoader(num_workers=0); TFRecord handles "
                "and fixed local batch boundaries must stay in the training process."
            )
        with ThreadPoolExecutor(
            max_workers=self.decode_workers,
            thread_name_prefix="planned-rlds-decode",
        ) as executor:
            for plan_path in self.plan_paths:
                with np.load(plan_path, allow_pickle=False) as plan:
                    trajectory_indices = plan["trajectory_index"][:, self.rank]
                    timesteps = plan["timestep"][:, self.rank]
                    loss_masks = plan["loss_mask"][:, self.rank]
                    csdr_masks = plan["csdr_mask"][:, self.rank]
                    prompt_indices = plan["prompt_index"][:, self.rank]
                    skip = min(remaining_skip, trajectory_indices.shape[0])
                    remaining_skip -= skip
                    for batch_index in range(skip, trajectory_indices.shape[0]):
                        batch_trajectories = trajectory_indices[batch_index]
                        if self.decode_workers > 1:
                            self._prime_episode_cache(batch_trajectories, executor)
                        for slot_index in range(batch_trajectories.shape[0]):
                            trajectory_index = int(batch_trajectories[slot_index])
                            trajectory = self.trajectories[trajectory_index]
                            item = self.window_transform(
                                self._episode(trajectory_index),
                                trajectory,
                                int(timesteps[batch_index, slot_index]),
                            )
                            item["sample_loss_mask"] = int(loss_masks[batch_index, slot_index])
                            item["csdr_sample_mask"] = int(csdr_masks[batch_index, slot_index])
                            item["canonical_prompt_id"] = int(prompt_indices[batch_index, slot_index])
                            yield item

    def __iter__(self):
        yield from _prefetch_in_order(self._iter_samples(), self.prefetch_size)

    def close(self) -> None:
        for decoder in self.decoders.values():
            decoder.close()
        self._episode_cache.clear()


class PlannedCollator:
    """Preserve plan masks while delegating model-specific tensor collation."""

    def __init__(self, base_collator) -> None:
        self.base_collator = base_collator

    def __call__(self, instances):
        output = self.base_collator(instances)
        output["sample_loss_mask"] = torch.tensor(
            [item["sample_loss_mask"] for item in instances], dtype=torch.float32
        )
        output["csdr_sample_mask"] = torch.tensor(
            [item["csdr_sample_mask"] for item in instances], dtype=torch.bool
        )
        output["canonical_prompt_ids"] = torch.tensor(
            [item["canonical_prompt_id"] for item in instances], dtype=torch.long
        )
        return output
