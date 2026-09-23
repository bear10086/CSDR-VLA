"""Exact planned-window adapters for the two current continuation baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from PIL import Image


LIBERO_IMAGE_AUGMENT = {
    "random_resized_crop": {"scale": [0.9, 0.9], "ratio": [1.0, 1.0]},
    "random_brightness": [0.2],
    "random_contrast": [0.8, 1.2],
    "random_saturation": [0.8, 1.2],
    "random_hue": [0.05],
    "augment_order": [
        "random_resized_crop",
        "random_brightness",
        "random_contrast",
        "random_saturation",
        "random_hue",
    ],
}


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _decode_text(value: Any) -> str:
    value = _numpy(value)
    if value.ndim == 0:
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _steps(episode: dict) -> list[dict]:
    steps = episode["steps"]
    if isinstance(steps, tf.data.Dataset):
        return list(tfds.as_numpy(steps))
    if isinstance(steps, (list, tuple)):
        return list(steps)
    raise TypeError(f"Unsupported RLDS steps container: {type(steps)!r}")


def normalize_q99(values: np.ndarray, statistics: dict) -> np.ndarray:
    """Match the OXE BOUNDS_Q99 transform, including its dimension mask."""
    values = np.asarray(values, dtype=np.float32)
    low = np.asarray(statistics["q01"], dtype=np.float32)
    high = np.asarray(statistics["q99"], dtype=np.float32)
    mask = np.asarray(statistics.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    normalized = np.clip(2.0 * (values - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0)
    output = np.where(mask, normalized, values)
    if "min" in statistics and "max" in statistics:
        constant = np.asarray(statistics["min"]) == np.asarray(statistics["max"])
        output = np.where(constant, 0.0, output)
    return output.astype(np.float32, copy=False)


class OXEImagePreprocessor:
    """Use the baseline's own dlimp resize and augmentation operations."""

    def __init__(
        self, resize_size: tuple[int, int], image_aug: bool, backend: str
    ) -> None:
        self.resize_size = tuple(resize_size)
        self.image_aug = bool(image_aug)
        self.backend = backend

    def __call__(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.backend == "openvla":
            from prismatic.vla.datasets.rlds import obs_transforms
        elif self.backend == "spatialvla":
            from data import obs_transforms
        else:
            raise ValueError(f"Unknown OXE image backend: {self.backend!r}")

        observation = {
            f"image_{name}": tf.convert_to_tensor(image, dtype=tf.uint8)
            for name, image in images.items()
        }
        observation["pad_mask_dict"] = {
            f"image_{name}": tf.constant(True) for name in images
        }
        observation = obs_transforms.decode_and_resize(
            observation,
            resize_size=self.resize_size,
            depth_resize_size={},
        )
        if self.image_aug:
            seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
            observation = obs_transforms.augment(
                observation,
                seed=seed,
                augment_kwargs=LIBERO_IMAGE_AUGMENT,
            )
        return {
            name: _numpy(observation[f"image_{name}"])
            for name in images
        }


class LiberoOpenVLAWindowTransform:
    """Reproduce OpenVLA-OFT LIBERO standardization before RLDSBatchTransform."""

    action_chunk_size = 8

    def __init__(
        self,
        batch_transform,
        statistics_path: Path,
        resize_size: tuple[int, int],
        image_aug: bool,
    ) -> None:
        self.batch_transform = batch_transform
        self.dataset_statistics = json.loads(Path(statistics_path).read_text(encoding="utf-8"))
        self.image_preprocessor = OXEImagePreprocessor(resize_size, image_aug, "openvla")

    def prepare_episode(self, episode: dict, trajectory: dict) -> dict:
        steps = _steps(episode)
        if len(steps) < self.action_chunk_size:
            raise ValueError(f"LIBERO episode is too short: {len(steps)}")
        raw_actions = np.stack([_numpy(step["action"]) for step in steps]).astype(np.float32)
        raw_states = np.stack([_numpy(step["observation"]["state"]) for step in steps]).astype(np.float32)
        actions = np.concatenate(
            [raw_actions[:, :6], 1.0 - np.clip(raw_actions[:, -1:], 0.0, 1.0)], axis=1
        )
        proprio = np.concatenate([raw_states[:, :6], raw_states[:, -2:]], axis=1)
        statistics = self.dataset_statistics[trajectory["dataset"]]
        return {
            "actions": normalize_q99(actions, statistics["action"]),
            "proprio": normalize_q99(proprio, statistics["proprio"]),
            "images": np.stack([_numpy(step["observation"]["image"]) for step in steps]),
            "wrist_images": np.stack(
                [_numpy(step["observation"]["wrist_image"]) for step in steps]
            ),
            "prompt": _decode_text(steps[0]["language_instruction"]),
        }

    def __call__(self, episode: dict, trajectory: dict, timestep: int) -> dict:
        end = timestep + self.action_chunk_size
        if not 0 <= timestep or end > len(episode["actions"]):
            raise IndexError(
                f"Invalid LIBERO window t={timestep}, episode_length={len(episode['actions'])}."
            )
        images = self.image_preprocessor(
            {
                "primary": episode["images"][timestep],
                "wrist": episode["wrist_images"][timestep],
            }
        )
        rlds_window = {
            "dataset_name": trajectory["dataset"].encode("utf-8"),
            "action": episode["actions"][timestep:end],
            "observation": {
                "image_primary": images["primary"][None],
                "image_wrist": images["wrist"][None],
                "proprio": episode["proprio"][timestep : timestep + 1],
            },
            "task": {"language_instruction": episode["prompt"].encode("utf-8")},
        }
        return self.batch_transform(rlds_window)


class BridgeSpatialVLAWindowTransform:
    """Reproduce SpatialVLA Bridge relabeling, normalization, chunking and processing."""

    action_chunk_size = 4

    def __init__(
        self,
        processor,
        statistics_path: Path,
        image_size: int = 224,
        max_length: int = 2048,
        image_aug: bool = True,
        include_action_valid_length: bool = False,
    ) -> None:
        self.processor = processor
        self.max_length = int(max_length)
        self.include_action_valid_length = bool(include_action_valid_length)
        self.dataset_statistics = json.loads(Path(statistics_path).read_text(encoding="utf-8"))
        self.image_preprocessor = OXEImagePreprocessor(
            (image_size, image_size), image_aug, "spatialvla"
        )

    def prepare_episode(self, episode: dict, trajectory: dict) -> dict:
        del trajectory
        steps = _steps(episode)
        if len(steps) < 3:
            raise ValueError(f"Bridge episode is too short: {len(steps)}")
        states = np.stack([_numpy(step["observation"]["state"]) for step in steps]).astype(np.float32)
        gripper = np.asarray(
            [_numpy(step["action"]["open_gripper"]).reshape(-1)[0] for step in steps],
            dtype=np.float32,
        )
        # SpatialVLA removes raw step 0, then relabels movement with reached state
        # and removes the resulting final no-action step.
        movement = states[2:, :6] - states[1:-1, :6]
        actions = np.concatenate([movement, gripper[1:-1, None]], axis=1)
        return {
            "actions": normalize_q99(actions, self.dataset_statistics["action"]),
            "images": np.stack([_numpy(step["observation"]["image"]) for step in steps[1:-1]]),
            "prompt": _decode_text(steps[1]["observation"]["natural_language_instruction"]),
        }

    def __call__(self, episode: dict, trajectory: dict, timestep: int) -> dict:
        del trajectory
        length = len(episode["actions"])
        if not 0 <= timestep < length:
            raise IndexError(f"Invalid Bridge window t={timestep}, episode_length={length}.")
        valid_length = min(self.action_chunk_size, length - timestep)
        indices = np.minimum(np.arange(timestep, timestep + self.action_chunk_size), length - 1)
        actions = episode["actions"][indices].copy()
        past_goal = np.arange(timestep, timestep + self.action_chunk_size) >= length
        actions[past_goal, :6] = 0.0
        # The gripper is an absolute action and therefore repeats at the goal.
        image = self.image_preprocessor({"primary": episode["images"][timestep]})["primary"]
        actions_tensor = torch.from_numpy(actions)
        result = self.processor(
            text=episode["prompt"].lower(),
            images=[Image.fromarray(image)],
            suffix_actions=actions_tensor,
            return_tensors="pt",
            padding=False,
            max_length=self.max_length,
            truncation=True,
            do_normalize=False,
        )
        window = {
            "input_ids": result["input_ids"][0],
            "labels": result["labels"][0],
            "token_type_ids": result["token_type_ids"][0],
            "attention_mask": result["attention_mask"][0],
            "pixel_values": result["pixel_values"],
            "intrinsic": result["intrinsic"],
            "actions": actions_tensor,
        }
        if self.include_action_valid_length:
            window["action_valid_length"] = torch.tensor(valid_length, dtype=torch.long)
        return window
