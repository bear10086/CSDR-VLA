#!/usr/bin/env python3
"""Scan RLDS episodes without decoding images and write trajectory metadata."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow_datasets as tfds

from planner import normalize_prompt, stable_prompt_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=TFDS_DIR",
        help="May be repeated for a multi-suite mixture.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--drop-first-dataset",
        action="append",
        default=[],
        help="Dataset whose standardization removes the first transition.",
    )
    parser.add_argument(
        "--trim-steps-dataset",
        action="append",
        default=[],
        metavar="NAME=COUNT",
        help="Total leading/trailing steps removed by dataset standardization.",
    )
    parser.add_argument("--max-trajectories", type=int)
    return parser.parse_args()


def image_skip_decoders(feature: Any) -> Any:
    if isinstance(feature, tfds.features.Image):
        return tfds.decode.SkipDecoding()
    if isinstance(feature, tfds.features.FeaturesDict):
        nested = {
            key: decoder
            for key, child in feature.items()
            if (decoder := image_skip_decoders(child)) is not None
        }
        return nested or None
    if isinstance(feature, tfds.features.Dataset):
        return image_skip_decoders(feature.feature)
    return None


def scalar_text(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def get_nested(mapping: dict, path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        value = value[key]
    return value


def prompt_and_embedding(first_step: dict) -> tuple[str, list[float] | None]:
    prompt_paths = (
        ("language_instruction",),
        ("observation", "natural_language_instruction"),
    )
    prompt = None
    for path in prompt_paths:
        try:
            prompt = scalar_text(get_nested(first_step, path))
            break
        except KeyError:
            continue
    if prompt is None:
        raise KeyError("No supported language instruction field in RLDS step.")
    embedding = None
    try:
        raw_embedding = get_nested(first_step, ("observation", "natural_language_embedding"))
        embedding = np.asarray(raw_embedding, dtype=np.float32).reshape(-1).tolist()
    except KeyError:
        pass
    return prompt, embedding


def scan_dataset(
    name: str,
    directory: Path,
    split: str,
    trim_steps: int,
    max_trajectories: int | None,
    manifest_handle,
    catalog: dict,
) -> dict:
    builder = tfds.builder_from_directory(str(directory))
    decoders = image_skip_decoders(builder.info.features)
    read_config = tfds.ReadConfig(add_tfds_id=True, try_autocache=False)
    dataset = builder.as_dataset(
        split=split,
        shuffle_files=False,
        decoders=decoders,
        read_config=read_config,
    )
    trajectory_count = 0
    raw_steps = 0
    windows = 0
    empty_trajectories = 0
    for episode_index, episode in enumerate(tfds.as_numpy(dataset)):
        if max_trajectories is not None and trajectory_count >= max_trajectories:
            break
        step_iterator = iter(episode["steps"])
        try:
            first_step = next(step_iterator)
        except StopIteration:
            empty_trajectories += 1
            continue
        prompt, embedding = prompt_and_embedding(first_step)
        step_count = 1 + sum(1 for _ in step_iterator)
        window_count = max(step_count - trim_steps, 0)
        normalized = normalize_prompt(prompt)
        prompt_id = stable_prompt_id(name, normalized)
        tfds_id = scalar_text(episode.get("tfds_id", f"{name}:{episode_index}"))
        metadata = episode.get("episode_metadata", {})
        source_path = scalar_text(metadata.get("file_path", "")) if metadata else ""
        record = {
            "dataset": name,
            "trajectory_id": tfds_id,
            "episode_index": episode_index,
            "source_path": source_path,
            "prompt": prompt,
            "normalized_prompt": normalized,
            "exact_prompt_id": prompt_id,
            "canonical_prompt_id": prompt_id,
            "raw_step_count": step_count,
            "window_count": window_count,
        }
        manifest_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        key = f"{name}\t{normalized}"
        if key not in catalog:
            catalog[key] = {
                "dataset": name,
                "prompt": prompt,
                "normalized_prompt": normalized,
                "exact_prompt_id": prompt_id,
                "trajectory_count": 0,
                "window_count": 0,
                "embedding": embedding,
            }
        catalog[key]["trajectory_count"] += 1
        catalog[key]["window_count"] += window_count
        trajectory_count += 1
        raw_steps += step_count
        windows += window_count
        if trajectory_count % 1000 == 0:
            print(
                f"{name}: trajectories={trajectory_count} windows={windows} "
                f"unique_prompts={sum(item['dataset'] == name for item in catalog.values())}",
                flush=True,
            )
    return {
        "dataset": name,
        "directory": str(directory),
        "split": split,
        "trim_steps": trim_steps,
        "trajectory_count": trajectory_count,
        "raw_step_count": raw_steps,
        "window_count": windows,
        "empty_trajectories": empty_trajectories,
    }


def main() -> None:
    args = parse_args()
    datasets = []
    for item in args.dataset:
        if "=" not in item:
            raise ValueError(f"Expected NAME=TFDS_DIR, got {item!r}.")
        name, directory = item.split("=", 1)
        datasets.append((name, Path(directory)))
    trim_by_dataset = {name: 1 for name in args.drop_first_dataset}
    for item in args.trim_steps_dataset:
        if "=" not in item:
            raise ValueError(f"Expected NAME=COUNT, got {item!r}.")
        name, count = item.split("=", 1)
        trim_by_dataset[name] = int(count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "trajectories.jsonl"
    catalog: dict[str, dict] = {}
    summaries = []
    with manifest_path.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
        for name, directory in datasets:
            summaries.append(
                scan_dataset(
                    name,
                    directory,
                    args.split,
                    trim_by_dataset.get(name, 0),
                    args.max_trajectories,
                    handle,
                    catalog,
                )
            )
    catalog_path = args.output_dir / "prompt_catalog.json"
    catalog_path.write_text(
        json.dumps(sorted(catalog.values(), key=lambda item: (item["dataset"], item["prompt"])), indent=2),
        encoding="utf-8",
    )
    summary = {
        "version": 1,
        "manifest": str(manifest_path),
        "prompt_catalog": str(catalog_path),
        "datasets": summaries,
        "total_trajectories": sum(item["trajectory_count"] for item in summaries),
        "total_windows": sum(item["window_count"] for item in summaries),
        "unique_exact_prompts": len(catalog),
    }
    (args.output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
