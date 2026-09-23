#!/usr/bin/env python3
"""Build compact NPZ batch plans from an RLDS trajectory manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from planner import TrajectoryRecord, build_full_coverage_plan, validate_full_coverage_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=10)
    parser.add_argument("--per-device-batch-size", type=int, required=True)
    parser.add_argument("--cohort-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--canonical-map", type=Path)
    return parser.parse_args()


def load_manifest(path: Path, canonical_map_path: Path | None) -> list[TrajectoryRecord]:
    canonical_map = {}
    if canonical_map_path is not None:
        for item in json.loads(canonical_map_path.read_text(encoding="utf-8")):
            canonical_map[(item["dataset"], item["normalized_prompt"])] = item["canonical_prompt_id"]
    trajectories = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                trajectories.append(
                    TrajectoryRecord(
                        dataset=item["dataset"],
                        trajectory_id=item["trajectory_id"],
                        prompt=item["prompt"],
                        canonical_prompt_id=canonical_map.get(
                            (item["dataset"], item["normalized_prompt"]),
                            item["canonical_prompt_id"],
                        ),
                        window_count=int(item["window_count"]),
                    )
                )
    return trajectories


def save_compact_plan(path: Path, plan: dict) -> None:
    batch_count = len(plan["batches"])
    world_size = plan["world_size"]
    batch_size = plan["per_device_batch_size"]
    shape = (batch_count, world_size, batch_size)
    trajectory_index = np.empty(shape, dtype=np.int32)
    timestep = np.empty(shape, dtype=np.int32)
    loss_mask = np.empty(shape, dtype=np.uint8)
    csdr_mask = np.empty(shape, dtype=np.uint8)
    prompt_ids = sorted(
        {
            item["prompt_id"]
            for batch in plan["batches"]
            for rank in batch["ranks"]
            for item in rank
        }
    )
    prompt_lookup = {value: index for index, value in enumerate(prompt_ids)}
    prompt_index = np.empty(shape, dtype=np.int32)
    batch_kind = np.empty(batch_count, dtype=np.uint8)
    for batch_index, batch in enumerate(plan["batches"]):
        batch_kind[batch_index] = int(batch["kind"] == "cohort")
        for rank_index, rank in enumerate(batch["ranks"]):
            for item_index, item in enumerate(rank):
                slot = (batch_index, rank_index, item_index)
                trajectory_index[slot] = item["trajectory_index"]
                timestep[slot] = item["timestep"]
                loss_mask[slot] = item["loss_mask"]
                csdr_mask[slot] = item["csdr_mask"]
                prompt_index[slot] = prompt_lookup[item["prompt_id"]]
    metadata = {key: value for key, value in plan.items() if key != "batches"}
    metadata["prompt_ids"] = prompt_ids
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        trajectory_index=trajectory_index,
        timestep=timestep,
        loss_mask=loss_mask,
        csdr_mask=csdr_mask,
        prompt_index=prompt_index,
        batch_kind=batch_kind,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=True)),
    )


def main() -> None:
    args = parse_args()
    trajectories = load_manifest(args.manifest, args.canonical_map)
    plan = build_full_coverage_plan(
        trajectories,
        world_size=args.world_size,
        per_device_batch_size=args.per_device_batch_size,
        cohort_count=args.cohort_count,
        seed=args.seed,
    )
    validation = validate_full_coverage_plan(plan, trajectories)
    save_compact_plan(args.output, plan)
    summary = {key: value for key, value in plan.items() if key != "batches"}
    summary.update({"batch_count": len(plan["batches"]), "validation": validation})
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
