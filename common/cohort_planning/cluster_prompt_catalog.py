#!/usr/bin/env python3
"""Conservatively merge semantically equivalent Bridge prompt strings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from planner import normalize_prompt, stable_prompt_id


STOPWORDS = {
    "a", "an", "please", "the", "you", "robot", "arm",
}
ACQUISITION_TAGS = {
    "cardboard", "fence", "cardboardfence", "distractor", "distractors",
}
RELATION_ALIASES = {
    "into": "in", "inside": "in", "within": "in",
    "onto": "on", "atop": "on",
    "beside": "next_to", "near": "next_to",
    "underneath": "under", "below": "under",
    "over": "above",
}
RELATIONS = {"in", "on", "next_to", "left", "right", "front", "behind", "under", "above", "off", "out"}


def tokens(text: str) -> list[str]:
    cleaned = "".join(character if character.isalnum() else " " for character in normalize_prompt(text))
    words = cleaned.split()
    combined = []
    index = 0
    while index < len(words):
        if index + 1 < len(words) and words[index] == "next" and words[index + 1] == "to":
            combined.append("next_to")
            index += 2
        else:
            combined.append(RELATION_ALIASES.get(words[index], words[index]))
            index += 1
    return combined


def signatures(text: str) -> tuple[frozenset[str], tuple[str, ...]]:
    words = tokens(text)
    relations = frozenset(word for word in words if word in RELATIONS)
    # Keep action, object, size, direction, and word order. Only articles,
    # politeness, and known Bridge collection tags may disappear. This blocks
    # high-embedding-similarity mistakes such as pig/duck or small/big.
    core = tuple(word for word in words if word not in STOPWORDS and word not in ACQUISITION_TAGS)
    return relations, core


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator else -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cosine-threshold", type=float, default=0.94)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = json.loads(args.catalog.read_text(encoding="utf-8"))
    entries.sort(key=lambda item: (-int(item["window_count"]), item["normalized_prompt"]))
    representatives: list[dict] = []
    signature_to_representatives: dict[tuple[str, tuple[str, ...]], list[int]] = defaultdict(list)
    assignments = []
    for entry in entries:
        relation_signature, core_signature = signatures(entry["normalized_prompt"])
        embedding = entry.get("embedding")
        vector = np.asarray(embedding, dtype=np.float32) if embedding is not None else None
        candidates = signature_to_representatives[(entry["dataset"], core_signature)]
        best_index = None
        best_similarity = -1.0
        if vector is not None:
            for index in candidates:
                representative_vector = representatives[index]["embedding"]
                if representative_vector is None:
                    continue
                similarity = cosine(vector, representative_vector)
                if similarity >= args.cosine_threshold and similarity > best_similarity:
                    best_index = index
                    best_similarity = similarity
        if best_index is None:
            best_index = len(representatives)
            representatives.append(
                {
                    "dataset": entry["dataset"],
                    "prompt": entry["prompt"],
                    "normalized_prompt": entry["normalized_prompt"],
                    "relations": relation_signature,
                    "core": core_signature,
                    "embedding": vector,
                }
            )
            signature_to_representatives[(entry["dataset"], core_signature)].append(best_index)
            best_similarity = 1.0
        representative = representatives[best_index]
        canonical_id = stable_prompt_id(entry["dataset"], representative["normalized_prompt"])
        assignments.append(
            {
                "dataset": entry["dataset"],
                "prompt": entry["prompt"],
                "normalized_prompt": entry["normalized_prompt"],
                "exact_prompt_id": entry["exact_prompt_id"],
                "canonical_prompt_id": canonical_id,
                "canonical_prompt": representative["prompt"],
                "cosine_similarity": best_similarity,
                "trajectory_count": entry["trajectory_count"],
                "window_count": entry["window_count"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = args.output_dir / "canonical_prompt_map.json"
    mapping_path.write_text(json.dumps(assignments, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    clusters: dict[str, list[dict]] = defaultdict(list)
    for item in assignments:
        clusters[item["canonical_prompt_id"]].append(item)
    audit_path = args.output_dir / "semantic_cluster_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["canonical_prompt_id", "cluster_size", "canonical_prompt", "prompt", "similarity", "windows"]
        )
        for canonical_id, members in sorted(clusters.items(), key=lambda item: -sum(x["window_count"] for x in item[1])):
            for member in members:
                writer.writerow(
                    [canonical_id, len(members), member["canonical_prompt"], member["prompt"], member["cosine_similarity"], member["window_count"]]
                )
    summary = {
        "exact_prompt_count": len(entries),
        "canonical_prompt_count": len(clusters),
        "merged_exact_prompts": len(entries) - len(clusters),
        "cosine_threshold": args.cosine_threshold,
        "core_signature_policy": "exact ordered core tokens after grammar aliases and acquisition-tag removal",
        "mapping": str(mapping_path),
        "audit": str(audit_path),
    }
    (args.output_dir / "semantic_cluster_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
