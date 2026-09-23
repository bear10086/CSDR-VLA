#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfrecord_store import build_tfrecord_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = build_tfrecord_index(args.dataset_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index, separators=(",", ":")) + "\n", encoding="utf-8")
    summary = {
        "files": len(index["files"]),
        "records": sum(len(item["records"]) for item in index["files"].values()),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

