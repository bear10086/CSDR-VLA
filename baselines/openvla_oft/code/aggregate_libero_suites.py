#!/usr/bin/env python3
"""Aggregate four LIBERO suite summaries into one checkpoint-level result."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite-summary",
        action="append",
        required=True,
        metavar="SUITE=PATH",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    suites = {}
    for item in args.suite_summary:
        suite, separator, raw_path = item.partition("=")
        if not separator or not suite or not raw_path:
            raise ValueError(f"Expected SUITE=PATH, got: {item}")
        with Path(raw_path).open(encoding="utf-8") as stream:
            summary = json.load(stream)
        if summary.get("num_tasks_done") != 10 or summary.get("episodes") != 500:
            raise ValueError(f"Incomplete suite result for {suite}: {raw_path}")
        suites[suite] = {
            "episodes": int(summary["episodes"]),
            "successes": int(summary["successes"]),
            "success_rate": float(summary["success_rate"]),
            "summary_path": str(Path(raw_path)),
        }

    if len(suites) != 4:
        raise ValueError(f"Expected four suites, got {sorted(suites)}")
    episodes = sum(item["episodes"] for item in suites.values())
    successes = sum(item["successes"] for item in suites.values())
    payload = {
        "num_suites": len(suites),
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "suites": suites,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
