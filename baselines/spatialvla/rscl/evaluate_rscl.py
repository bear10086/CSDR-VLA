"""Evaluate an RS-CL continuation checkpoint with native SpatialVLA evaluation.

The rollout implementation and task protocol are shared with this release's
SpatialVLA evaluator. This entry point only supplies the checkpoint and output
directory, so it does not introduce a separate evaluation protocol.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


RSCL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RSCL_ROOT.parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()

    command = [
        sys.executable,
        str(PROJECT_ROOT / "evaluate.py"),
        "spatialvla",
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--output",
        str(args.output.resolve()),
        "--gpus",
        args.gpus,
    ]
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
