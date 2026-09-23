#!/usr/bin/env python
"""Run one LIBERO task from an existing eval configuration.

This is a thin wrapper around experiments.robot.libero.run_libero_eval.run_task.
It lets a full LIBERO suite be split across multiple GPUs without changing the
underlying rollout logic.
"""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "LIBERO" / "libero"))

from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero import run_libero_eval as libero_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--local_log_dir", default="./experiments/logs")
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--run_id_note", default="single_task")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--save_inputs", action="store_true")
    parser.add_argument("--videos_per_task", type=int, default=5)
    parser.add_argument(
        "--allow_checkpoint_sync",
        action="store_true",
        help="Allow run_libero_eval.py to rewrite local checkpoint config/model files.",
    )
    return parser.parse_args()


def force_local_checkpoint_mode(allow_checkpoint_sync: bool) -> None:
    """Avoid HF network checks and concurrent checkpoint rewrites for local paths."""
    import experiments.robot.openvla_utils as openvla_utils

    openvla_utils.model_is_on_hf_hub = lambda _: False
    if not allow_checkpoint_sync:
        openvla_utils.update_auto_map = lambda _: None
        openvla_utils.check_model_logic_mismatch = lambda _: None


def main() -> None:
    args = parse_args()
    force_local_checkpoint_mode(args.allow_checkpoint_sync)

    cfg = libero_eval.GenerateConfig(
        model_family="openvla",
        pretrained_checkpoint=args.pretrained_checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        use_discrete_diffusion=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=8,
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
        local_log_dir=args.local_log_dir,
        use_wandb=False,
        seed=args.seed,
        save_videos=args.save_videos,
        save_inputs=args.save_inputs,
        videos_per_task=args.videos_per_task,
        run_id_note=f"{args.run_id_note}_task{args.task_id:02d}",
    )

    libero_eval.validate_config(cfg)
    libero_eval.set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, noisy_action_projector, processor = libero_eval.initialize_model(cfg)
    resize_size = libero_eval.get_image_resize_size(cfg)
    log_file, local_log_filepath, _ = libero_eval.setup_logging(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    if args.task_id < 0 or args.task_id >= task_suite.n_tasks:
        raise ValueError(f"task_id must be in [0, {task_suite.n_tasks}), got {args.task_id}")

    total_episodes, total_successes = libero_eval.run_task(
        cfg,
        task_suite,
        args.task_id,
        model,
        resize_size,
        processor,
        action_head,
        proprio_projector,
        noisy_action_projector,
        total_episodes=0,
        total_successes=0,
        log_file=log_file,
    )

    success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
    task = task_suite.get_task(args.task_id)
    summary = {
        "task_suite_name": cfg.task_suite_name,
        "task_id": args.task_id,
        "task_name": getattr(task, "name", ""),
        "language": getattr(task, "language", ""),
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": success_rate,
        "log_file": local_log_filepath,
    }

    libero_eval.log_message("Single-task final results:", log_file)
    libero_eval.log_message(json.dumps(summary, indent=2), log_file)
    if log_file:
        log_file.close()

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
