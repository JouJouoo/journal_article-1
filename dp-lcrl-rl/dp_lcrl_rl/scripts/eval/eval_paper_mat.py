#!/usr/bin/env python3
"""Evaluate a trained MAT/MAT-Dec checkpoint on the paper-aligned environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dp_lcrl_rl.runner.shared.paper_mat_runner import PaperMATRunner
from dp_lcrl_rl.scripts.train.train_paper_mat import (
    _apply_cli_aliases,
    _normalize_experiment_args,
    _set_global_seeds,
    build_arg_parser,
    configure_policy_class,
    make_env,
)


def _configure_low_resource_runtime() -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


def parse_args():
    parser = build_arg_parser()
    parser.add_argument("--output_dir", type=str, default="eval_runs")
    parser.add_argument("--eval_repeats", type=int, default=1, help="Number of evaluation episodes to run.")
    parser.add_argument("--export_eval_summary", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--eval_summary_filename", type=str, default="paper_eval_summary.html")
    args = parser.parse_args()
    _apply_cli_aliases(args)
    _normalize_experiment_args(args)

    if not args.model_dir:
        default_models_root = PROJECT_ROOT / "runs" / args.experiment_name / "models"
        if not default_models_root.exists():
            parser.error("--model_dir must be provided for evaluation (no default models directory found).")
        checkpoint_candidates = sorted(
            default_models_root.glob("transformer_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not checkpoint_candidates:
            parser.error("--model_dir must be provided for evaluation (no transformer_*.pt checkpoint found).")
        args.model_dir = str(checkpoint_candidates[0])

    args.num_agents = max(1, int(args.num_agents))
    args.min_agents = max(1, min(int(args.min_agents), args.num_agents))
    args.n_eval_rollout_threads = max(1, int(args.n_eval_rollout_threads or 1))
    args.n_rollout_threads = max(1, int(args.n_rollout_threads or 1))
    args.save_interval = 0
    args.use_eval = True
    return args


def main() -> None:
    args = parse_args()
    _set_global_seeds(args.seed)
    _configure_low_resource_runtime()
    configure_policy_class(args)

    if args.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    eval_envs = make_env(args, args.n_eval_rollout_threads)

    run_dir = PROJECT_ROOT / args.output_dir / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_dir).expanduser().resolve()
    if model_path.is_dir():
        candidates = sorted(
            model_path.glob("transformer_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No transformer_*.pt checkpoint found in {model_path}")
        model_path = candidates[0]
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    args.model_dir = str(model_path)

    config = {
        "all_args": args,
        "envs": eval_envs,
        "eval_envs": eval_envs,
        "device": device,
        "num_agents": args.num_agents,
        "run_dir": run_dir,
    }

    runner = PaperMATRunner(config)
    runner.log_eval_summary = bool(args.export_eval_summary)
    runner.restore(model_path)

    print(f"[Eval] Using checkpoint: {model_path}")
    for _ in range(max(1, int(args.eval_repeats))):
        runner.eval(args.episode_length * args.n_eval_rollout_threads)

    if runner.analytics.eval_history:
        avg_rewards = [item["average_episode_reward"] for item in runner.analytics.eval_history]
        print("Evaluation finished:")
        print(f"  repeats: {len(avg_rewards)}")
        print(f"  mean_episode_reward: {float(np.mean(avg_rewards)):.4f}")
    else:
        print("No evaluation results recorded.")

    if bool(args.export_eval_summary):
        report_path = runner.analytics.export_summary(summary_filename=args.eval_summary_filename)
        print(f"[Eval] Exported summary: {report_path}")

    eval_envs.close()


if __name__ == "__main__":
    main()
