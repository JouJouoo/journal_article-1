#!/usr/bin/env python3
"""Run Experiment 1: sequence modeling effectiveness across dynamic agent counts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]


METHOD_TO_ARCH = {
    "dp_lcrl": "transformer",
    "mlp_pad": "mlp_pad",
    "mappo_shared": "mappo_shared",
    "deepsets": "deepsets",
}


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_csv_methods(value: str) -> list[str]:
    methods = [item.strip().lower() for item in str(value).split(",") if item.strip()]
    unknown = [item for item in methods if item not in METHOD_TO_ARCH]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unsupported method(s): {', '.join(unknown)}. "
            f"Expected subset of {', '.join(METHOD_TO_ARCH)}."
        )
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DP-LCRL Experiment 1 baselines and active-count evaluation.")
    parser.add_argument(
        "--methods",
        type=_parse_csv_methods,
        default=_parse_csv_methods("dp_lcrl,mlp_pad,mappo_shared,deepsets"),
        help="Comma-separated methods: dp_lcrl, mlp_pad, mappo_shared, deepsets.",
    )
    parser.add_argument("--seeds", type=_parse_csv_ints, default=_parse_csv_ints("42,43,44"))
    parser.add_argument("--active_counts", type=_parse_csv_ints, default=_parse_csv_ints("5,10,15,20,25,30"))
    parser.add_argument("--experiment_prefix", type=str, default="exp1_sequence_modeling")
    parser.add_argument("--num_agents", type=int, default=30)
    parser.add_argument("--min_agents", type=int, default=5)
    parser.add_argument("--episode_length", type=int, default=24)
    parser.add_argument("--n_rollout_threads", type=int, default=4)
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1)
    parser.add_argument("--num_env_steps", type=int, default=240_000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--checkpoint_episode", type=int, default=None)
    parser.add_argument("--skip_training", action="store_true", default=False)
    parser.add_argument("--skip_eval", action="store_true", default=False)
    parser.add_argument("--report_dir", type=str, default="reports/experiment1_sequence_modeling")
    parser.add_argument("--ppo_epoch", type=int, default=5)
    parser.add_argument("--num_mini_batch", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_block", type=int, default=1)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--layer_N", type=int, default=2)
    return parser.parse_args()


def _run(cmd: list[str], cwd: Path) -> None:
    print("[Experiment1] " + " ".join(cmd), flush=True)
    env = dict(os.environ)
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("DP_LCRL_DISABLE_TENSORBOARD", "true")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _run_name(prefix: str, method: str, seed: int) -> str:
    return f"{prefix}_{method}_seed{seed}"


def main() -> None:
    args = parse_args()
    py = sys.executable

    for method in args.methods:
        arch = METHOD_TO_ARCH[method]
        for seed in args.seeds:
            run_name = _run_name(args.experiment_prefix, method, seed)
            if not args.skip_training:
                train_cmd = [
                    py,
                    "-m",
                    "dp_lcrl_rl.scripts.train.train_paper_mat",
                    "--experiment_name",
                    run_name,
                    "--policy_arch",
                    arch,
                    "--num_agents",
                    str(args.num_agents),
                    "--min_agents",
                    str(args.min_agents),
                    "--curriculum_min_agents",
                    str(args.min_agents),
                    "--scale_mode",
                    "random_scale",
                    "--episode_length",
                    str(args.episode_length),
                    "--n_rollout_threads",
                    str(args.n_rollout_threads),
                    "--n_eval_rollout_threads",
                    str(args.n_eval_rollout_threads),
                    "--num_env_steps",
                    str(args.num_env_steps),
                    "--seed",
                    str(seed),
                    "--ppo_epoch",
                    str(args.ppo_epoch),
                    "--num_mini_batch",
                    str(args.num_mini_batch),
                    "--hidden_size",
                    str(args.hidden_size),
                    "--n_embd",
                    str(args.n_embd),
                    "--n_block",
                    str(args.n_block),
                    "--n_head",
                    str(args.n_head),
                    "--layer_N",
                    str(args.layer_N),
                    "--use_eval",
                    "false",
                    "--save_interval",
                    "0",
                    "--use_wandb",
                    "false",
                ]
                _run(train_cmd, PROJECT_ROOT)

    if args.skip_eval:
        return

    eval_cmd = [
        py,
        "-m",
        "dp_lcrl_rl.scripts.eval.eval_agent_count_sweep",
        "--num_agents",
        str(args.num_agents),
        "--min_agents",
        str(args.min_agents),
        "--curriculum_min_agents",
        str(args.min_agents),
        "--episode_length",
        str(args.episode_length),
        "--n_eval_rollout_threads",
        str(args.n_eval_rollout_threads),
        "--hidden_size",
        str(args.hidden_size),
        "--n_embd",
        str(args.n_embd),
        "--n_block",
        str(args.n_block),
        "--n_head",
        str(args.n_head),
        "--layer_N",
        str(args.layer_N),
        "--eval_episodes",
        str(args.eval_episodes),
        "--agent_count_min",
        str(min(args.active_counts)),
        "--agent_count_max",
        str(max(args.active_counts)),
        "--output_dir",
        args.report_dir,
        "--report_name",
        args.experiment_prefix,
        "--title",
        "Experiment 1 Sequence Modeling Effectiveness",
    ]
    if args.checkpoint_episode is not None:
        eval_cmd.extend(["--checkpoint_episode", str(args.checkpoint_episode)])
    for count in args.active_counts:
        eval_cmd.extend(["--agent_count", str(count)])

    for method in args.methods:
        arch = METHOD_TO_ARCH[method]
        for seed in args.seeds:
            run_name = _run_name(args.experiment_prefix, method, seed)
            eval_cmd.extend(["--run_dir", str(PROJECT_ROOT / "runs" / run_name)])
            eval_cmd.extend(["--label", f"{method}-seed{seed}"])
            eval_cmd.extend(["--run_policy_arch", arch])

    _run(eval_cmd, PROJECT_ROOT)


if __name__ == "__main__":
    main()
