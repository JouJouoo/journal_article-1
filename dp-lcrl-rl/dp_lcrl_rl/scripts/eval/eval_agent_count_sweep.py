#!/usr/bin/env python3
"""Evaluate final checkpoints under fixed active-agent counts and render tables/plots."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dp_lcrl_rl.envs.p2ptrading.vec_env import ParallelPaperVecEnv
from dp_lcrl_rl.scripts.eval.eval_fixed_testset_convergence import (
    _aggregate_eval_summaries,
    _configure_low_resource_runtime,
    _collect_eval_metrics,
    _resolve_run_label,
)
from dp_lcrl_rl.scripts.train.train_paper_mat import (
    _apply_cli_aliases,
    _normalize_experiment_args,
    _set_global_seeds,
    build_arg_parser,
    make_env,
    resolve_policy_class,
)


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    parser.description = "Evaluate trained checkpoints under fixed active-agent counts."
    parser.add_argument(
        "--run_dir",
        action="append",
        required=True,
        help="One training run directory containing models/transformer_*.pt. Repeat for multiple seeds.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional display label for one --run_dir. Count must match run_dir count when provided.",
    )
    parser.add_argument(
        "--run_policy_arch",
        action="append",
        default=None,
        choices=["transformer", "dp_lcrl", "mlp_pad", "mappo_shared", "deepsets"],
        help=(
            "Optional policy architecture for one --run_dir. "
            "Use this when comparing heterogeneous Experiment 1 methods in one report."
        ),
    )
    parser.add_argument(
        "--checkpoint_episode",
        type=int,
        default=None,
        help="Checkpoint episode to evaluate. Defaults to the latest checkpoint in each run.",
    )
    parser.add_argument(
        "--agent_count_min",
        type=int,
        default=1,
        help="Minimum fixed active-agent count to evaluate.",
    )
    parser.add_argument(
        "--agent_count_max",
        type=int,
        default=None,
        help="Maximum fixed active-agent count to evaluate. Defaults to --num_agents.",
    )
    parser.add_argument(
        "--agent_count",
        action="append",
        type=int,
        default=None,
        help="Explicit active-agent count to evaluate. Repeat to use a sparse sweep such as 5, 10, 15, 20, 25, 30.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=20,
        help="Evaluation episodes per agent count for each run.",
    )
    parser.add_argument(
        "--fixed_eval_seed",
        type=int,
        default=20260417,
        help="Base seed used to generate the fixed test set.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/agent_count_sweep",
        help="Directory used to save plots and tables.",
    )
    parser.add_argument(
        "--report_name",
        type=str,
        default="agent_count_sweep",
        help="Prefix used for output files.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Fixed Active-Agent Count Sweep",
        help="Figure title.",
    )
    args = parser.parse_args()
    _apply_cli_aliases(args)
    _normalize_experiment_args(args)
    args.run_dir = [Path(item).expanduser().resolve() for item in args.run_dir]
    if args.label and len(args.label) != len(args.run_dir):
        parser.error("When --label is provided, its count must match --run_dir.")
    if args.run_policy_arch and len(args.run_policy_arch) != len(args.run_dir):
        parser.error("When --run_policy_arch is provided, its count must match --run_dir.")
    args.eval_episodes = max(1, int(args.eval_episodes))
    args.n_eval_rollout_threads = max(1, int(args.n_eval_rollout_threads or 1))
    args.agent_count_min = max(1, int(args.agent_count_min))
    max_count = int(args.num_agents) if args.agent_count_max is None else int(args.agent_count_max)
    args.agent_count_max = max(args.agent_count_min, min(max_count, int(args.num_agents)))
    if args.agent_count:
        counts = sorted({max(1, min(int(item), int(args.num_agents))) for item in args.agent_count})
        args.agent_count = counts
        args.agent_count_min = min(counts)
        args.agent_count_max = max(counts)
    args.save_interval = 0
    args.use_eval = False
    return args


def _parse_checkpoint_episode(path: Path) -> int:
    match = re.search(r"transformer_(\d+)\.pt$", path.name)
    if not match:
        raise ValueError(f"Unsupported checkpoint filename: {path.name}")
    return int(match.group(1))


def _resolve_checkpoint_path(run_dir: Path, checkpoint_episode: int | None) -> Path:
    model_dir = run_dir / "models"
    if checkpoint_episode is not None:
        path = model_dir / f"transformer_{int(checkpoint_episode)}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    candidates = sorted(model_dir.glob("transformer_*.pt"), key=_parse_checkpoint_episode)
    if not candidates:
        raise FileNotFoundError(f"No transformer_*.pt checkpoint found in {model_dir}")
    return candidates[-1]


def _build_policy(args: argparse.Namespace, device: torch.device):
    env = make_env(args, args.n_eval_rollout_threads)
    share_observation_space = (
        env.share_observation_space[0] if args.use_centralized_V else env.observation_space[0]
    )
    policy_class = resolve_policy_class(args)
    policy = policy_class(
        args,
        env.observation_space[0],
        share_observation_space,
        env.action_space[0],
        args.num_agents,
        device=device,
    )
    return env, policy


def _bind_fixed_active_count(envs: ParallelPaperVecEnv, active_count: int) -> None:
    active_count = max(1, min(int(active_count), int(envs.num_agents)))
    envs.set_min_agents(active_count)
    envs.set_step_churn_prob(0.0)

    for env in envs.envs:
        env.min_agents = active_count
        env.step_churn_prob = 0.0

        def _sample_exact_ids(*, _env=env, _count=active_count):
            if _count >= _env.max_agents:
                return list(range(_env.max_agents))
            ids = _env.rng.choice(_env.max_agents, size=_count, replace=False)
            return sorted(int(x) for x in ids.tolist())

        env._sample_active_ids = _sample_exact_ids  # type: ignore[attr-defined]


def evaluate_run(
    args: argparse.Namespace,
    run_dir: Path,
    label: str,
    active_counts: Sequence[int],
    device: torch.device,
) -> List[Dict[str, object]]:
    checkpoint_path = _resolve_checkpoint_path(run_dir, args.checkpoint_episode)
    checkpoint_episode = _parse_checkpoint_episode(checkpoint_path)
    records: List[Dict[str, object]] = []

    for active_count in active_counts:
        eval_args = copy.deepcopy(args)
        eval_args.seed = int(args.fixed_eval_seed)
        eval_args.min_agents = int(active_count)
        eval_args.step_churn_prob = 0.0
        _set_global_seeds(eval_args.seed)
        envs, policy = _build_policy(eval_args, device)
        _bind_fixed_active_count(envs, active_count)

        try:
            policy.restore(str(checkpoint_path))
            policy.eval()
            episode_summaries = _collect_eval_metrics(
                args=eval_args,
                policy=policy,
                envs=envs,
                eval_episodes=args.eval_episodes,
            )
        finally:
            envs.close()

        metrics = _aggregate_eval_summaries(episode_summaries)
        metrics.update(
            run_dir=str(run_dir),
            label=str(label),
            method=str(label).split("-seed", 1)[0] if "-seed" in str(label) else str(getattr(args, "policy_arch", label)),
            checkpoint_episode=int(checkpoint_episode),
            checkpoint_path=str(checkpoint_path),
            active_agent_count=int(active_count),
            policy_arch=str(getattr(args, "policy_arch", "transformer")),
        )
        records.append(metrics)
        print(
            "[AgentCountEval] {} | active_agents={} | reward={:.4f} | p2p={:.4f} | grid_buy={:.4f} | carbon={:.4f}".format(
                label,
                int(active_count),
                metrics["reward_mean"],
                metrics["p2p_mean"],
                metrics["grid_buy_mean"],
                metrics["carbon_mean"],
            )
        )

    return records


def _add_dynamic_drop(records: Sequence[Dict[str, object]]) -> None:
    by_label: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        by_label.setdefault(str(record["label"]), []).append(record)

    for bucket in by_label.values():
        max_count = max(int(item["active_agent_count"]) for item in bucket)
        refs = [item for item in bucket if int(item["active_agent_count"]) == max_count]
        if not refs:
            continue
        ref_reward = float(refs[0]["reward_mean"])
        denom = max(abs(ref_reward), 1e-9)
        for item in bucket:
            item["reference_active_agent_count"] = int(max_count)
            item["reward_drop_vs_max_count"] = float((ref_reward - float(item["reward_mean"])) / denom)


def _build_aggregate_rows(records: Sequence[Dict[str, object]]) -> List[Dict[str, float]]:
    by_count: Dict[tuple[str, int], List[Dict[str, object]]] = {}
    for record in records:
        method = str(record.get("method", record.get("policy_arch", "aggregate")))
        by_count.setdefault((method, int(record["active_agent_count"])), []).append(record)

    rows: List[Dict[str, float]] = []
    for method, active_count in sorted(by_count):
        bucket = by_count[(method, active_count)]

        def stats(key: str) -> tuple[float, float]:
            values = np.asarray([float(item[key]) for item in bucket], dtype=np.float64)
            return float(np.mean(values)), float(np.std(values))

        reward_mean, reward_std = stats("reward_mean")
        p2p_mean, p2p_std = stats("p2p_mean")
        grid_buy_mean, grid_buy_std = stats("grid_buy_mean")
        grid_sell_mean, grid_sell_std = stats("grid_sell_mean")
        carbon_mean, carbon_std = stats("carbon_mean")
        n_agents_mean, n_agents_std = stats("n_agents_mean")
        if "reward_drop_vs_max_count" in bucket[0]:
            reward_drop_mean, reward_drop_std = stats("reward_drop_vs_max_count")
        else:
            reward_drop_mean, reward_drop_std = 0.0, 0.0
        rows.append(
            {
                "method": method,
                "active_agent_count": int(active_count),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "reward_drop_vs_max_count_mean": reward_drop_mean,
                "reward_drop_vs_max_count_std": reward_drop_std,
                "p2p_mean": p2p_mean,
                "p2p_std": p2p_std,
                "grid_buy_mean": grid_buy_mean,
                "grid_buy_std": grid_buy_std,
                "grid_sell_mean": grid_sell_mean,
                "grid_sell_std": grid_sell_std,
                "carbon_mean": carbon_mean,
                "carbon_std": carbon_std,
                "n_agents_mean": n_agents_mean,
                "n_agents_std": n_agents_std,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    args: argparse.Namespace,
    records: Sequence[Dict[str, object]],
    aggregate_rows: Sequence[Dict[str, object]],
) -> None:
    labels = sorted({str(item["label"]) for item in records})
    checkpoint_episodes = sorted({int(item["checkpoint_episode"]) for item in records})
    lines = [
        "# Fixed Active-Agent Count Evaluation",
        "",
        f"- Runs: `{', '.join(labels)}`",
        f"- Checkpoint: `{', '.join(str(item) for item in checkpoint_episodes)}`",
        f"- Eval episodes per count: `{int(args.eval_episodes)}`",
        f"- n_eval_rollout_threads: `{int(args.n_eval_rollout_threads)}`",
        f"- Active-agent sweep: `{int(args.agent_count_min)}..{int(args.agent_count_max)}`",
        f"- Fixed eval seed: `{int(args.fixed_eval_seed)}`",
        "",
        "## Aggregate (mean ± std across runs)",
        "",
        "| Active Agents | Reward | P2P Mean | Grid Buy Mean | Grid Sell Mean | Carbon Mean |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {count} | {r:.4f} ± {rs:.4f} | {p:.4f} ± {ps:.4f} | {gb:.4f} ± {gbs:.4f} | {gs:.4f} ± {gss:.4f} | {c:.4f} ± {cs:.4f} |".format(
                count=int(row["active_agent_count"]),
                r=float(row["reward_mean"]),
                rs=float(row["reward_std"]),
                p=float(row["p2p_mean"]),
                ps=float(row["p2p_std"]),
                gb=float(row["grid_buy_mean"]),
                gbs=float(row["grid_buy_std"]),
                gs=float(row["grid_sell_mean"]),
                gss=float(row["grid_sell_std"]),
                c=float(row["carbon_mean"]),
                cs=float(row["carbon_std"]),
            )
        )

    lines.extend(
        [
            "",
            "## Per-Run Results",
            "",
            "| Run | Active Agents | Reward | P2P Mean | Grid Buy Mean | Grid Sell Mean | Carbon Mean |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in records:
        lines.append(
            "| {label} | {count} | {r:.4f} | {p:.4f} | {gb:.4f} | {gs:.4f} | {c:.4f} |".format(
                label=str(record["label"]),
                count=int(record["active_agent_count"]),
                r=float(record["reward_mean"]),
                p=float(record["p2p_mean"]),
                gb=float(record["grid_buy_mean"]),
                gs=float(record["grid_sell_mean"]),
                c=float(record["carbon_mean"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(
    path: Path,
    args: argparse.Namespace,
    records: Sequence[Dict[str, object]],
    aggregate_rows: Sequence[Dict[str, object]],
) -> None:
    labels = sorted({str(item["label"]) for item in records})
    checkpoint_episodes = sorted({int(item["checkpoint_episode"]) for item in records})
    explicit_counts = ", ".join(str(item) for item in getattr(args, "agent_count", []) or [])
    sweep_label = explicit_counts or f"{int(args.agent_count_min)}..{int(args.agent_count_max)}"
    lines = [
        "# Fixed Active-Agent Count Evaluation",
        "",
        f"- Runs: `{', '.join(labels)}`",
        f"- Checkpoint: `{', '.join(str(item) for item in checkpoint_episodes)}`",
        f"- Eval episodes per count: `{int(args.eval_episodes)}`",
        f"- n_eval_rollout_threads: `{int(args.n_eval_rollout_threads)}`",
        f"- Active-agent sweep: `{sweep_label}`",
        f"- Fixed eval seed: `{int(args.fixed_eval_seed)}`",
        "",
        "## Aggregate (mean +/- std across runs)",
        "",
        "| Method | Active Agents | Reward | Drop vs Max Count | P2P Mean | Grid Buy Mean | Grid Sell Mean | Carbon Mean |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {method} | {count} | {r:.4f} +/- {rs:.4f} | {drop:.4f} +/- {drops:.4f} | {p:.4f} +/- {ps:.4f} | {gb:.4f} +/- {gbs:.4f} | {gs:.4f} +/- {gss:.4f} | {c:.4f} +/- {cs:.4f} |".format(
                method=str(row.get("method", "aggregate")),
                count=int(row["active_agent_count"]),
                r=float(row["reward_mean"]),
                rs=float(row["reward_std"]),
                drop=float(row.get("reward_drop_vs_max_count_mean", 0.0)),
                drops=float(row.get("reward_drop_vs_max_count_std", 0.0)),
                p=float(row["p2p_mean"]),
                ps=float(row["p2p_std"]),
                gb=float(row["grid_buy_mean"]),
                gbs=float(row["grid_buy_std"]),
                gs=float(row["grid_sell_mean"]),
                gss=float(row["grid_sell_std"]),
                c=float(row["carbon_mean"]),
                cs=float(row["carbon_std"]),
            )
        )

    lines.extend(
        [
            "",
            "## Per-Run Results",
            "",
            "| Run | Active Agents | Reward | Drop vs Max Count | P2P Mean | Grid Buy Mean | Grid Sell Mean | Carbon Mean |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in records:
        lines.append(
            "| {label} | {count} | {r:.4f} | {drop:.4f} | {p:.4f} | {gb:.4f} | {gs:.4f} | {c:.4f} |".format(
                label=str(record["label"]),
                count=int(record["active_agent_count"]),
                r=float(record["reward_mean"]),
                drop=float(record.get("reward_drop_vs_max_count", 0.0)),
                p=float(record["p2p_mean"]),
                gb=float(record["grid_buy_mean"]),
                gs=float(record["grid_sell_mean"]),
                c=float(record["carbon_mean"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_overlay(records: Sequence[Dict[str, object]], output_path: Path, title: str) -> Path:
    by_label: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        by_label.setdefault(str(record["label"]), []).append(record)
    colors = ["#111111", "#d94801", "#2171b5", "#238b45", "#756bb1", "#dd1c77"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    fig.suptitle(title, fontsize=16)

    ax = axes[0, 0]
    for idx, (label, bucket) in enumerate(sorted(by_label.items())):
        bucket = sorted(bucket, key=lambda item: int(item["active_agent_count"]))
        xs = [int(item["active_agent_count"]) for item in bucket]
        ys = [float(item["reward_mean"]) for item in bucket]
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=colors[idx % len(colors)], label=label)
    ax.set_title("Reward")
    ax.set_xlabel("Active Agent Count")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for idx, (label, bucket) in enumerate(sorted(by_label.items())):
        bucket = sorted(bucket, key=lambda item: int(item["active_agent_count"]))
        xs = [int(item["active_agent_count"]) for item in bucket]
        ys = [float(item["p2p_mean"]) for item in bucket]
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=colors[idx % len(colors)], label=label)
    ax.set_title("P2P Volume Mean per Active Agent")
    ax.set_xlabel("Active Agent Count")
    ax.set_ylabel("Mean Volume")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for idx, (label, bucket) in enumerate(sorted(by_label.items())):
        bucket = sorted(bucket, key=lambda item: int(item["active_agent_count"]))
        xs = [int(item["active_agent_count"]) for item in bucket]
        buy = [float(item["grid_buy_mean"]) for item in bucket]
        sell = [float(item["grid_sell_mean"]) for item in bucket]
        color = colors[idx % len(colors)]
        ax.plot(xs, buy, marker="o", linewidth=2.0, color=color, label=f"{label} Grid Buy")
        ax.plot(xs, sell, marker="o", linewidth=1.8, linestyle="--", color=color, label=f"{label} Grid Sell")
    ax.set_title("Grid Trading Mean per Active Agent")
    ax.set_xlabel("Active Agent Count")
    ax.set_ylabel("Mean Volume")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[1, 1]
    for idx, (label, bucket) in enumerate(sorted(by_label.items())):
        bucket = sorted(bucket, key=lambda item: int(item["active_agent_count"]))
        xs = [int(item["active_agent_count"]) for item in bucket]
        ys = [float(item["carbon_mean"]) for item in bucket]
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=colors[idx % len(colors)], label=label)
    ax.set_title("Carbon Responsibility Episode Mean")
    ax.set_xlabel("Active Agent Count")
    ax.set_ylabel("Mean Carbon Responsibility")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_mean_std(aggregate_rows: Sequence[Dict[str, float]], output_path: Path, title: str) -> Path:
    xs = np.asarray([int(item["active_agent_count"]) for item in aggregate_rows], dtype=np.int32)
    reward_mean = np.asarray([float(item["reward_mean"]) for item in aggregate_rows], dtype=np.float64)
    reward_std = np.asarray([float(item["reward_std"]) for item in aggregate_rows], dtype=np.float64)
    p2p_mean = np.asarray([float(item["p2p_mean"]) for item in aggregate_rows], dtype=np.float64)
    p2p_std = np.asarray([float(item["p2p_std"]) for item in aggregate_rows], dtype=np.float64)
    grid_buy_mean = np.asarray([float(item["grid_buy_mean"]) for item in aggregate_rows], dtype=np.float64)
    grid_buy_std = np.asarray([float(item["grid_buy_std"]) for item in aggregate_rows], dtype=np.float64)
    grid_sell_mean = np.asarray([float(item["grid_sell_mean"]) for item in aggregate_rows], dtype=np.float64)
    grid_sell_std = np.asarray([float(item["grid_sell_std"]) for item in aggregate_rows], dtype=np.float64)
    carbon_mean = np.asarray([float(item["carbon_mean"]) for item in aggregate_rows], dtype=np.float64)
    carbon_std = np.asarray([float(item["carbon_std"]) for item in aggregate_rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    fig.suptitle(title, fontsize=16)

    ax = axes[0, 0]
    ax.plot(xs, reward_mean, marker="o", linewidth=2.2, color="#111111", label="Mean Reward")
    ax.fill_between(xs, reward_mean - reward_std, reward_mean + reward_std, color="#999999", alpha=0.25, label="±1 std")
    ax.set_title("Reward")
    ax.set_xlabel("Active Agent Count")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(xs, p2p_mean, marker="o", linewidth=2.2, color="#d94801", label="Mean P2P Volume")
    ax.fill_between(xs, p2p_mean - p2p_std, p2p_mean + p2p_std, color="#fdae6b", alpha=0.25, label="±1 std")
    ax.set_title("P2P Volume Mean per Active Agent")
    ax.set_xlabel("Active Agent Count")
    ax.set_ylabel("Mean Volume")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(xs, grid_buy_mean, marker="o", linewidth=2.2, color="#238b45", label="Grid Buy Mean")
    ax.fill_between(xs, grid_buy_mean - grid_buy_std, grid_buy_mean + grid_buy_std, color="#74c476", alpha=0.18)
    ax.plot(xs, grid_sell_mean, marker="o", linewidth=2.2, color="#756bb1", linestyle="--", label="Grid Sell Mean")
    ax.fill_between(xs, grid_sell_mean - grid_sell_std, grid_sell_mean + grid_sell_std, color="#bcbddc", alpha=0.18)
    ax.set_title("Grid Trading Mean per Active Agent")
    ax.set_xlabel("Active Agent Count")
    ax.set_ylabel("Mean Volume")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(xs, carbon_mean, marker="o", linewidth=2.2, color="#08519c", label="Carbon Responsibility Mean")
    ax.fill_between(xs, carbon_mean - carbon_std, carbon_mean + carbon_std, color="#9ecae1", alpha=0.25, label="±1 std")
    ax.set_title("Carbon Responsibility Episode Mean")
    ax.set_xlabel("Active Agent Count")
    ax.set_ylabel("Mean Carbon Responsibility")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_mean_std(aggregate_rows: Sequence[Dict[str, object]], output_path: Path, title: str) -> Path:
    by_method: Dict[str, List[Dict[str, object]]] = {}
    for row in aggregate_rows:
        by_method.setdefault(str(row.get("method", "aggregate")), []).append(row)
    colors = ["#111111", "#d94801", "#2171b5", "#238b45", "#756bb1", "#dd1c77"]
    metrics = [
        ("reward_mean", "reward_std", "Reward", "Mean Reward"),
        ("p2p_mean", "p2p_std", "P2P Volume Mean per Active Agent", "Mean Volume"),
        ("grid_buy_mean", "grid_buy_std", "Grid Buy Mean per Active Agent", "Mean Volume"),
        ("carbon_mean", "carbon_std", "Carbon Responsibility Episode Mean", "Mean Carbon Responsibility"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    fig.suptitle(title, fontsize=16)
    for ax, (mean_key, std_key, panel_title, ylabel) in zip(axes.ravel(), metrics):
        for idx, (method, bucket) in enumerate(sorted(by_method.items())):
            bucket = sorted(bucket, key=lambda item: int(item["active_agent_count"]))
            xs = np.asarray([int(item["active_agent_count"]) for item in bucket], dtype=np.int32)
            ys = np.asarray([float(item[mean_key]) for item in bucket], dtype=np.float64)
            std = np.asarray([float(item.get(std_key, 0.0)) for item in bucket], dtype=np.float64)
            color = colors[idx % len(colors)]
            ax.plot(xs, ys, marker="o", linewidth=2.0, color=color, label=method)
            if np.any(std > 0):
                ax.fill_between(xs, ys - std, ys + std, color=color, alpha=0.12)
        ax.set_title(panel_title)
        ax.set_xlabel("Active Agent Count")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_scatter_errorbar(
    aggregate_rows: Sequence[Dict[str, object]],
    output_path: Path,
    title: str,
) -> Path:
    by_method: Dict[str, List[Dict[str, object]]] = {}
    for row in aggregate_rows:
        by_method.setdefault(str(row.get("method", "aggregate")), []).append(row)

    if len(by_method) > 1:
        raise ValueError("Scatter error-bar plot expects one evaluated method.")

    rows = sorted(next(iter(by_method.values())), key=lambda item: int(item["active_agent_count"]))
    xs = np.asarray([int(item["active_agent_count"]) for item in rows], dtype=np.int32)

    def values(key: str) -> np.ndarray:
        return np.asarray([float(item.get(key, 0.0)) for item in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=180)
    fig.suptitle(title, fontsize=18, y=0.98)

    plot_specs = [
        (axes[0, 0], "reward_mean", "reward_std", "Reward", "Mean Reward", "#1a1a1a"),
        (
            axes[0, 1],
            "p2p_mean",
            "p2p_std",
            "P2P Trading Volume",
            "Mean Volume per Active Agent",
            "#238b45",
        ),
        (
            axes[1, 1],
            "carbon_mean",
            "carbon_std",
            "Carbon Responsibility",
            "Mean Carbon Responsibility",
            "#756bb1",
        ),
    ]

    for ax, mean_key, std_key, panel_title, ylabel, color in plot_specs:
        ax.errorbar(
            xs,
            values(mean_key),
            yerr=values(std_key),
            fmt="o",
            markersize=4.5,
            linewidth=1.2,
            elinewidth=1.2,
            capsize=3,
            color=color,
            ecolor=color,
            alpha=0.9,
        )
        ax.set_title(panel_title, fontsize=13)
        ax.set_xlabel("Active Agent Count")
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(1, int(xs.max()) + 1, 2))
        ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.errorbar(
        xs,
        values("grid_buy_mean"),
        yerr=values("grid_buy_std"),
        fmt="o",
        markersize=4.5,
        linewidth=1.2,
        elinewidth=1.2,
        capsize=3,
        color="#e6550d",
        ecolor="#e6550d",
        alpha=0.9,
        label="Grid Buy",
    )
    ax.errorbar(
        xs + 0.18,
        values("grid_sell_mean"),
        yerr=values("grid_sell_std"),
        fmt="o",
        markersize=4.5,
        linewidth=1.2,
        elinewidth=1.2,
        capsize=3,
        color="#3182bd",
        ecolor="#3182bd",
        alpha=0.9,
        label="Grid Sell",
    )
    ax.set_title("Grid Buy and Grid Sell", fontsize=13)
    ax.set_xlabel("Active Agent Count")
    ax.set_ylabel("Mean Volume per Active Agent")
    ax.set_xticks(np.arange(1, int(xs.max()) + 1, 2))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="center right")

    fig.text(
        0.5,
        0.02,
        "Each point is an independent fixed active-agent-count evaluation; "
        "error bars show standard deviation across seeds 0/1/2.",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), h_pad=2.0, w_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_dynamic_drop(aggregate_rows: Sequence[Dict[str, object]], output_path: Path, title: str) -> Path:
    by_method: Dict[str, List[Dict[str, object]]] = {}
    for row in aggregate_rows:
        by_method.setdefault(str(row.get("method", "aggregate")), []).append(row)
    colors = ["#111111", "#d94801", "#2171b5", "#238b45", "#756bb1", "#dd1c77"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=180)
    for idx, (method, bucket) in enumerate(sorted(by_method.items())):
        bucket = sorted(bucket, key=lambda item: int(item["active_agent_count"]))
        xs = np.asarray([int(item["active_agent_count"]) for item in bucket], dtype=np.int32)
        ys = np.asarray(
            [float(item.get("reward_drop_vs_max_count_mean", 0.0)) for item in bucket],
            dtype=np.float64,
        )
        std = np.asarray(
            [float(item.get("reward_drop_vs_max_count_std", 0.0)) for item in bucket],
            dtype=np.float64,
        )
        color = colors[idx % len(colors)]
        ax.plot(xs, ys, marker="o", linewidth=2.2, color=color, label=method)
        if np.any(std > 0):
            ax.fill_between(xs, ys - std, ys + std, color=color, alpha=0.12)
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Active Agent Count")
    ax.set_ylabel("Reward Drop vs Max Active Count")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    _configure_low_resource_runtime()
    for run_dir in args.run_dir:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    active_counts = (
        list(args.agent_count)
        if args.agent_count
        else list(range(int(args.agent_count_min), int(args.agent_count_max) + 1))
    )
    labels = [
        _resolve_run_label(run_dir, args.label[idx] if args.label else None)
        for idx, run_dir in enumerate(args.run_dir)
    ]

    if args.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    all_records: List[Dict[str, object]] = []
    for idx, (run_dir, label) in enumerate(zip(args.run_dir, labels)):
        eval_args = copy.deepcopy(args)
        if args.run_policy_arch:
            eval_args.policy_arch = str(args.run_policy_arch[idx])
        all_records.extend(
            evaluate_run(
                args=eval_args,
                run_dir=run_dir,
                label=label,
                active_counts=active_counts,
                device=device,
            )
        )

    _add_dynamic_drop(all_records)
    aggregate_rows = _build_aggregate_rows(all_records)

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = str(args.report_name)
    raw_csv = output_dir / f"{prefix}_raw.csv"
    aggregate_csv = output_dir / f"{prefix}_aggregate.csv"
    report_md = output_dir / f"{prefix}.md"
    report_json = output_dir / f"{prefix}.json"
    overlay_png = output_dir / f"{prefix}_overlay.png"
    mean_std_png = output_dir / f"{prefix}_mean_std.png"
    scatter_errorbar_png = output_dir / f"{prefix}_scatter_errorbar.png"
    dynamic_drop_png = output_dir / f"{prefix}_dynamic_drop.png"

    _write_csv(raw_csv, all_records)
    _write_csv(aggregate_csv, aggregate_rows)
    _write_markdown(report_md, args, all_records, aggregate_rows)
    report_json.write_text(
        json.dumps(
            {
                "config": {
                    "fixed_eval_seed": int(args.fixed_eval_seed),
                    "eval_episodes": int(args.eval_episodes),
                    "n_eval_rollout_threads": int(args.n_eval_rollout_threads),
                    "num_agents": int(args.num_agents),
                    "active_counts": active_counts,
                    "agent_count": list(args.agent_count or []),
                    "checkpoint_episode": int(args.checkpoint_episode) if args.checkpoint_episode is not None else None,
                    "run_dirs": [str(item) for item in args.run_dir],
                    "labels": labels,
                    "policy_arch": str(args.policy_arch),
                    "run_policy_arch": list(args.run_policy_arch or []),
                },
                "records": all_records,
                "aggregate": aggregate_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_overlay(all_records, overlay_png, f"{args.title} (Per-Seed)")
    _plot_scatter_errorbar(aggregate_rows, scatter_errorbar_png, str(args.title))
    _plot_mean_std(aggregate_rows, mean_std_png, f"{args.title} (Mean ± Std)")

    _plot_dynamic_drop(aggregate_rows, dynamic_drop_png, f"{args.title} Dynamic Reward Drop")

    print(f"saved_raw_csv={raw_csv}")
    print(f"saved_aggregate_csv={aggregate_csv}")
    print(f"saved_markdown={report_md}")
    print(f"saved_json={report_json}")
    print(f"saved_overlay_plot={overlay_png}")
    print(f"saved_mean_std_plot={mean_std_png}")
    print(f"saved_scatter_errorbar_plot={scatter_errorbar_png}")
    print(f"saved_dynamic_drop_plot={dynamic_drop_png}")


if __name__ == "__main__":
    main()
