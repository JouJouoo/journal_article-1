#!/usr/bin/env python3
"""Evaluate multiple checkpoints on a fixed test set and render paper-style convergence plots."""

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

from dp_lcrl_rl.runner.shared.base_runner import _t2n
from dp_lcrl_rl.runner.shared.training_analytics import PaperTrainingAnalytics
from dp_lcrl_rl.scripts.train.train_paper_mat import (
    _apply_cli_aliases,
    _normalize_experiment_args,
    _set_global_seeds,
    build_arg_parser,
    make_env,
    resolve_policy_class,
)


def _configure_low_resource_runtime() -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    parser.description = "Evaluate checkpoint convergence on a fixed test set."
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
        "--checkpoint",
        action="append",
        type=int,
        default=None,
        help="Explicit checkpoint episode number to evaluate, e.g. --checkpoint 1000 --checkpoint 2000.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=20,
        help="Number of fixed test episodes to evaluate for each checkpoint.",
    )
    parser.add_argument(
        "--fixed_eval_seed",
        type=int,
        default=20260415,
        help="Base seed used to generate the fixed test set.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/fixed_testset_convergence",
        help="Directory used to save plots and tables.",
    )
    parser.add_argument(
        "--report_name",
        type=str,
        default="fixed_testset_convergence",
        help="Prefix used for output files.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Fixed Test Set Convergence",
        help="Figure title.",
    )
    args = parser.parse_args()
    _apply_cli_aliases(args)
    _normalize_experiment_args(args)
    args.run_dir = [Path(item).expanduser().resolve() for item in args.run_dir]
    if args.label and len(args.label) != len(args.run_dir):
        parser.error("When --label is provided, its count must match --run_dir.")
    args.eval_episodes = max(1, int(args.eval_episodes))
    args.n_eval_rollout_threads = max(1, int(args.n_eval_rollout_threads or 1))
    args.save_interval = 0
    args.use_eval = False
    return args


def _parse_checkpoint_episode(path: Path) -> int:
    match = re.search(r"transformer_(\d+)\.pt$", path.name)
    if not match:
        raise ValueError(f"Unsupported checkpoint filename: {path.name}")
    return int(match.group(1))


def _resolve_run_label(run_dir: Path, explicit_label: str | None) -> str:
    if explicit_label:
        return str(explicit_label)
    lower = run_dir.name.lower()
    match = re.search(r"seed(\d+)", lower)
    if match:
        return f"Seed {match.group(1)}"
    return run_dir.name


def _discover_checkpoints(run_dirs: Sequence[Path], explicit: Sequence[int] | None) -> List[int]:
    if explicit:
        return sorted({int(item) for item in explicit})

    common: set[int] | None = None
    for run_dir in run_dirs:
        model_dir = run_dir / "models"
        checkpoints = {
            _parse_checkpoint_episode(path)
            for path in model_dir.glob("transformer_*.pt")
        }
        if not checkpoints:
            raise FileNotFoundError(f"No transformer_*.pt checkpoint found in {model_dir}")
        common = checkpoints if common is None else (common & checkpoints)
    resolved = sorted(common or [])
    if not resolved:
        raise ValueError("No common checkpoints found across the provided run directories.")
    return resolved


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


@torch.no_grad()
def _collect_eval_metrics(
    args: argparse.Namespace,
    policy,
    envs,
    eval_episodes: int,
) -> List[Dict[str, object]]:
    analytics = PaperTrainingAnalytics(
        run_dir=PROJECT_ROOT,
        num_agents=args.num_agents,
        n_threads=args.n_eval_rollout_threads,
        episode_length=args.episode_length,
        summary_filename="unused.html",
    )

    episodes_collected = 0
    batch_index = 0
    while episodes_collected < eval_episodes:
        obs = envs.reset()
        rnn_states = np.zeros(
            (args.n_eval_rollout_threads, args.num_agents, args.recurrent_N, args.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic = np.zeros_like(rnn_states)
        masks = np.ones((args.n_eval_rollout_threads, args.num_agents, 1), dtype=np.float32)
        agent_masks = np.asarray(envs.agent_masks, dtype=np.float32)[..., None]

        for _ in range(args.episode_length):
            obs_batch = np.concatenate(obs, axis=0)
            if args.use_centralized_V:
                share_batch = np.repeat(
                    np.asarray(envs.share_obs, dtype=np.float32)[:, None, :],
                    args.num_agents,
                    axis=1,
                )
                share_batch = np.concatenate(share_batch, axis=0)
            else:
                share_batch = obs_batch

            agent_mask_batch = np.concatenate(agent_masks, axis=0)
            _, actions, _, rnn_states, rnn_states_critic = policy.get_actions(
                share_batch,
                obs_batch,
                rnn_states,
                rnn_states_critic,
                masks,
                agent_mask=agent_mask_batch,
                deterministic=True,
            )
            action_array = np.array(np.split(_t2n(actions), args.n_eval_rollout_threads))
            obs, rewards, dones, infos = envs.step(action_array)
            analytics.push_step(infos, rewards, obs, actions=action_array, phase="eval")

            masks = (1.0 - np.asarray(dones, dtype=np.float32)).reshape(
                args.n_eval_rollout_threads,
                args.num_agents,
                1,
            )
            agent_masks = np.asarray(
                [np.asarray(info.get("agent_mask", [1.0] * args.num_agents), dtype=np.float32) for info in infos],
                dtype=np.float32,
            )[..., None]

        analytics.finalize_batch(batch_index)
        batch_index += 1
        episodes_collected += args.n_eval_rollout_threads

    eval_summaries = [item for item in analytics.episode_summaries if item.get("phase") == "eval"]
    return eval_summaries[:eval_episodes]


def _aggregate_eval_summaries(items: Sequence[Dict[str, object]]) -> Dict[str, float]:
    def values(key: str, fallback: str | None = None) -> np.ndarray:
        return np.asarray(
            [
                float(item.get(key, item.get(fallback, 0.0) if fallback else 0.0) or 0.0)
                for item in items
            ],
            dtype=np.float64,
        )

    reward = values("average_global_reward")
    p2p = values("p2p_volume_mean_active", "p2p_total_volume")
    grid_buy = values("grid_buy_mean_active", "grid_buy_total")
    grid_sell = values("grid_sell_mean_active", "grid_sell_total")
    carbon = values("carbon_responsibility_mean_active_episode", "load_responsibility_total")
    n_agents = values("n_agents_mean")

    return {
        "reward_mean": float(np.mean(reward)),
        "reward_std": float(np.std(reward)),
        "p2p_mean": float(np.mean(p2p)),
        "p2p_std": float(np.std(p2p)),
        "grid_buy_mean": float(np.mean(grid_buy)),
        "grid_buy_std": float(np.std(grid_buy)),
        "grid_sell_mean": float(np.mean(grid_sell)),
        "grid_sell_std": float(np.std(grid_sell)),
        "carbon_mean": float(np.mean(carbon)),
        "carbon_std": float(np.std(carbon)),
        "n_agents_mean": float(np.mean(n_agents)),
        "n_eval_episodes": int(len(items)),
    }


def evaluate_run(
    args: argparse.Namespace,
    run_dir: Path,
    label: str,
    checkpoint_episodes: Sequence[int],
    device: torch.device,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for checkpoint_episode in checkpoint_episodes:
        checkpoint_path = run_dir / "models" / f"transformer_{int(checkpoint_episode)}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        eval_args = copy.deepcopy(args)
        eval_args.seed = int(args.fixed_eval_seed)
        _set_global_seeds(eval_args.seed)
        envs, policy = _build_policy(eval_args, device)
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
            checkpoint_episode=int(checkpoint_episode),
            checkpoint_path=str(checkpoint_path),
        )
        records.append(metrics)
        print(
            "[FixedEval] {} @ {} | reward={:.4f} | p2p={:.4f} | grid_buy={:.4f} | carbon={:.4f}".format(
                label,
                checkpoint_episode,
                metrics["reward_mean"],
                metrics["p2p_mean"],
                metrics["grid_buy_mean"],
                metrics["carbon_mean"],
            )
        )
    return records


def _build_aggregate_rows(records: Sequence[Dict[str, object]]) -> List[Dict[str, float]]:
    by_checkpoint: Dict[int, List[Dict[str, object]]] = {}
    for record in records:
        by_checkpoint.setdefault(int(record["checkpoint_episode"]), []).append(record)

    rows: List[Dict[str, float]] = []
    for checkpoint_episode in sorted(by_checkpoint):
        bucket = by_checkpoint[checkpoint_episode]

        def stats(key: str) -> tuple[float, float]:
            values = np.asarray([float(item[key]) for item in bucket], dtype=np.float64)
            return float(np.mean(values)), float(np.std(values))

        reward_mean, reward_std = stats("reward_mean")
        p2p_mean, p2p_std = stats("p2p_mean")
        grid_buy_mean, grid_buy_std = stats("grid_buy_mean")
        grid_sell_mean, grid_sell_std = stats("grid_sell_mean")
        carbon_mean, carbon_std = stats("carbon_mean")
        rows.append(
            {
                "checkpoint_episode": int(checkpoint_episode),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "p2p_mean": p2p_mean,
                "p2p_std": p2p_std,
                "grid_buy_mean": grid_buy_mean,
                "grid_buy_std": grid_buy_std,
                "grid_sell_mean": grid_sell_mean,
                "grid_sell_std": grid_sell_std,
                "carbon_mean": carbon_mean,
                "carbon_std": carbon_std,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(
    path: Path,
    args: argparse.Namespace,
    records: Sequence[Dict[str, object]],
    aggregate_rows: Sequence[Dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fixed Test Set Convergence",
        "",
        f"- fixed_eval_seed: `{int(args.fixed_eval_seed)}`",
        f"- eval_episodes_per_checkpoint: `{int(args.eval_episodes)}`",
        f"- n_eval_rollout_threads: `{int(args.n_eval_rollout_threads)}`",
        f"- num_agents: `{int(args.num_agents)}`",
        f"- min_agents: `{int(args.min_agents)}`",
        f"- step_churn_prob: `{float(args.step_churn_prob):.4f}`",
        "",
        "## Aggregate (mean ± std across runs)",
        "",
        "| Checkpoint | Reward | P2P Mean | Grid Buy Mean | Grid Sell Mean | Carbon Mean |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {ckpt} | {r:.4f} ± {rs:.4f} | {p:.4f} ± {ps:.4f} | {gb:.4f} ± {gbs:.4f} | {gs:.4f} ± {gss:.4f} | {c:.4f} ± {cs:.4f} |".format(
                ckpt=int(row["checkpoint_episode"]),
                r=row["reward_mean"],
                rs=row["reward_std"],
                p=row["p2p_mean"],
                ps=row["p2p_std"],
                gb=row["grid_buy_mean"],
                gbs=row["grid_buy_std"],
                gs=row["grid_sell_mean"],
                gss=row["grid_sell_std"],
                c=row["carbon_mean"],
                cs=row["carbon_std"],
            )
        )

    lines.extend(
        [
            "",
            "## Per-Run Results",
            "",
            "| Run | Checkpoint | Reward | P2P Mean | Grid Buy Mean | Grid Sell Mean | Carbon Mean |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in records:
        lines.append(
            "| {label} | {ckpt} | {r:.4f} | {p:.4f} | {gb:.4f} | {gs:.4f} | {c:.4f} |".format(
                label=record["label"],
                ckpt=int(record["checkpoint_episode"]),
                r=float(record["reward_mean"]),
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
        bucket = sorted(bucket, key=lambda item: int(item["checkpoint_episode"]))
        xs = [int(item["checkpoint_episode"]) for item in bucket]
        ys = [float(item["reward_mean"]) for item in bucket]
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=colors[idx % len(colors)], label=label)
    ax.set_title("Reward on Fixed Test Set")
    ax.set_xlabel("Checkpoint Episode")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for idx, (label, bucket) in enumerate(sorted(by_label.items())):
        bucket = sorted(bucket, key=lambda item: int(item["checkpoint_episode"]))
        xs = [int(item["checkpoint_episode"]) for item in bucket]
        ys = [float(item["p2p_mean"]) for item in bucket]
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=colors[idx % len(colors)], label=label)
    ax.set_title("P2P Volume Mean per Active Agent")
    ax.set_xlabel("Checkpoint Episode")
    ax.set_ylabel("Mean Volume")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for idx, (label, bucket) in enumerate(sorted(by_label.items())):
        bucket = sorted(bucket, key=lambda item: int(item["checkpoint_episode"]))
        xs = [int(item["checkpoint_episode"]) for item in bucket]
        buy = [float(item["grid_buy_mean"]) for item in bucket]
        sell = [float(item["grid_sell_mean"]) for item in bucket]
        color = colors[idx % len(colors)]
        ax.plot(xs, buy, marker="o", linewidth=2.0, color=color, label=f"{label} Grid Buy")
        ax.plot(xs, sell, marker="o", linewidth=1.8, linestyle="--", color=color, label=f"{label} Grid Sell")
    ax.set_title("Grid Trading Mean per Active Agent")
    ax.set_xlabel("Checkpoint Episode")
    ax.set_ylabel("Mean Volume")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[1, 1]
    for idx, (label, bucket) in enumerate(sorted(by_label.items())):
        bucket = sorted(bucket, key=lambda item: int(item["checkpoint_episode"]))
        xs = [int(item["checkpoint_episode"]) for item in bucket]
        ys = [float(item["carbon_mean"]) for item in bucket]
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=colors[idx % len(colors)], label=label)
    ax.set_title("Carbon Responsibility Episode Mean")
    ax.set_xlabel("Checkpoint Episode")
    ax.set_ylabel("Mean Carbon Responsibility")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_mean_std(aggregate_rows: Sequence[Dict[str, float]], output_path: Path, title: str) -> Path:
    xs = np.asarray([int(item["checkpoint_episode"]) for item in aggregate_rows], dtype=np.int32)
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
    ax.set_title("Reward on Fixed Test Set")
    ax.set_xlabel("Checkpoint Episode")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(xs, p2p_mean, marker="o", linewidth=2.2, color="#d94801", label="Mean P2P Volume")
    ax.fill_between(xs, p2p_mean - p2p_std, p2p_mean + p2p_std, color="#fdae6b", alpha=0.25, label="±1 std")
    ax.set_title("P2P Volume Mean per Active Agent")
    ax.set_xlabel("Checkpoint Episode")
    ax.set_ylabel("Mean Volume")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(xs, grid_buy_mean, marker="o", linewidth=2.2, color="#238b45", label="Grid Buy Mean")
    ax.fill_between(xs, grid_buy_mean - grid_buy_std, grid_buy_mean + grid_buy_std, color="#74c476", alpha=0.18)
    ax.plot(xs, grid_sell_mean, marker="o", linewidth=2.2, color="#756bb1", linestyle="--", label="Grid Sell Mean")
    ax.fill_between(xs, grid_sell_mean - grid_sell_std, grid_sell_mean + grid_sell_std, color="#bcbddc", alpha=0.18)
    ax.set_title("Grid Trading Mean per Active Agent")
    ax.set_xlabel("Checkpoint Episode")
    ax.set_ylabel("Mean Volume")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(xs, carbon_mean, marker="o", linewidth=2.2, color="#08519c", label="Carbon Responsibility Mean")
    ax.fill_between(xs, carbon_mean - carbon_std, carbon_mean + carbon_std, color="#9ecae1", alpha=0.25, label="±1 std")
    ax.set_title("Carbon Responsibility Episode Mean")
    ax.set_xlabel("Checkpoint Episode")
    ax.set_ylabel("Mean Carbon Responsibility")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
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

    checkpoint_episodes = _discover_checkpoints(args.run_dir, args.checkpoint)
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
    for run_dir, label in zip(args.run_dir, labels):
        all_records.extend(
            evaluate_run(
                args=args,
                run_dir=run_dir,
                label=label,
                checkpoint_episodes=checkpoint_episodes,
                device=device,
            )
        )

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
                    "min_agents": int(args.min_agents),
                    "step_churn_prob": float(args.step_churn_prob),
                    "checkpoint_episodes": list(checkpoint_episodes),
                    "run_dirs": [str(item) for item in args.run_dir],
                    "labels": labels,
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
    _plot_mean_std(aggregate_rows, mean_std_png, f"{args.title} (Mean ± Std)")

    print(f"saved_raw_csv={raw_csv}")
    print(f"saved_aggregate_csv={aggregate_csv}")
    print(f"saved_markdown={report_md}")
    print(f"saved_json={report_json}")
    print(f"saved_overlay_plot={overlay_png}")
    print(f"saved_mean_std_plot={mean_std_png}")


if __name__ == "__main__":
    main()
