#!/usr/bin/env python3
"""Low-resource agent-count evaluation that writes lightweight CSV progress."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
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
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch

try:
    import importlib.metadata as _stdlib_metadata
    import importlib_metadata
except Exception:  # pragma: no cover - environment-specific compatibility
    importlib_metadata = None
    _stdlib_metadata = None


def _patch_importlib_metadata() -> None:
    if importlib_metadata is None or _stdlib_metadata is None:
        return
    if hasattr(importlib_metadata, "entry_points"):
        try:
            importlib_metadata.entry_points(group="pytest11")
            return
        except Exception:
            pass

    def _entry_points(**kwargs):
        eps = _stdlib_metadata.entry_points()
        group = kwargs.get("group")
        if group is None:
            return eps
        if hasattr(eps, "select"):
            return eps.select(group=group)
        if isinstance(eps, dict):
            return eps.get(group, ())
        return [ep for ep in eps if getattr(ep, "group", None) == group]

    importlib_metadata.entry_points = _entry_points


_patch_importlib_metadata()

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dp_lcrl_rl.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
from dp_lcrl_rl.runner.shared.base_runner import _t2n
from dp_lcrl_rl.runner.shared.training_analytics import PaperTrainingAnalytics
from dp_lcrl_rl.scripts.train.train_paper_mat import (
    _apply_cli_aliases,
    _normalize_experiment_args,
    _set_global_seeds,
    build_arg_parser,
    make_env,
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
    parser.description = "Low-resource agent-count sweep that writes aggregate CSV rows incrementally."
    parser.add_argument(
        "--run_dir",
        action="append",
        required=True,
        help="Training run directory. Repeat for multiple seeds.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional label for one --run_dir. Count must match run_dir count when provided.",
    )
    parser.add_argument(
        "--checkpoint_episode",
        type=int,
        default=None,
        help="Checkpoint episode to evaluate. Defaults to the latest checkpoint in each run.",
    )
    parser.add_argument("--agent_count_min", type=int, default=1)
    parser.add_argument("--agent_count_max", type=int, default=None)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--fixed_eval_seed", type=int, default=20260417)
    parser.add_argument(
        "--output_csv",
        type=str,
        default="reports/agent_count_sweep_safe/agent_count_sweep_safe.csv",
        help="Aggregate CSV updated after each active-agent count finishes.",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip counts already present in output_csv.",
    )
    args = parser.parse_args()
    _apply_cli_aliases(args)
    _normalize_experiment_args(args)

    args.run_dir = [Path(item).expanduser().resolve() for item in args.run_dir]
    if args.label and len(args.label) != len(args.run_dir):
        parser.error("When --label is provided, its count must match --run_dir.")

    args.eval_episodes = max(1, int(args.eval_episodes))
    args.n_eval_rollout_threads = max(1, int(args.n_eval_rollout_threads or 1))
    args.agent_count_min = max(1, int(args.agent_count_min))
    max_count = int(args.num_agents) if args.agent_count_max is None else int(args.agent_count_max)
    args.agent_count_max = max(args.agent_count_min, min(max_count, int(args.num_agents)))
    args.save_interval = 0
    args.use_eval = False

    output_csv = Path(str(args.output_csv).strip())
    if not output_csv.is_absolute():
        output_csv = (PROJECT_ROOT / output_csv).resolve()
    args.output_csv = output_csv
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


def _resolve_run_label(run_dir: Path, explicit_label: str | None) -> str:
    if explicit_label:
        return str(explicit_label)
    lower = run_dir.name.lower()
    match = re.search(r"seed(\d+)", lower)
    if match:
        return f"Seed{match.group(1)}"
    return run_dir.name


def _build_policy(args: argparse.Namespace, device: torch.device):
    env = make_env(args, args.n_eval_rollout_threads)
    share_observation_space = (
        env.share_observation_space[0] if args.use_centralized_V else env.observation_space[0]
    )
    policy = Policy(
        args,
        env.observation_space[0],
        share_observation_space,
        env.action_space[0],
        args.num_agents,
        device=device,
    )
    return env, policy


def _bind_fixed_active_count(envs, active_count: int) -> None:
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


@torch.no_grad()
def _collect_eval_summaries(
    args: argparse.Namespace,
    policy: Policy,
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

    return [item for item in analytics.episode_summaries if item.get("phase") == "eval"][:eval_episodes]


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
        "p2p_mean": float(np.mean(p2p)),
        "grid_buy_mean": float(np.mean(grid_buy)),
        "grid_sell_mean": float(np.mean(grid_sell)),
        "grid_trade_mean": float(np.mean(grid_buy + grid_sell)),
        "carbon_mean": float(np.mean(carbon)),
        "n_agents_mean": float(np.mean(n_agents)),
    }


def _load_completed_counts(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            int(row["active_agent_count"])
            for row in reader
            if row.get("active_agent_count") not in (None, "")
        }


def _append_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate_count(
    args: argparse.Namespace,
    run_dir: Path,
    label: str,
    active_count: int,
    device: torch.device,
) -> Dict[str, float]:
    checkpoint_path = _resolve_checkpoint_path(run_dir, args.checkpoint_episode)
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
        summaries = _collect_eval_summaries(
            args=eval_args,
            policy=policy,
            envs=envs,
            eval_episodes=args.eval_episodes,
        )
    finally:
        envs.close()
        del policy
        del envs
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = _aggregate_eval_summaries(summaries)
    metrics["checkpoint_episode"] = float(_parse_checkpoint_episode(checkpoint_path))
    print(
        "[SafeEval] {} | active_agents={} | reward={:.4f} | p2p={:.4f} | grid_total={:.4f} | carbon={:.4f}".format(
            label,
            int(active_count),
            metrics["reward_mean"],
            metrics["p2p_mean"],
            metrics["grid_trade_mean"],
            metrics["carbon_mean"],
        ),
        flush=True,
    )
    return metrics


def main() -> None:
    args = parse_args()
    _configure_low_resource_runtime()

    for run_dir in args.run_dir:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    labels = [
        _resolve_run_label(run_dir, args.label[idx] if args.label else None)
        for idx, run_dir in enumerate(args.run_dir)
    ]
    active_counts = list(range(int(args.agent_count_min), int(args.agent_count_max) + 1))
    completed = _load_completed_counts(args.output_csv) if args.skip_completed else set()

    for active_count in active_counts:
        if active_count in completed:
            print(f"[SafeEval] Skipping completed active_agents={active_count}", flush=True)
            continue

        bucket: List[Dict[str, float]] = []
        for run_dir, label in zip(args.run_dir, labels):
            bucket.append(evaluate_count(args=args, run_dir=run_dir, label=label, active_count=active_count, device=device))

        def mean_of(key: str) -> float:
            return float(np.mean([float(item[key]) for item in bucket]))

        row = {
            "active_agent_count": int(active_count),
            "reward_mean": mean_of("reward_mean"),
            "p2p_mean": mean_of("p2p_mean"),
            "grid_trade_mean": mean_of("grid_trade_mean"),
            "grid_buy_mean": mean_of("grid_buy_mean"),
            "grid_sell_mean": mean_of("grid_sell_mean"),
            "carbon_mean": mean_of("carbon_mean"),
            "seed_count": int(len(bucket)),
            "eval_episodes_per_seed": int(args.eval_episodes),
            "fixed_eval_seed": int(args.fixed_eval_seed),
            "checkpoint_episode": int(round(mean_of("checkpoint_episode"))),
        }
        _append_row(args.output_csv, row)
        print(
            "[SafeEval] Saved active_agents={} -> {}".format(
                int(active_count),
                args.output_csv,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
