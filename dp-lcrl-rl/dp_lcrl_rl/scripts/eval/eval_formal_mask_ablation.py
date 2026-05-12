#!/usr/bin/env python3
"""Native structured-mask ablation evaluation for the final paper setting.

The latest paper setting treats direct maximum-scale training as part of the
full method. This evaluator therefore compares:

1. Full / Ours: CMTM + structural mask + direct max-scale training.
2. w/o Structured Mask: CMTM + obs-only padding + direct max-scale training.

Evaluation is deliberately native: each checkpoint is evaluated with the same
mask mode it was trained with, rather than forcing a full-mask evaluator.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RUNTIME_TMP = PROJECT_ROOT / "tmp_eval_runtime"
RUNTIME_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(RUNTIME_TMP)
os.environ["TEMP"] = str(RUNTIME_TMP)
os.environ["WANDB_DIR"] = str(RUNTIME_TMP / "wandb")
os.environ["WANDB_DATA_DIR"] = str(RUNTIME_TMP / "wandb-data")
os.environ["WANDB_CACHE_DIR"] = str(RUNTIME_TMP / "wandb-cache")
os.environ["WANDB_CONFIG_DIR"] = str(RUNTIME_TMP / "wandb-config")

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


METHOD_ORDER = ["full_ours", "without_structured_mask"]
METHOD_LABELS = {
    "full_ours": "Full / Ours",
    "without_structured_mask": "w/o Structured Mask",
}
METHOD_COLORS = {
    "full_ours": "#111111",
    "without_structured_mask": "#c2410c",
}
METRIC_SPECS = {
    "reward_mean": ("Reward", True),
    "p2p_mean": ("P2P Trading Volume", True),
    "grid_trade_mean": ("Grid Trading Volume", False),
    "carbon_mean": ("Carbon Responsibility", False),
}


@dataclass(frozen=True)
class MethodRun:
    method: str
    label: str
    mask_mode: str
    seed: int
    run_dir: Path


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    category: str
    min_agents: int
    step_churn_prob: float
    exact_active_count: int | None = None
    inactive_noise_std: float = 0.0
    eval_episodes: int | None = None


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    parser.description = "Evaluate final Full/Ours vs w/o Structured Mask ablation."
    parser.add_argument("--output_dir", default="reports/formal_mask_ablation_20260428")
    parser.add_argument("--report_name", default="formal_mask_ablation")
    parser.add_argument("--full_run_dir", action="append", default=None)
    parser.add_argument("--mask_run_dir", action="append", default=None)
    parser.add_argument("--checkpoint_episode", type=int, default=10000)
    parser.add_argument("--agent_count_min", type=int, default=1)
    parser.add_argument("--agent_count_max", type=int, default=None)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--dynamic_eval_episodes", type=int, default=30)
    parser.add_argument("--noise_eval_episodes", type=int, default=40)
    parser.add_argument("--fixed_eval_seed", type=int, default=20260428)
    parser.add_argument("--inactive_noise_std", type=float, default=5.0)
    parser.add_argument("--force", action="store_true", help="Recompute existing raw rows.")
    args = parser.parse_args()
    _apply_cli_aliases(args)

    args.num_agents = int(args.num_agents or 30)
    args.agent_count_min = max(1, int(args.agent_count_min))
    max_count = int(args.num_agents) if args.agent_count_max is None else int(args.agent_count_max)
    args.agent_count_max = max(args.agent_count_min, min(max_count, int(args.num_agents)))
    args.eval_episodes = max(1, int(args.eval_episodes))
    args.dynamic_eval_episodes = max(1, int(args.dynamic_eval_episodes))
    args.noise_eval_episodes = max(1, int(args.noise_eval_episodes))
    args.inactive_noise_std = max(0.0, float(args.inactive_noise_std))

    # Normalize model hyperparameters, then override evaluation-specific fields.
    args.scale_mode = "direct_max"
    args.cmtm_mode = "full"
    args.mask_mode = "full"
    _normalize_experiment_args(args)
    args.cuda = False
    args.use_eval = False
    args.save_interval = 0
    args.n_rollout_threads = 1
    args.n_eval_rollout_threads = 1
    args.curriculum_warmup_episodes = 0
    args.curriculum_min_agents = args.num_agents
    return args


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _default_full_dirs() -> list[str]:
    return [
        "runs/paper_ablation_direct_max_10000ep_seed42_20260422",
        "runs/paper_ablation_direct_max_10000ep_seed43_20260422",
        "runs/paper_ablation_direct_max_10000ep_seed44_20260422",
    ]


def _default_mask_dirs() -> list[str]:
    return [
        "runs/paper_mask_ablation_obs_only_direct_10000ep_seed42_20260428",
        "runs/paper_mask_ablation_obs_only_direct_10000ep_seed43_20260428",
        "runs/paper_mask_ablation_obs_only_direct_10000ep_seed44_20260428",
    ]


def _parse_seed(path: Path, fallback: int) -> int:
    match = re.search(r"seed(\d+)", path.name.lower())
    return int(match.group(1)) if match else int(fallback)


def _method_runs(args: argparse.Namespace) -> list[MethodRun]:
    full_dirs = args.full_run_dir or _default_full_dirs()
    mask_dirs = args.mask_run_dir or _default_mask_dirs()
    runs: list[MethodRun] = []
    for idx, item in enumerate(full_dirs):
        path = _resolve_path(item)
        runs.append(
            MethodRun(
                method="full_ours",
                label=METHOD_LABELS["full_ours"],
                mask_mode="full",
                seed=_parse_seed(path, 42 + idx),
                run_dir=path,
            )
        )
    for idx, item in enumerate(mask_dirs):
        path = _resolve_path(item)
        runs.append(
            MethodRun(
                method="without_structured_mask",
                label=METHOD_LABELS["without_structured_mask"],
                mask_mode="obs_only",
                seed=_parse_seed(path, 42 + idx),
                run_dir=path,
            )
        )

    missing = [str(run.run_dir) for run in runs if not run.run_dir.exists()]
    if missing:
        raise FileNotFoundError("Missing run directories:\n" + "\n".join(missing))
    for run in runs:
        checkpoint = _resolve_checkpoint_path(run.run_dir, args.checkpoint_episode)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {run.label}: {checkpoint}")
    return sorted(runs, key=lambda row: (METHOD_ORDER.index(row.method), row.seed))


def _parse_checkpoint_episode(path: Path) -> int:
    match = re.search(r"transformer_(\d+)\.pt$", path.name)
    if not match:
        raise ValueError(f"Unsupported checkpoint filename: {path.name}")
    return int(match.group(1))


def _resolve_checkpoint_path(run_dir: Path, checkpoint_episode: int | None) -> Path:
    model_dir = run_dir / "models"
    if checkpoint_episode is not None:
        return model_dir / f"transformer_{int(checkpoint_episode)}.pt"
    candidates = sorted(model_dir.glob("transformer_*.pt"), key=_parse_checkpoint_episode)
    if not candidates:
        raise FileNotFoundError(f"No transformer_*.pt checkpoint found in {model_dir}")
    return candidates[-1]


def _output_dir(args: argparse.Namespace) -> Path:
    path = _resolve_path(args.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def _configure_low_resource_runtime() -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


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


def _bind_exact_active_count(envs, active_count: int) -> None:
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


def _scenarios(args: argparse.Namespace) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for count in range(int(args.agent_count_min), int(args.agent_count_max) + 1):
        scenarios.append(
            Scenario(
                name=f"scale_{count:02d}",
                label=f"{count} active agents",
                category="scale_sweep",
                min_agents=count,
                step_churn_prob=0.0,
                exact_active_count=count,
                eval_episodes=int(args.eval_episodes),
            )
        )

    scenarios.extend(
        [
            Scenario("dynamic_static30", "Static 30 agents", "dynamic", 30, 0.0, None, 0.0, args.dynamic_eval_episodes),
            Scenario("dynamic_var20", "Variable 20-30 agents", "dynamic", 20, 0.0, None, 0.0, args.dynamic_eval_episodes),
            Scenario("dynamic_churn20_p20", "20-30 agents, churn 0.20", "dynamic", 20, 0.20, None, 0.0, args.dynamic_eval_episodes),
            Scenario("dynamic_churn10_p30", "10-30 agents, churn 0.30", "dynamic", 10, 0.30, None, 0.0, args.dynamic_eval_episodes),
            Scenario("dynamic_churn05_p50", "5-30 agents, churn 0.50", "dynamic", 5, 0.50, None, 0.0, args.dynamic_eval_episodes),
            Scenario("noise_churn10_p30", "Inactive-noise stress", "inactive_noise", 10, 0.30, None, args.inactive_noise_std, args.noise_eval_episodes),
        ]
    )
    return scenarios


def _prepare_policy_inputs(args: argparse.Namespace, envs, obs: np.ndarray, agent_masks: np.ndarray, rng: np.random.Generator, noise_std: float):
    obs_for_policy = np.asarray(obs, dtype=np.float32).copy()
    if noise_std > 0.0:
        inactive = np.asarray(agent_masks, dtype=np.float32) < 0.5
        if np.any(inactive):
            noise = rng.normal(0.0, float(noise_std), size=obs_for_policy.shape).astype(np.float32)
            obs_for_policy[inactive] = noise[inactive]

    obs_batch = np.concatenate(obs_for_policy, axis=0)
    if args.use_centralized_V:
        share_obs = obs_for_policy.reshape(args.n_eval_rollout_threads, -1)
        share_batch = np.repeat(share_obs[:, None, :], args.num_agents, axis=1)
        share_batch = np.concatenate(share_batch, axis=0)
    else:
        share_batch = obs_batch
    return share_batch, obs_batch


@torch.no_grad()
def _collect_eval_metrics(
    args: argparse.Namespace,
    policy: Policy,
    envs,
    scenario: Scenario,
    eval_seed: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    analytics = PaperTrainingAnalytics(
        run_dir=PROJECT_ROOT,
        num_agents=args.num_agents,
        n_threads=args.n_eval_rollout_threads,
        episode_length=args.episode_length,
        summary_filename="unused.html",
    )
    rng = np.random.default_rng(int(eval_seed) + 104729)
    eval_episodes = int(scenario.eval_episodes or args.eval_episodes)
    episodes_collected = 0
    batch_index = 0
    attn_leakage = 0.0
    attn_calls = 0
    attn_entropy_values: list[float] = []
    inactive_action_abs_sum = 0.0
    inactive_action_count = 0

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
            flat_agent_mask = np.concatenate(agent_masks, axis=0)
            share_batch, obs_batch = _prepare_policy_inputs(
                args=args,
                envs=envs,
                obs=obs,
                agent_masks=agent_masks[..., 0],
                rng=rng,
                noise_std=float(scenario.inactive_noise_std),
            )
            _, actions, _, rnn_states, rnn_states_critic = policy.get_actions(
                share_batch,
                obs_batch,
                rnn_states,
                rnn_states_critic,
                masks,
                agent_mask=flat_agent_mask,
                deterministic=True,
            )
            stats = policy.pop_attn_stats()
            if isinstance(stats, dict):
                attn_leakage += float(stats.get("attn_leakage", 0.0) or 0.0)
                attn_calls += int(stats.get("attn_calls", 0) or 0)
                entropy = stats.get("attn_entropy")
                if entropy is not None:
                    arr = np.asarray(entropy, dtype=np.float64).reshape(-1)
                    if arr.size:
                        attn_entropy_values.extend(float(x) for x in arr if np.isfinite(x))

            action_array = np.array(np.split(_t2n(actions), args.n_eval_rollout_threads))
            inactive = agent_masks < 0.5
            if np.any(inactive):
                inactive_action_abs_sum += float(np.sum(np.abs(action_array) * inactive))
                inactive_action_count += int(np.sum(inactive) * action_array.shape[-1])

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
    diag = {
        "attn_leakage_per_call": float(attn_leakage / attn_calls) if attn_calls > 0 else float("nan"),
        "attn_calls": float(attn_calls),
        "attn_entropy_mean": float(np.mean(attn_entropy_values)) if attn_entropy_values else float("nan"),
        "inactive_action_abs_mean": (
            float(inactive_action_abs_sum / inactive_action_count)
            if inactive_action_count > 0
            else float("nan")
        ),
    }
    return eval_summaries[:eval_episodes], diag


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
        "grid_trade_mean": float(np.mean(grid_buy + grid_sell)),
        "grid_trade_std": float(np.std(grid_buy + grid_sell)),
        "carbon_mean": float(np.mean(carbon)),
        "carbon_std": float(np.std(carbon)),
        "n_agents_mean": float(np.mean(n_agents)),
        "n_eval_episodes": int(len(items)),
    }


def _row_key(row: dict[str, object]) -> tuple[str, int, str]:
    return (str(row["method"]), int(row["seed"]), str(row["scenario_name"]))


def _read_existing(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1 if arr.size > 1 else 0))


def _aggregate_by_scenario(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault((str(row["method"]), str(row["scenario_name"])), []).append(row)

    out: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        method_keys = [key for key in buckets if key[0] == method]
        for _, scenario_name in sorted(method_keys, key=lambda key: str(key[1])):
            bucket = buckets[(method, scenario_name)]
            first = bucket[0]
            result: dict[str, object] = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "scenario_name": scenario_name,
                "scenario_label": first["scenario_label"],
                "scenario_category": first["scenario_category"],
                "active_agent_count": first["active_agent_count"],
                "min_agents": first["min_agents"],
                "step_churn_prob": first["step_churn_prob"],
                "inactive_noise_std": first["inactive_noise_std"],
                "seed_count": len(bucket),
                "eval_episodes_per_seed": first["n_eval_episodes"],
            }
            for key in (
                "reward_mean",
                "p2p_mean",
                "grid_buy_mean",
                "grid_sell_mean",
                "grid_trade_mean",
                "carbon_mean",
                "n_agents_mean",
                "attn_leakage_per_call",
                "attn_entropy_mean",
                "inactive_action_abs_mean",
            ):
                mean, std = _mean_std(float(item[key]) for item in bucket)
                if key.endswith("_mean"):
                    std_key = key.replace("_mean", "_std")
                else:
                    std_key = f"{key}_std"
                result[key] = mean
                result[std_key] = std
            out.append(result)
    return out


def _scenario_lookup(rows: Sequence[dict[str, object]], method: str, category: str) -> list[dict[str, object]]:
    return [row for row in rows if row["method"] == method and row["scenario_category"] == category]


def _normalized_pair(values: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    finite = {k: v for k, v in values.items() if math.isfinite(float(v))}
    if not finite:
        return {k: float("nan") for k in values}
    lo = min(finite.values())
    hi = max(finite.values())
    if abs(hi - lo) < 1e-12:
        return {k: 0.5 for k in values}
    result = {}
    for key, value in values.items():
        if not math.isfinite(float(value)):
            result[key] = float("nan")
        elif higher_is_better:
            result[key] = float((value - lo) / (hi - lo))
        else:
            result[key] = float((hi - value) / (hi - lo))
    return result


def _aggregate_overall(by_scenario: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    method_rows: dict[str, dict[str, object]] = {method: {"method": method, "method_label": METHOD_LABELS[method]} for method in METHOD_ORDER}

    for method in METHOD_ORDER:
        scale_rows = _scenario_lookup(by_scenario, method, "scale_sweep")
        dyn_rows = _scenario_lookup(by_scenario, method, "dynamic")
        noise_rows = _scenario_lookup(by_scenario, method, "inactive_noise")
        baseline_noise = [
            row for row in dyn_rows if str(row["scenario_name"]) == "dynamic_churn10_p30"
        ]

        target = method_rows[method]
        for prefix, bucket in (("scale", scale_rows), ("dynamic", dyn_rows), ("noise", noise_rows)):
            for key in ("reward_mean", "p2p_mean", "grid_trade_mean", "carbon_mean", "inactive_action_abs_mean"):
                mean, std = _mean_std(float(item[key]) for item in bucket) if bucket else (float("nan"), float("nan"))
                target[f"{prefix}_{key}"] = mean
                target[f"{prefix}_{key.replace('_mean', '_std')}"] = std

        baseline_reward = float(baseline_noise[0]["reward_mean"]) if baseline_noise else float("nan")
        noise_reward = float(noise_rows[0]["reward_mean"]) if noise_rows else float("nan")
        target["inactive_noise_reward_drop"] = (
            baseline_reward - noise_reward
            if math.isfinite(baseline_reward) and math.isfinite(noise_reward)
            else float("nan")
        )
        target["inactive_noise_reward_delta"] = (
            noise_reward - baseline_reward
            if math.isfinite(baseline_reward) and math.isfinite(noise_reward)
            else float("nan")
        )
        target["inactive_noise_reward_sensitivity"] = (
            abs(noise_reward - baseline_reward)
            if math.isfinite(baseline_reward) and math.isfinite(noise_reward)
            else float("nan")
        )
        target["scale_agent_count_points"] = len(scale_rows)
        target["dynamic_scenario_count"] = len(dyn_rows)

    score_components: dict[str, dict[str, float]] = {}
    metrics = [
        ("scale_inactive_action_abs_mean", False),
        ("dynamic_inactive_action_abs_mean", False),
        ("inactive_noise_reward_sensitivity", False),
        ("noise_inactive_action_abs_mean", False),
    ]
    for key, higher in metrics:
        score_components[key] = _normalized_pair(
            {method: float(method_rows[method].get(key, float("nan"))) for method in METHOD_ORDER},
            higher_is_better=higher,
        )
    for method in METHOD_ORDER:
        values = [
            score_components[key][method]
            for key, _ in metrics
            if math.isfinite(float(score_components[key][method]))
        ]
        method_rows[method]["mask_isolation_score"] = float(np.mean(values)) if values else float("nan")
    return [method_rows[method] for method in METHOD_ORDER]


def _format_pm(mean: object, std: object, digits: int = 3) -> str:
    return f"{float(mean):.{digits}f} +/- {float(std):.{digits}f}"


def _write_word_table(path: Path, overall_rows: Sequence[dict[str, object]]) -> None:
    headers = [
        "Method",
        "Scale Reward",
        "Scale P2P",
        "Scale Grid",
        "Scale Carbon",
        "Dynamic Reward",
        "Noise Sens.",
        "Mask Score",
    ]
    body = []
    for row in overall_rows:
        body.append(
            [
                str(row["method_label"]),
                _format_pm(row["scale_reward_mean"], row["scale_reward_std"]),
                _format_pm(row["scale_p2p_mean"], row["scale_p2p_std"]),
                _format_pm(row["scale_grid_trade_mean"], row["scale_grid_trade_std"]),
                _format_pm(row["scale_carbon_mean"], row["scale_carbon_std"]),
                _format_pm(row["dynamic_reward_mean"], row["dynamic_reward_std"]),
                f"{float(row['inactive_noise_reward_sensitivity']):.3f}",
                f"{float(row['mask_isolation_score']):.3f}",
            ]
        )
    widths = [max(len(str(row[idx])) for row in [headers] + body) for idx in range(len(headers))]
    lines = ["  ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    for row in body:
        lines.append("  ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))
    lines.extend(
        [
            "",
            "Notes:",
            "Full / Ours = CMTM + structural mask + direct max-scale training.",
            "w/o Structured Mask = CMTM + obs-only padding + direct max-scale training.",
            "Scale metrics average the 1..30 fixed active-agent sweep after seed aggregation.",
            "Dynamic Reward averages variable-scale/churn scenarios.",
            "Noise Sens. is the absolute reward change under inactive-observation noise relative to the same churn test set; lower is better.",
            "Mask Score is a normalized average of inactive-action magnitude and inactive-noise sensitivity; higher is better.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_scale_sweep(by_scenario: Sequence[dict[str, object]], output_path: Path) -> None:
    scale_rows = [row for row in by_scenario if row["scenario_category"] == "scale_sweep"]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), dpi=220)
    for ax, (metric, (title, higher_is_better)) in zip(axes.flat, METRIC_SPECS.items()):
        for method in METHOD_ORDER:
            bucket = [row for row in scale_rows if row["method"] == method]
            bucket.sort(key=lambda item: int(item["active_agent_count"]))
            if not bucket:
                continue
            xs = np.asarray([int(item["active_agent_count"]) for item in bucket], dtype=np.float64)
            mean = np.asarray([float(item[metric]) for item in bucket], dtype=np.float64)
            std = np.asarray([float(item[metric.replace("_mean", "_std")]) for item in bucket], dtype=np.float64)
            ax.plot(xs, mean, marker="o", markersize=3.0, linewidth=2.0, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            ax.fill_between(xs, mean - std, mean + std, color=METHOD_COLORS[method], alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Active agent count")
        ax.set_ylabel("Mean value")
        ax.grid(True, alpha=0.25)
        if not higher_is_better:
            ax.text(0.02, 0.94, "lower is better", transform=ax.transAxes, fontsize=8, color="#555555")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Structured Mask Ablation: Fixed Active-Agent Sweep", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_stress_summary(by_scenario: Sequence[dict[str, object]], output_path: Path) -> None:
    rows = [
        row
        for row in by_scenario
        if row["scenario_category"] in {"dynamic", "inactive_noise"}
    ]
    scenario_order = []
    for row in rows:
        name = str(row["scenario_name"])
        if name not in scenario_order:
            scenario_order.append(name)
    x = np.arange(len(scenario_order), dtype=np.float64)
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.5), dpi=220, sharex=True)
    for offset_idx, method in enumerate(METHOD_ORDER):
        bucket = {str(row["scenario_name"]): row for row in rows if row["method"] == method}
        offset = (offset_idx - 0.5) * width
        reward = [float(bucket[name]["reward_mean"]) if name in bucket else np.nan for name in scenario_order]
        invalid = [float(bucket[name]["inactive_action_abs_mean"]) if name in bucket else np.nan for name in scenario_order]
        axes[0].bar(x + offset, reward, width=width, color=METHOD_COLORS[method], alpha=0.88, label=METHOD_LABELS[method])
        axes[1].bar(x + offset, invalid, width=width, color=METHOD_COLORS[method], alpha=0.88, label=METHOD_LABELS[method])
    labels = [
        str(next(row["scenario_label"] for row in rows if row["scenario_name"] == name))
        for name in scenario_order
    ]
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Dynamic and Inactive-Noise Robustness")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].set_ylabel("Inactive action magnitude")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].text(0.01, 0.92, "lower is better", transform=axes[1].transAxes, fontsize=8, color="#555555")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _load_training_curve(run_dir: Path, bin_size: int = 100) -> list[dict[str, float]]:
    summary_path = run_dir / "paper_training_summary.json"
    if not summary_path.exists():
        return []
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    items = payload.get("episode_summaries", [])
    buckets: dict[int, list[dict[str, object]]] = {}
    for item in items:
        if item.get("phase") != "train":
            continue
        iteration = int(item.get("iteration_index", item.get("episode_id", 0)) or 0)
        buckets.setdefault(iteration // int(bin_size), []).append(item)
    rows: list[dict[str, float]] = []
    for bucket_id in sorted(buckets):
        bucket = buckets[bucket_id]
        if not bucket:
            continue
        reward = [float(item.get("average_global_reward", 0.0) or 0.0) for item in bucket]
        p2p = [float(item.get("p2p_volume_mean_active", item.get("p2p_total_volume", 0.0)) or 0.0) for item in bucket]
        grid = [
            float(item.get("grid_buy_mean_active", item.get("grid_buy_total", 0.0)) or 0.0)
            + float(item.get("grid_sell_mean_active", item.get("grid_sell_total", 0.0)) or 0.0)
            for item in bucket
        ]
        carbon = [
            float(item.get("carbon_responsibility_mean_active_episode", item.get("load_responsibility_total", 0.0)) or 0.0)
            for item in bucket
        ]
        rows.append(
            {
                "iteration": float((bucket_id + 1) * bin_size),
                "reward_mean": float(np.mean(reward)),
                "p2p_mean": float(np.mean(p2p)),
                "grid_trade_mean": float(np.mean(grid)),
                "carbon_mean": float(np.mean(carbon)),
            }
        )
    return rows


def _plot_training_trends(runs: Sequence[MethodRun], output_path: Path) -> None:
    curves: dict[str, dict[float, list[dict[str, float]]]] = {method: {} for method in METHOD_ORDER}
    for run in runs:
        for row in _load_training_curve(run.run_dir):
            curves[run.method].setdefault(float(row["iteration"]), []).append(row)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), dpi=220)
    for ax, (metric, (title, higher_is_better)) in zip(axes.flat, METRIC_SPECS.items()):
        for method in METHOD_ORDER:
            points = []
            for iteration in sorted(curves[method]):
                bucket = curves[method][iteration]
                values = [float(item[metric]) for item in bucket]
                points.append((iteration, float(np.mean(values)), float(np.std(values, ddof=1 if len(values) > 1 else 0))))
            if not points:
                continue
            xs = np.asarray([item[0] for item in points], dtype=np.float64)
            mean = np.asarray([item[1] for item in points], dtype=np.float64)
            std = np.asarray([item[2] for item in points], dtype=np.float64)
            ax.plot(xs, mean, linewidth=2.0, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            ax.fill_between(xs, mean - std, mean + std, color=METHOD_COLORS[method], alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Training iteration")
        ax.grid(True, alpha=0.25)
        if not higher_is_better:
            ax.text(0.02, 0.94, "lower is better", transform=ax.transAxes, fontsize=8, color="#555555")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Structured Mask Ablation: Training Trends", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _refresh_outputs(
    output_dir: Path,
    prefix: str,
    raw_rows: Sequence[dict[str, object]],
    runs: Sequence[MethodRun],
    include_training_trends: bool = False,
) -> None:
    by_scenario = _aggregate_by_scenario(raw_rows)
    overall = _aggregate_overall(by_scenario)
    _write_csv(output_dir / f"{prefix}_by_scenario.csv", by_scenario)
    _write_csv(output_dir / f"{prefix}_overall.csv", overall)
    _write_word_table(output_dir / f"{prefix}_word_table.txt", overall)
    _plot_scale_sweep(by_scenario, output_dir / "figures" / f"{prefix}_scale_sweep.png")
    _plot_stress_summary(by_scenario, output_dir / "figures" / f"{prefix}_stress_summary.png")
    if include_training_trends:
        _plot_training_trends(runs, output_dir / "figures" / f"{prefix}_training_trends.png")


def _write_status(path: Path, payload: dict[str, object]) -> None:
    payload = dict(payload)
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _configure_low_resource_runtime()
    _set_global_seeds(int(args.fixed_eval_seed))

    output_dir = _output_dir(args)
    prefix = str(args.report_name)
    raw_csv = output_dir / f"{prefix}_raw.csv"
    status_json = output_dir / f"{prefix}_status.json"
    runs = _method_runs(args)
    scenarios = _scenarios(args)
    total = len(runs) * len(scenarios)

    rows = [] if args.force else _read_existing(raw_csv)
    done_keys = set() if args.force else {_row_key(row) for row in rows}
    device = torch.device("cpu")

    for run_index, run in enumerate(runs):
        for scenario_index, scenario in enumerate(scenarios):
            key = (run.method, run.seed, scenario.name)
            if key in done_keys:
                continue
            eval_args = copy.deepcopy(args)
            eval_args.mask_mode = run.mask_mode
            eval_args.cmtm_mode = "full"
            eval_args.scale_mode = "direct_max"
            eval_args.min_agents = int(scenario.min_agents)
            eval_args.step_churn_prob = float(scenario.step_churn_prob)
            eval_args.n_eval_rollout_threads = 1
            eval_args.n_rollout_threads = 1
            eval_args.use_eval = False
            eval_args.cuda = False
            seed_scenario_index = scenario_index
            if scenario.name == "noise_churn10_p30":
                seed_scenario_index = next(
                    idx for idx, item in enumerate(scenarios) if item.name == "dynamic_churn10_p30"
                )
            eval_args.seed = int(args.fixed_eval_seed) + seed_scenario_index * 1009
            _set_global_seeds(eval_args.seed)

            checkpoint_path = _resolve_checkpoint_path(run.run_dir, args.checkpoint_episode)
            status_payload = {
                "status": "running",
                "completed_rows": len(rows),
                "total_rows": total,
                "active_method": run.method,
                "active_label": run.label,
                "active_seed": run.seed,
                "active_scenario": scenario.name,
                "checkpoint_path": str(checkpoint_path),
            }
            _write_status(status_json, status_payload)

            envs, policy = _build_policy(eval_args, device)
            if scenario.exact_active_count is not None:
                _bind_exact_active_count(envs, int(scenario.exact_active_count))
            else:
                envs.set_min_agents(int(scenario.min_agents))
                envs.set_step_churn_prob(float(scenario.step_churn_prob))
            try:
                policy.restore(str(checkpoint_path))
                policy.eval()
                summaries, diag = _collect_eval_metrics(
                    args=eval_args,
                    policy=policy,
                    envs=envs,
                    scenario=scenario,
                    eval_seed=eval_args.seed,
                )
            finally:
                envs.close()

            metrics = _aggregate_eval_summaries(summaries)
            metrics.update(diag)
            row: dict[str, object] = {
                "method": run.method,
                "method_label": run.label,
                "mask_mode": run.mask_mode,
                "seed": int(run.seed),
                "run_dir": str(run.run_dir),
                "checkpoint_episode": int(args.checkpoint_episode),
                "checkpoint_path": str(checkpoint_path),
                "scenario_name": scenario.name,
                "scenario_label": scenario.label,
                "scenario_category": scenario.category,
                "active_agent_count": (
                    int(scenario.exact_active_count)
                    if scenario.exact_active_count is not None
                    else ""
                ),
                "min_agents": int(scenario.min_agents),
                "step_churn_prob": float(scenario.step_churn_prob),
                "inactive_noise_std": float(scenario.inactive_noise_std),
            }
            row.update(metrics)
            rows.append(row)
            rows.sort(key=lambda item: (METHOD_ORDER.index(str(item["method"])), int(item["seed"]), str(item["scenario_name"])))
            _write_csv(raw_csv, rows)
            _refresh_outputs(output_dir, prefix, rows, runs, include_training_trends=False)
            print(
                "[MaskEval] {label} seed={seed} scenario={scenario} reward={reward:.4f} p2p={p2p:.4f} grid={grid:.4f} carbon={carbon:.4f}".format(
                    label=run.label,
                    seed=run.seed,
                    scenario=scenario.name,
                    reward=float(row["reward_mean"]),
                    p2p=float(row["p2p_mean"]),
                    grid=float(row["grid_trade_mean"]),
                    carbon=float(row["carbon_mean"]),
                )
            )

    _refresh_outputs(output_dir, prefix, rows, runs, include_training_trends=True)
    _write_status(
        status_json,
        {
            "status": "completed",
            "completed_rows": len(rows),
            "total_rows": total,
            "raw_csv": str(raw_csv),
            "by_scenario_csv": str(output_dir / f"{prefix}_by_scenario.csv"),
            "overall_csv": str(output_dir / f"{prefix}_overall.csv"),
            "word_table": str(output_dir / f"{prefix}_word_table.txt"),
            "figures": [
                str(output_dir / "figures" / f"{prefix}_scale_sweep.png"),
                str(output_dir / "figures" / f"{prefix}_stress_summary.png"),
                str(output_dir / "figures" / f"{prefix}_training_trends.png"),
            ],
        },
    )


if __name__ == "__main__":
    main()
