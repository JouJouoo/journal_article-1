#!/usr/bin/env python3
"""Validate storage carbon-memory tracing with accounting-focused experiments."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

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

from dp_lcrl_rl.envs.p2ptrading.dp_lcrl_paper_env import DPLCRLPaperEnv  # noqa: E402
from dp_lcrl_rl.runner.shared.base_runner import _t2n  # noqa: E402
from dp_lcrl_rl.scripts.eval.eval_agent_count_sweep import (  # noqa: E402
    _build_policy,
    _configure_low_resource_runtime,
    _resolve_checkpoint_path,
)
from dp_lcrl_rl.scripts.train.train_paper_mat import (  # noqa: E402
    _apply_cli_aliases,
    _normalize_experiment_args,
    _set_global_seeds,
    build_arg_parser,
    make_env,
)


EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    parser.description = "Run CMTM storage-carbon-memory validation experiments."
    parser.add_argument("--manifest", required=True, help="formal_ablation_manifest.json.")
    parser.add_argument("--output_dir", default="reports/cmtm_memory_validation_20260427")
    parser.add_argument("--report_name", default="cmtm_memory_validation")
    parser.add_argument("--scripted_seeds", type=int, default=20)
    parser.add_argument("--policy_eval_episodes", type=int, default=20)
    parser.add_argument("--fixed_eval_seed", type=int, default=20260427)
    parser.add_argument("--checkpoint_episode", type=int, default=None)
    args = parser.parse_args()
    _apply_cli_aliases(args)
    args.num_agents = 30
    args.min_agents = 30
    args.curriculum_min_agents = 30
    args.curriculum_warmup_episodes = 0
    args.step_churn_prob = 0.0
    args.cmtm_mode = "full"
    args.mask_mode = "full"
    args.scale_mode = "curriculum"
    args.n_rollout_threads = 1
    args.n_eval_rollout_threads = 1
    args.use_eval = False
    args.save_interval = 0
    args.cuda = False
    _normalize_experiment_args(args)
    return args


def _output_dir(args: argparse.Namespace) -> Path:
    path = Path(args.output_dir).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1 if arr.size > 1 else 0))


def _sum_per_agent(info: dict[str, Any], key: str) -> float:
    return float(sum(float(item.get(key, 0.0) or 0.0) for item in info.get("per_agent", [])))


def _active_mean_per_agent(info: dict[str, Any], key: str) -> float:
    vals = [
        float(item.get(key, 0.0) or 0.0)
        for item in info.get("per_agent", [])
        if bool(item.get("active", False))
    ]
    return float(np.mean(vals)) if vals else 0.0


def _configure_delayed_profiles(env: DPLCRLPaperEnv, charge_steps: int = 8) -> None:
    env.pv_profiles = np.zeros((env.max_agents, env.horizon), dtype=np.float32)
    env.load_profiles = np.zeros((env.max_agents, env.horizon), dtype=np.float32)
    env.pv_profiles[:, :charge_steps] = 0.0
    env.load_profiles[:, :charge_steps] = 0.6
    env.pv_profiles[:, charge_steps:] = 4.0
    env.load_profiles[:, charge_steps:] = 6.5
    for agent_idx, state in enumerate(env.states):
        cap = float(env.agent_energy_caps[agent_idx])
        state.energy = 0.08 * cap
        if env.cmtm_mode == "full":
            state.carbon_mass = state.energy * float(env.market_spec.grid_carbon_factor)
            state.storage_intensity = float(env.market_spec.grid_carbon_factor)
        else:
            state.carbon_mass = 0.0
            state.storage_intensity = 0.0
    env._update_profiles_for_step(0)


def _scripted_actions(env: DPLCRLPaperEnv, step_idx: int, charge_steps: int = 8) -> list[np.ndarray]:
    ess_signal = -1.0 if step_idx < charge_steps else 1.0
    return [
        np.array([0.0, 0.0, ess_signal], dtype=np.float32)
        if env.agent_mask[agent_idx] > 0.5
        else np.zeros(3, dtype=np.float32)
        for agent_idx in range(env.max_agents)
    ]


def _run_scripted_delayed_episode(seed: int, num_agents: int = 30) -> tuple[dict[str, float], list[dict[str, float]]]:
    env = DPLCRLPaperEnv(
        max_agents=num_agents,
        min_agents=num_agents,
        horizon=24,
        seed=seed,
        step_churn_prob=0.0,
        cmtm_mode="full",
        storage_capacity_variance=0.0,
        storage_capacity_range=(12.0, 12.0),
    )
    env.reset()
    _configure_delayed_profiles(env, charge_steps=8)
    initial_storage_carbon = float(sum(state.carbon_mass for state in env.states))

    totals = {
        "source_injection": 0.0,
        "load_responsibility": 0.0,
        "charge_responsibility": 0.0,
        "storage_discharge_carbon": 0.0,
        "storage_delta": 0.0,
        "grid_export": 0.0,
        "reward_sum": 0.0,
        "storage_discharge_energy": 0.0,
        "storage_charge_energy": 0.0,
    }
    trace_rows: list[dict[str, float]] = []
    done = False
    step_idx = 0
    while not done:
        _, _, done, info = env.step(_scripted_actions(env, step_idx, charge_steps=8))
        carbon_trace = info.get("carbon_trace", {})
        storage_discharge_carbon = _sum_per_agent(info, "carbon_storage_discharge")
        storage_charge_energy = _sum_per_agent(info, "storage_charge")
        storage_discharge_energy = _sum_per_agent(info, "storage_discharge")
        storage_energy = float(sum(state.energy for state in env.states))
        storage_carbon_mass = float(sum(state.carbon_mass for state in env.states))
        row = {
            "seed": int(seed),
            "step": int(step_idx),
            "source_injection": float(carbon_trace.get("source_injection", 0.0)),
            "load_responsibility": float(carbon_trace.get("load_responsibility", 0.0)),
            "charge_responsibility": float(carbon_trace.get("charge_responsibility", 0.0)),
            "storage_delta": float(carbon_trace.get("storage_delta", 0.0)),
            "storage_discharge_carbon": storage_discharge_carbon,
            "storage_charge_energy": storage_charge_energy,
            "storage_discharge_energy": storage_discharge_energy,
            "storage_energy": storage_energy,
            "storage_carbon_mass": storage_carbon_mass,
            "storage_intensity_mean": _active_mean_per_agent(info, "C_storage_avg"),
            "load_carbon_intensity_mean": _active_mean_per_agent(info, "C_dynamic"),
            "reward": float(info.get("global_reward", 0.0)),
        }
        trace_rows.append(row)
        for key in totals:
            if key == "reward_sum":
                totals[key] += float(row["reward"])
            else:
                totals[key] += float(row.get(key, carbon_trace.get(key, 0.0)) or 0.0)
        step_idx += 1

    final_storage_carbon = float(sum(state.carbon_mass for state in env.states))
    no_pool_omitted = totals["storage_discharge_carbon"]
    delayed_recovery = totals["storage_discharge_carbon"] / max(
        initial_storage_carbon + totals["charge_responsibility"],
        EPS,
    )
    summary = {
        "seed": int(seed),
        "initial_storage_carbon": initial_storage_carbon,
        "final_storage_carbon": final_storage_carbon,
        "source_injection": totals["source_injection"],
        "load_responsibility": totals["load_responsibility"],
        "charge_responsibility": totals["charge_responsibility"],
        "storage_discharge_carbon_full": totals["storage_discharge_carbon"],
        "storage_discharge_carbon_no_pool": 0.0,
        "storage_discharge_energy": totals["storage_discharge_energy"],
        "storage_charge_energy": totals["storage_charge_energy"],
        "omitted_storage_carbon_no_pool": no_pool_omitted,
        "storage_carbon_leakage_ratio_no_pool": no_pool_omitted / max(totals["storage_discharge_carbon"], EPS),
        "delayed_carbon_recovery_ratio_full": delayed_recovery,
        "delayed_carbon_recovery_ratio_no_pool": 0.0,
        "reward_mean": totals["reward_sum"] / 24.0,
    }
    return summary, trace_rows


def _load_full_runs(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = []
    for item in payload.get("runs", []):
        if item.get("method") != "full" or item.get("status") != "completed":
            continue
        runs.append(
            {
                "seed": int(item["seed"]),
                "run_dir": Path(str(item["run_dir"])).expanduser().resolve(),
                "experiment_name": str(item["experiment_name"]),
            }
        )
    runs.sort(key=lambda row: int(row["seed"]))
    return runs


def _policy_counterfactual(args: argparse.Namespace, manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    device = torch.device("cpu")
    for run in _load_full_runs(manifest_path):
        eval_args = copy.deepcopy(args)
        eval_args.cmtm_mode = "full"
        eval_args.seed = int(args.fixed_eval_seed) + int(run["seed"])
        _set_global_seeds(eval_args.seed)
        full_envs, policy = _build_policy(eval_args, device)
        checkpoint = _resolve_checkpoint_path(Path(run["run_dir"]), args.checkpoint_episode)
        policy.restore(str(checkpoint))
        policy.eval()

        stateless_args = copy.deepcopy(eval_args)
        stateless_args.cmtm_mode = "stateless"
        stateless_envs = make_env(stateless_args, stateless_args.n_eval_rollout_threads)
        try:
            for episode_idx in range(int(args.policy_eval_episodes)):
                obs = full_envs.reset()
                stateless_envs.reset()
                rnn_states = np.zeros(
                    (eval_args.n_eval_rollout_threads, eval_args.num_agents, eval_args.recurrent_N, eval_args.hidden_size),
                    dtype=np.float32,
                )
                rnn_states_critic = np.zeros_like(rnn_states)
                masks = np.ones((eval_args.n_eval_rollout_threads, eval_args.num_agents, 1), dtype=np.float32)
                agent_masks = np.asarray(full_envs.agent_masks, dtype=np.float32)[..., None]

                episode_totals = {
                    "reward_full": 0.0,
                    "reward_stateless": 0.0,
                    "load_carbon_full": 0.0,
                    "load_carbon_stateless": 0.0,
                    "storage_discharge_carbon_full": 0.0,
                    "storage_discharge_carbon_stateless": 0.0,
                    "p2p_full": 0.0,
                    "grid_trade_full": 0.0,
                    "carbon_settlement_full": 0.0,
                    "carbon_settlement_stateless": 0.0,
                }

                for _ in range(eval_args.episode_length):
                    obs_batch = np.concatenate(obs, axis=0)
                    share_batch = np.repeat(
                        np.asarray(full_envs.share_obs, dtype=np.float32)[:, None, :],
                        eval_args.num_agents,
                        axis=1,
                    )
                    share_batch = np.concatenate(share_batch, axis=0)
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
                    action_array = np.array(np.split(_t2n(actions), eval_args.n_eval_rollout_threads))
                    obs, _, dones, full_infos = full_envs.step(action_array)
                    _, _, _, stateless_infos = stateless_envs.step(action_array)
                    full_info = full_infos[0]
                    stateless_info = stateless_infos[0]
                    full_trace = full_info.get("carbon_trace", {})
                    stateless_trace = stateless_info.get("carbon_trace", {})
                    full_market = full_info.get("market_summary", {})
                    episode_totals["reward_full"] += float(full_info.get("global_reward", 0.0))
                    episode_totals["reward_stateless"] += float(stateless_info.get("global_reward", 0.0))
                    episode_totals["load_carbon_full"] += float(full_trace.get("load_responsibility", 0.0))
                    episode_totals["load_carbon_stateless"] += float(stateless_trace.get("load_responsibility", 0.0))
                    episode_totals["storage_discharge_carbon_full"] += _sum_per_agent(full_info, "carbon_storage_discharge")
                    episode_totals["storage_discharge_carbon_stateless"] += _sum_per_agent(
                        stateless_info,
                        "carbon_storage_discharge",
                    )
                    episode_totals["p2p_full"] += float(full_market.get("p2p_mean_active", 0.0))
                    episode_totals["grid_trade_full"] += float(full_market.get("grid_buy_mean_active", 0.0)) + float(
                        full_market.get("grid_sell_mean_active", 0.0)
                    )
                    episode_totals["carbon_settlement_full"] += _sum_per_agent(full_info, "carbon_settlement")
                    episode_totals["carbon_settlement_stateless"] += _sum_per_agent(stateless_info, "carbon_settlement")
                    masks = (1.0 - np.asarray(dones, dtype=np.float32)).reshape(
                        eval_args.n_eval_rollout_threads,
                        eval_args.num_agents,
                        1,
                    )
                    agent_masks = np.asarray(
                        [
                            np.asarray(info.get("agent_mask", [1.0] * eval_args.num_agents), dtype=np.float32)
                            for info in full_infos
                        ],
                        dtype=np.float32,
                    )[..., None]

                omitted_zero_pool = episode_totals["storage_discharge_carbon_full"]
                rows.append(
                    {
                        "policy_seed": int(run["seed"]),
                        "episode": int(episode_idx),
                        "checkpoint": str(checkpoint),
                        "reward_full_mean_step": episode_totals["reward_full"] / float(eval_args.episode_length),
                        "reward_stateless_mean_step": episode_totals["reward_stateless"] / float(eval_args.episode_length),
                        "load_carbon_full": episode_totals["load_carbon_full"],
                        "load_carbon_stateless": episode_totals["load_carbon_stateless"],
                        "load_carbon_gap_full_minus_stateless": episode_totals["load_carbon_full"]
                        - episode_totals["load_carbon_stateless"],
                        "storage_discharge_carbon_full": episode_totals["storage_discharge_carbon_full"],
                        "storage_discharge_carbon_stateless": episode_totals["storage_discharge_carbon_stateless"],
                        "storage_discharge_carbon_no_pool": 0.0,
                        "omitted_storage_carbon_no_pool": omitted_zero_pool,
                        "storage_carbon_leakage_ratio_no_pool": omitted_zero_pool
                        / max(episode_totals["storage_discharge_carbon_full"], EPS),
                        "p2p_mean_step": episode_totals["p2p_full"] / float(eval_args.episode_length),
                        "grid_trade_mean_step": episode_totals["grid_trade_full"] / float(eval_args.episode_length),
                        "carbon_settlement_full": episode_totals["carbon_settlement_full"],
                        "carbon_settlement_stateless": episode_totals["carbon_settlement_stateless"],
                    }
                )
        finally:
            full_envs.close()
            stateless_envs.close()
    return rows


def _summarize(rows: Sequence[dict[str, Any]], keys: Sequence[str], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        mean, std = _mean_std(float(row[key]) for row in rows)
        out[f"{prefix}_{key}_mean"] = mean
        out[f"{prefix}_{key}_std"] = std
    return out


def _plot_delayed_trace(trace_rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    first_seed = int(trace_rows[0]["seed"])
    rows = [row for row in trace_rows if int(row["seed"]) == first_seed]
    xs = np.asarray([int(row["step"]) for row in rows], dtype=np.int64)
    charge_energy = np.asarray([float(row["storage_charge_energy"]) for row in rows], dtype=np.float64)
    discharge_energy = np.asarray([float(row["storage_discharge_energy"]) for row in rows], dtype=np.float64)
    storage_delta = np.asarray([float(row["storage_delta"]) for row in rows], dtype=np.float64)
    discharge_carbon = np.asarray([float(row["storage_discharge_carbon"]) for row in rows], dtype=np.float64)
    storage_intensity = np.asarray([float(row["storage_intensity_mean"]) for row in rows], dtype=np.float64)
    storage_energy = np.asarray([float(row.get("storage_energy", np.nan)) for row in rows], dtype=np.float64)
    storage_carbon_mass = np.asarray([float(row.get("storage_carbon_mass", np.nan)) for row in rows], dtype=np.float64)

    if np.all(np.isfinite(storage_carbon_mass)):
        storage_stock = np.maximum(storage_carbon_mass, 0.0)
    else:
        cumulative_delta = np.cumsum(storage_delta)
        initial_stock = max(0.0, -float(np.min(cumulative_delta)))
        storage_stock = initial_stock + cumulative_delta

    def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start: int | None = None
        for idx, active in enumerate(mask):
            if bool(active) and start is None:
                start = idx
            elif not bool(active) and start is not None:
                spans.append((start, idx - 1))
                start = None
        if start is not None:
            spans.append((start, len(mask) - 1))
        return spans

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    charge_color = "#2b8cbe"
    discharge_color = "#e34a33"
    stock_color = "#111111"
    attributed_color = "#fdae6b"
    intensity_color = "#31a354"
    charge_phase = "#dbeefa"
    hold_phase = "#eef7ea"
    discharge_phase = "#fde8cf"
    empty_phase = "#e9e9e9"
    charge_mask = charge_energy > EPS
    discharge_mask = discharge_energy > EPS
    if np.all(np.isfinite(storage_energy)):
        stored_mask = storage_energy > EPS
    else:
        stored_mask = storage_stock > EPS
    hold_mask = (~charge_mask) & (~discharge_mask) & stored_mask
    relevant_mask = charge_mask | discharge_mask | (storage_stock > EPS) | stored_mask
    if np.any(relevant_mask):
        relevant_xs = xs[relevant_mask]
        x_view_min = float(np.min(xs)) - 0.7
        x_view_max = min(float(np.max(xs)) + 0.7, float(np.max(relevant_xs)) + 0.9)
    else:
        x_view_min = float(np.min(xs)) - 0.7
        x_view_max = float(np.max(xs)) + 0.7

    fig = plt.figure(figsize=(10.2, 8.75), dpi=240, constrained_layout=True)
    grid = fig.add_gridspec(4, 1, height_ratios=[0.52, 1.0, 1.0, 1.0])
    legend_ax = fig.add_subplot(grid[0, 0])
    axes = np.asarray(
        [
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[2, 0]),
            fig.add_subplot(grid[3, 0]),
        ],
        dtype=object,
    )
    axes[1].sharex(axes[0])
    axes[2].sharex(axes[0])
    legend_ax.axis("off")
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.08, hspace=0.07)

    def _style_legend(legend: Any) -> None:
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("#d0d0d0")
        frame.set_linewidth(0.6)
        frame.set_alpha(1.0)
    for ax in axes:
        for start, end in _spans(hold_mask):
            ax.axvspan(xs[start] - 0.5, xs[end] + 0.5, color=hold_phase, alpha=0.78, lw=0, zorder=0)
        for start, end in _spans(charge_mask):
            ax.axvspan(xs[start] - 0.5, xs[end] + 0.5, color=charge_phase, alpha=0.75, lw=0, zorder=0)
        for start, end in _spans(discharge_mask):
            ax.axvspan(xs[start] - 0.5, xs[end] + 0.5, color=discharge_phase, alpha=0.8, lw=0, zorder=0)
        ax.grid(True, axis="y", color="#d9d9d9", lw=0.65, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)

    charge_bars = axes[0].bar(
        xs,
        charge_energy,
        width=0.64,
        color=charge_color,
        edgecolor="white",
        linewidth=0.5,
        label="Charge energy",
        zorder=3,
    )
    discharge_bars = axes[0].bar(
        xs,
        -discharge_energy,
        width=0.64,
        color=discharge_color,
        edgecolor="white",
        linewidth=0.5,
        label="Discharge energy",
        zorder=3,
    )
    energy_limit = max(float(np.max(charge_energy)), float(np.max(discharge_energy)), 1.0) * 1.18
    axes[0].axhline(0.0, color="#333333", lw=0.9, zorder=2)
    axes[0].set_ylim(-energy_limit, energy_limit)
    axes[0].set_ylabel("Energy (kWh)", fontsize=10)
    axes[0].set_title("(a) Storage charge and discharge schedule", loc="left", fontsize=11, fontweight="semibold")
    hold_spans = _spans(hold_mask)
    if hold_spans:
        hold_start, hold_end = max(hold_spans, key=lambda span: span[1] - span[0])
        axes[0].text(
            float(xs[hold_start] + xs[hold_end]) / 2.0,
            energy_limit * 0.54,
            "holding period\nstored carbon retained",
            ha="center",
            va="center",
            fontsize=8.2,
            color="#3f5f3b",
            bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#b8d4ad", "alpha": 0.92},
            zorder=6,
        )

    stock_line = axes[1].plot(
        xs,
        storage_stock,
        marker="o",
        ms=4.2,
        lw=1.9,
        color=stock_color,
        label="Storage carbon stock",
        zorder=4,
    )[0]
    carbon_bars = axes[1].bar(
        xs,
        discharge_carbon,
        width=0.54,
        color=attributed_color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.45,
        label="Discharge-attributed carbon",
        zorder=3,
    )
    carbon_limit = max(float(np.max(storage_stock)), float(np.max(discharge_carbon)), 1.0) * 1.12
    axes[1].set_ylim(0.0, carbon_limit)
    axes[1].set_ylabel("Carbon amount (kgCO2)", fontsize=10)
    axes[1].set_title("(b) Carbon retained in storage and released at discharge", loc="left", fontsize=11, fontweight="semibold")

    intensity_valid = stored_mask
    intensity_plot = np.where(intensity_valid, storage_intensity, np.nan)
    intensity_line = axes[2].plot(
        xs,
        intensity_plot,
        marker="s",
        ms=4.0,
        lw=1.9,
        color=intensity_color,
        label="Storage carbon intensity",
        zorder=4,
    )[0]
    has_empty_storage = bool(np.any(~intensity_valid))
    if has_empty_storage:
        empty_steps = xs[~intensity_valid]
        axes[2].axvspan(
            float(np.min(empty_steps)) - 0.5,
            x_view_max,
            color=empty_phase,
            alpha=0.9,
            lw=0,
            zorder=0,
        )
    valid_values = storage_intensity[intensity_valid]
    intensity_top = 0.8 if valid_values.size == 0 else max(0.8, float(np.max(valid_values)) + 0.06)
    axes[2].set_ylim(-0.03, min(1.05, intensity_top))
    axes[2].set_ylabel("Carbon intensity (kgCO2/kWh)", fontsize=10)
    axes[2].set_xlabel("Step")
    axes[2].set_title("(c) Carbon intensity of the remaining stored energy", loc="left", fontsize=11, fontweight="semibold")

    visible_xs = xs[(xs >= np.ceil(x_view_min)) & (xs <= np.floor(x_view_max))]
    tick_interval = 2 if visible_xs.size > 13 else 1
    axes[2].set_xticks(visible_xs[::tick_interval])
    axes[2].set_xlim(x_view_min, x_view_max)
    legend_handles = [
        Patch(facecolor=charge_color, edgecolor="white", linewidth=0.5),
        Patch(facecolor=discharge_color, edgecolor="white", linewidth=0.5),
        Patch(facecolor=charge_phase, edgecolor="#b9d6ea", linewidth=0.6, alpha=0.85),
        Patch(facecolor=hold_phase, edgecolor="#b8d4ad", linewidth=0.6, alpha=0.9),
        Patch(facecolor=discharge_phase, edgecolor="#eecb9d", linewidth=0.6, alpha=0.9),
        Line2D([0], [0], color=stock_color, marker="o", lw=1.9, ms=4.2),
        Patch(facecolor=attributed_color, edgecolor="white", linewidth=0.45, alpha=0.85),
        Line2D([0], [0], color=intensity_color, marker="s", lw=1.9, ms=4.0),
    ]
    legend_labels = [
        "Charge energy",
        "Discharge energy",
        "Charging window",
        "Holding window",
        "Discharging window",
        "Storage carbon stock",
        "Discharge-attributed carbon",
        "Storage carbon intensity",
    ]
    if has_empty_storage:
        legend_handles.append(Patch(facecolor=empty_phase, edgecolor="#bdbdbd", linewidth=0.6, alpha=0.9))
        legend_labels.append("Storage empty")
    legend = legend_ax.legend(
        legend_handles,
        legend_labels,
        loc="center",
        ncol=3,
        frameon=True,
        fontsize=8.3,
        handlelength=1.6,
        columnspacing=1.15,
        labelspacing=0.55,
        borderpad=0.5,
    )
    _style_legend(legend)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    fig.suptitle("Delayed Carbon Attribution Through Storage", fontsize=14, fontweight="semibold")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def _plot_summary(summary: dict[str, Any], output_path: Path) -> None:
    labels = ["Full CMTM", "No Storage Pool"]
    storage = [
        float(summary["scripted_storage_discharge_carbon_full_mean"]),
        0.0,
    ]
    recovery = [
        float(summary["scripted_delayed_carbon_recovery_ratio_full_mean"]),
        0.0,
    ]
    leakage = [
        0.0,
        float(summary["scripted_storage_carbon_leakage_ratio_no_pool_mean"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=220)
    panels = [
        (storage, "Recorded Storage-Discharge Carbon"),
        (recovery, "Delayed Carbon Recovery Ratio"),
        (leakage, "Carbon Leakage Ratio"),
    ]
    for ax, (values, title) in zip(axes, panels):
        ax.bar(labels, values, color=["#111111", "#d94801"])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_word_table(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "Experiment                         Key Metric                         Full CMTM          No Storage Pool",
        "---------------------------------  ---------------------------------  -----------------  ----------------",
        "Carbon conservation / leakage      Recorded storage-discharge carbon  "
        f"{summary['scripted_storage_discharge_carbon_full_mean']:.3f} +/- {summary['scripted_storage_discharge_carbon_full_std']:.3f}  "
        "0.000 +/- 0.000",
        "Carbon conservation / leakage      Carbon leakage ratio               "
        "0.000 +/- 0.000    "
        f"{summary['scripted_storage_carbon_leakage_ratio_no_pool_mean']:.3f} +/- {summary['scripted_storage_carbon_leakage_ratio_no_pool_std']:.3f}",
        "Delayed attribution                Delayed carbon recovery ratio      "
        f"{summary['scripted_delayed_carbon_recovery_ratio_full_mean']:.3f} +/- {summary['scripted_delayed_carbon_recovery_ratio_full_std']:.3f}  "
        "0.000 +/- 0.000",
        "Same-policy counterfactual         Omitted storage carbon             "
        "0.000 +/- 0.000    "
        f"{summary['policy_omitted_storage_carbon_no_pool_mean']:.3f} +/- {summary['policy_omitted_storage_carbon_no_pool_std']:.3f}",
        "",
        "Notes: No Storage Pool is a zero-memory counterfactual under the same trajectory.",
        "It records no historical carbon when stored energy is discharged, so omitted storage carbon is leakage.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    _configure_low_resource_runtime()
    output_dir = _output_dir(args)
    prefix = str(args.report_name)

    scripted_summaries: list[dict[str, Any]] = []
    delayed_trace_rows: list[dict[str, Any]] = []
    for idx in range(int(args.scripted_seeds)):
        summary, trace = _run_scripted_delayed_episode(seed=int(args.fixed_eval_seed) + idx, num_agents=int(args.num_agents))
        scripted_summaries.append(summary)
        delayed_trace_rows.extend(trace)

    policy_rows = _policy_counterfactual(args, Path(args.manifest).expanduser().resolve())

    summary: dict[str, Any] = {
        "scripted_seeds": int(args.scripted_seeds),
        "policy_eval_episodes": int(args.policy_eval_episodes),
        "fixed_eval_seed": int(args.fixed_eval_seed),
    }
    summary.update(
        _summarize(
            scripted_summaries,
            [
                "storage_discharge_carbon_full",
                "omitted_storage_carbon_no_pool",
                "storage_carbon_leakage_ratio_no_pool",
                "delayed_carbon_recovery_ratio_full",
                "delayed_carbon_recovery_ratio_no_pool",
                "load_responsibility",
                "charge_responsibility",
            ],
            "scripted",
        )
    )
    summary.update(
        _summarize(
            policy_rows,
            [
                "omitted_storage_carbon_no_pool",
                "storage_carbon_leakage_ratio_no_pool",
                "storage_discharge_carbon_full",
                "storage_discharge_carbon_stateless",
                "load_carbon_gap_full_minus_stateless",
                "reward_full_mean_step",
                "reward_stateless_mean_step",
            ],
            "policy",
        )
    )

    _write_csv(output_dir / f"{prefix}_carbon_leakage.csv", scripted_summaries)
    _write_csv(output_dir / f"{prefix}_delayed_trace.csv", delayed_trace_rows)
    _write_csv(output_dir / f"{prefix}_same_policy_counterfactual.csv", policy_rows)
    (output_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_word_table(output_dir / f"{prefix}_word_table.txt", summary)
    _plot_delayed_trace(delayed_trace_rows, output_dir / "figures" / f"{prefix}_delayed_trace.png")
    _plot_summary(summary, output_dir / "figures" / f"{prefix}_summary.png")
    print(f"saved_dir={output_dir}")
    print(f"word_table={output_dir / f'{prefix}_word_table.txt'}")


if __name__ == "__main__":
    main()
