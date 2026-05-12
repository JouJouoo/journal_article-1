#!/usr/bin/env python3
"""Extended CMTM validation: P2P chain, dynamic freeze-thaw, imbalance decomposition."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WANDB_DISABLED", "true")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dp_lcrl_rl.envs.p2ptrading.dp_lcrl_paper_env import DPLCRLPaperEnv
from dp_lcrl_rl.scripts.train.train_paper_mat import build_arg_parser

# 鈹€鈹€ Global style 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
})

EPS = 1e-9
RCP = {
    "charge_bg": "#e8f3ff",
    "discharge_bg": "#fff0df",
    "inactive_bg": "#f5f5f5",
    "cmtm": "#111111",
    "stateless": "#d94801",
    "grid": "#2b8cbe",
    "pv": "#31a354",
    "p2p": "#fdae6b",
}

# 鈹€鈹€ Helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def _mean_std(vals: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(vals), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1 if arr.size > 1 else 0))


def _sum_per_agent(info: dict, key: str) -> float:
    return float(sum(
        float(item.get(key, 0.0) or 0.0) for item in info.get("per_agent", [])
    ))


def _active_mean_per_agent(info: dict, key: str) -> float:
    vals = [
        float(item.get(key, 0.0) or 0.0) for item in info.get("per_agent", [])
        if bool(item.get("active", False))
    ]
    return float(np.mean(vals)) if vals else 0.0


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _annotate_bars(ax, values, fmt: str = ".3f"):
    for idx, v in enumerate(values):
        ax.text(idx, v, f"{v:{fmt}}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)


def _span_bg(ax, xs, charge_mask, discharge_mask, inactive_mask=None):
    """Draw background shading for charge/discharge/inactive regions."""
    charge_spans = _mask_to_spans(charge_mask)
    discharge_spans = _mask_to_spans(discharge_mask)
    inactive_spans = _mask_to_spans(inactive_mask) if inactive_mask is not None else []
    for s, e in charge_spans:
        ax.axvspan(xs[s] - 0.5, xs[e] + 0.5, color=RCP["charge_bg"], alpha=0.8, lw=0)
    for s, e in discharge_spans:
        ax.axvspan(xs[s] - 0.5, xs[e] + 0.5, color=RCP["discharge_bg"], alpha=0.85, lw=0)
    for s, e in inactive_spans:
        ax.axvspan(xs[s] - 0.5, xs[e] + 0.5, color=RCP["inactive_bg"], alpha=0.6, lw=0, zorder=-1)


def _mask_to_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans = []
    start = None
    for idx, active in enumerate(mask):
        if bool(active) and start is None:
            start = idx
        elif not bool(active) and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    return spans


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# EXPERIMENT 1: P2P Carbon Chain
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

def _p2p_profiles(env: DPLCRLPaperEnv, phase: str):
    """Override PV/load profiles for P2P chain scenario.

    Agent layout:
      0 鈥?PV-rich with storage (low-carbon seller)
      1 鈥?Grid-dependent with storage (high-carbon seller)
      2 鈥?Pure buyer
    """
    n = env.max_agents
    env.pv_profiles = np.zeros((n, env.horizon), dtype=np.float32)
    env.load_profiles = np.zeros((n, env.horizon), dtype=np.float32)
    if phase == "charge":
        env.pv_profiles[0, :] = 5.0
        env.load_profiles[0, :] = 1.0
        env.load_profiles[1, :] = 2.0
        env.load_profiles[2, :] = 1.5
    elif phase == "discharge":
        env.pv_profiles[0, :8] = 1.0
        env.pv_profiles[0, 8:] = 0.5
        env.load_profiles[0, :] = 3.0
        env.load_profiles[1, :] = 0.5
        env.load_profiles[2, :] = 5.0
    for state in env.states:
        cap = float(env.agent_energy_caps[state.id])
        if phase == "charge":
            state.energy = 0.1 * cap
            state.carbon_mass = state.energy * float(env.market_spec.grid_carbon_factor)
            state.storage_intensity = float(env.market_spec.grid_carbon_factor)
        else:
            # Discharge phase: Agent 0's storage charged from PV (low carbon),
            # Agent 1's storage charged from grid (high carbon).
            # We set initial carbon_mass to reflect this.
            state.energy = 0.85 * cap
            if state.id == 0:
                # Low-carbon storage (charged with PV + some grid mix)
                state.carbon_mass = state.energy * 0.15
                state.storage_intensity = 0.15
            elif state.id == 1:
                # High-carbon storage (charged from grid)
                state.carbon_mass = state.energy * 0.70
                state.storage_intensity = 0.70
            else:
                state.carbon_mass = 0.0
                state.storage_intensity = 0.0
    env._update_profiles_for_step(0)


def _p2p_scripted_actions(env: DPLCRLPaperEnv, step: int, phase: str) -> list[np.ndarray]:
    actions = []
    for agent_idx in range(env.max_agents):
        if env.agent_mask[agent_idx] < 0.5:
            actions.append(np.zeros(3, dtype=np.float32))
            continue
        if phase == "charge":
            if agent_idx == 0:
                actions.append(np.array([0.0, 0.0, -0.8], dtype=np.float32))
            elif agent_idx == 1:
                actions.append(np.array([0.0, 0.0, -0.8], dtype=np.float32))
            else:
                actions.append(np.array([-0.5, 0.2, 0.0], dtype=np.float32))
        else:
            if agent_idx == 0:
                actions.append(np.array([0.8, -0.5, 0.8], dtype=np.float32))
            elif agent_idx == 1:
                actions.append(np.array([0.8, -0.3, 0.8], dtype=np.float32))
            else:
                actions.append(np.array([-0.9, 0.6, 0.0], dtype=np.float32))
    return actions


def _run_p2p_chain(seed: int, num_agents: int = 3, cmtm_mode: str = "full") -> dict:
    env = DPLCRLPaperEnv(
        max_agents=num_agents, min_agents=num_agents, horizon=24, seed=seed,
        cmtm_mode=cmtm_mode,
        storage_capacity_variance=0.0, storage_capacity_range=(12.0, 12.0),
        step_churn_prob=0.0,
    )
    env.reset()
    env.agent_mask[:] = 1.0
    env.active_ids = list(range(num_agents))

    # Phase 1: charge (steps 0-7)
    _p2p_profiles(env, "charge")
    for s in range(8):
        env.step(_p2p_scripted_actions(env, s, "charge"))

    # Phase 2: discharge + P2P (steps 8-23)
    _p2p_profiles(env, "discharge")
    trace = []
    for s in range(8, 24):
        obs, rewards, done, info = env.step(_p2p_scripted_actions(env, s, "discharge"))
        per = info.get("per_agent", [])
        # Use C_sell (output intensity) as carbon intensity; always available for active agents
        c_sell_0 = next((p["C_sell"] for p in per if p.get("active") and p["agent_id"] == 0), 0.0)
        c_sell_1 = next((p["C_sell"] for p in per if p.get("active") and p["agent_id"] == 1), 0.0)
        # P2P import carbon for buyer (agent 2)
        buyer_p2p = next((p["carbon_p2p_import"] for p in per if p.get("active") and p["agent_id"] == 2), 0.0)
        buyer_carbon_load = next((p["carbon_load_responsibility"] for p in per if p.get("active") and p["agent_id"] == 2), 0.0)
        trace.append({
            "step": s,
            "seller_0_output_intensity": float(c_sell_0),
            "seller_1_output_intensity": float(c_sell_1),
            "buyer_carbon_p2p_import": float(buyer_p2p),
            "buyer_carbon_load": float(buyer_carbon_load),
            "cmtm_mode": cmtm_mode,
        })
    env.close()
    return trace


def experiment_p2p_chain(output_dir: Path, num_seeds: int = 5) -> dict:
    """Run P2P carbon chain experiment (full vs stateless) and produce figures."""
    fixed_seed = 20260427
    full_traces = []
    stateless_traces = []
    for idx in range(num_seeds):
        full_traces.extend(_run_p2p_chain(fixed_seed + idx, cmtm_mode="full"))
        stateless_traces.extend(_run_p2p_chain(fixed_seed + idx, cmtm_mode="stateless"))

    _write_csv(output_dir / "p2p_chain_full_trace.csv", full_traces)
    _write_csv(output_dir / "p2p_chain_stateless_trace.csv", stateless_traces)

    # Aggregate
    steps = sorted(set(r["step"] for r in full_traces))
    agg_full = {}
    agg_stateless = {}
    for s in steps:
        rows = [r for r in full_traces if r["step"] == s]
        agg_full[s] = {k: float(np.mean([r[k] for r in rows])) for k in ["buyer_carbon_p2p_import", "buyer_carbon_load", "seller_0_output_intensity", "seller_1_output_intensity"]}
        rows_s = [r for r in stateless_traces if r["step"] == s]
        agg_stateless[s] = {k: float(np.mean([r[k] for r in rows_s])) for k in ["buyer_carbon_p2p_import", "buyer_carbon_load", "seller_0_output_intensity", "seller_1_output_intensity"]}

    xs = np.array(steps)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=200)

    # Panel A: Seller carbon intensities
    ax = axes[0, 0]
    full_0 = [agg_full[s]["seller_0_output_intensity"] for s in steps]
    full_1 = [agg_full[s]["seller_1_output_intensity"] for s in steps]
    sl_0 = [agg_stateless[s]["seller_0_output_intensity"] for s in steps]
    sl_1 = [agg_stateless[s]["seller_1_output_intensity"] for s in steps]
    ax.plot(xs, full_0, "s-", color="#31a354", label="Seller 0 (low-C) 鈥?Full CMTM", ms=4)
    ax.plot(xs, full_1, "s-", color="#d94801", label="Seller 1 (high-C) 鈥?Full CMTM", ms=4)
    ax.plot(xs, sl_0, "o--", color="#74c476", label="Seller 0 鈥?Stateless", ms=3, alpha=0.7)
    ax.plot(xs, sl_1, "o--", color="#fc9272", label="Seller 1 鈥?Stateless", ms=3, alpha=0.7)
    ax.set_ylabel("P2P Carbon Intensity (kgCO2/kWh)")
    ax.set_title("P2P carbon intensities by Seller")
    ax.legend(frameon=False, fontsize=8)

    # Panel B: Buyer carbon load
    ax = axes[0, 1]
    full_load = [agg_full[s]["buyer_carbon_load"] for s in steps]
    sl_load = [agg_stateless[s]["buyer_carbon_load"] for s in steps]
    ax.bar(xs - 0.15, full_load, width=0.28, color=RCP["cmtm"], label="Full CMTM")
    ax.bar(xs + 0.15, sl_load, width=0.28, color=RCP["stateless"], label="Stateless")
    ax.set_ylabel("Buyer Load Carbon (kgCO鈧?")
    ax.set_title("Buyer (Agent 2) Carbon Footprint")
    ax.legend(frameon=False)

    # Panel C: Buyer P2P import carbon
    ax = axes[1, 0]
    full_p2p = [agg_full[s]["buyer_carbon_p2p_import"] for s in steps]
    sl_p2p = [agg_stateless[s]["buyer_carbon_p2p_import"] for s in steps]
    ax.plot(xs, full_p2p, "o-", color=RCP["cmtm"], label="Full CMTM", ms=4)
    ax.plot(xs, sl_p2p, "o--", color=RCP["stateless"], label="Stateless", ms=4)
    ax.set_ylabel("P2P Import Carbon (kgCO鈧?")
    ax.set_title("Buyer P2P Carbon Import")
    ax.legend(frameon=False)

    # Panel D: Carbon Intensity Gap (Full - Stateless)
    ax = axes[1, 1]
    intensity_gap_0 = np.array(full_0) - np.array(sl_0)
    intensity_gap_1 = np.array(full_1) - np.array(sl_1)
    ax.plot(xs, intensity_gap_0, "s-", color="#31a354", label="Seller 0 gap", ms=4)
    ax.plot(xs, intensity_gap_1, "s-", color="#d94801", label="Seller 1 gap", ms=4)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("Carbon Intensity Gap (Full - Stateless)")
    ax.set_title("CMTM Effect on carbon intensities")
    ax.legend(frameon=False)

    fig.suptitle("Experiment 1: P2P Carbon Chain Validation", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "p2p_chain.png", bbox_inches="tight")
    plt.close(fig)

    # Bar summary
    fig2, axes2 = plt.subplots(1, 3, figsize=(11, 4.2), dpi=200)
    total_load_full = sum(agg_full[s]["buyer_carbon_load"] for s in steps)
    total_load_sl = sum(agg_stateless[s]["buyer_carbon_load"] for s in steps)
    avg_intensity_0_full = float(np.mean(full_0))
    avg_intensity_1_full = float(np.mean(full_1))
    avg_intensity_0_sl = float(np.mean(sl_0))
    avg_intensity_1_sl = float(np.mean(sl_1))

    # (1) Total buyer load carbon
    ax = axes2[0]
    ax.bar(["Full CMTM", "Stateless"], [total_load_full, total_load_sl], color=[RCP["cmtm"], RCP["stateless"]])
    ax.set_title("Total Buyer Load Carbon")
    _annotate_bars(ax, [total_load_full, total_load_sl])

    # (2) Avg P2P carbon intensities
    ax = axes2[1]
    x = np.arange(2)
    w = 0.3
    ax.bar(x[0] - w/2, avg_intensity_0_full, w, color="#31a354", label="Seller 0 Full")
    ax.bar(x[0] + w/2, avg_intensity_0_sl, w, color="#74c476", label="Seller 0 Stateless", alpha=0.7)
    ax.bar(x[1] - w/2, avg_intensity_1_full, w, color="#d94801", label="Seller 1 Full")
    ax.bar(x[1] + w/2, avg_intensity_1_sl, w, color="#fc9272", label="Seller 1 Stateless", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Seller 0 (low-C)", "Seller 1 (high-C)"])
    ax.set_title("Avg P2P carbon intensity")
    ax.legend(frameon=False, fontsize=7)

    # (3) Gap
    ax = axes2[2]
    ax.bar(["Seller 0 gap", "Seller 1 gap"],
           [float(np.mean(intensity_gap_0)), float(np.mean(intensity_gap_1))],
           color=["#31a354", "#d94801"])
    ax.set_title("CMTM Effect (Full 鈭?Stateless)")
    ax.axhline(0, color="gray", lw=0.5)
    _annotate_bars(ax, [float(np.mean(intensity_gap_0)), float(np.mean(intensity_gap_1))])

    fig2.suptitle("P2P Chain 鈥?Aggregate Summary", fontsize=12, y=1.02)
    fig2.tight_layout()
    fig2.savefig(output_dir / "figures" / "p2p_chain_summary.png", bbox_inches="tight")
    plt.close(fig2)

    summary = {
        "num_seeds": num_seeds,
        "total_load_carbon_full": total_load_full,
        "total_load_carbon_stateless": total_load_sl,
        "avg_seller0_intensity_full": avg_intensity_0_full,
        "avg_seller1_intensity_full": avg_intensity_1_full,
        "avg_seller0_intensity_stateless": avg_intensity_0_sl,
        "avg_seller1_intensity_stateless": avg_intensity_1_sl,
        "intensity_gap_seller0_mean": float(np.mean(intensity_gap_0)),
        "intensity_gap_seller1_mean": float(np.mean(intensity_gap_1)),
        "intensity_gap_seller0_std": float(np.std(intensity_gap_0, ddof=1)),
        "intensity_gap_seller1_std": float(np.std(intensity_gap_1, ddof=1)),
    }
    (output_dir / "p2p_chain_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# EXPERIMENT 2: Dynamic Participation Freeze-Thaw
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

def _freeze_profiles(env: DPLCRLPaperEnv):
    """Override profiles for freeze-thaw scenario.

    Phase 1 (steps 0-5): All active, charge storage
    Phase 2 (steps 6-15): Agents 2-4 become inactive
    Phase 3 (steps 16-23): All active again, discharge
    """
    n = env.max_agents
    env.pv_profiles = np.zeros((n, env.horizon), dtype=np.float32)
    env.load_profiles = np.zeros((n, env.horizon), dtype=np.float32)
    env.pv_profiles[0, :6] = 4.0
    env.pv_profiles[0, 6:16] = 1.0
    env.pv_profiles[0, 16:] = 0.5
    env.load_profiles[:, :6] = 1.0
    env.load_profiles[:, 6:16] = 0.5
    env.load_profiles[:, 16:] = 3.0


def _freeze_actions(env: DPLCRLPaperEnv, step: int) -> list[np.ndarray]:
    phase_charge = step < 6
    phase_inactive = 6 <= step < 16
    actions = []
    for agent_idx in range(env.max_agents):
        if env.agent_mask[agent_idx] < 0.5:
            actions.append(np.zeros(3, dtype=np.float32))
        elif phase_charge:
            actions.append(np.array([0.0, 0.0, -0.8], dtype=np.float32))
        elif phase_inactive:
            actions.append(np.zeros(3, dtype=np.float32))
        else:
            actions.append(np.array([0.0, 0.0, 0.8], dtype=np.float32))
    return actions


def _run_freeze_thaw(seed: int, num_agents: int = 5, cmtm_mode: str = "full") -> list[dict]:
    env = DPLCRLPaperEnv(
        max_agents=num_agents, min_agents=num_agents, horizon=24, seed=seed,
        cmtm_mode=cmtm_mode,
        storage_capacity_variance=0.0, storage_capacity_range=(12.0, 12.0),
        step_churn_prob=0.0,
    )
    env.reset()

    # Override agent_mask manually for dynamic participation
    env.agent_mask[:] = 1.0
    env.active_ids = list(range(num_agents))

    _freeze_profiles(env)
    trace = []
    for s in range(24):
        # Apply dynamic participation schedule
        if 6 <= s < 16:
            # Agents 2-4 become inactive
            new_mask = np.ones(num_agents, dtype=np.float32)
            new_mask[2:5] = 0.0
            env.agent_mask[:] = new_mask
            env.active_ids = [int(i) for i in np.flatnonzero(new_mask > 0.5)]
        else:
            env.agent_mask[:] = 1.0
            env.active_ids = list(range(num_agents))

        actions = _freeze_actions(env, s)
        obs, rewards, done, info = env.step(actions)

        row = {
            "step": s,
            "phase": "charge" if s < 6 else ("freeze" if s < 16 else "discharge"),
            "cmtm_mode": cmtm_mode,
        }
        for agent_idx in range(num_agents):
            state = env.states[agent_idx]
            row[f"agent{agent_idx}_carbon_mass"] = float(state.carbon_mass)
            row[f"agent{agent_idx}_storage_intensity"] = float(state.storage_intensity)
            row[f"agent{agent_idx}_energy"] = float(state.energy)
            row[f"agent{agent_idx}_active"] = float(env.agent_mask[agent_idx])
        trace.append(row)
    env.close()
    return trace


def experiment_freeze_thaw(output_dir: Path, num_seeds: int = 5) -> dict:
    fixed_seed = 20260427
    full_traces = []
    sl_traces = []
    for idx in range(num_seeds):
        full_traces.extend(_run_freeze_thaw(fixed_seed + idx, cmtm_mode="full"))
        sl_traces.extend(_run_freeze_thaw(fixed_seed + idx, cmtm_mode="stateless"))

    _write_csv(output_dir / "freeze_thaw_full_trace.csv", full_traces)
    _write_csv(output_dir / "freeze_thaw_stateless_trace.csv", sl_traces)

    # Aggregate over seeds
    steps = list(range(24))
    # For agents 2 (freeze agent), track carbon_mass across steps
    agg_full = {}
    for s in steps:
        rows = [r for r in full_traces if r["step"] == s]
        agg_full[s] = {k: float(np.mean([r[k] for r in rows])) for k in [
            "agent0_carbon_mass", "agent2_carbon_mass", "agent0_storage_intensity",
            "agent2_storage_intensity", "agent0_energy", "agent2_energy"
        ]}

    # Plot
    xs = np.array(steps)
    charge_mask = xs < 6
    inactive_mask = (6 <= xs) & (xs < 16)
    discharge_mask = xs >= 16

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), dpi=200)

    # Row 1: Agent 0 (always active) carbon mass + intensity
    ax = axes[0, 0]
    _span_bg(ax, xs, charge_mask, discharge_mask, inactive_mask)
    ax.plot(xs, [agg_full[s]["agent0_carbon_mass"] for s in steps], "o-", color=RCP["cmtm"], ms=4)
    ax.set_ylabel("Carbon Mass (kgCO鈧?")
    ax.set_title("Agent 0 (Always Active) 鈥?Carbon Mass")

    ax = axes[0, 1]
    _span_bg(ax, xs, charge_mask, discharge_mask, inactive_mask)
    ax.plot(xs, [agg_full[s]["agent0_storage_intensity"] for s in steps], "s-", color="#31a354", ms=4)
    ax.set_ylabel("Storage Carbon Intensity")
    ax.set_title("Agent 0 鈥?Storage Intensity")
    _annotate_phases(ax, charge=6, inactive=16, discharge_start=16)

    # Row 2: Agent 2 (freeze agent) carbon mass
    ax = axes[1, 0]
    _span_bg(ax, xs, charge_mask, discharge_mask, inactive_mask)
    ax.plot(xs, [agg_full[s]["agent2_carbon_mass"] for s in steps], "o-", color=RCP["cmtm"], ms=4, label="Full CMTM")
    # Also plot stateless for comparison
    agg_sl = {}
    for s in steps:
        rows = [r for r in sl_traces if r["step"] == s]
        agg_sl[s] = float(np.mean([r["agent2_carbon_mass"] for r in rows])) if rows else 0.0
    ax.plot(xs, [agg_sl[s] for s in steps], "o--", color=RCP["stateless"], ms=4, label="Stateless")
    ax.axvline(5.5, color="red", ls=":", lw=1, alpha=0.5)
    ax.axvline(15.5, color="green", ls=":", lw=1, alpha=0.5)
    ax.set_ylabel("Carbon Mass (kgCO鈧?")
    ax.set_title("Agent 2 (Freeze-Thaw) 鈥?Carbon Mass")
    ax.legend(frameon=False)

    # Highlight freeze period: carbon mass should be flat
    freeze_indices = [s for s in steps if 6 <= s < 16]
    freeze_vals = [agg_full[s]["agent2_carbon_mass"] for s in freeze_indices]
    ax = axes[1, 1]
    _span_bg(ax, np.array(freeze_indices), np.zeros(len(freeze_indices)), np.zeros(len(freeze_indices)))
    ax.plot(freeze_indices, freeze_vals, "o-", color=RCP["cmtm"], ms=5)
    ax.set_ylabel("Carbon Mass During Freeze")
    ax.set_title("Agent 2 鈥?Carbon Mass Frozen?")
    # Compute flatness
    if len(freeze_vals) >= 2:
        freeze_std = float(np.std(freeze_vals, ddof=1))
        ax.text(0.5, 0.9, f"Std during freeze: {freeze_std:.6f}", transform=ax.transAxes,
                ha="center", fontsize=10,
                color="green" if freeze_std < 1e-4 else "red")

    # Row 3: Energy and intensity for agent 2
    ax = axes[2, 0]
    _span_bg(ax, xs, charge_mask, discharge_mask, inactive_mask)
    ax.plot(xs, [agg_full[s]["agent2_energy"] for s in steps], "o-", color="#2b8cbe", ms=4)
    ax.set_ylabel("Storage Energy (kWh)")
    ax.set_title("Agent 2 鈥?Storage Energy")

    ax = axes[2, 1]
    _span_bg(ax, xs, charge_mask, discharge_mask, inactive_mask)
    ax.plot(xs, [agg_full[s]["agent2_storage_intensity"] for s in steps], "s-", color="#31a354", ms=4)
    ax.set_ylabel("Storage Carbon Intensity")
    ax.set_title("Agent 2 鈥?Storage Intensity (Preserved?)")

    fig.suptitle("Experiment 2: Dynamic Participation Freeze-Thaw Validation", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "freeze_thaw.png", bbox_inches="tight")
    plt.close(fig)

    # Summary bar chart
    pre_freeze_carbon = float(np.mean([agg_full[s]["agent2_carbon_mass"] for s in [5]]))
    during_freeze_carbon = float(np.mean([agg_full[s]["agent2_carbon_mass"] for s in range(6, 16)]))
    post_freeze_carbon = float(np.mean([agg_full[s]["agent2_carbon_mass"] for s in [16]]))
    pre_intensity = float(np.mean([agg_full[s]["agent2_storage_intensity"] for s in [5]]))
    post_intensity = float(np.mean([agg_full[s]["agent2_storage_intensity"] for s in [16]]))

    fig2, axes2 = plt.subplots(1, 3, figsize=(11, 4), dpi=200)
    ax = axes2[0]
    ax.bar(["Pre-freeze", "During freeze", "Post-freeze"],
           [pre_freeze_carbon, during_freeze_carbon, post_freeze_carbon],
           color=["#31a354", "#d94801", "#31a354"])
    ax.set_title("Agent 2 Carbon Mass")
    _annotate_bars(ax, [pre_freeze_carbon, during_freeze_carbon, post_freeze_carbon])

    ax = axes2[1]
    ax.bar(["Pre-freeze", "Post-freeze"], [pre_intensity, post_intensity],
           color=["#31a354", "#2b8cbe"])
    ax.set_title("Storage Intensity (Preserved?)")
    _annotate_bars(ax, [pre_intensity, post_intensity])

    ax = axes2[2]
    ax.bar(["Carbon Mass\nDrift During\nFreeze"], [freeze_std if len(freeze_vals) >= 2 else 0],
           color=["green" if freeze_std < 1e-4 else "red"])
    ax.set_title("Freeze Stability")
    _annotate_bars(ax, [freeze_std if len(freeze_vals) >= 2 else 0], ".6f")

    fig2.suptitle("Freeze-Thaw 鈥?Aggregate Summary", fontsize=12, y=1.02)
    fig2.tight_layout()
    fig2.savefig(output_dir / "figures" / "freeze_thaw_summary.png", bbox_inches="tight")
    plt.close(fig2)

    summary = {
        "num_seeds": num_seeds,
        "pre_freeze_carbon_mass": pre_freeze_carbon,
        "during_freeze_carbon_mass": during_freeze_carbon,
        "post_freeze_carbon_mass": post_freeze_carbon,
        "pre_freeze_storage_intensity": pre_intensity,
        "post_freeze_storage_intensity": post_intensity,
        "freeze_std": freeze_std,
        "freeze_preserved": freeze_std < 1e-4,
    }
    (output_dir / "freeze_thaw_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _annotate_phases(ax, charge: int, inactive: int, discharge_start: int):
    """Add phase labels."""
    y0, y1 = ax.get_ylim()
    mid = (y0 + y1) / 2
    ax.text(charge / 2, y1 * 0.9, "charge", ha="center", fontsize=9, color="#1f78b4")
    ax.text((charge + inactive) / 2, y1 * 0.9, "freeze", ha="center", fontsize=9, color="#d94801")
    ax.text((inactive + 24) / 2, y1 * 0.9, "discharge", ha="center", fontsize=9, color="#b30000")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# EXPERIMENT 3: Carbon Imbalance Decomposition (storage pool conservation)
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

def _imbalance_profiles(env: DPLCRLPaperEnv, charge_steps: int = 8):
    """Profiles for scripted charge-then-discharge, same as original."""
    n = env.max_agents
    env.pv_profiles = np.zeros((n, env.horizon), dtype=np.float32)
    env.load_profiles = np.zeros((n, env.horizon), dtype=np.float32)
    env.pv_profiles[:, :charge_steps] = 0.0
    env.load_profiles[:, :charge_steps] = 0.6
    env.pv_profiles[:, charge_steps:] = 4.0
    env.load_profiles[:, charge_steps:] = 6.5
    for state in env.states:
        cap = float(env.agent_energy_caps[state.id])
        state.energy = 0.08 * cap
        state.carbon_mass = state.energy * float(env.market_spec.grid_carbon_factor)
        state.storage_intensity = float(env.market_spec.grid_carbon_factor)
    env._update_profiles_for_step(0)


def _imbalance_scripted_actions(env: DPLCRLPaperEnv, step: int, charge_steps: int = 8) -> list[np.ndarray]:
    ess_signal = -1.0 if step < charge_steps else 1.0
    return [
        np.array([0.0, 0.0, ess_signal], dtype=np.float32)
        if env.agent_mask[agent_idx] > 0.5
        else np.zeros(3, dtype=np.float32)
        for agent_idx in range(env.max_agents)
    ]


def _run_imbalance(seed: int, num_agents: int = 10, charge_steps: int = 8) -> dict:
    """Run scripted charge-discharge, decomposing storage carbon pool."""
    env = DPLCRLPaperEnv(
        max_agents=num_agents, min_agents=num_agents, horizon=24, seed=seed,
        cmtm_mode="full",
        storage_capacity_variance=0.0, storage_capacity_range=(12.0, 12.0),
        step_churn_prob=0.0,
    )
    env.reset()
    env.agent_mask[:] = 1.0
    env.active_ids = list(range(num_agents))
    _imbalance_profiles(env, charge_steps)

    initial_storage_carbon = float(sum(state.carbon_mass for state in env.states))
    total_charge_injection = 0.0
    total_discharge_removal = 0.0
    trace_rows = []

    for s in range(24):
        obs, rewards, done, info = env.step(_imbalance_scripted_actions(env, s, charge_steps))
        per_agent = info["per_agent"]
        step_charge_in = sum(float(p["carbon_charge_responsibility"]) for p in per_agent if p.get("active", False))
        step_discharge_out = sum(float(p["carbon_storage_discharge"]) for p in per_agent if p.get("active", False))
        total_charge_injection += step_charge_in
        total_discharge_removal += step_discharge_out
        trace_rows.append({
            "step": s,
            "charge_carbon_in": step_charge_in,
            "discharge_carbon_out": step_discharge_out,
            "storage_carbon_mass": float(sum(state.carbon_mass for state in env.states)),
            "storage_energy": float(sum(state.energy for state in env.states)),
        })

    final_storage_carbon = float(sum(state.carbon_mass for state in env.states))
    env.close()

    total_input = initial_storage_carbon + total_charge_injection
    total_output = final_storage_carbon + total_discharge_removal
    residual = total_input - total_output
    residual_pct = residual / max(abs(total_input), EPS) * 100
    recovery_ratio = total_discharge_removal / max(total_input, EPS)

    summary = {
        "seed": seed,
        "num_agents": num_agents,
        "initial_storage_carbon": initial_storage_carbon,
        "total_charge_injection": total_charge_injection,
        "total_discharge_removal": total_discharge_removal,
        "final_storage_carbon": final_storage_carbon,
        "total_input": total_input,
        "total_accounted_output": total_output,
        "residual": residual,
        "residual_pct": residual_pct,
        "recovery_ratio": recovery_ratio,
    }
    return summary, trace_rows


def experiment_imbalance(output_dir: Path, num_seeds: int = 20) -> dict:
    fixed_seed = 20260427
    summaries = []
    all_traces = []
    for idx in range(num_seeds):
        s, trace = _run_imbalance(fixed_seed + idx, num_agents=10)
        summaries.append(s)
        all_traces.extend(trace)

    _write_csv(output_dir / "imbalance_trace.csv", all_traces)
    _write_csv(output_dir / "imbalance_summary.csv", summaries)

    # Aggregate
    keys = ["initial_storage_carbon", "total_charge_injection",
            "total_discharge_removal", "final_storage_carbon",
            "total_input", "total_accounted_output",
            "residual", "residual_pct", "recovery_ratio"]
    agg = {k: {"mean": 0.0, "std": 0.0} for k in keys}
    for k in keys:
        vals = [s[k] for s in summaries]
        agg[k]["mean"], agg[k]["std"] = _mean_std(vals)

    # Per-step traces
    steps = sorted(set(r["step"] for r in all_traces))
    xs = np.array(steps)
    charge_mask = xs < 8
    discharge_mask = xs >= 8

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=200)

    ax = axes[0]
    _span_bg(ax, xs, charge_mask, discharge_mask)
    step_means = {}
    for s in steps:
        rows = [r for r in all_traces if r["step"] == s]
        step_means[s] = {k: float(np.mean([r[k] for r in rows])) for k in
                         ["charge_carbon_in", "discharge_carbon_out", "storage_carbon_mass"]}
    cum_charge = np.cumsum([step_means[s]["charge_carbon_in"] for s in steps])
    cum_discharge = np.cumsum([step_means[s]["discharge_carbon_out"] for s in steps])
    ax.fill_between(xs, 0, [step_means[s]["storage_carbon_mass"] for s in steps],
                    alpha=0.3, color="#31a354", label="Storage carbon mass")
    ax.plot(xs, cum_charge, "s-", color="#2b8cbe", ms=4, label="Cumulative charge injection")
    ax.plot(xs, cum_discharge, "o-", color="#d94801", ms=4, label="Cumulative discharge removal")
    ax.set_ylabel("Carbon (kgCO鈧?")
    ax.set_title("Storage Carbon Pool Evolution")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    _span_bg(ax, xs, charge_mask, discharge_mask)
    ax.bar(xs, [step_means[s]["charge_carbon_in"] for s in steps], width=0.6,
           alpha=0.7, color="#2b8cbe", label="Charge injection")
    ax.bar(xs, [-step_means[s]["discharge_carbon_out"] for s in steps], width=0.6,
           alpha=0.7, color="#d94801", label="Discharge removal")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("Carbon Flow (kgCO鈧?step)")
    ax.set_title("Per-Step Storage Carbon Flows")
    ax.legend(frameon=False)

    ax = axes[2]
    _span_bg(ax, xs, charge_mask, discharge_mask)
    ax.plot(xs, [step_means[s]["storage_carbon_mass"] for s in steps], "o-",
            color="#31a354", ms=4, label="Storage carbon mass")
    ax.set_ylabel("Carbon Mass (kgCO鈧?")
    ax.set_title("Storage Carbon Mass Over Time")
    ax.legend(frameon=False)

    fig.suptitle("Experiment 3: Storage Carbon Pool Conservation", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "imbalance.png", bbox_inches="tight")
    plt.close(fig)

    # Summary bar chart
    fig2, axes2 = plt.subplots(1, 4, figsize=(14, 4.2), dpi=200)

    ax = axes2[0]
    recovery_mean = agg["recovery_ratio"]["mean"]
    recovery_std = agg["recovery_ratio"]["std"]
    ax.bar(["Delayed Carbon\nRecovery Ratio"], [recovery_mean], yerr=[recovery_std],
           color=RCP["cmtm"], capsize=5)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_ylim(0, 1.2)
    ax.set_title("Recovery Ratio\n(discharge / input)")

    ax = axes2[1]
    residual_mean = agg["residual"]["mean"]
    residual_std = agg["residual"]["std"]
    ax.bar(["Carbon Pool\nResidual"], [residual_mean], yerr=[residual_std],
           color="#d94801", capsize=5)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("Storage Pool Imbalance\n(input 鈭?output)")

    ax = axes2[2]
    input_mean = agg["total_input"]["mean"]
    out_mean = agg["total_accounted_output"]["mean"]
    ax.bar(["Total Input\n(initial + charge)", "Total Output\n(final + discharge)"],
           [input_mean, out_mean], color=["#2b8cbe", "#31a354"])
    _annotate_bars(ax, [input_mean, out_mean])
    ax.set_title("Carbon Conservation\n(Input vs Output)")

    ax = axes2[3]
    components = {
        "Discharged": agg["total_discharge_removal"]["mean"],
        "Final storage": agg["final_storage_carbon"]["mean"],
        "Residual": abs(agg["residual"]["mean"]),
    }
    colors_vals = [components.get("Discharged", 0), components.get("Final storage", 0), components.get("Residual", 0)]
    wedges, texts, autotexts = ax.pie(
        [components["Discharged"], components["Final storage"], components["Residual"]],
        labels=["Discharged", "Final\nstorage", "Residual"],
        autopct="%1.1f%%", colors=[RCP["cmtm"], "#31a354", "#d94801"],
        startangle=90, textprops={"fontsize": 8}
    )
    ax.set_title("Storage Carbon Output\nDecomposition")

    fig2.suptitle("Carbon Imbalance 鈥?Aggregate Summary", fontsize=12, y=1.02)
    fig2.tight_layout()
    fig2.savefig(output_dir / "figures" / "imbalance_summary.png", bbox_inches="tight")
    plt.close(fig2)

    result = {"num_seeds": num_seeds}
    for k in keys:
        result[f"{k}_mean"] = agg[k]["mean"]
        result[f"{k}_std"] = agg[k]["std"]
    (output_dir / "imbalance_summary.json").write_text(json.dumps(result, indent=2))
    return result


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# MAIN
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

def _parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    parser.description = "Extended CMTM validation experiments"
    parser.add_argument("--output_dir", default="reports/cmtm_extended")
    parser.add_argument("--experiments", nargs="+",
                        default=["p2p_chain", "freeze_thaw", "imbalance"],
                        choices=["p2p_chain", "freeze_thaw", "imbalance"])
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--imbalance_seeds", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    results = {}

    if "p2p_chain" in args.experiments:
        print("[Experiment 1] P2P Carbon Chain...")
        results["p2p_chain"] = experiment_p2p_chain(output_dir, num_seeds=args.num_seeds)

    if "freeze_thaw" in args.experiments:
        print("[Experiment 2] Dynamic Freeze-Thaw...")
        results["freeze_thaw"] = experiment_freeze_thaw(output_dir, num_seeds=args.num_seeds)

    if "imbalance" in args.experiments:
        print("[Experiment 3] Carbon Imbalance Decomposition...")
        results["imbalance"] = experiment_imbalance(output_dir, num_seeds=args.imbalance_seeds)

    (output_dir / "all_results.json").write_text(json.dumps(results, indent=2))
    print(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()

