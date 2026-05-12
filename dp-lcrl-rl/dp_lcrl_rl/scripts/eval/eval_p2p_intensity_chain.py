#!/usr/bin/env python3
"""Validate multi-hop P2P carbon-intensity propagation under full CMTM."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WANDB_DISABLED", "true")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dp_lcrl_rl.envs.p2ptrading.dp_lcrl_paper_env import DPLCRLPaperEnv


AGENT_NAMES = {
    0: "A low-C\nsource",
    1: "D high-C\nsource",
    2: "B relay\nnode",
    3: "C final\nbuyer",
}
EDGE_ORDER = ["A->B", "D->B", "B->C"]
EDGE_META = {
    "A->B": {"seller": 0, "buyer": 2, "color": "#2ca25f", "legend": "A -> B (low-C)"},
    "D->B": {"seller": 1, "buyer": 2, "color": "#e6550d", "legend": "D -> B (high-C)"},
    "B->C": {"seller": 2, "buyer": 3, "color": "#3182bd", "legend": "B -> C (relay)"},
}


def _phase(step: int) -> str:
    if step < 4:
        return "upstream"
    if step < 6:
        return "hold"
    if step < 10:
        return "downstream"
    return "idle"


def _configure_profiles(env: DPLCRLPaperEnv) -> None:
    env.pv_profiles = np.zeros((env.max_agents, env.horizon), dtype=np.float32)
    env.load_profiles = np.zeros((env.max_agents, env.horizon), dtype=np.float32)

    # Phase 1: A and D sell to B, and B stores the mixed-carbon input.
    env.pv_profiles[0, :4] = 1.7
    env.load_profiles[0, :4] = 0.1
    env.load_profiles[1, :4] = 0.1
    env.load_profiles[2, :4] = 0.1
    env.load_profiles[3, :4] = 0.1

    # Phase 2: no trade. The relay node should preserve its stored carbon intensity.
    env.load_profiles[:, 4:6] = 0.1

    # Phase 3: B discharges the mixed-carbon storage and sells to C.
    env.load_profiles[0, 6:10] = 0.1
    env.load_profiles[1, 6:10] = 0.1
    env.load_profiles[2, 6:10] = 0.1
    env.load_profiles[3, 6:10] = 3.0

    env.load_profiles[:, 10:] = 0.1

    for state in env.states:
        cap = float(env.agent_energy_caps[state.id])
        state.energy = 0.05 * cap
        state.carbon_mass = state.energy * float(env.market_spec.grid_carbon_factor)
        state.storage_intensity = float(env.market_spec.grid_carbon_factor)
        state.dynamic_intensity = float(env.market_spec.grid_carbon_factor)
        state.sell_intensity = float(env.market_spec.grid_carbon_factor)

    # D is a high-carbon source with enough grid-like stored energy for upstream sales.
    env.states[1].energy = 0.90 * float(env.agent_energy_caps[1])
    env.states[1].carbon_mass = env.states[1].energy * float(env.market_spec.grid_carbon_factor)
    env.states[1].storage_intensity = float(env.market_spec.grid_carbon_factor)

    # B starts nearly empty so its later carbon intensity is mostly inherited from A/D.
    env.states[2].energy = 0.02 * float(env.agent_energy_caps[2])
    env.states[2].carbon_mass = 0.0
    env.states[2].storage_intensity = 0.0
    env._update_profiles_for_step(0)


def _actions(env: DPLCRLPaperEnv, step: int) -> list[np.ndarray]:
    actions = [np.zeros(3, dtype=np.float32) for _ in range(env.max_agents)]
    phase = _phase(step)
    if phase == "upstream":
        # A: low-carbon PV seller. D: high-carbon storage seller. B: relay buyer + charger.
        actions[0] = np.array([1.0, -0.80, 0.0], dtype=np.float32)
        actions[1] = np.array([1.0, -0.65, 0.55], dtype=np.float32)
        actions[2] = np.array([-1.0, 0.85, -1.0], dtype=np.float32)
    elif phase == "downstream":
        # B sells its stored mixed-carbon electricity to the final buyer C.
        actions[2] = np.array([1.0, -0.70, 1.0], dtype=np.float32)
        actions[3] = np.array([-1.0, 0.90, 0.0], dtype=np.float32)
    return actions


def _run_once(seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    env = DPLCRLPaperEnv(
        max_agents=4,
        min_agents=4,
        horizon=12,
        seed=seed,
        cmtm_mode="full",
        step_churn_prob=0.0,
        storage_capacity_variance=0.0,
        storage_capacity_range=(12.0, 12.0),
    )
    # This mechanism test treats PV as zero-carbon rather than negative-carbon
    # credit. Otherwise the low-carbon source can offset the high-carbon source
    # and obscure the carbon-intensity propagation being tested.
    env.market_spec.pv_carbon_factor = 0.0
    env.reset()
    env.agent_mask[:] = 1.0
    env.active_ids = list(range(4))
    _configure_profiles(env)

    edge_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    done = False
    step = 0
    while not done:
        _, _, done, info = env.step(_actions(env, step))
        phase = _phase(step)
        per_agent = {int(row["agent_id"]): row for row in info.get("per_agent", [])}
        records = info.get("settlement_records", [])
        for edge_name, meta in EDGE_META.items():
            record = next(
                (
                    row
                    for row in records
                    if int(row.get("seller_id", -1)) == meta["seller"]
                    and int(row.get("buyer_id", -1)) == meta["buyer"]
                ),
                None,
            )
            edge_rows.append(
                {
                    "seed": int(seed),
                    "step": int(step),
                    "phase": phase,
                    "edge": edge_name,
                    "seller": int(meta["seller"]),
                    "buyer": int(meta["buyer"]),
                    "quantity": float(record.get("quantity", 0.0)) if record else 0.0,
                    "carbon_intensity": float(record.get("carbon_intensity", 0.0)) if record else 0.0,
                    "carbon_responsibility": float(record.get("carbon_responsibility", 0.0)) if record else 0.0,
                }
            )
        state_rows.append(
            {
                "seed": int(seed),
                "step": int(step),
                "phase": phase,
                "b_storage_energy": float(env.states[2].energy),
                "b_carbon_mass": float(env.states[2].carbon_mass),
                "b_storage_intensity": float(env.states[2].storage_intensity),
                "b_sell_intensity": float(per_agent[2].get("C_sell", 0.0)),
                "c_p2p_carbon_import": float(per_agent[3].get("carbon_p2p_import", 0.0)),
                "c_load_carbon": float(per_agent[3].get("carbon_load_responsibility", 0.0)),
            }
        )
        step += 1
    env.close()
    return edge_rows, state_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(edge_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for edge in EDGE_ORDER:
        rows = [r for r in edge_rows if r["edge"] == edge and float(r["quantity"]) > 1e-9]
        qty = float(sum(float(r["quantity"]) for r in rows))
        carbon = float(sum(float(r["carbon_responsibility"]) for r in rows))
        summary[f"{edge}_quantity"] = qty
        summary[f"{edge}_carbon"] = carbon
        summary[f"{edge}_weighted_intensity"] = carbon / qty if qty > 1e-9 else 0.0

    hold_rows = [r for r in state_rows if r["phase"] == "hold"]
    downstream_rows = [r for r in state_rows if r["phase"] == "downstream"]
    summary["b_storage_intensity_after_upstream"] = float(
        np.mean([float(r["b_storage_intensity"]) for r in hold_rows])
    )
    summary["b_storage_intensity_during_downstream"] = float(
        np.mean([float(r["b_storage_intensity"]) for r in downstream_rows])
    )
    summary["c_total_p2p_carbon_import"] = float(sum(float(r["c_p2p_carbon_import"]) for r in state_rows))
    summary["c_total_load_carbon"] = float(sum(float(r["c_load_carbon"]) for r in state_rows))
    summary["num_seeds"] = int(len({int(r["seed"]) for r in state_rows}))
    return summary


def _mean_by_step(rows: list[dict[str, Any]], key: str, *, edge: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    if edge is not None:
        rows = [r for r in rows if r.get("edge") == edge]
    steps = sorted({int(r["step"]) for r in rows})
    vals = []
    for step in steps:
        bucket = [float(r[key]) for r in rows if int(r["step"]) == step]
        vals.append(float(np.mean(bucket)) if bucket else 0.0)
    return np.asarray(steps, dtype=np.int32), np.asarray(vals, dtype=np.float64)


def _shade_phases(ax: plt.Axes) -> None:
    ax.axvspan(-0.5, 3.5, color="#e8f3ff", alpha=0.85, lw=0)
    ax.axvspan(3.5, 5.5, color="#f5f5f5", alpha=0.85, lw=0)
    ax.axvspan(5.5, 9.5, color="#fff0df", alpha=0.90, lw=0)


def _plot(output_dir: Path, edge_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), dpi=220)
    fig.suptitle("P2P Carbon-Intensity Propagation Chain Under Full CMTM", fontsize=14, y=0.99)

    ax = axes[0, 0]
    ax.set_title("(a) Controlled carbon-intensity flow")
    ax.axis("off")
    positions = {0: (0.10, 0.70), 1: (0.10, 0.25), 2: (0.50, 0.48), 3: (0.88, 0.48)}
    for agent_id, (x, y) in positions.items():
        ax.scatter([x], [y], s=1450, color="#ffffff", edgecolor="#333333", linewidth=1.3, zorder=3)
        ax.text(x, y, AGENT_NAMES[agent_id], ha="center", va="center", fontsize=9, zorder=4)
    for edge, meta in EDGE_META.items():
        s = meta["seller"]
        b = meta["buyer"]
        x0, y0 = positions[s]
        x1, y1 = positions[b]
        qty = float(summary[f"{edge}_quantity"])
        intensity = float(summary[f"{edge}_weighted_intensity"])
        width = 1.6 + 2.3 * qty / max(1e-9, max(float(summary[f'{e}_quantity']) for e in EDGE_ORDER))
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=width, color=meta["color"], alpha=0.85, shrinkA=22, shrinkB=22),
        )
        ax.text(
            (x0 + x1) / 2.0,
            (y0 + y1) / 2.0 + (0.08 if edge != "D->B" else -0.08),
            f"{edge}\nq={qty:.2f}, I={intensity:.3f}",
            color=meta["color"],
            ha="center",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    ax.set_title("(b) Carbon intensity on P2P edges")
    _shade_phases(ax)
    for edge in EDGE_ORDER:
        xs, ys = _mean_by_step(edge_rows, "carbon_intensity", edge=edge)
        qty_xs, qty = _mean_by_step(edge_rows, "quantity", edge=edge)
        mask = qty > 1e-9
        ax.plot(xs[mask], ys[mask], "o-", color=EDGE_META[edge]["color"], lw=2.0, ms=5, label=EDGE_META[edge]["legend"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Carbon intensity (kgCO2/kWh)")
    ax.set_ylim(-0.02, 0.78)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.set_title("(c) Relay node B stores and releases inherited carbon intensity")
    _shade_phases(ax)
    xs, b_intensity = _mean_by_step(state_rows, "b_storage_intensity")
    _, b_mass = _mean_by_step(state_rows, "b_carbon_mass")
    ax.plot(xs, b_intensity, "s-", color="#2ca25f", lw=2.1, ms=4, label="B storage carbon intensity")
    ax.set_xlabel("Step")
    ax.set_ylabel("Storage intensity (kgCO2/kWh)", color="#2ca25f")
    ax.tick_params(axis="y", labelcolor="#2ca25f")
    ax.set_ylim(-0.02, 0.78)
    ax2 = ax.twinx()
    ax2.plot(xs, b_mass, "o-", color="#111111", lw=1.9, ms=4, label="B carbon mass")
    ax2.set_ylabel("B storage carbon mass (kgCO2)", color="#111111")
    ax2.tick_params(axis="y", labelcolor="#111111")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="upper right")

    ax = axes[1, 1]
    ax.set_title("(d) Final buyer C receives propagated carbon")
    _shade_phases(ax)
    xs, c_import = _mean_by_step(state_rows, "c_p2p_carbon_import")
    _, c_load = _mean_by_step(state_rows, "c_load_carbon")
    bars = ax.bar(xs, c_import, width=0.65, color="#3182bd", alpha=0.82, label="C P2P carbon import")
    ax.plot(xs, np.cumsum(c_import), "o-", color="#08519c", lw=2.0, ms=4, label="C cumulative P2P carbon")
    ax.plot(xs, c_load, "s--", color="#444444", lw=1.5, ms=4, label="C load carbon")
    ax.set_xlabel("Step")
    ax.set_ylabel("Carbon amount (kgCO2)")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat[1:]:
        ax.text(1.5, ax.get_ylim()[1] * 0.92, "A/D -> B", ha="center", color="#2171b5", fontsize=8)
        ax.text(4.5, ax.get_ylim()[1] * 0.92, "hold", ha="center", color="#555555", fontsize=8)
        ax.text(7.5, ax.get_ylim()[1] * 0.92, "B -> C", ha="center", color="#b30000", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = output_dir / "figures" / "p2p_intensity_chain_full_cmtm.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P2P carbon-intensity chain validation under full CMTM.")
    parser.add_argument("--output_dir", default="reports/cmtm_validation_100k_full_20260507/p2p_intensity_chain")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--fixed_seed", type=int, default=20260507)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    edge_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for idx in range(int(args.num_seeds)):
        edge, state = _run_once(seed=int(args.fixed_seed) + idx)
        edge_rows.extend(edge)
        state_rows.extend(state)

    summary = _aggregate(edge_rows, state_rows)
    _write_csv(output_dir / "p2p_intensity_chain_edges.csv", edge_rows)
    _write_csv(output_dir / "p2p_intensity_chain_states.csv", state_rows)
    (output_dir / "p2p_intensity_chain_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure_path = _plot(output_dir, edge_rows, state_rows, summary)
    print(f"saved_dir={output_dir}")
    print(f"figure={figure_path}")


if __name__ == "__main__":
    main()
