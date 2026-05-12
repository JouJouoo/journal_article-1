#!/usr/bin/env python3
"""Render a 30-agent P2P carbon-responsibility flow heatmap for one episode."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
)


EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    parser.description = "Evaluate one 30-agent episode and plot the P2P carbon-responsibility flow matrix."
    parser.add_argument("--manifest", default=None, help="Manifest containing the trained full-CMTM runs.")
    parser.add_argument("--run_dir", default=None, help="Direct path to one trained run directory.")
    parser.add_argument("--run_seed", type=int, default=0, help="Training seed to select from the manifest.")
    parser.add_argument("--checkpoint_episode", type=int, default=None, help="Checkpoint episode to evaluate.")
    parser.add_argument("--fixed_eval_seed", type=int, default=20260508, help="Episode seed for this heatmap.")
    parser.add_argument(
        "--stochastic_policy",
        action="store_true",
        help="Sample policy actions for one reproducible episode instead of using deterministic action means.",
    )
    parser.add_argument(
        "--flow_metric",
        choices=["actual", "signed_relative", "low_carbon_contribution"],
        default="signed_relative",
        help=(
            "Heatmap value. actual = energy * seller carbon intensity; "
            "signed_relative = energy * (seller intensity - grid baseline), negative for low-carbon P2P; "
            "low_carbon_contribution = energy * (grid baseline - seller intensity), positive for low-carbon P2P."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="reports/p2p_carbon_flow_heatmap",
        help="Output directory for the heatmap, matrix CSV, and edge table.",
    )
    parser.add_argument("--report_name", default="p2p_carbon_responsibility_flow")
    args = parser.parse_args()
    if not args.manifest and not args.run_dir:
        parser.error("Either --manifest or --run_dir must be provided.")
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


def _load_run_dir(manifest_path: Path, run_seed: int) -> Path:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in payload.get("runs", []):
        if item.get("method") != "full" or item.get("status") != "completed":
            continue
        if int(item.get("seed", -1)) == int(run_seed):
            return Path(str(item["run_dir"])).expanduser().resolve()
    raise FileNotFoundError(f"No completed full-CMTM run with seed={run_seed} in {manifest_path}")


def _select_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(str(args.run_dir)).expanduser().resolve()
    return _load_run_dir(Path(args.manifest).expanduser().resolve(), int(args.run_seed))


def _write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seller_id"] + [f"buyer_{idx}" for idx in range(matrix.shape[1])])
        for seller_idx in range(matrix.shape[0]):
            writer.writerow([seller_idx] + [float(value) for value in matrix[seller_idx]])


def _write_edge_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "step",
        "seller_id",
        "buyer_id",
        "p2p_energy",
        "price",
        "carbon_intensity",
        "actual_carbon_responsibility",
        "signed_carbon_responsibility",
        "low_carbon_contribution",
        "plotted_carbon_flow",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, 0.0) for key in fields})


def _run_episode(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    device = torch.device("cpu")
    _set_global_seeds(int(args.fixed_eval_seed))
    envs, policy = _build_policy(args, device)
    carbon_matrix = np.zeros((int(args.num_agents), int(args.num_agents)), dtype=np.float64)
    energy_matrix = np.zeros_like(carbon_matrix)
    actual_carbon_matrix = np.zeros_like(carbon_matrix)
    low_carbon_matrix = np.zeros_like(carbon_matrix)
    grid_carbon_factor = float(args.grid_carbon_factor)
    edge_rows: list[dict[str, Any]] = []
    reward_sum = 0.0
    p2p_volume_sum = 0.0
    grid_buy_sum = 0.0
    grid_sell_sum = 0.0
    load_carbon_sum = 0.0
    steps_run = 0

    try:
        policy.restore(str(checkpoint_path))
        policy.eval()
        obs = envs.reset()
        rnn_states = np.zeros(
            (args.n_eval_rollout_threads, args.num_agents, args.recurrent_N, args.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic = np.zeros_like(rnn_states)
        masks = np.ones((args.n_eval_rollout_threads, args.num_agents, 1), dtype=np.float32)
        agent_masks = np.asarray(envs.agent_masks, dtype=np.float32)[..., None]

        for step in range(int(args.episode_length)):
            obs_batch = np.concatenate(obs, axis=0)
            if args.use_centralized_V:
                share_batch = np.repeat(
                    np.asarray(envs.share_obs, dtype=np.float32)[:, None, :],
                    int(args.num_agents),
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
                deterministic=not bool(args.stochastic_policy),
            )
            action_array = np.array(np.split(_t2n(actions), int(args.n_eval_rollout_threads)))
            obs, _, dones, infos = envs.step(action_array)
            info = infos[0]

            for record in info.get("settlement_records", []):
                if record.get("record_type") != "p2p":
                    continue
                seller_id = int(record.get("seller_id", -1))
                buyer_id = int(record.get("buyer_id", -1))
                quantity = float(record.get("quantity", 0.0) or 0.0)
                intensity = float(record.get("carbon_intensity", 0.0) or 0.0)
                actual_responsibility = float(record.get("carbon_responsibility", quantity * intensity) or 0.0)
                signed_responsibility = float(quantity * (intensity - grid_carbon_factor))
                low_carbon_contribution = -signed_responsibility
                if args.flow_metric == "actual":
                    plotted_flow = actual_responsibility
                elif args.flow_metric == "low_carbon_contribution":
                    plotted_flow = low_carbon_contribution
                else:
                    plotted_flow = signed_responsibility
                if 0 <= seller_id < int(args.num_agents) and 0 <= buyer_id < int(args.num_agents):
                    energy_matrix[seller_id, buyer_id] += quantity
                    actual_carbon_matrix[seller_id, buyer_id] += actual_responsibility
                    low_carbon_matrix[seller_id, buyer_id] += low_carbon_contribution
                    carbon_matrix[seller_id, buyer_id] += plotted_flow
                edge_rows.append(
                    {
                        "step": int(step),
                        "seller_id": seller_id,
                        "buyer_id": buyer_id,
                        "p2p_energy": quantity,
                        "price": float(record.get("price", 0.0) or 0.0),
                        "carbon_intensity": intensity,
                        "actual_carbon_responsibility": actual_responsibility,
                        "signed_carbon_responsibility": signed_responsibility,
                        "low_carbon_contribution": low_carbon_contribution,
                        "plotted_carbon_flow": plotted_flow,
                    }
                )

            market = info.get("market_summary", {})
            trace = info.get("carbon_trace", {})
            reward_sum += float(info.get("global_reward", 0.0) or 0.0)
            p2p_volume_sum += float(market.get("p2p_total_volume", 0.0) or 0.0)
            grid_buy_sum += float(market.get("grid_buy_total", 0.0) or 0.0)
            grid_sell_sum += float(market.get("grid_sell_total", 0.0) or 0.0)
            load_carbon_sum += float(trace.get("load_responsibility", 0.0) or 0.0)
            steps_run += 1

            masks = (1.0 - np.asarray(dones, dtype=np.float32)).reshape(
                int(args.n_eval_rollout_threads),
                int(args.num_agents),
                1,
            )
            agent_masks = np.asarray(
                [
                    np.asarray(info_row.get("agent_mask", [1.0] * int(args.num_agents)), dtype=np.float32)
                    for info_row in infos
                ],
                dtype=np.float32,
            )[..., None]
            if bool(np.all(dones)):
                break
    finally:
        envs.close()

    nonzero_edges = int(np.count_nonzero(np.abs(carbon_matrix) > EPS))
    top_edges = []
    for seller_idx, buyer_idx in np.argwhere(np.abs(carbon_matrix) > EPS):
        top_edges.append(
            {
                "seller_id": int(seller_idx),
                "buyer_id": int(buyer_idx),
                "carbon_flow": float(carbon_matrix[seller_idx, buyer_idx]),
                "actual_carbon_responsibility": float(actual_carbon_matrix[seller_idx, buyer_idx]),
                "low_carbon_contribution": float(low_carbon_matrix[seller_idx, buyer_idx]),
                "p2p_energy": float(energy_matrix[seller_idx, buyer_idx]),
            }
        )
    top_edges.sort(key=lambda item: abs(float(item["carbon_flow"])), reverse=True)
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "fixed_eval_seed": int(args.fixed_eval_seed),
        "stochastic_policy": bool(args.stochastic_policy),
        "flow_metric": str(args.flow_metric),
        "grid_carbon_factor": grid_carbon_factor,
        "episode_steps": int(steps_run),
        "total_plotted_p2p_carbon_flow": float(np.sum(carbon_matrix)),
        "total_abs_plotted_p2p_carbon_flow": float(np.sum(np.abs(carbon_matrix))),
        "total_actual_p2p_carbon_responsibility": float(np.sum(actual_carbon_matrix)),
        "total_p2p_low_carbon_contribution": float(np.sum(low_carbon_matrix)),
        "total_p2p_energy": float(np.sum(energy_matrix)),
        "nonzero_p2p_edges": nonzero_edges,
        "reward_mean_step": float(reward_sum / max(1, steps_run)),
        "p2p_volume_total": float(p2p_volume_sum),
        "grid_buy_total": float(grid_buy_sum),
        "grid_sell_total": float(grid_sell_sum),
        "load_carbon_total": float(load_carbon_sum),
        "top_edges": top_edges[:10],
    }
    return carbon_matrix, energy_matrix, edge_rows, summary


def _plot_heatmap(matrix: np.ndarray, output_path: Path, summary: dict[str, Any]) -> None:
    num_agents = matrix.shape[0]
    metric = str(summary.get("flow_metric", "signed_relative"))
    has_flow = bool(np.max(np.abs(matrix)) > EPS)
    if metric == "signed_relative":
        cmap = plt.get_cmap("coolwarm").copy()
        limit = max(float(np.max(np.abs(matrix))), 1.0)
        plot_data = matrix
        vmin, vmax = -limit, limit
        cbar_label = "Signed P2P carbon responsibility vs grid (kgCO2)"
        title = "Signed P2P Carbon-Responsibility Flow in One 30-Agent Episode"
        note = (
            "Cell (i, j) = P2P energy from seller i to buyer j times "
            "(seller carbon intensity - grid baseline). Negative values indicate low-carbon responsibility reduction."
        )
    elif metric == "low_carbon_contribution":
        cmap = plt.get_cmap("YlGn").copy()
        cmap.set_bad("#f7f7f7")
        plot_data = np.ma.masked_where(matrix <= EPS, matrix)
        vmin, vmax = None, None
        cbar_label = "P2P low-carbon contribution transferred (kgCO2)"
        title = "P2P Low-Carbon Contribution Flow in One 30-Agent Episode"
        note = (
            "Cell (i, j) = P2P energy from seller i to buyer j times "
            "(grid baseline - seller carbon intensity). White cells indicate zero contribution."
        )
    else:
        title = "Actual P2P Carbon-Responsibility Flow in One 30-Agent Episode"
        if float(np.min(matrix)) < -EPS:
            cmap = plt.get_cmap("coolwarm").copy()
            limit = max(float(np.max(np.abs(matrix))), 1.0)
            plot_data = matrix
            vmin, vmax = -limit, limit
            cbar_label = "Actual P2P carbon responsibility transferred (kgCO2)"
            note = (
                "Cell (i, j) = P2P energy from seller i to buyer j times seller carbon intensity. "
                "Negative values indicate carbon-offset responsibility carried by low-carbon/PV-origin energy."
            )
        else:
            cmap = plt.get_cmap("YlOrRd").copy()
            cmap.set_bad("#f7f7f7")
            plot_data = np.ma.masked_where(matrix <= EPS, matrix)
            vmin, vmax = None, None
            cbar_label = "Actual P2P carbon responsibility transferred (kgCO2)"
            note = (
                "Cell (i, j) = P2P energy from seller i to buyer j times seller carbon intensity. "
                "White cells indicate zero actual carbon transfer."
            )

    fig, ax = plt.subplots(figsize=(9.2, 8.4), dpi=240)
    if has_flow:
        image = ax.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
        cbar.set_label(cbar_label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    else:
        image = ax.imshow(np.zeros_like(matrix), cmap="Greys", vmin=0.0, vmax=1.0, aspect="equal")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
        cbar.set_label(f"No {metric} flow observed", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    ticks = np.arange(num_agents)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(idx) for idx in ticks], fontsize=6)
    ax.set_yticklabels([str(idx) for idx in ticks], fontsize=6)
    ax.set_xlabel("Buyer / receiving agent", fontsize=10)
    ax.set_ylabel("Seller / sending agent", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="semibold")
    ax.set_xticks(np.arange(-0.5, num_agents, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_agents, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)

    top_edges = summary.get("top_edges", [])[:8]
    for item in top_edges:
        seller_idx = int(item["seller_id"])
        buyer_idx = int(item["buyer_id"])
        value = float(item["carbon_flow"])
        if abs(value) <= EPS:
            continue
        ax.text(
            buyer_idx,
            seller_idx,
            f"{value:+.1f}" if metric in {"signed_relative", "actual"} and value < 0.0 else f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=5.5,
            color="#111111",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
        )
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _configure_low_resource_runtime()
    output_dir = _output_dir(args)
    run_dir = _select_run_dir(args)
    checkpoint_path = _resolve_checkpoint_path(run_dir, args.checkpoint_episode)
    carbon_matrix, energy_matrix, edge_rows, summary = _run_episode(args, run_dir, checkpoint_path)

    prefix = str(args.report_name)
    _write_matrix_csv(output_dir / f"{prefix}_matrix.csv", carbon_matrix)
    _write_matrix_csv(output_dir / f"{prefix}_energy_matrix.csv", energy_matrix)
    _write_edge_csv(output_dir / f"{prefix}_edges.csv", edge_rows)
    (output_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure_path = output_dir / "figures" / f"{prefix}_heatmap.png"
    _plot_heatmap(carbon_matrix, figure_path, summary)

    print(f"saved_dir={output_dir}")
    print(f"figure={figure_path}")
    print(f"flow_metric={summary['flow_metric']}")
    print(f"total_plotted_p2p_carbon_flow={summary['total_plotted_p2p_carbon_flow']:.6f}")
    print(f"total_actual_p2p_carbon_responsibility={summary['total_actual_p2p_carbon_responsibility']:.6f}")
    print(f"total_p2p_low_carbon_contribution={summary['total_p2p_low_carbon_contribution']:.6f}")
    print(f"nonzero_p2p_edges={summary['nonzero_p2p_edges']}")


if __name__ == "__main__":
    main()
