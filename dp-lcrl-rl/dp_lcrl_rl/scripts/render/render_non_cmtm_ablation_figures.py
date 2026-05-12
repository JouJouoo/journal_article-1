"""Render ablation figures without the CMTM-stateless variant."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["full", "mask_obs_only", "direct_max"]
METHOD_LABELS = {
    "full": "Full",
    "mask_obs_only": "Mask ObsOnly",
    "direct_max": "Direct Max Scale",
}
COLORS = {
    "full": "#111111",
    "mask_obs_only": "#2171b5",
    "direct_max": "#238b45",
}

TRAIN_METRICS = {
    "reward": ("Reward", "Reward"),
    "p2p": ("P2P Trading Volume", "Mean volume"),
    "grid_trade": ("Grid Trading Volume", "Mean volume"),
    "carbon_resp": ("Carbon Responsibility", "Mean carbon responsibility"),
}

EVAL_METRICS = {
    "reward_mean": ("Reward", True),
    "p2p_mean": ("P2P Trading Volume", True),
    "grid_trade_mean": ("Grid Trading Volume", False),
    "carbon_mean": ("Carbon Responsibility", False),
}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _method_rows(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("method_key") == method or row.get("method") == method]


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _plot_training_metric(rows: list[dict[str, Any]], metric: str, output_path: Path) -> None:
    title, ylabel = TRAIN_METRICS[metric]
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=220)
    for method in METHOD_ORDER:
        bucket = sorted(_method_rows(rows, method), key=lambda item: int(float(item["iteration"])))
        xs = np.asarray([_float(row, "iteration") for row in bucket], dtype=np.float64)
        mean = np.asarray([_float(row, f"{metric}_mean") for row in bucket], dtype=np.float64)
        std = np.asarray([_float(row, f"{metric}_std") for row in bucket], dtype=np.float64)
        ax.plot(xs, mean, linewidth=2.2, label=METHOD_LABELS[method], color=COLORS[method])
        ax.fill_between(xs, mean - std, mean + std, color=COLORS[method], alpha=0.14, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_training_combined(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), dpi=220)
    for ax, metric in zip(axes.flat, TRAIN_METRICS):
        title, ylabel = TRAIN_METRICS[metric]
        for method in METHOD_ORDER:
            bucket = sorted(_method_rows(rows, method), key=lambda item: int(float(item["iteration"])))
            xs = np.asarray([_float(row, "iteration") for row in bucket], dtype=np.float64)
            mean = np.asarray([_float(row, f"{metric}_mean") for row in bucket], dtype=np.float64)
            std = np.asarray([_float(row, f"{metric}_std") for row in bucket], dtype=np.float64)
            ax.plot(xs, mean, linewidth=2.0, label=METHOD_LABELS[method], color=COLORS[method])
            ax.fill_between(xs, mean - std, mean + std, color=COLORS[method], alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Core-Module Ablation Training Trends (3 seeds, mean +/- std)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_eval_by_count(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), dpi=220)
    for ax, (metric, (title, higher_is_better)) in zip(axes.flat, EVAL_METRICS.items()):
        for method in METHOD_ORDER:
            bucket = sorted(_method_rows(rows, method), key=lambda item: int(float(item["active_agent_count"])))
            xs = np.asarray([_float(row, "active_agent_count") for row in bucket], dtype=np.float64)
            mean = np.asarray([_float(row, metric) for row in bucket], dtype=np.float64)
            std = np.asarray([_float(row, metric.replace("_mean", "_std")) for row in bucket], dtype=np.float64)
            ax.plot(xs, mean, marker="o", markersize=3.2, linewidth=2.0, color=COLORS[method], label=METHOD_LABELS[method])
            ax.fill_between(xs, mean - std, mean + std, color=COLORS[method], alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Active agent count")
        ax.set_ylabel("Mean value")
        ax.grid(True, alpha=0.25)
        if not higher_is_better:
            ax.text(0.02, 0.94, "lower is better", transform=ax.transAxes, fontsize=8, color="#555555")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Unified Core-Module Ablation Evaluation", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_dynamic_by_scenario(rows: list[dict[str, Any]], output_path: Path) -> None:
    scenario_order = ["episode_var20", "churn20_p10", "churn20_p20", "churn20_p30"]
    scenarios = [item for item in scenario_order if any(row.get("scenario") == item for row in rows)]
    if not scenarios:
        scenarios = sorted({str(row["scenario"]) for row in rows})
    x_lookup = {scenario: idx for idx, scenario in enumerate(scenarios)}

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), dpi=220)
    for ax, (metric, (title, higher_is_better)) in zip(axes.flat, EVAL_METRICS.items()):
        for method in METHOD_ORDER:
            bucket = [row for row in _method_rows(rows, method) if str(row["scenario"]) in x_lookup]
            bucket = sorted(bucket, key=lambda item: x_lookup[str(item["scenario"])])
            xs = np.asarray([x_lookup[str(row["scenario"])] for row in bucket], dtype=np.float64)
            mean = np.asarray([_float(row, metric) for row in bucket], dtype=np.float64)
            std = np.asarray([_float(row, metric.replace("_mean", "_std")) for row in bucket], dtype=np.float64)
            ax.plot(xs, mean, marker="o", linewidth=2.0, color=COLORS[method], label=METHOD_LABELS[method])
            ax.fill_between(xs, mean - std, mean + std, color=COLORS[method], alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xticks(range(len(scenarios)), scenarios, rotation=15)
        ax.set_ylabel("Mean value")
        ax.grid(True, alpha=0.25)
        if not higher_is_better:
            ax.text(0.02, 0.94, "lower is better", transform=ax.transAxes, fontsize=8, color="#555555")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Dynamic Participation Stress Evaluation", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ablation figures without the CMTM-stateless variant.")
    parser.add_argument("--train_curve_csv", default="reports/formal_ablation_3seed_20260422/aggregated/formal_ablation_3seed_curve_points.csv")
    parser.add_argument("--unified_by_count_csv", default="reports/formal_ablation_unified_eval_20260425/formal_ablation_unified_eval_by_count.csv")
    parser.add_argument("--dynamic_by_scenario_csv", default="reports/formal_ablation_dynamic_eval_20260425/formal_ablation_dynamic_eval_by_scenario.csv")
    args = parser.parse_args()

    train_csv = Path(args.train_curve_csv).resolve()
    unified_csv = Path(args.unified_by_count_csv).resolve()
    dynamic_csv = Path(args.dynamic_by_scenario_csv).resolve()

    train_rows = _read_csv(train_csv)
    train_fig_dir = train_csv.parent / "figures"
    _plot_training_combined(train_rows, train_fig_dir / "formal_ablation_3seed_trends_combined_non_cmtm.png")
    for metric in TRAIN_METRICS:
        _plot_training_metric(train_rows, metric, train_fig_dir / f"formal_ablation_3seed_trend_{metric}_non_cmtm.png")

    unified_rows = _read_csv(unified_csv)
    _plot_eval_by_count(unified_rows, unified_csv.parent / "figures" / "formal_ablation_unified_eval_by_count_non_cmtm.png")

    dynamic_rows = _read_csv(dynamic_csv)
    _plot_dynamic_by_scenario(dynamic_rows, dynamic_csv.parent / "figures" / "formal_ablation_dynamic_eval_by_scenario_non_cmtm.png")

    print(f"training_figures={train_fig_dir}")
    print(f"unified_figure={unified_csv.parent / 'figures' / 'formal_ablation_unified_eval_by_count_non_cmtm.png'}")
    print(f"dynamic_figure={dynamic_csv.parent / 'figures' / 'formal_ablation_dynamic_eval_by_scenario_non_cmtm.png'}")


if __name__ == "__main__":
    main()
