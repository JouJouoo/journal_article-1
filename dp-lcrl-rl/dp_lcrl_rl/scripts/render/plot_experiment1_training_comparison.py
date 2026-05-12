#!/usr/bin/env python3
"""Plot grouped Experiment 1 training curves across methods and seeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("dp_lcrl", "mlp_pad", "mappo_shared", "deepsets")


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values.astype(np.float64)
    window = min(int(window), int(values.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values.astype(np.float64), (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _method_from_path(path: Path) -> str:
    lower = path.parent.name.lower()
    for method in METHODS:
        if f"_{method}_" in lower:
            return method
    return path.parent.name


def _aggregate_episode_metrics(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    buckets: dict[int, dict[str, list[float]]] = {}
    for item in payload.get("episode_summaries", []):
        if item.get("phase") != "train":
            continue
        idx = int(item.get("iteration_index", 0))
        bucket = buckets.setdefault(
            idx,
            {"reward": [], "p2p": [], "grid_trade": [], "carbon": []},
        )
        grid_buy = float(item.get("grid_buy_mean_active", item.get("grid_buy_total", 0.0)) or 0.0)
        grid_sell = float(item.get("grid_sell_mean_active", item.get("grid_sell_total", 0.0)) or 0.0)
        bucket["reward"].append(float(item.get("average_global_reward", 0.0) or 0.0))
        bucket["p2p"].append(float(item.get("p2p_volume_mean_active", item.get("p2p_total_volume", 0.0)) or 0.0))
        bucket["grid_trade"].append(grid_buy + grid_sell)
        bucket["carbon"].append(
            float(
                item.get(
                    "carbon_responsibility_mean_active_episode",
                    item.get("load_responsibility_total", 0.0),
                )
                or 0.0
            )
        )
    iterations = np.asarray(sorted(buckets.keys()), dtype=np.int32)
    if iterations.size == 0:
        raise ValueError(f"No training episode summaries found: {path}")
    metrics = {"iteration": iterations}
    for key in ("reward", "p2p", "grid_trade", "carbon"):
        metrics[key] = np.asarray(
            [float(np.mean(buckets[int(idx)][key])) for idx in iterations],
            dtype=np.float64,
        )
    return metrics


def _load_grouped(paths: List[Path]) -> Dict[str, List[Dict[str, np.ndarray]]]:
    grouped: Dict[str, List[Dict[str, np.ndarray]]] = {}
    for path in paths:
        method = _method_from_path(path)
        grouped.setdefault(method, []).append(_aggregate_episode_metrics(path))
    return grouped


def _mean_std(runs: List[Dict[str, np.ndarray]], metric: str, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_x = runs[0]["iteration"] + 1
    stacked = []
    for run in runs:
        if run["iteration"].shape != runs[0]["iteration"].shape or not np.array_equal(run["iteration"], runs[0]["iteration"]):
            raise ValueError("All runs for one method must share the same iteration axis.")
        stacked.append(_moving_average(run[metric], window))
    values = np.stack(stacked, axis=0)
    return base_x, np.mean(values, axis=0), np.std(values, axis=0)


def plot(paths: List[Path], output: Path, smoothing_window: int, title: str) -> Path:
    grouped = _load_grouped(paths)
    colors = {
        "dp_lcrl": "#111111",
        "mlp_pad": "#d94801",
        "mappo_shared": "#2171b5",
        "deepsets": "#238b45",
    }
    panels = [
        ("reward", "Reward Curve", "Reward"),
        ("p2p", "P2P Volume Mean per Active Agent", "Mean Volume"),
        ("grid_trade", "Grid Trading Mean per Active Agent", "Mean Volume"),
        ("carbon", "Carbon Responsibility Episode Mean", "Mean Carbon Responsibility"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    fig.suptitle(title, fontsize=16)

    for ax, (metric, panel_title, ylabel) in zip(axes.ravel(), panels):
        for method in sorted(grouped):
            xs, mean, std = _mean_std(grouped[method], metric, smoothing_window)
            color = colors.get(method, "#756bb1")
            ax.plot(xs, mean, linewidth=2.2, color=color, label=method)
            ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.14)
        ax.set_title(f"{panel_title} (MA{int(smoothing_window)})")
        ax.set_xlabel("Training Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Experiment 1 training comparison curves.")
    parser.add_argument("--summary_json", action="append", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--smoothing_window", type=int, default=5)
    parser.add_argument("--title", type=str, default="Experiment 1 Training Curves")
    args = parser.parse_args()

    paths = [Path(item).expanduser().resolve() for item in args.summary_json]
    result = plot(
        paths=paths,
        output=Path(args.output).expanduser().resolve(),
        smoothing_window=max(1, int(args.smoothing_window)),
        title=str(args.title),
    )
    print(f"saved_plot={result}")


if __name__ == "__main__":
    main()

