"""Plot multi-seed training curves as per-seed overlays or mean/std bands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values.astype(np.float64)
    window = min(int(window), int(values.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values.astype(np.float64), (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _aggregate_episode_metrics(payload: dict) -> Dict[str, np.ndarray]:
    episode_summaries = payload.get("episode_summaries", [])
    buckets: dict[int, dict[str, list[float]]] = {}
    for item in episode_summaries:
        if item.get("phase") != "train":
            continue
        idx = int(item.get("iteration_index", 0))
        bucket = buckets.setdefault(
            idx,
            {
                "reward": [],
                "p2p": [],
                "grid_buy": [],
                "grid_sell": [],
                "carbon_load": [],
            },
        )
        bucket["reward"].append(float(item.get("average_global_reward", 0.0)))
        bucket["p2p"].append(float(item.get("p2p_volume_mean_active", item.get("p2p_total_volume", 0.0))))
        bucket["grid_buy"].append(float(item.get("grid_buy_mean_active", item.get("grid_buy_total", 0.0))))
        bucket["grid_sell"].append(float(item.get("grid_sell_mean_active", item.get("grid_sell_total", 0.0))))
        bucket["carbon_load"].append(
            float(
                item.get(
                    "carbon_responsibility_mean_active_episode",
                    item.get("load_responsibility_total", 0.0),
                )
            )
        )

    iterations = np.array(sorted(buckets.keys()), dtype=np.int32)
    if iterations.size == 0:
        raise ValueError("No train episode summaries found in summary JSON.")

    metrics = {"iteration": iterations}
    for key in ("reward", "p2p", "grid_buy", "grid_sell", "carbon_load"):
        metrics[key] = np.array(
            [float(np.mean(buckets[int(idx)][key])) for idx in iterations],
            dtype=np.float64,
        )
    return metrics


def _load_runs(summary_jsons: List[Path]) -> List[Dict[str, np.ndarray]]:
    runs = []
    for summary_json in summary_jsons:
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        run = _aggregate_episode_metrics(payload)
        run["label"] = np.array([summary_json.parent.name], dtype=object)
        runs.append(run)

    base_iterations = runs[0]["iteration"]
    for run in runs[1:]:
        if run["iteration"].shape != base_iterations.shape or not np.array_equal(run["iteration"], base_iterations):
            raise ValueError("All runs must share the same iteration axis for multi-seed plotting.")
    return runs


def _stack_runs(runs: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    stacked = {"iteration": runs[0]["iteration"]}
    for key in ("reward", "p2p", "grid_buy", "grid_sell", "carbon_load"):
        stacked[key] = np.stack([run[key] for run in runs], axis=0)
    return stacked


def _mean_std_smooth(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    smoothed = np.stack([_moving_average(row, window) for row in values], axis=0)
    return np.mean(smoothed, axis=0), np.std(smoothed, axis=0)


def _resolve_labels(runs: List[Dict[str, np.ndarray]], labels: List[str] | None) -> List[str]:
    if labels:
        if len(labels) != len(runs):
            raise ValueError("When --label is provided, its count must match --summary_json.")
        return [str(label) for label in labels]

    resolved: List[str] = []
    for idx, run in enumerate(runs):
        raw = str(run["label"][0]) if "label" in run else f"run{idx + 1}"
        lower = raw.lower()
        if "seed" in lower:
            tail = lower.split("seed", 1)[1]
            digits = "".join(ch for ch in tail if ch.isdigit())
            if digits:
                resolved.append(f"Seed {digits}")
                continue
        resolved.append(raw)
    return resolved


def _plot_overlay(
    runs: List[Dict[str, np.ndarray]],
    labels: List[str],
    output_path: Path,
    smoothing_window: int,
    title: str,
) -> Path:
    xs = runs[0]["iteration"] + 1
    colors = ["#111111", "#d94801", "#2171b5", "#238b45", "#756bb1", "#dd1c77"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    fig.suptitle(title, fontsize=16)

    panels = [
        ("reward", "Reward Curve", "Reward", "#111111"),
        ("p2p", "P2P Volume Mean per Active Agent", "Mean Volume", "#d94801"),
        ("grid_buy", "Grid Buy Mean per Active Agent", "Mean Volume", "#e6550d"),
        ("grid_sell", "Grid Sell Mean per Active Agent", "Mean Volume", "#2171b5"),
        ("carbon_load", "Carbon Responsibility Episode Mean", "Mean Carbon Responsibility", "#08519c"),
    ]

    for ax, (metric_key, panel_title, y_label, _default_color) in zip(axes.flat, panels):
        for idx, (run, label) in enumerate(zip(runs, labels)):
            ax.plot(
                xs,
                _moving_average(run[metric_key], smoothing_window),
                linewidth=2.0,
                color=colors[idx % len(colors)],
                label=label,
            )
        ax.set_title(f"{panel_title} (MA{int(smoothing_window)})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_mean_std(
    runs: List[Dict[str, np.ndarray]],
    output_path: Path,
    smoothing_window: int,
    title: str,
) -> Path:
    series = _stack_runs(runs)
    xs = series["iteration"] + 1

    reward_mean, reward_std = _mean_std_smooth(series["reward"], smoothing_window)
    p2p_mean, p2p_std = _mean_std_smooth(series["p2p"], smoothing_window)
    grid_buy_mean, grid_buy_std = _mean_std_smooth(series["grid_buy"], smoothing_window)
    grid_sell_mean, grid_sell_std = _mean_std_smooth(series["grid_sell"], smoothing_window)
    carbon_mean, carbon_std = _mean_std_smooth(series["carbon_load"], smoothing_window)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    fig.suptitle(title, fontsize=16)

    panels = [
        ("Reward Curve", "Reward", reward_mean, reward_std, "#111111", "#999999", "Mean Reward"),
        ("P2P Volume Mean per Active Agent", "Mean Volume", p2p_mean, p2p_std, "#d94801", "#fdae6b", "Mean P2P Volume"),
        (
            "Grid Buy and Grid Sell",
            "Mean Volume",
            (grid_buy_mean, grid_sell_mean),
            (grid_buy_std, grid_sell_std),
            ("#e6550d", "#2171b5"),
            ("#fdae6b", "#9ecae1"),
            ("Grid Buy", "Grid Sell"),
        ),
        ("Carbon Responsibility Episode Mean", "Mean Carbon Responsibility", carbon_mean, carbon_std, "#08519c", "#9ecae1", "Mean Carbon Responsibility"),
    ]

    for ax, (panel_title, y_label, mean_vals, std_vals, line_color, fill_color, curve_label) in zip(axes.flat, panels):
        if isinstance(mean_vals, tuple):
            for mean_item, std_item, line_item, fill_item, label_item in zip(
                mean_vals, std_vals, line_color, fill_color, curve_label
            ):
                ax.plot(xs, mean_item, color=line_item, linewidth=2.2, label=label_item)
                ax.fill_between(
                    xs,
                    mean_item - std_item,
                    mean_item + std_item,
                    color=fill_item,
                    alpha=0.18,
                    label=f"{label_item} +/- 1 std",
                )
        else:
            ax.plot(xs, mean_vals, color=line_color, linewidth=2.2, label=curve_label)
            ax.fill_between(xs, mean_vals - std_vals, mean_vals + std_vals, color=fill_color, alpha=0.25, label="+/- 1 std")
        ax.set_title(f"{panel_title} (MA{int(smoothing_window)})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_curves(
    summary_jsons: List[Path],
    output_path: Path,
    smoothing_window: int,
    title: str,
    mode: str = "overlay",
    labels: List[str] | None = None,
) -> Path:
    runs = _load_runs(summary_jsons)
    resolved_labels = _resolve_labels(runs, labels)
    if mode == "overlay":
        return _plot_overlay(runs, resolved_labels, output_path, smoothing_window, title)
    if mode == "mean_std":
        return _plot_mean_std(runs, output_path, smoothing_window, title)
    raise ValueError(f"Unsupported mode={mode}. Expected 'overlay' or 'mean_std'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multi-seed DP-LCRL training curves.")
    parser.add_argument("--summary_json", action="append", required=True, help="Path to one paper_training_summary.json")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default="Multi-Seed Training Curves")
    parser.add_argument("--smoothing_window", type=int, default=100)
    parser.add_argument("--mode", type=str, default="overlay", choices=["overlay", "mean_std"])
    parser.add_argument("--label", action="append", default=None, help="Optional display label for one summary_json.")
    args = parser.parse_args()

    summary_jsons = [Path(item).expanduser().resolve() for item in args.summary_json]
    for path in summary_jsons:
        if not path.exists():
            raise FileNotFoundError(f"Summary JSON not found: {path}")

    result = plot_curves(
        summary_jsons=summary_jsons,
        output_path=Path(args.output).expanduser().resolve(),
        smoothing_window=max(1, int(args.smoothing_window)),
        title=str(args.title),
        mode=str(args.mode),
        labels=args.label,
    )
    print(f"saved_plot={result}")


if __name__ == "__main__":
    main()
