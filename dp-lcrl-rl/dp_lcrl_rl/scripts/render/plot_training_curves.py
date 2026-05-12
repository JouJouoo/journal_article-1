"""Render training curves from a paper summary JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values
    window = min(window, values.size)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _aggregate_episode_metrics(payload: dict) -> dict[str, np.ndarray]:
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
                "carbon": [],
            },
        )
        bucket["reward"].append(float(item.get("average_global_reward", 0.0)))
        bucket["p2p"].append(float(item.get("p2p_volume_mean_active", item.get("p2p_total_volume", 0.0))))
        bucket["grid_buy"].append(float(item.get("grid_buy_mean_active", item.get("grid_buy_total", 0.0))))
        bucket["grid_sell"].append(float(item.get("grid_sell_mean_active", item.get("grid_sell_total", 0.0))))
        bucket["carbon"].append(
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
    for key in ("reward", "p2p", "grid_buy", "grid_sell", "carbon"):
        metrics[key] = np.array(
            [float(np.mean(buckets[int(idx)][key])) for idx in iterations],
            dtype=np.float64,
        )
    metrics["grid_trade"] = metrics["grid_buy"] + metrics["grid_sell"]
    return metrics


def plot_dashboard(summary_json: Path, output_path: Path, smoothing_window: int) -> Path:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    episode_metrics = _aggregate_episode_metrics(payload)

    xs = episode_metrics["iteration"] + 1
    smooth_reward = _moving_average(episode_metrics["reward"], smoothing_window)
    smooth_p2p = _moving_average(episode_metrics["p2p"], smoothing_window)
    smooth_grid_trade = _moving_average(episode_metrics["grid_trade"], smoothing_window)
    smooth_carbon = _moving_average(episode_metrics["carbon"], smoothing_window)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160)
    fig.suptitle("DP-LCRL Training Curves", fontsize=16)

    ax = axes[0, 0]
    ax.plot(xs, episode_metrics["reward"], color="#9ecae1", linewidth=1.0, alpha=0.5, label="Raw")
    ax.plot(xs, smooth_reward, color="#08519c", linewidth=2.0, label=f"MA({smoothing_window})")
    ax.set_title("Average Global Reward")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(xs, episode_metrics["p2p"], color="#fdae6b", linewidth=1.0, alpha=0.5, label="Raw")
    ax.plot(xs, smooth_p2p, color="#e6550d", linewidth=2.0, label=f"MA({smoothing_window})")
    ax.set_title("P2P Trading Mean per Active Agent")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Volume")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(xs, episode_metrics["grid_trade"], color="#74c476", linewidth=1.0, alpha=0.5, label="Raw")
    ax.plot(xs, smooth_grid_trade, color="#238b45", linewidth=2.0, label=f"MA({smoothing_window})")
    ax.set_title("Grid Trading Mean per Active Agent")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Volume")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(xs, episode_metrics["carbon"], color="#9ecae1", linewidth=1.0, alpha=0.5, label="Episode Mean")
    ax.plot(xs, smooth_carbon, color="#08519c", linewidth=2.0, label=f"MA({smoothing_window})")
    ax.set_title("Carbon Responsibility (Episode Mean)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Carbon Responsibility")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DP-LCRL training curves from a summary JSON.")
    parser.add_argument("--summary_json", type=str, required=True, help="Path to paper_training_summary.json")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path. Defaults to <summary_dir>/training_curves.png",
    )
    parser.add_argument("--smoothing_window", type=int, default=25, help="Moving average window size.")
    args = parser.parse_args()

    summary_json = Path(args.summary_json).expanduser().resolve()
    if not summary_json.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_json}")

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else summary_json.with_name("training_curves.png")
    )
    result = plot_dashboard(summary_json, output, max(1, int(args.smoothing_window)))
    print(f"saved_plot={result}")


if __name__ == "__main__":
    main()
