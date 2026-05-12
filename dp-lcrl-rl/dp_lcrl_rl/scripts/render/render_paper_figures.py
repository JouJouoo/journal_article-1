"""Generate publication-quality paper figures from training and ablation data.

Usage:
  python render_paper_figures.py training --seed_dirs <dir1> <dir2> <dir3> --output_dir <dir>
  python render_paper_figures.py ablation --by_count_csv <path> --output_dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED_IDS = [42, 43, 44]
SEED_COLORS = {42: "#E41A1C", 43: "#377EB8", 44: "#4DAF4A"}
SEED_LABELS = {42: "Seed 42", 43: "Seed 43", 44: "Seed 44"}

METHOD_ORDER = ["full", "cmtm_stateless", "mask_obs_only", "direct_max"]
METHOD_LABELS = {
    "full": "Full",
    "cmtm_stateless": "CMTM Stateless",
    "mask_obs_only": "Mask ObsOnly",
    "direct_max": "Direct Max Scale",
}
METHOD_COLORS = {
    "full": "#111111",
    "cmtm_stateless": "#D94801",
    "mask_obs_only": "#2171B5",
    "direct_max": "#238B45",
}

FIGSIZE_SINGLE = (7.2, 4.8)
FIGSIZE_COMBINED = (14, 8)
DPI = 220
SMOOTHING_WINDOW = 100
RAW_ALPHA = 0.15
STD_ALPHA = 0.12

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def _configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "axes.linewidth": 1.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values.astype(np.float64)
    window = min(int(window), int(values.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values.astype(np.float64), (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Requirement 1: Training curves
# ---------------------------------------------------------------------------


def _load_training_summary(summary_path: Path) -> dict[str, np.ndarray]:
    """Load a single training summary JSON and aggregate episode data per iteration."""
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    buckets: dict[int, dict[str, list[float]]] = {}
    for item in payload.get("episode_summaries", []):
        if item.get("phase") != "train":
            continue
        iteration = int(item.get("iteration_index", 0))
        bucket = buckets.setdefault(iteration, {
            "reward": [],
            "p2p": [],
            "grid_buy": [],
            "grid_sell": [],
            "carbon_resp": [],
        })
        bucket["reward"].append(float(item.get("average_global_reward", 0.0)))
        bucket["p2p"].append(float(item.get("p2p_volume_mean_active", item.get("p2p_total_volume", 0.0))))
        bucket["grid_buy"].append(float(item.get("grid_buy_mean_active", item.get("grid_buy_total", 0.0))))
        bucket["grid_sell"].append(float(item.get("grid_sell_mean_active", item.get("grid_sell_total", 0.0))))
        bucket["carbon_resp"].append(
            float(item.get(
                "carbon_responsibility_mean_active_episode",
                item.get("load_responsibility_total", 0.0),
            ))
        )

    iterations = np.array(sorted(buckets.keys()), dtype=np.int32)
    if iterations.size == 0:
        raise ValueError(f"No train episode summaries found in: {summary_path}")

    metrics: dict[str, np.ndarray] = {"iteration": iterations}
    for key in ("reward", "p2p", "grid_buy", "grid_sell", "carbon_resp"):
        metrics[key] = np.array(
            [float(np.mean(buckets[int(idx)][key])) for idx in iterations],
            dtype=np.float64,
        )
    metrics["grid_trade"] = metrics["grid_buy"] + metrics["grid_sell"]
    return metrics


def _load_seed_data(seed_dirs: list[Path]) -> dict[str, Any]:
    """Load training data from 3 seed directories."""
    data = {
        "iteration": None,
        "raw": {},   # (seed_idx, metric) -> np.ndarray
        "smooth": {},  # (seed_idx, metric) -> np.ndarray
    }
    for path in seed_dirs:
        summary_path = path / "paper_training_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing training summary: {summary_path}")
        metrics = _load_training_summary(summary_path)
        # Infer seed from path name
        seed = None
        for sid in SEED_IDS:
            if f"seed{sid}" in str(path).lower():
                seed = sid
                break
        if seed is None:
            raise ValueError(f"Cannot infer seed from path: {path}")

        if data["iteration"] is None:
            data["iteration"] = metrics["iteration"] + 1  # 1-indexed
        for key in ("reward", "p2p", "grid_trade", "carbon_resp"):
            raw = metrics[key]
            smooth = _moving_average(raw, SMOOTHING_WINDOW)
            data["raw"][(seed, key)] = raw
            data["smooth"][(seed, key)] = smooth
    return data


def _plot_training_metric(
    data: dict[str, Any],
    metric_key: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot a single training metric as mean +/- std across 3 seeds."""
    _configure_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, dpi=DPI)
    xs = data["iteration"]

    # Stack smoothed curves from all seeds
    smooth_arrays = []
    for seed in SEED_IDS:
        smooth = data["smooth"].get((seed, metric_key))
        if smooth is not None:
            smooth_arrays.append(smooth)

    if len(smooth_arrays) >= 2:
        # Truncate all to the shortest length
        min_len = min(len(s) for s in smooth_arrays)
        stacked = np.stack([s[:min_len] for s in smooth_arrays], axis=0)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0, ddof=1)
        plot_xs = xs[:min_len]

        # Mean line
        ax.plot(plot_xs, mean, color="#111111", linewidth=2.2, label="Mean", zorder=3)
        # Std shading
        ax.fill_between(plot_xs, mean - std, mean + std,
                        color="#111111", alpha=STD_ALPHA, linewidth=0, zorder=2,
                        label="± 1 Std")
        # Semi-transparent individual raw curves
        for seed in SEED_IDS:
            raw = data["raw"].get((seed, metric_key))
            if raw is not None:
                ax.plot(xs, raw, color=SEED_COLORS[seed], alpha=RAW_ALPHA,
                        linewidth=0.6, zorder=1)
    else:
        # Fallback: plot single available seed
        single = smooth_arrays[0] if smooth_arrays else None
        if single is not None:
            ax.plot(xs[:len(single)], single, color="#111111", linewidth=2.2)

    ax.set_title(title)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


TRAINING_METRICS = [
    ("reward", "Average Reward", "Training Reward Curve"),
    ("p2p", "P2P Trade Volume per Active Agent", "P2P Trading Volume"),
    ("grid_trade", "Grid Trade Volume per Active Agent", "Grid Trading Volume"),
    ("carbon_resp", "Carbon Responsibility per Active Agent", "Carbon Responsibility"),
]


def run_training(args: argparse.Namespace) -> None:
    seed_dirs = [Path(p).expanduser().resolve() for p in args.seed_dirs]
    output_dir = Path(args.output_dir).expanduser().resolve()

    data = _load_seed_data(seed_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)

    filenames = [
        "training_reward.png",
        "training_p2p_volume.png",
        "training_grid_trade.png",
        "training_carbon_responsibility.png",
    ]
    for (metric_key, y_label, title), fname in zip(TRAINING_METRICS, filenames):
        out_path = output_dir / fname
        _plot_training_metric(data, metric_key, y_label, title, out_path)
        print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Requirement 2: Agent count ablation sweep
# ---------------------------------------------------------------------------


def _load_ablation_csv(csv_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load ablation by-count CSV and group rows by method key."""
    groups: dict[str, list[dict[str, Any]]] = {m: [] for m in METHOD_ORDER}
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("method", "").strip()
            if method not in groups:
                continue
            groups[method].append({
                "count": int(row.get("active_agent_count", 0)),
                "reward_mean": float(row.get("reward_mean", 0.0)),
                "reward_std": float(row.get("reward_std", 0.0)),
                "p2p_mean": float(row.get("p2p_mean", 0.0)),
                "p2p_std": float(row.get("p2p_std", 0.0)),
                "grid_trade_mean": float(row.get("grid_trade_mean", 0.0)),
                "grid_trade_std": float(row.get("grid_trade_std", 0.0)),
                "carbon_mean": float(row.get("carbon_mean", 0.0)),
                "carbon_std": float(row.get("carbon_std", 0.0)),
            })
    # Sort each group by count
    for method in groups:
        groups[method].sort(key=lambda r: r["count"])
    return groups


ABLATION_SUBPLOTS = [
    ("reward", "Average Reward", "Average Reward"),
    ("p2p", "P2P Trade Volume per Agent", "P2P Trading Volume"),
    ("grid_trade", "Grid Trade Volume per Agent", "Grid Trading Volume"),
    ("carbon", "Carbon Responsibility per Agent", "Carbon Responsibility"),
]


def _plot_ablation_combined(
    groups: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    """Create 2x2 combined figure comparing 4 methods across agent counts."""
    _configure_style()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_COMBINED, dpi=DPI)
    fig.suptitle("Ablation: Performance vs. Active Agent Count", fontsize=15)

    for ax, (csv_key, y_label, title) in zip(axes.flat, ABLATION_SUBPLOTS):
        for method in METHOD_ORDER:
            rows = groups[method]
            if not rows:
                continue
            xs = np.array([r["count"] for r in rows], dtype=np.float64)
            mean = np.array([r[f"{csv_key}_mean"] for r in rows], dtype=np.float64)
            std = np.array([r[f"{csv_key}_std"] for r in rows], dtype=np.float64)
            color = METHOD_COLORS[method]
            ax.plot(xs, mean, color=color, linewidth=2.0,
                    label=METHOD_LABELS[method], zorder=2)
            ax.fill_between(xs, mean - std, mean + std,
                            color=color, alpha=STD_ALPHA, linewidth=0, zorder=1)

        ax.set_title(title)
        ax.set_xlabel("Active Agent Count")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, 31)
        ax.set_xticks(range(0, 31, 5))

    # Shared legend at top
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_ablation(args: argparse.Namespace) -> None:
    csv_path = Path(args.by_count_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    groups = _load_ablation_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ablation_agent_count_combined.png"
    _plot_ablation_combined(groups, out_path)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render publication-quality figures for the DP-LCRL paper."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True,
                                       help="Figure type to generate")

    # Training curves
    train_parser = subparsers.add_parser("training", help="Training curves (Requirement 1)")
    train_parser.add_argument("--seed_dirs", nargs=3, required=True,
                              help="3 directories containing paper_training_summary.json")
    train_parser.add_argument("--output_dir", required=True,
                              help="Output directory for PNG files")
    train_parser.add_argument("--smoothing_window", type=int, default=SMOOTHING_WINDOW,
                              help=f"Moving average window (default: {SMOOTHING_WINDOW})")

    # Ablation sweep
    abl_parser = subparsers.add_parser("ablation", help="Agent count ablation sweep (Requirement 2)")
    abl_parser.add_argument("--by_count_csv", required=True,
                            help="Path to formal_ablation_unified_eval_by_count.csv")
    abl_parser.add_argument("--output_dir", required=True,
                            help="Output directory for PNG files")

    args = parser.parse_args()

    if args.mode == "training":
        print("Generating training curve figures...")
        run_training(args)
        print("Done.")
    elif args.mode == "ablation":
        print("Generating ablation sweep figure...")
        run_ablation(args)
        print("Done.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
