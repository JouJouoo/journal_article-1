"""Aggregate and plot the formal 3-seed ablation training results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["full", "cmtm_stateless", "mask_obs_only", "direct_max"]
METHOD_LABELS = {
    "full": "Full",
    "cmtm_stateless": "CMTM Stateless",
    "mask_obs_only": "Mask ObsOnly",
    "direct_max": "Direct Max Scale",
}
METRICS = {
    "reward": ("Reward", "Reward"),
    "p2p": ("P2P Trading Volume", "Mean volume"),
    "grid_trade": ("Grid Trading Volume", "Mean volume"),
    "carbon_resp": ("Carbon Responsibility", "Mean carbon responsibility"),
}


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values.astype(np.float64)
    window = min(int(window), int(values.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values.astype(np.float64), (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return math.nan, math.nan
    return float(np.mean(arr)), float(np.std(arr, ddof=1 if arr.size > 1 else 0))


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{_fmt(mean, digits)} +/- {_fmt(std, digits)}"


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if not runs:
        raise ValueError(f"No runs found in manifest: {manifest_path}")
    return payload


def _extract_seed(run: dict[str, Any]) -> int:
    if "seed" in run:
        return int(run["seed"])
    match = re.search(r"seed(\d+)", str(run.get("experiment_name", "")))
    if not match:
        raise ValueError(f"Could not infer seed from run: {run}")
    return int(match.group(1))


def _summary_path(run: dict[str, Any]) -> Path:
    run_dir = Path(str(run["run_dir"]))
    summary_path = run_dir / "paper_training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing training summary: {summary_path}")
    return summary_path


def _load_iteration_metrics(summary_path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    buckets: dict[int, dict[str, list[float]]] = {}
    for item in payload.get("episode_summaries", []):
        if item.get("phase") != "train":
            continue
        iteration = int(item.get("iteration_index", 0))
        bucket = buckets.setdefault(
            iteration,
            {
                "reward": [],
                "p2p": [],
                "grid_buy": [],
                "grid_sell": [],
                "carbon_resp": [],
            },
        )
        bucket["reward"].append(float(item.get("average_global_reward", 0.0)))
        bucket["p2p"].append(float(item.get("p2p_volume_mean_active", item.get("p2p_total_volume", 0.0))))
        bucket["grid_buy"].append(float(item.get("grid_buy_mean_active", item.get("grid_buy_total", 0.0))))
        bucket["grid_sell"].append(float(item.get("grid_sell_mean_active", item.get("grid_sell_total", 0.0))))
        bucket["carbon_resp"].append(
            float(
                item.get(
                    "carbon_responsibility_mean_active_episode",
                    item.get("load_responsibility_total", 0.0),
                )
            )
        )

    iterations = np.array(sorted(buckets.keys()), dtype=np.int32)
    if iterations.size == 0:
        raise ValueError(f"No training episode summaries found in: {summary_path}")

    metrics: dict[str, np.ndarray] = {"iteration": iterations}
    for key in ("reward", "p2p", "grid_buy", "grid_sell", "carbon_resp"):
        metrics[key] = np.array(
            [float(np.mean(buckets[int(idx)][key])) for idx in iterations],
            dtype=np.float64,
        )
    metrics["grid_trade"] = metrics["grid_buy"] + metrics["grid_sell"]
    return metrics


def _collect_runs(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {method: [] for method in METHOD_ORDER}
    for run in manifest.get("runs", []):
        method = str(run.get("method", ""))
        if method not in grouped:
            continue
        if run.get("status") != "completed":
            raise RuntimeError(f"Run is not completed: {run.get('experiment_name')} status={run.get('status')}")
        grouped[method].append(
            {
                "method": method,
                "seed": _extract_seed(run),
                "experiment_name": run.get("experiment_name"),
                "summary_path": _summary_path(run),
                "metrics": _load_iteration_metrics(_summary_path(run)),
            }
        )

    missing = [method for method in METHOD_ORDER if len(grouped[method]) != 3]
    if missing:
        details = ", ".join(f"{method}={len(grouped[method])}" for method in missing)
        raise RuntimeError(f"Expected 3 completed seeds per method, got: {details}")

    for method in grouped:
        grouped[method].sort(key=lambda item: int(item["seed"]))
    return grouped


def _tail_mean(values: np.ndarray, window: int) -> float:
    tail = values[-min(int(window), int(values.size)) :]
    return float(np.mean(tail))


def _summarize(grouped: dict[str, list[dict[str, Any]]], tail_window: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []

    for method in METHOD_ORDER:
        runs = grouped[method]
        by_metric: dict[str, list[float]] = {key: [] for key in METRICS}
        all_by_metric: dict[str, list[float]] = {key: [] for key in METRICS}

        for run in runs:
            row: dict[str, Any] = {
                "method": METHOD_LABELS[method],
                "method_key": method,
                "seed": run["seed"],
                "experiment_name": run["experiment_name"],
            }
            metrics = run["metrics"]
            for key in METRICS:
                final_value = _tail_mean(metrics[key], tail_window)
                all_value = float(np.mean(metrics[key]))
                row[f"final{tail_window}_{key}"] = final_value
                row[f"all_mean_{key}"] = all_value
                by_metric[key].append(final_value)
                all_by_metric[key].append(all_value)
            seed_rows.append(row)

        method_row: dict[str, Any] = {
            "method": METHOD_LABELS[method],
            "method_key": method,
            "seeds": ",".join(str(run["seed"]) for run in runs),
        }
        for key in METRICS:
            final_mean, final_std = _mean_std(by_metric[key])
            all_mean, all_std = _mean_std(all_by_metric[key])
            method_row[f"final{tail_window}_{key}_mean"] = final_mean
            method_row[f"final{tail_window}_{key}_std"] = final_std
            method_row[f"all_mean_{key}_mean"] = all_mean
            method_row[f"all_mean_{key}_std"] = all_std
        method_rows.append(method_row)

    baseline = next(row for row in method_rows if row["method_key"] == "full")
    for row in method_rows:
        for key in METRICS:
            base = float(baseline[f"final{tail_window}_{key}_mean"])
            value = float(row[f"final{tail_window}_{key}_mean"])
            row[f"delta_final{tail_window}_{key}_vs_full_pct"] = 100.0 * (value - base) / (base or 1e-9)
    return seed_rows, method_rows


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8-sig")
        return
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _word_table(method_rows: list[dict[str, Any]], tail_window: int) -> str:
    header = [
        "Method",
        f"Reward (Final-{tail_window})",
        f"P2P Volume (Final-{tail_window})",
        f"Grid Trade (Final-{tail_window})",
        f"Carbon Resp. (Final-{tail_window})",
    ]
    body = []
    for row in method_rows:
        body.append(
            [
                str(row["method"]),
                _fmt_pm(row[f"final{tail_window}_reward_mean"], row[f"final{tail_window}_reward_std"]),
                _fmt_pm(row[f"final{tail_window}_p2p_mean"], row[f"final{tail_window}_p2p_std"]),
                _fmt_pm(row[f"final{tail_window}_grid_trade_mean"], row[f"final{tail_window}_grid_trade_std"]),
                _fmt_pm(row[f"final{tail_window}_carbon_resp_mean"], row[f"final{tail_window}_carbon_resp_std"]),
            ]
        )
    widths = [max(len(str(row[idx])) for row in [header] + body) for idx in range(len(header))]
    lines = ["  ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(header))]
    lines.append("  ".join("-" * width for width in widths))
    for row in body:
        lines.append("  ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))
    lines.append("")
    lines.append(
        f"Notes: Values are mean +/- std across 3 seeds. Each seed value is the mean of the last {tail_window} training iterations."
    )
    lines.append("Grid Trade = grid_buy_mean_active + grid_sell_mean_active.")
    return "\n".join(lines) + "\n"


def _curve_frame(grouped: dict[str, list[dict[str, Any]]], smoothing_window: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        runs = grouped[method]
        iterations = runs[0]["metrics"]["iteration"]
        for run in runs[1:]:
            if not np.array_equal(iterations, run["metrics"]["iteration"]):
                raise RuntimeError(f"Iteration axis mismatch in method: {method}")
        for idx, iteration in enumerate(iterations):
            row: dict[str, Any] = {
                "iteration": int(iteration) + 1,
                "method": METHOD_LABELS[method],
                "method_key": method,
            }
            for key in METRICS:
                smoothed = [_moving_average(run["metrics"][key], smoothing_window)[idx] for run in runs]
                mean, std = _mean_std([float(value) for value in smoothed])
                row[f"{key}_mean"] = mean
                row[f"{key}_std"] = std
            rows.append(row)
    return rows


def _plot_metric(curve_rows: list[dict[str, Any]], metric: str, output_path: Path) -> None:
    colors = {
        "full": "#111111",
        "cmtm_stateless": "#d94801",
        "mask_obs_only": "#2171b5",
        "direct_max": "#238b45",
    }
    title, ylabel = METRICS[metric]
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=220)
    for method in METHOD_ORDER:
        selected = [row for row in curve_rows if row["method_key"] == method]
        xs = np.array([row["iteration"] for row in selected], dtype=np.float64)
        mean = np.array([row[f"{metric}_mean"] for row in selected], dtype=np.float64)
        std = np.array([row[f"{metric}_std"] for row in selected], dtype=np.float64)
        ax.plot(xs, mean, linewidth=2.2, label=METHOD_LABELS[method], color=colors[method])
        ax.fill_between(xs, mean - std, mean + std, color=colors[method], alpha=0.14, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_combined(curve_rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), dpi=220)
    for ax, metric in zip(axes.flat, METRICS):
        colors = {
            "full": "#111111",
            "cmtm_stateless": "#d94801",
            "mask_obs_only": "#2171b5",
            "direct_max": "#238b45",
        }
        title, ylabel = METRICS[metric]
        for method in METHOD_ORDER:
            selected = [row for row in curve_rows if row["method_key"] == method]
            xs = np.array([row["iteration"] for row in selected], dtype=np.float64)
            mean = np.array([row[f"{metric}_mean"] for row in selected], dtype=np.float64)
            std = np.array([row[f"{metric}_std"] for row in selected], dtype=np.float64)
            ax.plot(xs, mean, linewidth=2.0, label=METHOD_LABELS[method], color=colors[method])
            ax.fill_between(xs, mean - std, mean + std, color=colors[method], alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Formal Ablation Training Trends (3 seeds, mean +/- std)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render formal 3-seed ablation summary and trend figures.")
    parser.add_argument("--manifest", required=True, help="Path to formal_ablation_manifest.json.")
    parser.add_argument("--output_dir", required=True, help="Directory for summary tables and figures.")
    parser.add_argument("--tail_window", type=int, default=100, help="Final-iteration window for paper table values.")
    parser.add_argument("--smoothing_window", type=int, default=100, help="Moving-average window for trend curves.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    tail_window = max(1, int(args.tail_window))
    smoothing_window = max(1, int(args.smoothing_window))

    manifest = _load_manifest(manifest_path)
    grouped = _collect_runs(manifest)
    seed_rows, method_rows = _summarize(grouped, tail_window)
    curve_rows = _curve_frame(grouped, smoothing_window)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(method_rows, output_dir / "formal_ablation_3seed_method_summary.csv")
    _write_csv(seed_rows, output_dir / "formal_ablation_3seed_seed_summary.csv")
    _write_csv(curve_rows, output_dir / "formal_ablation_3seed_curve_points.csv")
    (output_dir / "formal_ablation_3seed_word_table.txt").write_text(
        _word_table(method_rows, tail_window),
        encoding="utf-8",
    )
    (output_dir / "formal_ablation_3seed_summary.json").write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "tail_window": tail_window,
                "smoothing_window": smoothing_window,
                "method_rows": method_rows,
                "seed_rows": seed_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    figures_dir = output_dir / "figures"
    _plot_combined(curve_rows, figures_dir / "formal_ablation_3seed_trends_combined.png")
    for metric in METRICS:
        _plot_metric(curve_rows, metric, figures_dir / f"formal_ablation_3seed_trend_{metric}.png")

    print(f"saved_dir={output_dir}")
    print(f"word_table={output_dir / 'formal_ablation_3seed_word_table.txt'}")
    print(f"combined_figure={figures_dir / 'formal_ablation_3seed_trends_combined.png'}")


if __name__ == "__main__":
    main()
