"""Render comparison tables and figures for controlled ablation experiments."""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values.astype(np.float64)
    window = min(int(window), int(values.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values.astype(np.float64), (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _resolve_summary_path(path_text: str) -> Path:
    path = Path(str(path_text).strip()).expanduser().resolve()
    if path.is_file():
        return path
    summary_json = path / "paper_training_summary.json"
    if summary_json.exists():
        return summary_json
    raise FileNotFoundError(f"Could not find paper_training_summary.json under: {path}")


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
                "agents": [],
                "grid_buy": [],
                "grid_sell": [],
                "carbon_load": [],
            },
        )
        bucket["reward"].append(float(item.get("average_global_reward", 0.0)))
        bucket["p2p"].append(float(item.get("p2p_volume_mean_active", item.get("p2p_total_volume", 0.0))))
        bucket["agents"].append(float(item.get("n_agents_mean", 0.0)))
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
    for key in ("reward", "p2p", "agents", "grid_buy", "grid_sell", "carbon_load"):
        metrics[key] = np.array(
            [float(np.mean(buckets[int(idx)][key])) for idx in iterations],
            dtype=np.float64,
        )
    metrics["grid_trade"] = metrics["grid_buy"] + metrics["grid_sell"]
    return metrics


def _collect_training_metrics(payload: dict) -> dict[str, np.ndarray]:
    history = payload.get("training_history", [])
    iterations = np.array([int(item.get("iteration_index", 0)) for item in history], dtype=np.int32)
    return {
        "iteration": iterations,
        "policy_loss": np.array(
            [float((item.get("metrics") or {}).get("policy_loss", 0.0)) for item in history],
            dtype=np.float64,
        ),
        "value_loss": np.array(
            [float((item.get("metrics") or {}).get("value_loss", 0.0)) for item in history],
            dtype=np.float64,
        ),
        "entropy": np.array(
            [float((item.get("metrics") or {}).get("dist_entropy", 0.0)) for item in history],
            dtype=np.float64,
        ),
        "fps": np.array(
            [float((item.get("metrics") or {}).get("fps_policy", 0.0)) for item in history],
            dtype=np.float64,
        ),
    }


def _convergence_iteration(smoothed_reward: np.ndarray) -> int | None:
    if smoothed_reward.size == 0:
        return None
    tail_window = min(50, int(smoothed_reward.size))
    target = float(np.mean(smoothed_reward[-tail_window:]))
    threshold = 0.95 * target if target >= 0.0 else 1.05 * target
    hits = np.where(smoothed_reward >= threshold)[0] if target >= 0.0 else np.where(smoothed_reward <= threshold)[0]
    if hits.size == 0:
        return None
    return int(hits[0] + 1)


def _tail_mean(values: np.ndarray, window: int) -> float:
    if values.size == 0:
        return 0.0
    tail = values[-min(window, int(values.size)) :]
    return float(np.mean(tail))


def _tail_std(values: np.ndarray, window: int) -> float:
    if values.size == 0:
        return 0.0
    tail = values[-min(window, int(values.size)) :]
    return float(np.std(tail))


def _format_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4f}"


def _load_runs(specs: List[str], smoothing_window: int) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    runs: List[Dict[str, object]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --run value '{spec}'. Expected LABEL=PATH.")
        label, raw_path = spec.split("=", 1)
        summary_path = _resolve_summary_path(raw_path)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        episode_metrics = _aggregate_episode_metrics(payload)
        training_metrics = _collect_training_metrics(payload)
        smoothed_reward = _moving_average(episode_metrics["reward"], smoothing_window)
        runs.append(
            {
                "label": label.strip(),
                "summary_path": summary_path,
                "episode_metrics": episode_metrics,
                "training_metrics": training_metrics,
                "smooth_reward": smoothed_reward,
            }
        )

    baseline = next((run for run in runs if run["label"] == "Full"), runs[0] if runs else None)
    baseline_metrics = {}
    if baseline is not None:
        ep = baseline["episode_metrics"]
        baseline_metrics = {
            "reward": float(np.mean(ep["reward"])),
            "p2p": float(np.mean(ep["p2p"])),
            "grid_trade": float(np.mean(ep["grid_trade"])),
            "carbon_load": float(np.mean(ep["carbon_load"])),
        }
    return runs, baseline_metrics


def _build_table_rows(runs: List[Dict[str, object]], baseline_metrics: Dict[str, float]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run in runs:
        ep = run["episode_metrics"]
        tr = run["training_metrics"]
        smooth_reward = run["smooth_reward"]
        row = {
            "Method": run["label"],
            "Avg Reward": float(np.mean(ep["reward"])),
            "Final100 Reward": _tail_mean(ep["reward"], 100),
            "Final100 Reward Std": _tail_std(ep["reward"], 100),
            "Avg P2P Mean": float(np.mean(ep["p2p"])),
            "Final100 P2P Mean": _tail_mean(ep["p2p"], 100),
            "Avg Grid Trade Mean": float(np.mean(ep["grid_trade"])),
            "Final100 Grid Trade Mean": _tail_mean(ep["grid_trade"], 100),
            "Avg Carbon Responsibility": float(np.mean(ep["carbon_load"])),
            "Final100 Carbon Responsibility": _tail_mean(ep["carbon_load"], 100),
            "Avg Active Agents": float(np.mean(ep["agents"])),
            "Convergence Iter (95%)": _convergence_iteration(smooth_reward),
            "Mean Policy Loss": float(np.mean(tr["policy_loss"])) if tr["policy_loss"].size else 0.0,
            "Mean Value Loss": float(np.mean(tr["value_loss"])) if tr["value_loss"].size else 0.0,
            "Mean Entropy": float(np.mean(tr["entropy"])) if tr["entropy"].size else 0.0,
            "Mean FPS": float(np.mean(tr["fps"])) if tr["fps"].size else 0.0,
            "Summary JSON": str(run["summary_path"]),
        }
        if baseline_metrics:
            reward_base = baseline_metrics.get("reward", 0.0) or 1e-9
            p2p_base = baseline_metrics.get("p2p", 0.0) or 1e-9
            grid_trade_base = baseline_metrics.get("grid_trade", 0.0) or 1e-9
            carbon_base = baseline_metrics.get("carbon_load", 0.0) or 1e-9
            row["Delta Reward vs Full (%)"] = 100.0 * (row["Avg Reward"] - reward_base) / reward_base
            row["Delta P2P vs Full (%)"] = 100.0 * (row["Avg P2P Mean"] - p2p_base) / p2p_base
            row["Delta Grid Trade vs Full (%)"] = 100.0 * (row["Avg Grid Trade Mean"] - grid_trade_base) / grid_trade_base
            row["Delta Carbon vs Full (%)"] = 100.0 * (row["Avg Carbon Responsibility"] - carbon_base) / carbon_base
        rows.append(row)
    return rows


def _write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [_format_value(row.get(col)) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(rows: List[Dict[str, object]], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    header_cells = "".join(f"<th>{escape(col)}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{escape(_format_value(row.get(col)))}</td>" for col in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; position: sticky; top: 0; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <table>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def _plot_learning_curves(runs: List[Dict[str, object]], output_path: Path, smoothing_window: int) -> None:
    colors = ["#111111", "#d94801", "#2171b5", "#238b45", "#756bb1", "#dd1c77"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    plot_specs = [
        ("reward", "Average Global Reward", "Reward"),
        ("p2p", "P2P Trading Mean per Active Agent", "Mean Volume"),
        ("grid_trade", "Grid Trading Mean per Active Agent", "Mean Volume"),
        ("carbon_load", "Carbon Responsibility Episode Mean", "Mean Carbon Responsibility"),
    ]

    for ax, (metric_key, title, y_label) in zip(axes.flat, plot_specs):
        for idx, run in enumerate(runs):
            metric_source = run["episode_metrics"]
            xs = metric_source["iteration"] + 1
            ys = _moving_average(metric_source[metric_key], smoothing_window)
            ax.plot(xs, ys, linewidth=2.0, color=colors[idx % len(colors)], label=run["label"])
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(runs)), frameon=False)
    fig.suptitle(f"Controlled Ablation Comparison (MA{int(smoothing_window)})", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_bar_metrics(rows: List[Dict[str, object]], output_path: Path) -> None:
    methods = [str(row["Method"]) for row in rows]
    reward = np.array([float(row["Final100 Reward"]) for row in rows], dtype=np.float64)
    p2p = np.array([float(row["Final100 P2P Mean"]) for row in rows], dtype=np.float64)
    grid_trade = np.array([float(row["Final100 Grid Trade Mean"]) for row in rows], dtype=np.float64)
    carbon = np.array([float(row["Final100 Carbon Responsibility"]) for row in rows], dtype=np.float64)

    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), dpi=180)
    panels = [
        (reward, "Final-100 Reward", "#111111"),
        (p2p, "Final-100 P2P Mean", "#d94801"),
        (grid_trade, "Final-100 Grid Trade Mean", "#238b45"),
        (carbon, "Final-100 Carbon Responsibility", "#08519c"),
    ]
    for ax, (values, title, color) in zip(axes, panels):
        bars = ax.bar(x, values, color=color, alpha=0.85)
        ax.set_title(title)
        ax.set_xticks(x, methods, rotation=15)
        ax.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_summary_json(rows: List[Dict[str, object]], runs: List[Dict[str, object]], output_path: Path) -> None:
    payload = {
        "table_rows": rows,
        "runs": [
            {
                "label": run["label"],
                "summary_path": str(run["summary_path"]),
            }
            for run in runs
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a controlled-ablation report from multiple run summaries.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in LABEL=PATH form. PATH can be a run directory or paper_training_summary.json",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write the report files into.")
    parser.add_argument("--title", type=str, default="Controlled Ablation Report")
    parser.add_argument("--smoothing_window", type=int, default=25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    runs, baseline_metrics = _load_runs(args.run, max(1, int(args.smoothing_window)))
    rows = _build_table_rows(runs, baseline_metrics)

    _write_csv(rows, output_dir / "ablation_comparison.csv")
    _write_markdown(rows, output_dir / "ablation_comparison.md")
    _write_html(rows, output_dir / "ablation_comparison.html", title=str(args.title))
    _plot_learning_curves(runs, output_dir / "ablation_learning_curves.png", max(1, int(args.smoothing_window)))
    _plot_bar_metrics(rows, output_dir / "ablation_key_metrics.png")
    _write_summary_json(rows, runs, output_dir / "ablation_comparison.json")

    print(f"saved_dir={output_dir}")


if __name__ == "__main__":
    main()
