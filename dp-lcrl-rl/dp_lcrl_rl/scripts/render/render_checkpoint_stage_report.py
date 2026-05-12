"""Render tables and figures for selected training checkpoints across multiple runs."""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _resolve_summary_path(path_text: str) -> Path:
    path = Path(str(path_text).strip()).expanduser().resolve()
    if path.is_file():
        return path
    summary_json = path / "paper_training_summary.json"
    if summary_json.exists():
        return summary_json
    raise FileNotFoundError(f"Could not find paper_training_summary.json under: {path}")


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
                "carbon": [],
                "agents": [],
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
        bucket["agents"].append(float(item.get("n_agents_mean", 0.0)))

    iterations = np.array(sorted(buckets.keys()), dtype=np.int32)
    if iterations.size == 0:
        raise ValueError("No train episode summaries found in summary JSON.")

    metrics = {"iteration": iterations}
    for key in ("reward", "p2p", "grid_buy", "grid_sell", "carbon", "agents"):
        metrics[key] = np.array(
            [float(np.mean(buckets[int(idx)][key])) for idx in iterations],
            dtype=np.float64,
        )
    metrics["grid_trade"] = metrics["grid_buy"] + metrics["grid_sell"]
    return metrics


def _load_runs(specs: Sequence[str]) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --run value '{spec}'. Expected LABEL=PATH.")
        label, raw_path = spec.split("=", 1)
        summary_path = _resolve_summary_path(raw_path)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        runs.append(
            {
                "label": label.strip(),
                "summary_path": summary_path,
                "metrics": _aggregate_episode_metrics(payload),
            }
        )
    return runs


def _window_slice(iterations: np.ndarray, checkpoint_episode: int, window: int) -> np.ndarray:
    target_iter = int(checkpoint_episode) - 1
    valid = np.where(iterations <= target_iter)[0]
    if valid.size == 0:
        raise ValueError(f"No iterations found before checkpoint {checkpoint_episode}.")
    end = int(valid[-1])
    start = max(0, end - max(1, int(window)) + 1)
    return np.arange(start, end + 1, dtype=np.int32)


def _collect_rows(runs: Sequence[Dict[str, object]], checkpoints: Sequence[int], window: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run in runs:
        metrics = run["metrics"]
        iterations = np.asarray(metrics["iteration"], dtype=np.int32)
        for checkpoint in checkpoints:
            idxs = _window_slice(iterations, int(checkpoint), window)
            row = {
                "Run": run["label"],
                "Checkpoint": int(checkpoint),
                "Window": int(len(idxs)),
                "Reward Mean": float(np.mean(metrics["reward"][idxs])),
                "Reward Std": float(np.std(metrics["reward"][idxs])),
                "P2P Mean": float(np.mean(metrics["p2p"][idxs])),
                "P2P Std": float(np.std(metrics["p2p"][idxs])),
                "Grid Trade Mean": float(np.mean(metrics["grid_trade"][idxs])),
                "Grid Trade Std": float(np.std(metrics["grid_trade"][idxs])),
                "Carbon Mean": float(np.mean(metrics["carbon"][idxs])),
                "Carbon Std": float(np.std(metrics["carbon"][idxs])),
                "Active Agents Mean": float(np.mean(metrics["agents"][idxs])),
                "Summary JSON": str(run["summary_path"]),
            }
            rows.append(row)
    return rows


def _aggregate_rows(rows: Sequence[Dict[str, object]], checkpoints: Sequence[int]) -> List[Dict[str, object]]:
    grouped: dict[int, list[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["Checkpoint"]), []).append(row)

    out: List[Dict[str, object]] = []
    for checkpoint in checkpoints:
        bucket = grouped.get(int(checkpoint), [])
        if not bucket:
            continue

        def stats(key: str) -> Tuple[float, float]:
            values = np.asarray([float(item[key]) for item in bucket], dtype=np.float64)
            return float(np.mean(values)), float(np.std(values))

        reward_mean, reward_std = stats("Reward Mean")
        p2p_mean, p2p_std = stats("P2P Mean")
        grid_trade_mean, grid_trade_std = stats("Grid Trade Mean")
        carbon_mean, carbon_std = stats("Carbon Mean")
        out.append(
            {
                "Checkpoint": int(checkpoint),
                "Reward Mean": reward_mean,
                "Reward Std": reward_std,
                "P2P Mean": p2p_mean,
                "P2P Std": p2p_std,
                "Grid Trade Mean": grid_trade_mean,
                "Grid Trade Std": grid_trade_std,
                "Carbon Mean": carbon_mean,
                "Carbon Std": carbon_std,
            }
        )
    return out


def _format_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4f}"


def _write_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(
    raw_rows: Sequence[Dict[str, object]],
    aggregate_rows: Sequence[Dict[str, object]],
    output_path: Path,
    window: int,
) -> None:
    lines = [
        "# Checkpoint Stage Report",
        "",
        f"- statistic: `window-{int(window)}` trailing average ending at each checkpoint",
        "",
        "## Aggregate (mean +/- std across seeds)",
        "",
        "| Checkpoint | Reward | P2P Mean | Grid Trade Mean | Carbon Mean |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {ckpt} | {r:.4f} +/- {rs:.4f} | {p:.4f} +/- {ps:.4f} | {g:.4f} +/- {gs:.4f} | {c:.4f} +/- {cs:.4f} |".format(
                ckpt=int(row["Checkpoint"]),
                r=float(row["Reward Mean"]),
                rs=float(row["Reward Std"]),
                p=float(row["P2P Mean"]),
                ps=float(row["P2P Std"]),
                g=float(row["Grid Trade Mean"]),
                gs=float(row["Grid Trade Std"]),
                c=float(row["Carbon Mean"]),
                cs=float(row["Carbon Std"]),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Seed Results",
            "",
            "| Run | Checkpoint | Reward Mean | P2P Mean | Grid Trade Mean | Carbon Mean |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in raw_rows:
        lines.append(
            "| {run} | {ckpt} | {r:.4f} | {p:.4f} | {g:.4f} | {c:.4f} |".format(
                run=str(row["Run"]),
                ckpt=int(row["Checkpoint"]),
                r=float(row["Reward Mean"]),
                p=float(row["P2P Mean"]),
                g=float(row["Grid Trade Mean"]),
                c=float(row["Carbon Mean"]),
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(rows: Sequence[Dict[str, object]], output_path: Path, title: str) -> None:
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
    th {{ background: #f5f5f5; }}
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def _plot_per_seed(rows: Sequence[Dict[str, object]], output_path: Path, title: str) -> None:
    by_run: dict[str, list[Dict[str, object]]] = {}
    for row in rows:
        by_run.setdefault(str(row["Run"]), []).append(row)
    colors = ["#111111", "#d94801", "#2171b5", "#238b45", "#756bb1", "#dd1c77"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    fig.suptitle(title, fontsize=16)
    specs = [
        ("Reward Mean", "Reward"),
        ("P2P Mean", "P2P Volume Mean per Active Agent"),
        ("Grid Trade Mean", "Grid Trading Mean per Active Agent"),
        ("Carbon Mean", "Carbon Responsibility Episode Mean"),
    ]
    for ax, (metric_key, panel_title) in zip(axes.flat, specs):
        for idx, (run, bucket) in enumerate(sorted(by_run.items())):
            bucket = sorted(bucket, key=lambda item: int(item["Checkpoint"]))
            xs = [int(item["Checkpoint"]) for item in bucket]
            ys = [float(item[metric_key]) for item in bucket]
            ax.plot(xs, ys, marker="o", linewidth=2.0, color=colors[idx % len(colors)], label=run)
        ax.set_title(panel_title)
        ax.set_xlabel("Checkpoint Episode")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_aggregate(rows: Sequence[Dict[str, object]], output_path: Path, title: str) -> None:
    xs = np.asarray([int(item["Checkpoint"]) for item in rows], dtype=np.int32)
    reward_mean = np.asarray([float(item["Reward Mean"]) for item in rows], dtype=np.float64)
    reward_std = np.asarray([float(item["Reward Std"]) for item in rows], dtype=np.float64)
    p2p_mean = np.asarray([float(item["P2P Mean"]) for item in rows], dtype=np.float64)
    p2p_std = np.asarray([float(item["P2P Std"]) for item in rows], dtype=np.float64)
    grid_trade_mean = np.asarray([float(item["Grid Trade Mean"]) for item in rows], dtype=np.float64)
    grid_trade_std = np.asarray([float(item["Grid Trade Std"]) for item in rows], dtype=np.float64)
    carbon_mean = np.asarray([float(item["Carbon Mean"]) for item in rows], dtype=np.float64)
    carbon_std = np.asarray([float(item["Carbon Std"]) for item in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    fig.suptitle(title, fontsize=16)
    panels = [
        ("Reward", reward_mean, reward_std, "#111111", "#999999"),
        ("P2P Volume Mean per Active Agent", p2p_mean, p2p_std, "#d94801", "#fdae6b"),
        ("Grid Trading Mean per Active Agent", grid_trade_mean, grid_trade_std, "#238b45", "#74c476"),
        ("Carbon Responsibility Episode Mean", carbon_mean, carbon_std, "#08519c", "#9ecae1"),
    ]
    for ax, (panel_title, ys, errs, line_color, fill_color) in zip(axes.flat, panels):
        ax.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2.0, color=line_color, capsize=4)
        ax.fill_between(xs, ys - errs, ys + errs, color=fill_color, alpha=0.20)
        ax.set_title(panel_title)
        ax.set_xlabel("Checkpoint Episode")
        ax.grid(True, alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_summary_json(
    raw_rows: Sequence[Dict[str, object]],
    aggregate_rows: Sequence[Dict[str, object]],
    output_path: Path,
    checkpoints: Sequence[int],
    window: int,
) -> None:
    payload = {
        "checkpoints": [int(item) for item in checkpoints],
        "window": int(window),
        "raw_rows": list(raw_rows),
        "aggregate_rows": list(aggregate_rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render checkpoint-stage comparison report.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in LABEL=PATH form. PATH can be a run directory or paper_training_summary.json",
    )
    parser.add_argument("--checkpoint", action="append", type=int, required=True, help="Checkpoint episode.")
    parser.add_argument("--window", type=int, default=100, help="Trailing window size used for statistics.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--title", type=str, default="Checkpoint Stage Report")
    args = parser.parse_args()

    checkpoints = sorted({int(item) for item in args.checkpoint})
    runs = _load_runs(args.run)
    raw_rows = _collect_rows(runs, checkpoints, max(1, int(args.window)))
    aggregate_rows = _aggregate_rows(raw_rows, checkpoints)

    output_dir = Path(args.output_dir).expanduser().resolve()
    _write_csv(raw_rows, output_dir / "checkpoint_stage_raw.csv")
    _write_csv(aggregate_rows, output_dir / "checkpoint_stage_aggregate.csv")
    _write_markdown(raw_rows, aggregate_rows, output_dir / "checkpoint_stage_report.md", max(1, int(args.window)))
    _write_html(raw_rows, output_dir / "checkpoint_stage_raw.html", title=f"{args.title} Raw")
    _plot_per_seed(raw_rows, output_dir / "checkpoint_stage_per_seed.png", f"{args.title} (Per Seed)")
    _plot_aggregate(aggregate_rows, output_dir / "checkpoint_stage_aggregate.png", f"{args.title} (Mean +/- Std)")
    _write_summary_json(
        raw_rows,
        aggregate_rows,
        output_dir / "checkpoint_stage_report.json",
        checkpoints,
        max(1, int(args.window)),
    )

    print(f"saved_dir={output_dir}")


if __name__ == "__main__":
    main()
