"""Aggregate unified evaluation results for the mask-stress experiment."""

from __future__ import annotations

import argparse
import json
import re
from html import escape
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


DIR_PATTERN = re.compile(
    r"^maskstress_eval_(?P<method>[a-z_]+)_seed(?P<seed>\d+)_churn(?P<churn>\d+p\d+)_.*$",
    re.IGNORECASE,
)


def _label_method(method_tag: str) -> str:
    tag = str(method_tag).strip().lower()
    if tag == "full":
        return "Full"
    if tag in {"obs_only", "mask_obsonly"}:
        return "Mask ObsOnly"
    return method_tag


def _parse_churn(churn_tag: str) -> float:
    return float(str(churn_tag).replace("p", "."))


def _load_eval_metrics(summary_path: Path) -> Dict[str, float]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = [item for item in payload.get("episode_summaries", []) if item.get("phase") == "eval"]
    if not episodes:
        raise ValueError(f"No eval episode summaries found in {summary_path}")

    return {
        "reward": float(np.mean([float(item.get("average_global_reward", 0.0)) for item in episodes])),
        "p2p": float(np.mean([float(item.get("p2p_total_volume", 0.0)) for item in episodes])),
        "grid_buy": float(np.mean([float(item.get("grid_buy_total", 0.0)) for item in episodes])),
        "carbon_load": float(np.mean([float(item.get("load_responsibility_total", 0.0)) for item in episodes])),
        "agents": float(np.mean([float(item.get("n_agents_mean", 0.0)) for item in episodes])),
        "episodes": float(len(episodes)),
    }


def _scan_eval_root(eval_root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for summary_path in sorted(eval_root.rglob("paper_eval_summary.json")):
        match = DIR_PATTERN.match(summary_path.parent.name)
        if not match:
            continue
        method_tag = match.group("method").lower()
        seed = int(match.group("seed"))
        churn_tag = match.group("churn")
        metrics = _load_eval_metrics(summary_path)
        records.append(
            {
                "method_tag": method_tag,
                "method": _label_method(method_tag),
                "seed": seed,
                "churn_tag": churn_tag,
                "churn": _parse_churn(churn_tag),
                "summary_path": str(summary_path),
                **metrics,
            }
        )
    if not records:
        raise FileNotFoundError(f"No matching paper_eval_summary.json files found under {eval_root}")
    return records


def _group_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, float], List[Dict[str, object]]] = {}
    for record in records:
        key = (str(record["method"]), float(record["churn"]))
        grouped.setdefault(key, []).append(record)

    rows: List[Dict[str, object]] = []
    for (method, churn), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        row: Dict[str, object] = {
            "Method": method,
            "Churn": churn,
            "Seeds": len(items),
            "Seed List": ",".join(str(int(item["seed"])) for item in sorted(items, key=lambda x: int(x["seed"]))),
        }
        for metric in ("reward", "p2p", "grid_buy", "carbon_load", "agents"):
            values = np.array([float(item[metric]) for item in items], dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=0))
        rows.append(row)
    return rows


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4f}"


def _write_markdown(rows: List[Dict[str, object]], output_path: Path) -> None:
    columns = [
        "Method",
        "Churn",
        "Seeds",
        "Seed List",
        "reward_mean",
        "reward_std",
        "p2p_mean",
        "p2p_std",
        "grid_buy_mean",
        "grid_buy_std",
        "carbon_load_mean",
        "carbon_load_std",
        "agents_mean",
        "agents_std",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col)) for col in columns) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    import csv

    columns = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_html(rows: List[Dict[str, object]], output_path: Path, title: str) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    head = "".join(f"<th>{escape(col)}</th>" for col in cols)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{escape(_fmt(row.get(col)))}</td>" for col in cols) + "</tr>")
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
    <thead><tr>{head}</tr></thead>
    <tbody>{''.join(body)}</tbody>
  </table>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


def _write_json(records: List[Dict[str, object]], rows: List[Dict[str, object]], output_path: Path) -> None:
    payload = {
        "records": records,
        "summary_rows": rows,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot(rows: List[Dict[str, object]], output_path: Path) -> None:
    methods = sorted({str(row["Method"]) for row in rows})
    churns = sorted({float(row["Churn"]) for row in rows})
    color_map = {
        "Full": "#111111",
        "Mask ObsOnly": "#2171b5",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180)
    plot_specs = [
        ("reward", "Eval Reward"),
        ("p2p", "Eval P2P Volume"),
        ("grid_buy", "Eval Grid Buy"),
        ("carbon_load", "Eval Carbon Load"),
    ]

    for ax, (metric, title) in zip(axes.flat, plot_specs):
        for method in methods:
            method_rows = {float(row["Churn"]): row for row in rows if str(row["Method"]) == method}
            xs = np.array([churn for churn in churns if churn in method_rows], dtype=np.float64)
            ys = np.array([float(method_rows[churn][f"{metric}_mean"]) for churn in xs], dtype=np.float64)
            es = np.array([float(method_rows[churn][f"{metric}_std"]) for churn in xs], dtype=np.float64)
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                marker="o",
                linewidth=2.0,
                capsize=4,
                color=color_map.get(method, None),
                label=method,
            )
        ax.set_title(title)
        ax.set_xlabel("Step Churn Probability")
        ax.grid(True, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(methods)), frameon=False)
    fig.suptitle("Mask Stress Unified Evaluation", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the mask-stress unified evaluation report.")
    parser.add_argument("--eval_root", type=str, required=True, help="Root directory containing eval run folders.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write report files into.")
    parser.add_argument("--title", type=str, default="Mask Stress Unified Evaluation")
    args = parser.parse_args()

    eval_root = Path(args.eval_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _scan_eval_root(eval_root)
    rows = _group_records(records)

    _write_csv(rows, output_dir / "mask_stress_eval_summary.csv")
    _write_markdown(rows, output_dir / "mask_stress_eval_summary.md")
    _write_html(rows, output_dir / "mask_stress_eval_summary.html", str(args.title))
    _write_json(records, rows, output_dir / "mask_stress_eval_summary.json")
    _plot(rows, output_dir / "mask_stress_eval_curves.png")

    print(f"saved_dir={output_dir}")


if __name__ == "__main__":
    main()
