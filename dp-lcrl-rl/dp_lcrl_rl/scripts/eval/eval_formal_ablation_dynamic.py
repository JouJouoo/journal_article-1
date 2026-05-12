#!/usr/bin/env python3
"""Unified dynamic-participation evaluation for formal ablation checkpoints."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dp_lcrl_rl.scripts.eval.eval_agent_count_sweep import (  # noqa: E402
    _aggregate_eval_summaries,
    _build_policy,
    _collect_eval_metrics,
    _configure_low_resource_runtime,
    _resolve_checkpoint_path,
)
from dp_lcrl_rl.scripts.train.train_paper_mat import (  # noqa: E402
    _apply_cli_aliases,
    _normalize_experiment_args,
    _set_global_seeds,
    build_arg_parser,
)


METHOD_ORDER = ["full", "cmtm_stateless", "mask_obs_only", "direct_max"]
METHOD_LABELS = {
    "full": "Full",
    "cmtm_stateless": "CMTM Stateless",
    "mask_obs_only": "Mask ObsOnly",
    "direct_max": "Direct Max Scale",
}
METRICS = {
    "reward_mean": ("Reward", True),
    "p2p_mean": ("P2P Trading Volume", True),
    "grid_trade_mean": ("Grid Trading Volume", False),
    "carbon_mean": ("Carbon Responsibility", False),
}


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    parser.description = "Evaluate formal ablation checkpoints under dynamic active-agent scenarios."
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", default="reports/formal_ablation_dynamic_eval")
    parser.add_argument("--report_name", default="formal_ablation_dynamic_eval")
    parser.add_argument("--checkpoint_episode", type=int, default=None)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--fixed_eval_seed", type=int, default=20260425)
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Scenario as name:min_agents:step_churn_prob, e.g. churn20:20:0.20.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    _apply_cli_aliases(args)
    args.cmtm_mode = "full"
    args.mask_mode = "full"
    args.scale_mode = "curriculum"
    _normalize_experiment_args(args)
    args.eval_episodes = max(1, int(args.eval_episodes))
    args.n_eval_rollout_threads = 1
    args.n_rollout_threads = 1
    args.save_interval = 0
    args.use_eval = False
    args.cuda = False
    return args


def _output_dir(args: argparse.Namespace) -> Path:
    path = Path(args.output_dir).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _scenarios(args: argparse.Namespace) -> list[dict[str, object]]:
    raw = args.scenario or [
        "episode_var20:20:0.00",
        "churn20_p10:20:0.10",
        "churn20_p20:20:0.20",
        "churn20_p30:20:0.30",
    ]
    parsed: list[dict[str, object]] = []
    for item in raw:
        name, min_agents, churn = str(item).split(":")
        parsed.append(
            {
                "scenario": name,
                "min_agents": max(1, min(int(min_agents), int(args.num_agents))),
                "step_churn_prob": max(0.0, min(1.0, float(churn))),
            }
        )
    return parsed


def _load_manifest(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = []
    for item in payload.get("runs", []):
        method = str(item.get("method", ""))
        if method not in METHOD_ORDER:
            continue
        if item.get("status") != "completed":
            raise RuntimeError(f"Run is not completed: {item.get('experiment_name')} status={item.get('status')}")
        runs.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "seed": int(item["seed"]),
                "experiment_name": str(item["experiment_name"]),
                "run_dir": Path(str(item["run_dir"])).expanduser().resolve(),
            }
        )
    runs.sort(key=lambda row: (METHOD_ORDER.index(str(row["method"])), int(row["seed"])))
    return runs


def _read_existing(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1 if arr.size > 1 else 0))


def _key(row: dict[str, object]) -> tuple[str, int, str]:
    return str(row["method"]), int(row["seed"]), str(row["scenario"])


def _aggregate_by_scenario(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault((str(row["method"]), str(row["scenario"])), []).append(row)
    out = []
    scenarios = sorted({scenario for _, scenario in buckets})
    for scenario in scenarios:
        for method in METHOD_ORDER:
            bucket = buckets.get((method, scenario), [])
            if not bucket:
                continue
            result: dict[str, object] = {
                "scenario": scenario,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "seed_count": len(bucket),
                "min_agents": int(bucket[0]["min_agents"]),
                "step_churn_prob": float(bucket[0]["step_churn_prob"]),
                "eval_episodes_per_seed": int(bucket[0]["eval_episodes"]),
            }
            for metric in ("reward_mean", "p2p_mean", "grid_buy_mean", "grid_sell_mean", "grid_trade_mean", "carbon_mean"):
                mean, std = _mean_std(float(item[metric]) for item in bucket)
                result[metric] = mean
                result[metric.replace("_mean", "_std")] = std
            out.append(result)
    return out


def _aggregate_overall(rows_by_scenario: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[str, list[dict[str, object]]] = {}
    for row in rows_by_scenario:
        buckets.setdefault(str(row["method"]), []).append(row)
    out = []
    for method in METHOD_ORDER:
        bucket = buckets.get(method, [])
        if not bucket:
            continue
        result: dict[str, object] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "scenario_count": len(bucket),
        }
        for metric in ("reward_mean", "p2p_mean", "grid_trade_mean", "carbon_mean"):
            mean, std = _mean_std(float(item[metric]) for item in bucket)
            result[metric] = mean
            result[metric.replace("_mean", "_std")] = std
        out.append(result)
    return out


def _write_word_table(path: Path, rows: Sequence[dict[str, object]]) -> None:
    headers = ["Method", "Reward", "P2P Volume", "Grid Trade", "Carbon Resp."]
    body = []
    for row in rows:
        body.append(
            [
                str(row["method_label"]),
                f"{float(row['reward_mean']):.3f} +/- {float(row['reward_std']):.3f}",
                f"{float(row['p2p_mean']):.3f} +/- {float(row['p2p_std']):.3f}",
                f"{float(row['grid_trade_mean']):.3f} +/- {float(row['grid_trade_std']):.3f}",
                f"{float(row['carbon_mean']):.3f} +/- {float(row['carbon_std']):.3f}",
            ]
        )
    widths = [max(len(str(row[idx])) for row in [headers] + body) for idx in range(len(headers))]
    lines = ["  ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    for row in body:
        lines.append("  ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))
    lines.extend(
        [
            "",
            "Notes: all checkpoints are evaluated with full CMTM and full mask under dynamic active-agent scenarios.",
            "Values are mean +/- std over scenarios after seed aggregation.",
            "Grid Trade = grid_buy_mean + grid_sell_mean.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(rows_by_scenario: Sequence[dict[str, object]], output_path: Path) -> None:
    colors = {
        "full": "#111111",
        "cmtm_stateless": "#d94801",
        "mask_obs_only": "#2171b5",
        "direct_max": "#238b45",
    }
    scenarios = sorted({str(row["scenario"]) for row in rows_by_scenario})
    x_lookup = {scenario: idx for idx, scenario in enumerate(scenarios)}
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), dpi=220)
    for ax, (metric, (title, higher_is_better)) in zip(axes.flat, METRICS.items()):
        for method in METHOD_ORDER:
            bucket = [row for row in rows_by_scenario if row["method"] == method]
            if not bucket:
                continue
            bucket = sorted(bucket, key=lambda item: x_lookup[str(item["scenario"])])
            xs = np.asarray([x_lookup[str(item["scenario"])] for item in bucket], dtype=np.float64)
            mean = np.asarray([float(item[metric]) for item in bucket], dtype=np.float64)
            std = np.asarray([float(item[metric.replace("_mean", "_std")]) for item in bucket], dtype=np.float64)
            ax.plot(xs, mean, marker="o", linewidth=2.0, color=colors[method], label=METHOD_LABELS[method])
            ax.fill_between(xs, mean - std, mean + std, color=colors[method], alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xticks(range(len(scenarios)), scenarios, rotation=15)
        ax.set_ylabel("Mean value")
        ax.grid(True, alpha=0.25)
        if not higher_is_better:
            ax.text(0.02, 0.94, "lower is better", transform=ax.transAxes, fontsize=8, color="#555555")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Dynamic Participation Stress Evaluation", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _refresh(output_dir: Path, prefix: str, rows: Sequence[dict[str, object]]) -> None:
    by_scenario = _aggregate_by_scenario(rows)
    overall = _aggregate_overall(by_scenario)
    _write_csv(output_dir / f"{prefix}_by_scenario.csv", by_scenario)
    _write_csv(output_dir / f"{prefix}_overall.csv", overall)
    _write_word_table(output_dir / f"{prefix}_word_table.txt", overall)
    _plot(by_scenario, output_dir / "figures" / f"{prefix}_by_scenario.png")


def _write_status(path: Path, payload: dict[str, object]) -> None:
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _configure_low_resource_runtime()
    output_dir = _output_dir(args)
    prefix = str(args.report_name)
    raw_csv = output_dir / f"{prefix}_raw.csv"
    status_json = output_dir / f"{prefix}_status.json"
    runs = _load_manifest(Path(args.manifest).expanduser().resolve())
    scenarios = _scenarios(args)
    total = len(runs) * len(scenarios)
    rows = [] if args.force else _read_existing(raw_csv)
    done = {_key(row) for row in rows}
    failures = []
    device = torch.device("cpu")

    for run in runs:
        checkpoint = _resolve_checkpoint_path(Path(run["run_dir"]), args.checkpoint_episode)
        for scenario in scenarios:
            key = (str(run["method"]), int(run["seed"]), str(scenario["scenario"]))
            if key in done:
                continue
            active = {**scenario, "method": run["method"], "seed": int(run["seed"])}
            _write_status(
                status_json,
                {
                    "status": "running",
                    "total_rows": total,
                    "completed_rows": len(done),
                    "failed_rows": len(failures),
                    "active": active,
                    "output_dir": str(output_dir),
                },
            )
            try:
                eval_args = copy.deepcopy(args)
                eval_args.seed = int(args.fixed_eval_seed) + int(run["seed"]) + int(float(scenario["step_churn_prob"]) * 100)
                eval_args.min_agents = int(scenario["min_agents"])
                eval_args.curriculum_min_agents = int(scenario["min_agents"])
                eval_args.step_churn_prob = float(scenario["step_churn_prob"])
                _set_global_seeds(eval_args.seed)
                envs, policy = _build_policy(eval_args, device)
                try:
                    policy.restore(str(checkpoint))
                    policy.eval()
                    summaries = _collect_eval_metrics(eval_args, policy, envs, int(args.eval_episodes))
                finally:
                    envs.close()
                metrics = _aggregate_eval_summaries(summaries)
                row = {
                    "scenario": scenario["scenario"],
                    "min_agents": int(scenario["min_agents"]),
                    "step_churn_prob": float(scenario["step_churn_prob"]),
                    "method": run["method"],
                    "method_label": run["method_label"],
                    "seed": int(run["seed"]),
                    "experiment_name": run["experiment_name"],
                    "checkpoint_path": str(checkpoint),
                    "eval_episodes": int(args.eval_episodes),
                    "fixed_eval_seed": int(args.fixed_eval_seed),
                    **metrics,
                }
                row["grid_trade_mean"] = float(row["grid_buy_mean"]) + float(row["grid_sell_mean"])
                row["grid_trade_std"] = float(
                    np.sqrt(float(row["grid_buy_std"]) ** 2 + float(row["grid_sell_std"]) ** 2)
                )
                rows.append(row)
                done.add(key)
                _write_csv(raw_csv, rows)
                _refresh(output_dir, prefix, rows)
                print(
                    "[DynamicEval] {scenario} {method} seed={seed} reward={reward:.4f} p2p={p2p:.4f} grid={grid:.4f} carbon={carbon:.4f}".format(
                        scenario=scenario["scenario"],
                        method=run["method_label"],
                        seed=int(run["seed"]),
                        reward=float(row["reward_mean"]),
                        p2p=float(row["p2p_mean"]),
                        grid=float(row["grid_trade_mean"]),
                        carbon=float(row["carbon_mean"]),
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append({**active, "error": repr(exc)})
                _write_status(
                    status_json,
                    {
                        "status": "failed",
                        "total_rows": total,
                        "completed_rows": len(done),
                        "failed_rows": len(failures),
                        "active": active,
                        "failures": failures,
                        "output_dir": str(output_dir),
                    },
                )
                raise

    _refresh(output_dir, prefix, rows)
    _write_status(
        status_json,
        {
            "status": "completed",
            "total_rows": total,
            "completed_rows": len(done),
            "failed_rows": len(failures),
            "active": None,
            "failures": failures,
            "output_dir": str(output_dir),
        },
    )
    print(f"saved_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
