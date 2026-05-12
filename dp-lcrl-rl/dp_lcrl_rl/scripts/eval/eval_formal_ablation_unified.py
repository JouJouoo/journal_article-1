#!/usr/bin/env python3
"""Unified fixed-test evaluation for the formal 3-seed ablation checkpoints."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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
    _bind_fixed_active_count,
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
    parser.description = "Evaluate formal ablation checkpoints under one unified full evaluator."
    parser.add_argument("--manifest", required=True, help="formal_ablation_manifest.json.")
    parser.add_argument("--output_dir", default="reports/formal_ablation_unified_eval", help="Output directory.")
    parser.add_argument("--report_name", default="formal_ablation_unified_eval", help="Output file prefix.")
    parser.add_argument("--checkpoint_episode", type=int, default=None, help="Defaults to latest checkpoint per run.")
    parser.add_argument("--agent_count_min", type=int, default=1)
    parser.add_argument("--agent_count_max", type=int, default=None)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--fixed_eval_seed", type=int, default=20260425)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=METHOD_ORDER,
        choices=METHOD_ORDER,
        help="Subset of methods to evaluate.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute rows already present in the raw CSV.")
    args = parser.parse_args()
    _apply_cli_aliases(args)

    # The evaluator is intentionally unified across all checkpoints.
    args.cmtm_mode = "full"
    args.mask_mode = "full"
    args.scale_mode = "curriculum"
    _normalize_experiment_args(args)
    args.num_agents = max(1, int(args.num_agents))
    args.agent_count_min = max(1, int(args.agent_count_min))
    max_count = int(args.num_agents) if args.agent_count_max is None else int(args.agent_count_max)
    args.agent_count_max = max(args.agent_count_min, min(max_count, int(args.num_agents)))
    args.eval_episodes = max(1, int(args.eval_episodes))
    args.n_eval_rollout_threads = 1
    args.n_rollout_threads = 1
    args.step_churn_prob = 0.0
    args.min_agents = args.num_agents
    args.curriculum_min_agents = args.num_agents
    args.curriculum_warmup_episodes = 0
    args.use_eval = False
    args.save_interval = 0
    args.cuda = False
    return args


def _output_dir(args: argparse.Namespace) -> Path:
    path = Path(args.output_dir).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_manifest(path: Path, methods: Sequence[str]) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = []
    wanted = set(methods)
    for item in payload.get("runs", []):
        method = str(item.get("method", ""))
        if method not in wanted:
            continue
        if item.get("status") != "completed":
            raise RuntimeError(f"Run is not completed: {item.get('experiment_name')} status={item.get('status')}")
        run_dir = Path(str(item["run_dir"])).expanduser().resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        runs.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "seed": int(item["seed"]),
                "experiment_name": str(item["experiment_name"]),
                "run_dir": run_dir,
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
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row_key(row: dict[str, object]) -> tuple[str, int, int]:
    return (str(row["method"]), int(row["seed"]), int(row["active_agent_count"]))


def _active_counts(args: argparse.Namespace) -> list[int]:
    return list(range(int(args.agent_count_min), int(args.agent_count_max) + 1))


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1 if arr.size > 1 else 0))


def _aggregate_by_count(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault((str(row["method"]), int(row["active_agent_count"])), []).append(row)

    out: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        counts = sorted({count for (m, count) in buckets if m == method})
        for count in counts:
            bucket = buckets[(method, count)]
            result: dict[str, object] = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "active_agent_count": int(count),
                "seed_count": len(bucket),
                "eval_episodes_per_seed": int(bucket[0]["eval_episodes"]),
            }
            for key in ("reward_mean", "p2p_mean", "grid_buy_mean", "grid_sell_mean", "grid_trade_mean", "carbon_mean"):
                mean, std = _mean_std(float(item[key]) for item in bucket)
                result[key] = mean
                result[key.replace("_mean", "_std")] = std
            out.append(result)
    return out


def _aggregate_overall(rows_by_count: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[str, list[dict[str, object]]] = {}
    for row in rows_by_count:
        buckets.setdefault(str(row["method"]), []).append(row)

    out: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        if method not in buckets:
            continue
        bucket = buckets[method]
        result: dict[str, object] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "agent_count_points": len(bucket),
        }
        for key in ("reward_mean", "p2p_mean", "grid_trade_mean", "carbon_mean"):
            mean, std = _mean_std(float(item[key]) for item in bucket)
            result[key] = mean
            result[key.replace("_mean", "_std")] = std
        out.append(result)
    return out


def _write_word_table(path: Path, overall_rows: Sequence[dict[str, object]]) -> None:
    headers = ["Method", "Reward", "P2P Volume", "Grid Trade", "Carbon Resp."]
    body = []
    for row in overall_rows:
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
            "Notes: all checkpoints are evaluated under the same full CMTM and full mask evaluator.",
            "Values are mean +/- std over active-agent counts after seed aggregation at each count.",
            "Grid Trade = grid_buy_mean + grid_sell_mean.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_by_count(rows_by_count: Sequence[dict[str, object]], output_path: Path) -> None:
    colors = {
        "full": "#111111",
        "cmtm_stateless": "#d94801",
        "mask_obs_only": "#2171b5",
        "direct_max": "#238b45",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), dpi=220)
    for ax, (metric, (title, higher_is_better)) in zip(axes.flat, METRICS.items()):
        for method in METHOD_ORDER:
            bucket = [row for row in rows_by_count if row["method"] == method]
            if not bucket:
                continue
            bucket = sorted(bucket, key=lambda item: int(item["active_agent_count"]))
            xs = np.asarray([int(item["active_agent_count"]) for item in bucket], dtype=np.float64)
            mean = np.asarray([float(item[metric]) for item in bucket], dtype=np.float64)
            std_key = metric.replace("_mean", "_std")
            std = np.asarray([float(item[std_key]) for item in bucket], dtype=np.float64)
            ax.plot(xs, mean, marker="o", markersize=3.2, linewidth=2.0, color=colors[method], label=METHOD_LABELS[method])
            ax.fill_between(xs, mean - std, mean + std, color=colors[method], alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Active agent count")
        ax.set_ylabel("Mean value")
        ax.grid(True, alpha=0.25)
        if not higher_is_better:
            ax.text(0.02, 0.94, "lower is better", transform=ax.transAxes, fontsize=8, color="#555555")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Unified Formal Ablation Evaluation", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_status(path: Path, payload: dict[str, object]) -> None:
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _refresh_outputs(output_dir: Path, prefix: str, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    by_count = _aggregate_by_count(rows)
    overall = _aggregate_overall(by_count)
    _write_csv(output_dir / f"{prefix}_by_count.csv", by_count)
    _write_csv(output_dir / f"{prefix}_overall.csv", overall)
    _write_word_table(output_dir / f"{prefix}_word_table.txt", overall)
    _plot_by_count(by_count, output_dir / "figures" / f"{prefix}_by_count.png")


def main() -> None:
    args = parse_args()
    _configure_low_resource_runtime()
    _set_global_seeds(int(args.fixed_eval_seed))

    output_dir = _output_dir(args)
    prefix = str(args.report_name)
    raw_csv = output_dir / f"{prefix}_raw.csv"
    status_json = output_dir / f"{prefix}_status.json"
    manifest_path = Path(args.manifest).expanduser().resolve()
    runs = _load_manifest(manifest_path, args.methods)
    counts = _active_counts(args)
    total = len(runs) * len(counts)

    existing_rows = [] if args.force else _read_existing(raw_csv)
    done_keys = {_row_key(row) for row in existing_rows}
    rows: list[dict[str, object]] = list(existing_rows)
    device = torch.device("cpu")

    _write_status(
        status_json,
        {
            "status": "running",
            "total_rows": total,
            "completed_rows": len(done_keys),
            "failed_rows": 0,
            "active": None,
            "output_dir": str(output_dir),
            "raw_csv": str(raw_csv),
            "eval_episodes": int(args.eval_episodes),
            "active_counts": counts,
            "methods": list(args.methods),
        },
    )

    failures: list[dict[str, object]] = []
    for run in runs:
        checkpoint = _resolve_checkpoint_path(Path(run["run_dir"]), args.checkpoint_episode)
        for active_count in counts:
            key = (str(run["method"]), int(run["seed"]), int(active_count))
            if key in done_keys:
                continue
            active_payload = {
                "method": run["method"],
                "seed": int(run["seed"]),
                "active_agent_count": int(active_count),
                "checkpoint": str(checkpoint),
            }
            _write_status(
                status_json,
                {
                    "status": "running",
                    "total_rows": total,
                    "completed_rows": len(done_keys),
                    "failed_rows": len(failures),
                    "active": active_payload,
                    "output_dir": str(output_dir),
                    "raw_csv": str(raw_csv),
                    "eval_episodes": int(args.eval_episodes),
                    "active_counts": counts,
                    "methods": list(args.methods),
                    "failures": failures[-10:],
                },
            )
            try:
                eval_args = copy.deepcopy(args)
                eval_args.seed = int(args.fixed_eval_seed) + int(active_count)
                eval_args.min_agents = int(active_count)
                eval_args.step_churn_prob = 0.0
                _set_global_seeds(eval_args.seed)
                envs, policy = _build_policy(eval_args, device)
                _bind_fixed_active_count(envs, active_count)
                try:
                    policy.restore(str(checkpoint))
                    policy.eval()
                    episode_summaries = _collect_eval_metrics(
                        args=eval_args,
                        policy=policy,
                        envs=envs,
                        eval_episodes=int(args.eval_episodes),
                    )
                finally:
                    envs.close()
                metrics = _aggregate_eval_summaries(episode_summaries)
                row = {
                    "method": run["method"],
                    "method_label": run["method_label"],
                    "seed": int(run["seed"]),
                    "experiment_name": run["experiment_name"],
                    "active_agent_count": int(active_count),
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
                done_keys.add(key)
                _write_csv(raw_csv, rows)
                _refresh_outputs(output_dir, prefix, rows)
                print(
                    "[UnifiedEval] {method} seed={seed} agents={agents} reward={reward:.4f} p2p={p2p:.4f} grid={grid:.4f} carbon={carbon:.4f}".format(
                        method=run["method_label"],
                        seed=int(run["seed"]),
                        agents=int(active_count),
                        reward=float(row["reward_mean"]),
                        p2p=float(row["p2p_mean"]),
                        grid=float(row["grid_trade_mean"]),
                        carbon=float(row["carbon_mean"]),
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failure = {**active_payload, "error": repr(exc)}
                failures.append(failure)
                _write_status(
                    status_json,
                    {
                        "status": "failed",
                        "total_rows": total,
                        "completed_rows": len(done_keys),
                        "failed_rows": len(failures),
                        "active": active_payload,
                        "output_dir": str(output_dir),
                        "raw_csv": str(raw_csv),
                        "eval_episodes": int(args.eval_episodes),
                        "active_counts": counts,
                        "methods": list(args.methods),
                        "failures": failures,
                    },
                )
                raise

    _refresh_outputs(output_dir, prefix, rows)
    _write_status(
        status_json,
        {
            "status": "completed",
            "total_rows": total,
            "completed_rows": len(done_keys),
            "failed_rows": len(failures),
            "active": None,
            "output_dir": str(output_dir),
            "raw_csv": str(raw_csv),
            "eval_episodes": int(args.eval_episodes),
            "active_counts": counts,
            "methods": list(args.methods),
            "failures": failures,
        },
    )
    print(f"saved_dir={output_dir}", flush=True)
    print(f"status_json={status_json}", flush=True)


if __name__ == "__main__":
    main()
