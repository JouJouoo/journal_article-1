#!/usr/bin/env python3
"""Run the final structured-mask ablation experiment sequentially.

This script is intentionally conservative: it trains only the missing
obs-only/direct-max runs, one seed at a time, then launches a native evaluator.
Existing direct-max runs are reused as Full / Ours.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "formal_mask_ablation_20260428"
LOG_DIR = REPORT_DIR / "logs"
MANIFEST_PATH = REPORT_DIR / "formal_mask_ablation_manifest.json"
DESIGN_PATH = REPORT_DIR / "formal_mask_ablation_design.txt"

SEEDS = [42, 43, 44]
FULL_RUNS = {
    42: PROJECT_ROOT / "runs" / "paper_ablation_direct_max_10000ep_seed42_20260422",
    43: PROJECT_ROOT / "runs" / "paper_ablation_direct_max_10000ep_seed43_20260422",
    44: PROJECT_ROOT / "runs" / "paper_ablation_direct_max_10000ep_seed44_20260422",
}
MASK_RUNS = {
    seed: PROJECT_ROOT / "runs" / f"paper_mask_ablation_obs_only_direct_10000ep_seed{seed}_20260428"
    for seed in SEEDS
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _final_checkpoint(run_dir: Path) -> Path:
    return run_dir / "models" / "transformer_10000.pt"


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    tmp_dir = PROJECT_ROOT / "tmp_eval_runtime"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"
    env["DP_LCRL_DISABLE_TENSORBOARD"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["TMP"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)
    env["WANDB_DIR"] = str(tmp_dir / "wandb")
    env["WANDB_DATA_DIR"] = str(tmp_dir / "wandb-data")
    env["WANDB_CACHE_DIR"] = str(tmp_dir / "wandb-cache")
    env["WANDB_CONFIG_DIR"] = str(tmp_dir / "wandb-config")
    return env


def _train_command(seed: int) -> list[str]:
    exp = f"paper_mask_ablation_obs_only_direct_10000ep_seed{seed}_20260428"
    return [
        sys.executable,
        "-m",
        "dp_lcrl_rl.scripts.train.train_paper_mat",
        "--algorithm_name",
        "mat",
        "--experiment_name",
        exp,
        "--seed",
        str(seed),
        "--num_agents",
        "30",
        "--min_agents",
        "30",
        "--curriculum_min_agents",
        "30",
        "--curriculum_warmup_episodes",
        "0",
        "--scale_mode",
        "direct_max",
        "--cmtm_mode",
        "full",
        "--mask_mode",
        "obs_only",
        "--step_churn_prob",
        "0.0",
        "--episode_length",
        "24",
        "--n_rollout_threads",
        "4",
        "--n_eval_rollout_threads",
        "1",
        "--num_env_steps",
        "960000",
        "--save_interval",
        "1000",
        "--use_wandb",
        "false",
        "--use_eval",
        "false",
    ]


def _eval_command() -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "dp_lcrl_rl.scripts.eval.eval_formal_mask_ablation",
        "--output_dir",
        str(REPORT_DIR),
        "--report_name",
        "formal_mask_ablation",
        "--checkpoint_episode",
        "10000",
        "--agent_count_min",
        "1",
        "--agent_count_max",
        "30",
        "--eval_episodes",
        "20",
        "--dynamic_eval_episodes",
        "30",
        "--noise_eval_episodes",
        "40",
        "--inactive_noise_std",
        "5.0",
        "--fixed_eval_seed",
        "20260428",
    ]
    for seed in SEEDS:
        cmd.extend(["--full_run_dir", str(FULL_RUNS[seed])])
    for seed in SEEDS:
        cmd.extend(["--mask_run_dir", str(MASK_RUNS[seed])])
    return cmd


def _initial_manifest() -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        runs.append(
            {
                "method": "full_ours",
                "method_label": "Full / Ours",
                "seed": seed,
                "mask_mode": "full",
                "scale_mode": "direct_max",
                "cmtm_mode": "full",
                "run_dir": str(FULL_RUNS[seed]),
                "final_model": str(_final_checkpoint(FULL_RUNS[seed])),
                "status": "completed" if _final_checkpoint(FULL_RUNS[seed]).exists() else "missing",
                "source": "reused_existing_direct_max_run",
            }
        )
    for seed in SEEDS:
        runs.append(
            {
                "method": "without_structured_mask",
                "method_label": "w/o Structured Mask",
                "seed": seed,
                "mask_mode": "obs_only",
                "scale_mode": "direct_max",
                "cmtm_mode": "full",
                "run_dir": str(MASK_RUNS[seed]),
                "final_model": str(_final_checkpoint(MASK_RUNS[seed])),
                "status": "completed" if _final_checkpoint(MASK_RUNS[seed]).exists() else "pending",
                "command": _train_command(seed),
                "log_file": str(LOG_DIR / f"train_obs_only_direct_seed{seed}.log"),
            }
        )
    return {
        "experiment": "formal_mask_ablation_20260428",
        "created_at": _now(),
        "updated_at": _now(),
        "status": "initialized",
        "report_dir": str(REPORT_DIR),
        "design_file": str(DESIGN_PATH),
        "runs": runs,
        "evaluation": {
            "status": "pending",
            "command": _eval_command(),
            "log_file": str(LOG_DIR / "formal_mask_ablation_eval.log"),
        },
    }


def _load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if payload.get("status") == "failed":
            return _initial_manifest()
        return payload
    return _initial_manifest()


def _write_manifest(payload: dict[str, Any]) -> None:
    payload["updated_at"] = _now()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_design() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "Formal structured-mask ablation design (2026-04-28)",
        "",
        "Latest method definition:",
        "Full / Ours = CMTM + structural mask + direct maximum-scale training.",
        "Direct maximum-scale training is no longer treated as an ablation term.",
        "",
        "Compared methods:",
        "1. Full / Ours: reuse completed direct_max seed 42/43/44 checkpoints.",
        "2. w/o Structured Mask: train obs_only + direct_max seed 42/43/44 checkpoints.",
        "",
        "Evaluation protocol:",
        "1. Fixed scale sweep: exactly 1, 2, ..., 30 active agents.",
        "2. Dynamic participation: variable scale and churn scenarios.",
        "3. Inactive-noise stress: inject noise only into inactive padded observations.",
        "4. Training trends: reward, P2P trade, grid trade, and carbon responsibility over training.",
        "",
        "Primary metrics:",
        "Reward mean, P2P trading volume mean, grid trading volume mean, carbon responsibility mean.",
        "",
        "Interpretation:",
        "The mask module is expected to show its value under variable participation and inactive-slot perturbations.",
        "The inactive-noise stress test avoids relying only on zero-padded inactive observations, which made the older obs_only ablation too weak.",
        "",
    ]
    DESIGN_PATH.write_text("\n".join(lines), encoding="utf-8")


def _update_run_status(manifest: dict[str, Any], seed: int, **updates: Any) -> None:
    for row in manifest["runs"]:
        if row.get("method") == "without_structured_mask" and int(row.get("seed", -1)) == int(seed):
            row.update(updates)
            return
    raise KeyError(f"Run entry not found for seed {seed}")


def _run_process(cmd: list[str], log_file: Path, manifest: dict[str, Any], status_label: str) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n\n[{_now()}] START {status_label}\n")
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=_base_env(),
            text=True,
        )
        while True:
            return_code = process.poll()
            if return_code is not None:
                log.write(f"\n[{_now()}] END {status_label} returncode={return_code}\n")
                log.flush()
                return int(return_code)
            manifest["status"] = status_label
            manifest["active_pid"] = int(process.pid)
            _write_manifest(manifest)
            time.sleep(60)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _write_design()

    manifest = _load_manifest()
    _write_manifest(manifest)

    missing_full = [
        str(_final_checkpoint(FULL_RUNS[seed]))
        for seed in SEEDS
        if not _final_checkpoint(FULL_RUNS[seed]).exists()
    ]
    if missing_full:
        manifest["status"] = "failed"
        manifest["failure"] = "Missing reused Full / Ours checkpoints: " + "; ".join(missing_full)
        _write_manifest(manifest)
        raise FileNotFoundError(manifest["failure"])

    for seed in SEEDS:
        final_model = _final_checkpoint(MASK_RUNS[seed])
        if final_model.exists():
            _update_run_status(
                manifest,
                seed,
                status="completed",
                skipped=True,
                finished_at=_now(),
            )
            _write_manifest(manifest)
            continue

        log_file = LOG_DIR / f"train_obs_only_direct_seed{seed}.log"
        _update_run_status(
            manifest,
            seed,
            status="running",
            started_at=_now(),
            log_file=str(log_file),
        )
        manifest["status"] = f"training_seed_{seed}"
        _write_manifest(manifest)
        return_code = _run_process(_train_command(seed), log_file, manifest, f"training_seed_{seed}")
        if return_code != 0:
            _update_run_status(
                manifest,
                seed,
                status="failed",
                finished_at=_now(),
                returncode=return_code,
            )
            manifest["status"] = "failed"
            manifest["failure"] = f"Training failed for seed {seed}, returncode={return_code}"
            _write_manifest(manifest)
            raise RuntimeError(manifest["failure"])
        if not final_model.exists():
            _update_run_status(
                manifest,
                seed,
                status="failed",
                finished_at=_now(),
                returncode=return_code,
            )
            manifest["status"] = "failed"
            manifest["failure"] = f"Training finished but final checkpoint is missing: {final_model}"
            _write_manifest(manifest)
            raise FileNotFoundError(manifest["failure"])
        _update_run_status(
            manifest,
            seed,
            status="completed",
            finished_at=_now(),
            returncode=return_code,
        )
        _write_manifest(manifest)

    eval_log = LOG_DIR / "formal_mask_ablation_eval.log"
    manifest["status"] = "evaluating"
    manifest["evaluation"]["status"] = "running"
    manifest["evaluation"]["started_at"] = _now()
    manifest["evaluation"]["log_file"] = str(eval_log)
    _write_manifest(manifest)
    return_code = _run_process(_eval_command(), eval_log, manifest, "evaluating")
    manifest["evaluation"]["returncode"] = return_code
    manifest["evaluation"]["finished_at"] = _now()
    if return_code != 0:
        manifest["status"] = "failed"
        manifest["evaluation"]["status"] = "failed"
        manifest["failure"] = f"Evaluation failed, returncode={return_code}"
        _write_manifest(manifest)
        raise RuntimeError(manifest["failure"])

    manifest["status"] = "completed"
    manifest["evaluation"]["status"] = "completed"
    manifest["outputs"] = {
        "raw_csv": str(REPORT_DIR / "formal_mask_ablation_raw.csv"),
        "by_scenario_csv": str(REPORT_DIR / "formal_mask_ablation_by_scenario.csv"),
        "overall_csv": str(REPORT_DIR / "formal_mask_ablation_overall.csv"),
        "word_table": str(REPORT_DIR / "formal_mask_ablation_word_table.txt"),
        "figures": [
            str(REPORT_DIR / "figures" / "formal_mask_ablation_scale_sweep.png"),
            str(REPORT_DIR / "figures" / "formal_mask_ablation_stress_summary.png"),
            str(REPORT_DIR / "figures" / "formal_mask_ablation_training_trends.png"),
        ],
    }
    _write_manifest(manifest)


if __name__ == "__main__":
    main()
