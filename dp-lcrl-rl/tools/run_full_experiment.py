#!/usr/bin/env python3
"""Parallel launcher for the complete DP-LCRL experiment plan.

Runs training for all methods/seeds in parallel batches, then all evaluations.
Usage:
    python tools/run_full_experiment.py --cuda
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "C:\\Users\\zrway\\.conda\\envs\\DP-LCRL\\python.exe"
DATE_TAG = "20260430"


@dataclass
class MethodSpec:
    key: str
    template: str
    extra_args: List[str]
    seeds: List[int]


METHODS = [
    MethodSpec("full", "paper_ablation_full_10000ep_seed{seed}_{tag}",
               ["--cmtm_mode", "full", "--mask_mode", "full", "--scale_mode", "curriculum"],
               seeds=[42, 43, 44]),
    MethodSpec("cmtm_stateless", "paper_ablation_cmtm_stateless_10000ep_seed{seed}_{tag}",
               ["--cmtm_mode", "stateless", "--mask_mode", "full", "--scale_mode", "curriculum"],
               seeds=[42]),
    MethodSpec("mask_obs_only", "paper_ablation_mask_obs_only_10000ep_seed{seed}_{tag}",
               ["--cmtm_mode", "full", "--mask_mode", "obs_only", "--scale_mode", "curriculum"],
               seeds=[42]),
    MethodSpec("direct_max", "paper_ablation_direct_max_10000ep_seed{seed}_{tag}",
               ["--cmtm_mode", "full", "--mask_mode", "full", "--scale_mode", "direct_max"],
               seeds=[42]),
    MethodSpec("mask_obs_only_direct", "paper_mask_ablation_obs_only_direct_10000ep_seed{seed}_{tag}",
               ["--cmtm_mode", "full", "--mask_mode", "obs_only", "--scale_mode", "direct_max"],
               seeds=[42]),
]

N_PARALLEL = 3  # Optimal parallelism on RTX 3090


def build_common_args() -> List[str]:
    return [
        PYTHON, "-m", "dp_lcrl_rl.scripts.train.train_paper_mat",
        "--algorithm_name", "mat",
        "--num_agents", "30",
        "--min_agents", "20",
        "--curriculum_min_agents", "20",
        "--curriculum_warmup_episodes", "2000",
        "--step_churn_prob", "0.0",
        "--episode_length", "24",
        "--n_rollout_threads", "4",
        "--n_eval_rollout_threads", "1",
        "--num_env_steps", "120000",
        "--save_interval", "250",
        "--use_wandb", "false",
    ]


def build_train_runs(args: argparse.Namespace) -> List[Dict]:
    tag = str(args.date_tag)
    common = build_common_args()
    runs = []
    for method in METHODS:
        for seed in method.seeds:
            exp_name = method.template.format(seed=seed, tag=tag)
            run_dir = REPO_ROOT / "runs" / exp_name
            model_path = run_dir / "models" / f"transformer_{_final_checkpoint_episode()}.pt"
            log_dir = REPO_ROOT / "reports" / f"full_experiment_{tag}" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{exp_name}.log"
            cmd = [
                *common,
                "--experiment_name", exp_name,
                "--seed", str(seed),
                *method.extra_args,
            ]
            if args.cuda:
                cmd.append("--cuda")
            runs.append({
                "method": method.key,
                "seed": seed,
                "experiment_name": exp_name,
                "run_dir": str(run_dir),
                "final_model": str(model_path),
                "log_path": str(log_path),
                "command": cmd,
                "status": "pending",
                "returncode": None,
            })
    return runs


def _final_checkpoint_episode() -> int:
    return 1250  # 120000 / 24 / 4


def build_eval_commands(args: argparse.Namespace) -> Dict[str, List[str]]:
    tag = str(args.date_tag)
    manifest = str(REPO_ROOT / "reports" / f"full_experiment_{tag}" / "formal_ablation_manifest.json")

    cmds = {}

    # Eval 1: Unified ablation (agent count sweep)
    cmds["unified_ablation"] = [
        PYTHON, "-m", "dp_lcrl_rl.scripts.eval.eval_formal_ablation_unified",
        "--manifest", manifest,
        "--output_dir", str(REPO_ROOT / "reports" / f"formal_ablation_unified_eval_{tag}"),
        "--agent_count_min", "1", "--agent_count_max", "30",
        "--eval_episodes", "20",
    ]

    # Eval 2: Dynamic participation
    cmds["dynamic_eval"] = [
        PYTHON, "-m", "dp_lcrl_rl.scripts.eval.eval_formal_ablation_dynamic",
        "--manifest", manifest,
        "--output_dir", str(REPO_ROOT / "reports" / f"formal_ablation_dynamic_eval_{tag}"),
        "--eval_episodes", "20",
    ]

    # Eval 3: CMTM memory validation
    cmds["cmtm_validation"] = [
        PYTHON, "-m", "dp_lcrl_rl.scripts.eval.eval_cmtm_memory_validation",
        "--manifest", manifest,
        "--output_dir", str(REPO_ROOT / "reports" / f"cmtm_memory_validation_{tag}"),
    ]

    # Eval 4: Structured mask ablation
    full_seeds = [m.seeds for m in METHODS if m.key == "direct_max"][0]
    mask_seeds = [m.seeds for m in METHODS if m.key == "mask_obs_only_direct"][0]
    full_dirs = [
        str(REPO_ROOT / "runs" / f"paper_ablation_direct_max_10000ep_seed{s}_{tag}")
        for s in full_seeds
    ]
    mask_dirs = [
        str(REPO_ROOT / "runs" / f"paper_mask_ablation_obs_only_direct_10000ep_seed{s}_{tag}")
        for s in mask_seeds
    ]
    cmds["mask_ablation"] = [
        PYTHON, "-m", "dp_lcrl_rl.scripts.eval.eval_formal_mask_ablation",
        "--output_dir", str(REPO_ROOT / "reports" / f"formal_mask_ablation_{tag}"),
        "--checkpoint_episode", "1250",
        *[a for d in full_dirs for a in ("--full_run_dir", d)],
        *[a for d in mask_dirs for a in ("--mask_run_dir", d)],
    ]

    # Eval 5: Fixed testset convergence (on Full method only)
    full_seeds = [m.seeds for m in METHODS if m.key == "full"][0]
    cmds["convergence"] = [
        PYTHON, "-m", "dp_lcrl_rl.scripts.eval.eval_fixed_testset_convergence",
        *[a for s in full_seeds for a in (
            "--run_dir", str(REPO_ROOT / "runs" / f"paper_ablation_full_10000ep_seed{s}_{tag}"))],
        "--output_dir", str(REPO_ROOT / "reports" / f"fixed_testset_convergence_{tag}"),
    ] + [a for s in full_seeds for a in ("--label", f"Full seed{s}")]

    return cmds


def run_training_process(run: Dict) -> Dict:
    """Run one training process and return updated status."""
    log_path = Path(str(run["log_path"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"

    exp_name = str(run["experiment_name"])
    cmd = list(run["command"])

    print(f"[TRAIN] Starting {exp_name} ...")
    start = time.time()

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"START: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("CMD: " + " ".join(cmd) + "\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            log.write(line)
            print(f"  [{exp_name}] {line}", end="")
        proc.wait()

    elapsed = time.time() - start
    rc = proc.returncode
    status = "completed" if rc == 0 else "failed"
    run["status"] = status
    run["returncode"] = rc
    run["elapsed_sec"] = round(elapsed, 1)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"END: {time.strftime('%Y-%m-%d %H:%M:%S')} rc={rc} elapsed={elapsed:.1f}s\n")

    print(f"[TRAIN] {exp_name} -> {status} ({elapsed:.0f}s)")
    return run


def write_manifest(path: Path, runs: Sequence[Dict], status: str = "running") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date_tag": DATE_TAG,
        "repo_root": str(REPO_ROOT),
        "output_dir": str(path.parent),
        "final_checkpoint_episode": _final_checkpoint_episode(),
        "status": status,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": list(runs),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_evaluation(name: str, cmd: List[str]) -> Dict:
    """Run one evaluation script and return status."""
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"

    log_dir = REPO_ROOT / "reports" / f"full_experiment_{DATE_TAG}" / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    print(f"\n{'='*60}")
    print(f"[EVAL] Starting {name} ...")
    print(f"{'='*60}")
    start = time.time()

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"START: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("CMD: " + " ".join(cmd) + "\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            log.write(line)
            print(f"  [{name}] {line}", end="")
        proc.wait()

    elapsed = time.time() - start
    rc = proc.returncode
    status = "completed" if rc == 0 else "failed"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"END: rc={rc} elapsed={elapsed:.1f}s\n")

    print(f"[EVAL] {name} -> {status} ({elapsed:.0f}s)")
    return {"name": name, "status": status, "returncode": rc, "elapsed_sec": round(elapsed, 1)}


def main():
    parser = argparse.ArgumentParser(description="Run full DP-LCRL experiment plan in parallel.")
    parser.add_argument("--cuda", action="store_true", default=True, help="Use CUDA GPU")
    parser.add_argument("--date_tag", default=DATE_TAG)
    parser.add_argument("--n_parallel", type=int, default=N_PARALLEL, help="Concurrent training processes")
    parser.add_argument("--skip_train", action="store_true", help="Skip training, run only evaluations")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluations, run only training")
    args = parser.parse_args()

    tag = str(args.date_tag)
    report_dir = REPO_ROOT / "reports" / f"full_experiment_{tag}"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / "formal_ablation_manifest.json"

    total_start = time.time()

    # =========================================================
    # PHASE 1: TRAINING
    # =========================================================
    if not args.skip_train:
        runs = build_train_runs(args)
        print(f"\n{'='*60}")
        print(f"PHASE 1: Training {len(runs)} runs in parallel batches of {args.n_parallel}")
        print(f"{'='*60}")

        # Check which runs already have final models (to skip)
        pending_runs = []
        for run in runs:
            model = Path(str(run["final_model"]))
            if model.exists():
                run["status"] = "skipped"
                run["returncode"] = 0
                print(f"  [SKIP] {run['experiment_name']} - model exists")
            else:
                pending_runs.append(run)

        # Train in parallel batches
        completed = [r for r in runs if r["status"] == "skipped"]
        batch_num = 0
        while pending_runs:
            batch = pending_runs[:args.n_parallel]
            pending_runs = pending_runs[args.n_parallel:]
            batch_num += 1
            print(f"\n--- Batch {batch_num}: {len(batch)} runs ---")
            for r in batch:
                print(f"  {r['experiment_name']}")

            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {executor.submit(run_training_process, r): r for r in batch}
                for future in as_completed(futures):
                    result = future.result()
                    completed.append(result)
                    write_manifest(manifest_path, completed, status="training")

            write_manifest(manifest_path, completed, status="training_partial")

        # Final training status
        successes = sum(1 for r in completed if r["returncode"] == 0)
        failures = sum(1 for r in completed if r["returncode"] != 0)
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE: {successes} success, {failures} failed")
        print(f"{'='*60}")

        if failures > 0:
            print("FAILED RUNS:")
            for r in completed:
                if r["returncode"] != 0:
                    print(f"  {r['experiment_name']} rc={r['returncode']}")

        write_manifest(manifest_path, completed, status="completed")
    else:
        print("[SKIP] Training skipped (--skip_train)")
        # Load existing manifest
        if manifest_path.exists():
            completed = json.loads(manifest_path.read_text(encoding="utf-8")).get("runs", [])
        else:
            completed = []

    # =========================================================
    # PHASE 2: EVALUATION
    # =========================================================
    if not args.skip_eval:
        eval_cmds = build_eval_commands(args)
        print(f"\n{'='*60}")
        print(f"PHASE 2: Running {len(eval_cmds)} evaluations")
        print(f"{'='*60}")

        # Run evaluations sequentially (each uses CPU heavily for env simulation)
        # But some can be parallelized
        eval_results = []

        # Run the lightweight evals first in parallel
        parallel_evals = ["dynamic_eval", "cmtm_validation"]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(run_evaluation, name, eval_cmds[name]): name
                for name in parallel_evals if name in eval_cmds
            }
            for future in as_completed(futures):
                eval_results.append(future.result())

        # Run the heavier evals
        for name in ["unified_ablation", "mask_ablation", "convergence"]:
            if name in eval_cmds:
                result = run_evaluation(name, eval_cmds[name])
                eval_results.append(result)

        successes = sum(1 for r in eval_results if r["status"] == "completed")
        failures = sum(1 for r in eval_results if r["status"] == "failed")
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE: {successes} success, {failures} failed")
        if failures > 0:
            for r in eval_results:
                if r["status"] != "completed":
                    print(f"  FAILED: {r['name']} rc={r['returncode']}")
        print(f"{'='*60}")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"FULL EXPERIMENT COMPLETE in {total_elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    print(f"Manifest: {manifest_path}")
    print(f"Reports: {report_dir}")


if __name__ == "__main__":
    main()
