#!/usr/bin/env python3
"""Launch the formal 3-seed ablation runs sequentially with resumable logging."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MethodSpec:
    key: str
    experiment_template: str
    extra_args: List[str]


METHOD_SPECS = [
    MethodSpec(
        key="full",
        experiment_template="paper_ablation_full_10000ep_seed{seed}_{date_tag}",
        extra_args=["--cmtm_mode", "full", "--mask_mode", "full", "--scale_mode", "curriculum"],
    ),
    MethodSpec(
        key="cmtm_stateless",
        experiment_template="paper_ablation_cmtm_stateless_10000ep_seed{seed}_{date_tag}",
        extra_args=["--cmtm_mode", "stateless", "--mask_mode", "full", "--scale_mode", "curriculum"],
    ),
    MethodSpec(
        key="mask_obs_only",
        experiment_template="paper_ablation_mask_obs_only_10000ep_seed{seed}_{date_tag}",
        extra_args=["--cmtm_mode", "full", "--mask_mode", "obs_only", "--scale_mode", "curriculum"],
    ),
    MethodSpec(
        key="direct_max",
        experiment_template="paper_ablation_direct_max_10000ep_seed{seed}_{date_tag}",
        extra_args=["--cmtm_mode", "full", "--mask_mode", "full", "--scale_mode", "direct_max"],
    ),
]


def _build_common_args(args: argparse.Namespace) -> List[str]:
    return [
        "-m",
        "dp_lcrl_rl.scripts.train.train_paper_mat",
        "--algorithm_name",
        "mat",
        "--num_agents",
        str(int(args.num_agents)),
        "--min_agents",
        str(int(args.min_agents)),
        "--curriculum_min_agents",
        str(int(args.curriculum_min_agents)),
        "--curriculum_warmup_episodes",
        str(int(args.curriculum_warmup_episodes)),
        "--step_churn_prob",
        str(float(args.step_churn_prob)),
        "--episode_length",
        str(int(args.episode_length)),
        "--n_rollout_threads",
        str(int(args.n_rollout_threads)),
        "--n_eval_rollout_threads",
        str(int(args.n_eval_rollout_threads)),
        "--num_env_steps",
        str(int(args.num_env_steps)),
        "--save_interval",
        str(int(args.save_interval)),
        "--use_wandb",
        "false",
    ]


def _final_checkpoint_episode(args: argparse.Namespace) -> int:
    return max(
        1,
        int(args.num_env_steps) // int(args.episode_length) // max(1, int(args.n_rollout_threads)),
    )


def _build_runs(args: argparse.Namespace) -> List[Dict[str, object]]:
    checkpoint_episode = _final_checkpoint_episode(args)
    common_args = _build_common_args(args)
    runs: List[Dict[str, object]] = []
    for method in METHOD_SPECS:
        for seed in args.seeds:
            experiment_name = method.experiment_template.format(seed=int(seed), date_tag=str(args.date_tag))
            run_dir = REPO_ROOT / "runs" / experiment_name
            model_path = run_dir / "models" / f"transformer_{checkpoint_episode}.pt"
            log_path = Path(args.output_dir) / "logs" / f"{experiment_name}.log"
            cmd = [
                str(Path(args.python).resolve()),
                *common_args,
                "--experiment_name",
                experiment_name,
                "--seed",
                str(int(seed)),
                *method.extra_args,
            ]
            if bool(args.cuda):
                cmd.append("--cuda")
            runs.append(
                {
                    "method": method.key,
                    "seed": int(seed),
                    "experiment_name": experiment_name,
                    "run_dir": str(run_dir),
                    "final_model": str(model_path),
                    "log_path": str(log_path),
                    "command": cmd,
                    "status": "pending",
                    "returncode": None,
                    "started_at": None,
                    "finished_at": None,
                }
            )
    return runs


def _write_manifest(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _launcher_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _run_one(run: Dict[str, object], env: Dict[str, str]) -> int:
    log_path = Path(str(run["log_path"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n=== START {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_file.write("COMMAND: " + " ".join(str(part) for part in run["command"]) + "\n")
        log_file.flush()

        process = subprocess.Popen(
            list(run["command"]),
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        process.wait()
        log_file.write(f"=== END {time.strftime('%Y-%m-%d %H:%M:%S')} rc={process.returncode} ===\n")
        log_file.flush()
        return int(process.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the formal 3-seed ablation experiment sequentially.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python interpreter used for training.")
    parser.add_argument("--date_tag", type=str, default="20260422")
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "reports" / "formal_ablation_3seed_20260422"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--num_agents", type=int, default=30)
    parser.add_argument("--min_agents", type=int, default=20)
    parser.add_argument("--curriculum_min_agents", type=int, default=20)
    parser.add_argument("--curriculum_warmup_episodes", type=int, default=2000)
    parser.add_argument("--step_churn_prob", type=float, default=0.0)
    parser.add_argument("--episode_length", type=int, default=24)
    parser.add_argument("--n_rollout_threads", type=int, default=4)
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1)
    parser.add_argument("--num_env_steps", type=int, default=960000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / "formal_ablation_manifest.json"
    env = _launcher_env()
    runs = _build_runs(args)
    payload: Dict[str, object] = {
        "date_tag": str(args.date_tag),
        "repo_root": str(REPO_ROOT),
        "output_dir": str(output_dir),
        "final_checkpoint_episode": _final_checkpoint_episode(args),
        "runs": runs,
    }
    _write_manifest(manifest_path, payload)

    for run in runs:
        final_model = Path(str(run["final_model"]))
        if final_model.exists():
            run["status"] = "skipped_existing"
            run["returncode"] = 0
            run["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _write_manifest(manifest_path, payload)
            print(f"[skip] {run['experiment_name']} already has {final_model.name}")
            continue

        if args.dry_run:
            print("[dry-run] " + " ".join(str(part) for part in run["command"]))
            continue

        run["status"] = "running"
        run["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_manifest(manifest_path, payload)
        print(f"[launch] {run['experiment_name']}")
        returncode = _run_one(run, env)
        run["returncode"] = int(returncode)
        run["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        run["status"] = "completed" if returncode == 0 else "failed"
        _write_manifest(manifest_path, payload)
        if returncode != 0:
            raise SystemExit(returncode)

    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
