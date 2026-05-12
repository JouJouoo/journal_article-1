#!/usr/bin/env python3
"""Train MAT/MAT-Dec on the paper-aligned DP-LCRL environment."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
for parent in THIS_FILE.parents:
    if (parent / "dp_lcrl_rl" / "__init__.py").exists():
        repo_root = parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        break
else:
    repo_root = THIS_FILE.parent

import dp_lcrl_rl.runner.shared.base_runner as base_runner  # type: ignore
from dp_lcrl_rl.algorithms.baselines.baseline_policy import BaselinePolicy  # type: ignore
from dp_lcrl_rl.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy  # type: ignore
from dp_lcrl_rl.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo  # type: ignore
from dp_lcrl_rl.envs.p2ptrading import MarketSpec, ParallelPaperVecEnv, StorageSpec  # type: ignore
from dp_lcrl_rl.runner.shared.paper_mat_runner import PaperMATRunner  # type: ignore

base_runner.Policy = Policy
base_runner.TrainAlgo = TrainAlgo


def resolve_policy_class(args: argparse.Namespace):
    arch = str(getattr(args, "policy_arch", "transformer")).strip().lower()
    if arch in {"transformer", "dp_lcrl"}:
        return Policy
    if arch in {"mlp_pad", "mappo_shared", "deepsets"}:
        return BaselinePolicy
    raise ValueError(
        f"Unsupported policy_arch={arch}. "
        "Expected one of ['transformer', 'mlp_pad', 'mappo_shared', 'deepsets']."
    )


def configure_policy_class(args: argparse.Namespace):
    policy_class = resolve_policy_class(args)
    base_runner.Policy = policy_class
    return policy_class


def _bool_arg(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_user_path(path_value: str) -> Path:
    raw_path = Path(str(path_value).strip()).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    repo_candidate = (repo_root / raw_path).resolve()
    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return repo_candidate


def _set_global_seeds(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _normalize_experiment_args(args: argparse.Namespace) -> argparse.Namespace:
    args.num_agents = max(1, int(args.num_agents))
    args.min_agents = max(1, min(int(args.min_agents), args.num_agents))
    args.curriculum_min_agents = max(1, min(int(args.curriculum_min_agents), args.num_agents))
    args.cmtm_mode = str(getattr(args, "cmtm_mode", "full")).strip().lower()
    args.mask_mode = str(getattr(args, "mask_mode", "full")).strip().lower()
    args.scale_mode = str(getattr(args, "scale_mode", "curriculum")).strip().lower()
    args.p2p_reward_weight = max(0.0, float(getattr(args, "p2p_reward_weight", 0.10)))
    args.grid_buy_penalty_weight = max(0.0, float(getattr(args, "grid_buy_penalty_weight", 0.10)))
    args.unmatched_penalty_weight = max(0.0, float(getattr(args, "unmatched_penalty_weight", 0.05)))
    args.episode_offset = max(0, int(getattr(args, "episode_offset", 0)))
    args.late_anneal_start_episode = max(0, int(getattr(args, "late_anneal_start_episode", 0)))
    late_lr_final = getattr(args, "late_lr_final", None)
    args.late_lr_final = None if late_lr_final is None else max(0.0, float(late_lr_final))
    late_entropy_final = getattr(args, "late_entropy_coef_final", None)
    args.late_entropy_coef_final = (
        None if late_entropy_final is None else max(0.0, float(late_entropy_final))
    )

    if args.scale_mode == "direct_max":
        args.curriculum_warmup_episodes = 0
        args.curriculum_min_agents = args.num_agents
        args.min_agents = args.num_agents
    else:
        args.min_agents = min(args.min_agents, args.curriculum_min_agents)

    return args


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MAT or MAT-Dec on the dynamic-participation paper-aligned P2P environment."
    )

    parser.add_argument("--exp", type=str, default=None, help="Alias of --experiment_name.")
    parser.add_argument("--agents", type=int, default=None, help="Alias of --num_agents.")
    parser.add_argument("--threads", type=int, default=None, help="Alias of --n_rollout_threads.")
    parser.add_argument("--eval_threads", type=int, default=None, help="Alias of --n_eval_rollout_threads.")
    parser.add_argument("--horizon", type=int, default=None, help="Alias of --episode_length.")
    parser.add_argument("--steps", type=int, default=None, help="Alias of --num_env_steps.")
    parser.add_argument("--algo", type=str, default=None, choices=["mat", "mat_dec"], help="Alias of --algorithm_name.")

    parser.add_argument("--env_name", type=str, default="dp_lcrl_p2p")
    parser.add_argument("--algorithm_name", type=str, default="mat", choices=["mat", "mat_dec"])
    parser.add_argument(
        "--policy_arch",
        type=str,
        default="transformer",
        choices=["transformer", "dp_lcrl", "mlp_pad", "mappo_shared", "deepsets"],
        help="Policy encoder used for Experiment 1 baselines.",
    )
    parser.add_argument("--experiment_name", type=str, default="paper_default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--mps", action="store_true", default=False)
    parser.add_argument("--use_render", type=_bool_arg, default=False)

    parser.add_argument("--num_agents", type=int, default=30)
    parser.add_argument("--min_agents", type=int, default=20)
    parser.add_argument("--curriculum_min_agents", type=int, default=20)
    parser.add_argument(
        "--cmtm_mode",
        type=str,
        default="full",
        choices=["full", "stateless"],
        help="Storage carbon tracing mode: full CMTM or stateless single-period accounting.",
    )
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="full",
        choices=["full", "obs_only"],
        help="Agent-mask mode: full masking or observation-only padding without attention/action masking.",
    )
    parser.add_argument(
        "--scale_mode",
        type=str,
        default="curriculum",
        choices=["curriculum", "direct_max", "random_scale"],
        help="Scale training mode: progressive curriculum, direct training at maximum scale, or random instance-count sampling.",
    )
    parser.add_argument(
        "--no_id_emb",
        action="store_true",
        default=False,
        help="Disable agent identity embedding (ablation). When set, agent_id_emb is skipped.",
    )
    parser.add_argument(
        "--curriculum_warmup_episodes",
        type=int,
        default=2000,
        help="Episodes used to linearly raise the minimum active agent count to num_agents.",
    )
    parser.add_argument("--episode_length", type=int, default=24)
    parser.add_argument("--n_rollout_threads", type=int, default=4)
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1)
    parser.add_argument("--n_render_rollout_threads", type=int, default=0)
    parser.add_argument("--num_env_steps", type=int, default=240_000)
    parser.add_argument(
        "--episode_offset",
        type=int,
        default=0,
        help="Global episode offset used for resumed training, logging, and checkpoint numbering.",
    )

    parser.add_argument("--step_churn_prob", type=float, default=0.0)
    parser.add_argument("--dynamic_carbon_price", type=_bool_arg, default=False)
    parser.add_argument("--carbon_price", type=float, default=0.08)
    parser.add_argument("--carbon_price_alpha", type=float, default=0.2)
    parser.add_argument(
        "--p2p_reward_weight",
        type=float,
        default=0.10,
        help="Reward bonus weight applied to each agent's executed P2P trading volume.",
    )
    parser.add_argument(
        "--grid_buy_penalty_weight",
        type=float,
        default=0.10,
        help="Penalty weight applied to each agent's grid-buy volume in the reward.",
    )
    parser.add_argument(
        "--unmatched_penalty_weight",
        type=float,
        default=0.05,
        help="Penalty weight applied to each agent's unmatched planned P2P volume in the reward.",
    )
    parser.add_argument("--grid_carbon_factor", type=float, default=0.7)
    parser.add_argument("--pv_carbon_factor", type=float, default=0.0)
    parser.add_argument("--buy_price_base", type=float, default=1.0)
    parser.add_argument("--sell_price_base", type=float, default=0.2)
    parser.add_argument("--buy_price_amp", type=float, default=0.25)
    parser.add_argument("--sell_price_amp", type=float, default=0.10)
    parser.add_argument("--spread_min", type=float, default=0.08)
    parser.add_argument("--quote_price_min", type=float, default=0.0)
    parser.add_argument("--quote_price_max", type=float, default=1.5)

    parser.add_argument("--storage_e_max", type=float, default=12.0)
    parser.add_argument("--storage_p_ch_max", type=float, default=3.0)
    parser.add_argument("--storage_p_dis_max", type=float, default=3.0)
    parser.add_argument("--storage_eta_ch", type=float, default=0.95)
    parser.add_argument("--storage_eta_dis", type=float, default=0.95)
    parser.add_argument("--storage_cycle_cost", type=float, default=0.01)
    parser.add_argument("--storage_capacity_variance", type=float, default=0.25)
    parser.add_argument("--storage_capacity_min", type=float, default=6.0)
    parser.add_argument("--storage_capacity_max", type=float, default=18.0)

    parser.add_argument("--pv_peak_scale", type=float, default=7.5)
    parser.add_argument("--pv_phase_jitter", type=float, default=0.15)
    parser.add_argument("--load_phase_jitter", type=float, default=0.25)
    parser.add_argument("--load_base", type=float, default=2.2)
    parser.add_argument("--load_peak", type=float, default=4.5)

    parser.add_argument("--use_centralized_V", type=_bool_arg, default=True)
    parser.add_argument("--use_obs_instead_of_state", type=_bool_arg, default=False)
    parser.add_argument("--use_linear_lr_decay", type=_bool_arg, default=True)
    parser.add_argument("--recurrent_N", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--use_gae", type=_bool_arg, default=True)
    parser.add_argument("--use_valuenorm", type=_bool_arg, default=True)
    parser.add_argument("--use_proper_time_limits", type=_bool_arg, default=False)

    parser.add_argument("--use_wandb", type=_bool_arg, default=False)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--use_eval", type=_bool_arg, default=False)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--summary_filename", type=str, default="paper_training_summary.html")
    parser.add_argument("--export_summary_interval", type=int, default=0)
    parser.add_argument("--export_last_episode_summary", type=_bool_arg, default=False)
    parser.add_argument("--last_episode_summary_filename", type=str, default="paper_training_summary.last.html")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--opti_eps", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--ppo_epoch", type=int, default=5)
    parser.add_argument("--num_mini_batch", type=int, default=2)
    parser.add_argument("--entropy_coef", type=float, default=0.03)
    parser.add_argument(
        "--late_anneal_start_episode",
        type=int,
        default=0,
        help="Episode index where late-stage linear annealing of lr/entropy begins. 0 disables it.",
    )
    parser.add_argument(
        "--late_lr_final",
        type=float,
        default=None,
        help="Target learning rate reached at the final episode during late-stage annealing.",
    )
    parser.add_argument(
        "--late_entropy_coef_final",
        type=float,
        default=None,
        help="Target entropy coefficient reached at the final episode during late-stage annealing.",
    )
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--use_clipped_value_loss", type=_bool_arg, default=True)
    parser.add_argument("--use_huber_loss", type=_bool_arg, default=True)
    parser.add_argument("--use_value_active_masks", type=_bool_arg, default=False)
    parser.add_argument("--use_policy_active_masks", type=_bool_arg, default=False)
    parser.add_argument("--use_popart", type=_bool_arg, default=False)
    parser.add_argument("--data_chunk_length", type=int, default=10)
    parser.add_argument("--huber_delta", type=float, default=10.0)
    parser.add_argument("--use_max_grad_norm", type=_bool_arg, default=True)
    parser.add_argument("--use_recurrent_policy", type=_bool_arg, default=True)
    parser.add_argument("--use_naive_recurrent_policy", type=_bool_arg, default=False)

    parser.add_argument("--encode_state", type=_bool_arg, default=False)
    parser.add_argument("--n_block", type=int, default=1)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--dec_actor", type=_bool_arg, default=False)
    parser.add_argument("--share_actor", type=_bool_arg, default=False)
    parser.add_argument("--log_std_min", type=float, default=-2.0)
    parser.add_argument("--log_std_max", type=float, default=2.0)

    parser.add_argument("--stacked_frames", type=int, default=1)
    parser.add_argument("--layer_N", type=int, default=2)
    parser.add_argument("--use_feature_normalization", type=_bool_arg, default=False)
    parser.add_argument("--use_orthogonal", type=_bool_arg, default=True)
    parser.add_argument("--use_ReLU", type=_bool_arg, default=True)

    return parser


def _apply_cli_aliases(args: argparse.Namespace) -> None:
    if getattr(args, "exp", None):
        args.experiment_name = str(args.exp)
    if getattr(args, "agents", None) is not None:
        args.num_agents = int(args.agents)
    if getattr(args, "threads", None) is not None:
        args.n_rollout_threads = int(args.threads)
    if getattr(args, "eval_threads", None) is not None:
        args.n_eval_rollout_threads = int(args.eval_threads)
    if getattr(args, "horizon", None) is not None:
        args.episode_length = int(args.horizon)
    if getattr(args, "steps", None) is not None:
        args.num_env_steps = int(args.steps)
    if getattr(args, "algo", None):
        args.algorithm_name = str(args.algo)
    if str(getattr(args, "algorithm_name", "mat")).strip().lower() == "mat_dec":
        args.dec_actor = True


def make_env(args: argparse.Namespace, n_threads: int) -> ParallelPaperVecEnv:
    storage_spec = StorageSpec(
        eta_ch=float(args.storage_eta_ch),
        eta_dis=float(args.storage_eta_dis),
        e_max=float(args.storage_e_max),
        p_ch_max=float(args.storage_p_ch_max),
        p_dis_max=float(args.storage_p_dis_max),
        cycle_cost=float(args.storage_cycle_cost),
    )
    market_spec = MarketSpec(
        base_buy=float(args.buy_price_base),
        base_sell=float(args.sell_price_base),
        buy_amp=float(args.buy_price_amp),
        sell_amp=float(args.sell_price_amp),
        spread_min=float(args.spread_min),
        carbon_price=float(args.carbon_price),
        carbon_price_alpha=float(args.carbon_price_alpha),
        dynamic_carbon_price=bool(args.dynamic_carbon_price),
        grid_carbon_factor=float(args.grid_carbon_factor),
        pv_carbon_factor=float(args.pv_carbon_factor),
    )
    return ParallelPaperVecEnv(
        n_threads=int(n_threads),
        num_agents=int(args.num_agents),
        seed=int(args.seed),
        horizon=int(args.episode_length),
        min_agents=int(args.min_agents),
        step_churn_prob=float(args.step_churn_prob),
        storage_spec=storage_spec,
        market_spec=market_spec,
        price_quote_range=(
            float(min(args.quote_price_min, args.quote_price_max)),
            float(max(args.quote_price_min, args.quote_price_max)),
        ),
        storage_capacity_variance=float(args.storage_capacity_variance),
        storage_capacity_range=(
            float(min(args.storage_capacity_min, args.storage_capacity_max)),
            float(max(args.storage_capacity_min, args.storage_capacity_max)),
        ),
        pv_peak_scale=float(args.pv_peak_scale),
        pv_phase_jitter=float(args.pv_phase_jitter),
        load_phase_jitter=float(args.load_phase_jitter),
        load_base=float(args.load_base),
        load_peak=float(args.load_peak),
        dynamic_carbon_price=bool(args.dynamic_carbon_price),
        cmtm_mode=str(args.cmtm_mode),
        mask_mode=str(args.mask_mode),
        p2p_reward_weight=float(args.p2p_reward_weight),
        grid_buy_penalty_weight=float(args.grid_buy_penalty_weight),
        unmatched_penalty_weight=float(args.unmatched_penalty_weight),
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _apply_cli_aliases(args)
    _normalize_experiment_args(args)
    _set_global_seeds(args.seed)
    configure_policy_class(args)

    if args.model_dir:
        model_path = _resolve_user_path(args.model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"model_dir path not found: {model_path}")
        args.model_dir = str(model_path)

    if args.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.set_num_threads(1)

    envs = make_env(args, args.n_rollout_threads)
    eval_envs = make_env(args, args.n_eval_rollout_threads) if args.use_eval else None

    run_dir = Path(repo_root) / "runs" / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "all_args": args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "num_agents": args.num_agents,
        "run_dir": run_dir,
    }

    runner = PaperMATRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
