from pathlib import Path

import numpy as np
import torch

from dp_lcrl_rl.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from dp_lcrl_rl.envs.p2ptrading import DPLCRLPaperEnv
from dp_lcrl_rl.runner.shared.paper_mat_runner import PaperMATRunner
from dp_lcrl_rl.scripts.train.train_paper_mat import (
    _normalize_experiment_args,
    build_arg_parser,
    make_env,
)


def _small_args():
    args = build_arg_parser().parse_args([])
    args.num_agents = 4
    args.min_agents = 2
    args.curriculum_min_agents = 2
    args.curriculum_warmup_episodes = 10
    args.episode_length = 2
    args.n_rollout_threads = 1
    args.n_eval_rollout_threads = 1
    args.n_render_rollout_threads = 0
    args.num_env_steps = 4
    args.hidden_size = 32
    args.n_embd = 16
    args.n_block = 1
    args.n_head = 1
    args.layer_N = 1
    args.save_interval = 0
    args.export_summary_interval = 0
    args.log_interval = 100
    args.use_wandb = False
    args.use_eval = False
    return _normalize_experiment_args(args)


def test_stateless_cmtm_keeps_env_runnable_without_storage_memory():
    env = DPLCRLPaperEnv(max_agents=3, min_agents=3, horizon=2, seed=7, cmtm_mode="stateless")
    env.reset()

    actions = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    _, rewards, _, info = env.step(actions)

    assert info["cmtm_mode"] == "stateless"
    assert len(rewards) == 3
    for agent_info in info["per_agent"]:
        if not agent_info["active"]:
            continue
        assert agent_info["C_storage_avg"] == 0.0
    for state in env.states:
        assert state.carbon_mass == 0.0
        assert state.storage_intensity == 0.0


def test_obs_only_mask_mode_disables_transformer_side_masking():
    args = _small_args()
    args.mask_mode = "obs_only"

    env = DPLCRLPaperEnv(max_agents=args.num_agents, min_agents=args.min_agents, horizon=2, seed=5)
    policy = TransformerPolicy(
        args,
        env.observation_space,
        env.cent_observation_space,
        env.action_space,
        args.num_agents,
        device=torch.device("cpu"),
    )

    input_mask = np.array([[1.0], [0.0], [1.0], [0.0]], dtype=np.float32)
    resolved_mask = policy._transformer_agent_mask(input_mask, batch_size=1)

    assert resolved_mask.shape == (1, args.num_agents)
    assert np.allclose(resolved_mask, 1.0)


def test_direct_max_scale_mode_locks_training_env_to_full_size(tmp_path: Path):
    args = _small_args()
    args.scale_mode = "direct_max"
    args.min_agents = 2
    args.curriculum_min_agents = 2
    _normalize_experiment_args(args)

    envs = make_env(args, args.n_rollout_threads)
    config = {
        "all_args": args,
        "envs": envs,
        "eval_envs": None,
        "device": torch.device("cpu"),
        "num_agents": args.num_agents,
        "run_dir": tmp_path,
    }

    runner = PaperMATRunner(config)
    try:
        assert runner.scale_mode == "direct_max"
        assert runner.envs.min_agents == args.num_agents
        assert runner.envs.envs[0].min_agents == args.num_agents
    finally:
        runner.envs.close()
