from pathlib import Path

import numpy as np

from dp_lcrl_rl.envs.p2ptrading import DPLCRLPaperEnv
from dp_lcrl_rl.runner.shared.training_analytics import PaperTrainingAnalytics


def _configure_two_agent_trade(env: DPLCRLPaperEnv) -> None:
    env.agent_mask[:] = 1.0
    env.active_ids = [0, 1]
    env.pv_profiles[:] = 0.0
    env.load_profiles[:] = 0.0
    env.pv_profiles[0, 0] = 5.0
    env.load_profiles[0, 0] = 1.0
    env.pv_profiles[1, 0] = 0.2
    env.load_profiles[1, 0] = 4.2


def test_p2p_reward_weight_increases_reward_when_trade_executes():
    env_base = DPLCRLPaperEnv(max_agents=2, min_agents=2, horizon=1, seed=23, p2p_reward_weight=0.0)
    env_bonus = DPLCRLPaperEnv(max_agents=2, min_agents=2, horizon=1, seed=23, p2p_reward_weight=0.5)
    env_base.reset()
    env_bonus.reset()
    _configure_two_agent_trade(env_base)
    _configure_two_agent_trade(env_bonus)

    actions = [
        np.array([0.9, -0.3, 0.0], dtype=np.float32),
        np.array([-0.9, 0.8, 0.0], dtype=np.float32),
    ]

    _, rewards_base, _, info_base = env_base.step(actions)
    _, rewards_bonus, _, info_bonus = env_bonus.step(actions)

    assert info_base["market_summary"]["p2p_total_volume"] > 0.0
    assert info_bonus["market_summary"]["p2p_total_volume"] == info_base["market_summary"]["p2p_total_volume"]
    assert sum(rewards_bonus) > sum(rewards_base)
    assert info_bonus["per_agent"][0]["p2p_reward_bonus"] >= 0.0
    assert info_bonus["per_agent"][1]["p2p_reward_bonus"] >= 0.0


def test_grid_buy_penalty_reduces_reward_for_grid_dependent_agent():
    env_base = DPLCRLPaperEnv(
        max_agents=1,
        min_agents=1,
        horizon=1,
        seed=31,
        p2p_reward_weight=0.0,
        grid_buy_penalty_weight=0.0,
    )
    env_penalty = DPLCRLPaperEnv(
        max_agents=1,
        min_agents=1,
        horizon=1,
        seed=31,
        p2p_reward_weight=0.0,
        grid_buy_penalty_weight=0.2,
    )
    env_base.reset()
    env_penalty.reset()
    for env in (env_base, env_penalty):
        env.agent_mask[:] = 1.0
        env.active_ids = [0]
        env.pv_profiles[:] = 0.0
        env.load_profiles[:] = 0.0
        env.load_profiles[0, 0] = 4.0

    actions = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
    _, rewards_base, _, info_base = env_base.step(actions)
    _, rewards_penalty, _, info_penalty = env_penalty.step(actions)

    assert info_base["per_agent"][0]["grid_buy"] > 0.0
    assert info_penalty["per_agent"][0]["grid_buy_penalty"] > 0.0
    assert rewards_penalty[0] < rewards_base[0]


def test_unmatched_penalty_reduces_reward_when_orders_do_not_clear():
    env_base = DPLCRLPaperEnv(
        max_agents=2,
        min_agents=2,
        horizon=1,
        seed=37,
        p2p_reward_weight=0.0,
        unmatched_penalty_weight=0.0,
    )
    env_penalty = DPLCRLPaperEnv(
        max_agents=2,
        min_agents=2,
        horizon=1,
        seed=37,
        p2p_reward_weight=0.0,
        unmatched_penalty_weight=0.2,
    )
    env_base.reset()
    env_penalty.reset()
    _configure_two_agent_trade(env_base)
    _configure_two_agent_trade(env_penalty)

    actions = [
        np.array([0.9, 1.0, 0.0], dtype=np.float32),
        np.array([-0.9, -1.0, 0.0], dtype=np.float32),
    ]

    _, rewards_base, _, info_base = env_base.step(actions)
    _, rewards_penalty, _, info_penalty = env_penalty.step(actions)

    assert info_base["market_summary"]["p2p_total_volume"] == 0.0
    assert info_penalty["market_summary"]["unmatched_penalty_total"] > 0.0
    assert sum(rewards_penalty) < sum(rewards_base)


def test_episode_summary_uses_active_agent_means_and_last_step_carbon(tmpdir):
    analytics = PaperTrainingAnalytics(run_dir=Path(str(tmpdir)), num_agents=4, n_threads=1, episode_length=2)
    zero_matrix = np.zeros((4, 4), dtype=np.float32)
    records = [
        {
            "phase": "train",
            "n_active_agents": 2,
            "global_reward": 3.0,
            "market_summary": {
                "p2p_mean_active": 1.5,
                "grid_buy_mean_active": 2.0,
                "grid_sell_mean_active": 0.5,
                "carbon_price": 50.0,
            },
            "carbon_trace": {"load_responsibility_mean_active": 4.0, "p2p_import": 1.2, "source_injection": 2.5},
            "p2p_matrix": zero_matrix,
            "carbon_flow_matrix": zero_matrix,
        },
        {
            "phase": "train",
            "n_active_agents": 4,
            "global_reward": 5.0,
            "market_summary": {
                "p2p_mean_active": 2.5,
                "grid_buy_mean_active": 1.0,
                "grid_sell_mean_active": 0.25,
                "carbon_price": 60.0,
            },
            "carbon_trace": {"load_responsibility_mean_active": 3.0, "p2p_import": 0.8, "source_injection": 1.5},
            "p2p_matrix": zero_matrix,
            "carbon_flow_matrix": zero_matrix,
        },
    ]

    summary = analytics._summarize_episode(records)

    assert summary["average_global_reward"] == 4.0
    assert summary["p2p_volume_mean_active"] == 2.0
    assert summary["grid_buy_mean_active"] == 1.5
    assert summary["grid_sell_mean_active"] == 0.375
    assert summary["carbon_responsibility_mean_active_episode"] == 3.5
    assert summary["p2p_total_volume"] == summary["p2p_volume_mean_active"]
    assert summary["load_responsibility_total"] == summary["carbon_responsibility_mean_active_episode"]
    assert "p2p_matrix_sum" not in summary
    assert "carbon_matrix_sum" not in summary


def test_attention_entropy_mean_is_streamed_without_retaining_full_history(tmpdir):
    analytics = PaperTrainingAnalytics(run_dir=Path(str(tmpdir)), num_agents=2, n_threads=1, episode_length=1)

    analytics.record_attention_samples(np.array([1.0, 2.0], dtype=np.float32))
    analytics.record_attention_samples(np.array([3.0], dtype=np.float32))

    payload = analytics._build_payload()

    assert analytics.attn_entropy_count == 3
    assert analytics.attn_entropy_sum == 6.0
    assert payload["overall_metrics"]["attention_entropy_mean"] == 2.0
