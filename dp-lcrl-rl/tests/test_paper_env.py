import numpy as np

from dp_lcrl_rl.envs.p2ptrading import DPLCRLPaperEnv


def test_paper_env_uses_three_dim_actions_and_agent_mask():
    env = DPLCRLPaperEnv(max_agents=6, min_agents=3, horizon=4, seed=3)
    obs = env.reset()

    assert len(obs) == 6
    assert env.action_space.shape == (3,)
    assert env.observation_space.shape == (12,)
    assert 3 <= int(np.sum(env.agent_mask)) <= 6

    inactive = np.where(env.agent_mask < 0.5)[0]
    if inactive.size:
        assert np.allclose(obs[int(inactive[0])], 0.0)


def test_paper_env_tracks_cmtm_without_blockchain():
    env = DPLCRLPaperEnv(max_agents=2, min_agents=2, horizon=2, seed=11)
    env.reset()
    env.agent_mask[:] = 1.0
    env.active_ids = [0, 1]

    env.states[0].pv = 5.0
    env.states[0].load = 1.0
    env.states[1].pv = 0.2
    env.states[1].load = 3.5

    actions = [
        np.array([0.9, -0.2, 0.2], dtype=np.float32),
        np.array([-0.9, 0.6, 0.0], dtype=np.float32),
    ]
    _, rewards, _, info = env.step(actions)

    assert len(rewards) == 2
    assert "blockchain" not in info
    assert "carbon_flow_matrix" in info
    assert "settlement_records" in info
    assert "responsibility_state_updates" in info
    assert len(info["per_agent"]) == 2
    assert info["per_agent"][0]["C_sell"] >= 0.0
    assert info["market_summary"]["p2p_total_volume"] >= 0.0
    assert "p2p_mean_active" in info["market_summary"]
    assert "grid_buy_mean_active" in info["market_summary"]
    assert "grid_buy_penalty_total" in info["market_summary"]
    assert "unmatched_penalty_total" in info["market_summary"]
    assert "load_responsibility_mean_active" in info["carbon_trace"]
