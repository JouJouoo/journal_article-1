import numpy as np

from dp_lcrl_rl.envs.p2ptrading import ParallelPaperVecEnv


def test_parallel_env_reset_and_step_shapes():
    env = ParallelPaperVecEnv(n_threads=2, num_agents=5, seed=17, horizon=4, min_agents=3)
    obs = env.reset()

    assert obs.shape == (2, 5, 12)
    assert env.share_obs.shape == (2, 60)
    assert env.agent_masks.shape == (2, 5)

    actions = np.zeros((2, 5, 3), dtype=np.float32)
    next_obs, rewards, dones, infos = env.step(actions)

    assert next_obs.shape == (2, 5, 12)
    assert rewards.shape == (2, 5)
    assert dones.shape == (2, 5)
    assert len(infos) == 2
    assert "cent_obs" in infos[0]


def test_parallel_env_min_agent_schedule_updates_subenvs():
    env = ParallelPaperVecEnv(n_threads=1, num_agents=6, seed=5, horizon=4, min_agents=2)
    env.set_min_agents(4)

    assert env.min_agents == 4
    assert env.envs[0].min_agents == 4
