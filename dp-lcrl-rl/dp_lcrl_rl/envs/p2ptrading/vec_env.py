"""Vectorized wrapper for the paper-aligned P2P trading environment."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from dp_lcrl_rl.envs.p2ptrading.dp_lcrl_paper_env import DPLCRLPaperEnv, MarketSpec, StorageSpec


class ParallelPaperVecEnv:
    """Thin vectorized wrapper used by the MAT runner.

    Each sub-environment keeps a fixed maximum agent dimension and uses
    `agent_mask` to represent dynamic participation, matching the paper's
    fixed-size training interface.
    """

    def __init__(
        self,
        n_threads: int,
        num_agents: int,
        seed: int,
        horizon: int,
        min_agents: Optional[int] = None,
        step_churn_prob: float = 0.0,
        storage_spec: Optional[StorageSpec] = None,
        market_spec: Optional[MarketSpec] = None,
        price_quote_range: Optional[tuple[float, float]] = None,
        storage_capacity_variance: float = 0.25,
        storage_capacity_range: Optional[tuple[float, float]] = (6.0, 18.0),
        pv_peak_scale: float = 7.5,
        pv_phase_jitter: float = 0.15,
        load_phase_jitter: float = 0.25,
        load_base: float = 2.2,
        load_peak: float = 4.5,
        dynamic_carbon_price: Optional[bool] = None,
        cmtm_mode: str = "full",
        mask_mode: str = "full",
        p2p_reward_weight: float = 0.10,
        grid_buy_penalty_weight: float = 0.10,
        unmatched_penalty_weight: float = 0.05,
    ) -> None:
        self.n_threads = int(max(1, n_threads))
        self.num_agents = int(max(1, num_agents))
        self.horizon = int(max(1, horizon))
        self.min_agents = (
            self.num_agents if min_agents is None else max(1, min(int(min_agents), self.num_agents))
        )
        self.step_churn_prob = float(max(0.0, min(1.0, step_churn_prob)))
        self._base_seed = int(seed)
        self.cmtm_mode = str(cmtm_mode).strip().lower()
        self.mask_mode = str(mask_mode).strip().lower()
        self.p2p_reward_weight = float(max(0.0, p2p_reward_weight))
        self.grid_buy_penalty_weight = float(max(0.0, grid_buy_penalty_weight))
        self.unmatched_penalty_weight = float(max(0.0, unmatched_penalty_weight))

        self.envs: List[DPLCRLPaperEnv] = []
        for thread_idx in range(self.n_threads):
            env_seed = self._base_seed + thread_idx * 7919
            self.envs.append(
                DPLCRLPaperEnv(
                    max_agents=self.num_agents,
                    horizon=self.horizon,
                    seed=env_seed,
                    min_agents=self.min_agents,
                    step_churn_prob=self.step_churn_prob,
                    storage_spec=storage_spec,
                    market_spec=market_spec,
                    price_quote_range=price_quote_range,
                    storage_capacity_variance=storage_capacity_variance,
                    storage_capacity_range=storage_capacity_range,
                    pv_peak_scale=pv_peak_scale,
                    pv_phase_jitter=pv_phase_jitter,
                    load_phase_jitter=load_phase_jitter,
                    load_base=load_base,
                    load_peak=load_peak,
                    dynamic_carbon_price=dynamic_carbon_price,
                    cmtm_mode=self.cmtm_mode,
                    mask_mode=self.mask_mode,
                    p2p_reward_weight=self.p2p_reward_weight,
                    grid_buy_penalty_weight=self.grid_buy_penalty_weight,
                    unmatched_penalty_weight=self.unmatched_penalty_weight,
                )
            )

        sample_env = self.envs[0]
        self.obs_dim = int(sample_env.observation_space.shape[0])
        self.share_obs_dim = int(sample_env.cent_observation_space.shape[0])
        self.observation_space = [sample_env.observation_space for _ in range(self.num_agents)]
        self.share_observation_space = [sample_env.cent_observation_space for _ in range(self.num_agents)]
        self.action_space = [sample_env.action_space for _ in range(self.num_agents)]

        self.obs = np.zeros((self.n_threads, self.num_agents, self.obs_dim), dtype=np.float32)
        self.share_obs = np.zeros((self.n_threads, self.share_obs_dim), dtype=np.float32)
        self.agent_masks = np.ones((self.n_threads, self.num_agents), dtype=np.float32)
        self.infos: List[Dict[str, object]] = [{} for _ in range(self.n_threads)]

    def set_min_agents(self, min_agents: int) -> None:
        self.min_agents = max(1, min(int(min_agents), self.num_agents))
        for env in self.envs:
            env.min_agents = self.min_agents

    def set_step_churn_prob(self, step_churn_prob: float) -> None:
        self.step_churn_prob = float(max(0.0, min(1.0, step_churn_prob)))
        for env in self.envs:
            env.step_churn_prob = self.step_churn_prob

    def _sync_from_env(
        self,
        thread_idx: int,
        obs: Sequence[np.ndarray],
        info: Optional[Dict[str, object]],
    ) -> None:
        obs_array = np.asarray(obs, dtype=np.float32)
        if obs_array.shape != (self.num_agents, self.obs_dim):
            raise ValueError(
                f"Thread {thread_idx} observation shape mismatch: "
                f"expected {(self.num_agents, self.obs_dim)}, got {obs_array.shape}."
            )
        self.obs[thread_idx] = obs_array

        info_dict: Dict[str, object] = dict(info or {})
        cent_obs = np.asarray(info_dict.get("cent_obs", obs_array.reshape(-1)), dtype=np.float32).reshape(-1)
        if cent_obs.shape[0] != self.share_obs_dim:
            raise ValueError(
                f"Thread {thread_idx} cent_obs shape mismatch: "
                f"expected {self.share_obs_dim}, got {cent_obs.shape[0]}."
            )
        self.share_obs[thread_idx] = cent_obs

        agent_mask = np.asarray(
            info_dict.get("agent_mask", self.envs[thread_idx].agent_mask.tolist()),
            dtype=np.float32,
        ).reshape(-1)
        if agent_mask.shape[0] != self.num_agents:
            raise ValueError(
                f"Thread {thread_idx} agent_mask shape mismatch: "
                f"expected {self.num_agents}, got {agent_mask.shape[0]}."
            )
        self.agent_masks[thread_idx] = agent_mask
        self.infos[thread_idx] = info_dict

    def reset(self) -> np.ndarray:
        for thread_idx, env in enumerate(self.envs):
            env.min_agents = self.min_agents
            env.step_churn_prob = self.step_churn_prob
            obs = env.reset()
            self._sync_from_env(thread_idx, obs, env.last_info)
        return self.obs.copy()

    def step(self, actions: np.ndarray):
        actions_array = np.asarray(actions, dtype=np.float32)
        if actions_array.shape != (self.n_threads, self.num_agents, self.action_space[0].shape[0]):
            raise ValueError(
                f"Action batch shape mismatch: expected "
                f"{(self.n_threads, self.num_agents, self.action_space[0].shape[0])}, "
                f"got {actions_array.shape}."
            )

        reward_batches: List[np.ndarray] = []
        done_batches: List[np.ndarray] = []
        info_batches: List[Dict[str, object]] = []

        for thread_idx, env in enumerate(self.envs):
            obs, rewards, done, info = env.step(actions_array[thread_idx])
            self._sync_from_env(thread_idx, obs, info)
            reward_batches.append(np.asarray(rewards, dtype=np.float32))
            done_batches.append(np.full(self.num_agents, bool(done), dtype=bool))
            info_batches.append(dict(info))

        rewards_array = np.stack(reward_batches, axis=0)
        dones_array = np.stack(done_batches, axis=0)
        return self.obs.copy(), rewards_array, dones_array, info_batches

    def close(self) -> None:
        for env in self.envs:
            env.close()
