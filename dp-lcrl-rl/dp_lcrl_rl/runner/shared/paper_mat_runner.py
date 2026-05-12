"""Paper-aligned MAT runner without blockchain dependencies."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from dp_lcrl_rl.runner.shared.base_runner import Runner, _t2n
from dp_lcrl_rl.runner.shared.training_analytics import PaperTrainingAnalytics


class PaperMATRunner(Runner):
    """Runner for MAT/MAT-Dec on the paper-aligned P2P environment."""

    def __init__(self, config):
        super().__init__(config)
        summary_filename = getattr(self.all_args, "summary_filename", "paper_training_summary.html")
        self.curriculum_min_agents = int(getattr(self.all_args, "curriculum_min_agents", self.num_agents))
        self.curriculum_warmup = int(getattr(self.all_args, "curriculum_warmup_episodes", 0))
        self.scale_mode = str(getattr(self.all_args, "scale_mode", "curriculum")).strip().lower()
        if self.scale_mode not in {"curriculum", "direct_max", "random_scale"}:
            raise ValueError(
                f"Unsupported scale_mode={self.scale_mode}. Expected one of ['curriculum', 'direct_max', 'random_scale']."
            )
        self.export_summary_interval = max(0, int(getattr(self.all_args, "export_summary_interval", 0)))
        self.export_last_episode_summary = bool(getattr(self.all_args, "export_last_episode_summary", False))
        self.last_episode_summary_filename = str(
            getattr(self.all_args, "last_episode_summary_filename", "paper_training_summary.last.html")
        )
        self.log_eval_summary = bool(getattr(self.all_args, "export_eval_summary", False))
        self.eval_summary_filename = str(
            getattr(self.all_args, "eval_summary_filename", "paper_eval_summary.html")
        )
        self.base_lr = float(getattr(self.all_args, "lr", 1e-4))
        self.base_entropy_coef = float(getattr(self.all_args, "entropy_coef", 0.03))
        self.episode_offset = max(0, int(getattr(self.all_args, "episode_offset", 0)))
        self.late_anneal_start_episode = max(
            0,
            int(getattr(self.all_args, "late_anneal_start_episode", 0)),
        )
        self.late_lr_final = getattr(self.all_args, "late_lr_final", None)
        self.late_entropy_coef_final = getattr(self.all_args, "late_entropy_coef_final", None)

        self.analytics = PaperTrainingAnalytics(
            run_dir=Path(self.run_dir),
            num_agents=self.num_agents,
            n_threads=self.n_rollout_threads,
            episode_length=self.episode_length,
            summary_filename=summary_filename,
        )
        self._attn_rollout_mass = 0.0
        self._attn_rollout_calls = 0
        self._attn_entropy_batch: List[np.ndarray] = []

        if self.scale_mode == "direct_max" and hasattr(self.envs, "set_min_agents"):
            self.envs.set_min_agents(self.num_agents)

    def _apply_scale_schedule(self, episode_idx: int) -> None:
        if self.scale_mode == "direct_max" or not hasattr(self.envs, "set_min_agents"):
            return
        if self.curriculum_warmup <= 0 or self.curriculum_min_agents >= self.num_agents:
            return

        if self.scale_mode == "random_scale":
            target_min = random.randint(self.curriculum_min_agents, self.num_agents)
        else:
            progress = min(1.0, float(episode_idx + 1) / float(max(1, self.curriculum_warmup)))
            target_min = int(
                round(self.curriculum_min_agents + progress * (self.num_agents - self.curriculum_min_agents))
            )
        target_min = max(1, min(target_min, self.num_agents))
        self.envs.set_min_agents(target_min)

    def _prime_buffer_from_envs(self) -> None:
        obs = np.asarray(self.envs.obs, dtype=np.float32)
        share = np.asarray(self.envs.share_obs, dtype=np.float32)
        if self.use_centralized_V:
            share = np.repeat(share[:, None, :], self.num_agents, axis=1)
        else:
            share = obs.copy()
        agent_masks = np.asarray(self.envs.agent_masks, dtype=np.float32)[..., None]

        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share.copy()
        self.buffer.rnn_states[0] = np.zeros_like(self.buffer.rnn_states[0])
        self.buffer.rnn_states_critic[0] = np.zeros_like(self.buffer.rnn_states_critic[0])
        self.buffer.masks[0] = np.ones_like(self.buffer.masks[0])
        self.buffer.active_masks[0] = agent_masks.copy()

    def warmup(self):
        self.envs.reset()
        self._prime_buffer_from_envs()

    @torch.no_grad()
    def collect(self, step: int) -> Dict[str, np.ndarray]:
        self.trainer.prep_rollout()

        obs_batch = np.concatenate(self.buffer.obs[step], axis=0)
        share_batch = np.concatenate(self.buffer.share_obs[step], axis=0)
        rnn_states = np.concatenate(self.buffer.rnn_states[step], axis=0)
        rnn_states_critic = np.concatenate(self.buffer.rnn_states_critic[step], axis=0)
        masks = np.concatenate(self.buffer.masks[step], axis=0)
        active_masks = np.concatenate(self.buffer.active_masks[step], axis=0)

        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.policy.get_actions(
            share_batch,
            obs_batch,
            rnn_states,
            rnn_states_critic,
            masks,
            agent_mask=active_masks,
        )

        if hasattr(self.policy, "pop_attn_stats"):
            self._update_rollout_attn_stats(self.policy.pop_attn_stats())

        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        obs, rewards, dones, infos = self.envs.step(actions.copy())
        self.analytics.set_thread_count(self.n_rollout_threads)
        self.analytics.push_step(infos, rewards, obs, actions=actions, phase="train")

        if self.use_centralized_V:
            share_obs_flat = np.asarray(
                [np.asarray(info.get("cent_obs"), dtype=np.float32) for info in infos],
                dtype=np.float32,
            )
            share_obs = np.repeat(share_obs_flat[:, None, :], self.num_agents, axis=1)
        else:
            share_obs = np.asarray(obs, dtype=np.float32)

        rewards = np.asarray(rewards, dtype=np.float32).reshape(self.n_rollout_threads, self.num_agents, 1)
        dones = np.asarray(dones, dtype=bool)
        masks_next = (1.0 - dones.astype(np.float32)).reshape(self.n_rollout_threads, self.num_agents, 1)
        agent_masks_next = np.asarray(
            [np.asarray(info.get("agent_mask", [1.0] * self.num_agents), dtype=np.float32) for info in infos],
            dtype=np.float32,
        )[..., None]

        rewards = rewards * agent_masks_next
        masks_next = masks_next * agent_masks_next

        return {
            "obs": np.asarray(obs, dtype=np.float32),
            "share_obs": np.asarray(share_obs, dtype=np.float32),
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "actions": actions,
            "action_log_probs": action_log_probs,
            "rnn_states": rnn_states,
            "rnn_states_critic": rnn_states_critic,
            "masks": masks_next,
            "agent_mask": agent_masks_next,
        }

    def insert(self, data: Dict[str, np.ndarray]) -> None:
        self.buffer.insert(
            data["share_obs"],
            data["obs"],
            data["rnn_states"],
            data["rnn_states_critic"],
            data["actions"],
            data["action_log_probs"],
            data["values"],
            data["rewards"],
            data["masks"],
            active_masks=data.get("agent_mask"),
        )

    def _reset_envs(self) -> None:
        self.envs.reset()
        self._prime_buffer_from_envs()

    def _update_rollout_attn_stats(self, stats: Optional[Dict[str, float]]) -> None:
        if not stats:
            return
        self._attn_rollout_mass += float(stats.get("attn_leakage", 0.0) or 0.0)
        self._attn_rollout_calls += int(stats.get("attn_calls", 0) or 0)
        entropy = stats.get("attn_entropy")
        if entropy is not None:
            arr = np.asarray(entropy, dtype=np.float32).ravel()
            if arr.size:
                self._attn_entropy_batch.append(arr)

    @staticmethod
    def _late_stage_linear(
        episode_idx: int,
        total_episodes: int,
        start_episode: int,
        start_value: float,
        end_value: float,
    ) -> float:
        if total_episodes <= 1 or episode_idx < start_episode:
            return float(start_value)
        span = max(1, total_episodes - start_episode - 1)
        progress = min(1.0, float(episode_idx - start_episode) / float(span))
        return float(start_value + progress * (end_value - start_value))

    def _apply_optimization_schedule(self, episode_idx: int, total_episodes: int) -> tuple[float, float]:
        late_enabled = (
            self.late_anneal_start_episode > 0
            and (
                self.late_lr_final is not None
                or self.late_entropy_coef_final is not None
            )
        )
        if late_enabled:
            target_lr = self.base_lr if self.late_lr_final is None else float(self.late_lr_final)
            target_entropy = (
                self.base_entropy_coef
                if self.late_entropy_coef_final is None
                else float(self.late_entropy_coef_final)
            )
            current_lr = self._late_stage_linear(
                episode_idx=episode_idx,
                total_episodes=total_episodes,
                start_episode=self.late_anneal_start_episode,
                start_value=self.base_lr,
                end_value=target_lr,
            )
            current_entropy = self._late_stage_linear(
                episode_idx=episode_idx,
                total_episodes=total_episodes,
                start_episode=self.late_anneal_start_episode,
                start_value=self.base_entropy_coef,
                end_value=target_entropy,
            )
            self.policy.set_lr(current_lr)
            self.trainer.set_entropy_coef(current_entropy)
            return current_lr, current_entropy

        if self.use_linear_lr_decay:
            self.policy.lr_decay(episode_idx, total_episodes)
        current_lr = self.policy.get_lr() if hasattr(self.policy, "get_lr") else self.base_lr
        current_entropy = float(getattr(self.trainer, "entropy_coef", self.base_entropy_coef))
        return current_lr, current_entropy

    def run(self) -> None:
        self.warmup()
        episodes = max(1, int(self.num_env_steps) // self.episode_length // self.n_rollout_threads)
        total_target_episodes = self.episode_offset + episodes
        total_num_steps = 0

        for ep in range(episodes):
            global_ep = self.episode_offset + ep
            current_lr, current_entropy = self._apply_optimization_schedule(global_ep, total_target_episodes)

            self._apply_scale_schedule(global_ep)

            if ep > 0:
                self._reset_envs()

            iteration_start = time.time()
            for step in range(self.episode_length):
                step_data = self.collect(step)
                self.insert(step_data)
                total_num_steps += int(np.asarray(step_data["agent_mask"], dtype=np.float32).sum())

            self.analytics.finalize_batch(global_ep)
            self.compute()
            train_infos = self.train()

            if self._attn_rollout_calls > 0:
                train_infos["attn_leakage_rollout"] = self._attn_rollout_mass / float(self._attn_rollout_calls)
            else:
                train_infos["attn_leakage_rollout"] = 0.0
            self._attn_rollout_mass = 0.0
            self._attn_rollout_calls = 0

            if self._attn_entropy_batch:
                entropy_concat = np.concatenate(self._attn_entropy_batch)
                train_infos["attn_entropy_mean"] = float(np.mean(entropy_concat))
                self.analytics.record_attention_samples(entropy_concat)
                self._attn_entropy_batch = []
            else:
                train_infos["attn_entropy_mean"] = 0.0

            iter_duration = max(time.time() - iteration_start, 1e-6)
            episode_stats = self.analytics.recent_episode_stats(self.n_rollout_threads, phase="train")
            active_agents_mean = episode_stats.get("average_n_agents", float(self.num_agents))
            frames_collected = active_agents_mean * self.episode_length * self.n_rollout_threads
            resource_metrics = {
                "fps_policy": float(frames_collected / iter_duration),
                "iter_duration_s": float(iter_duration),
                "avg_n_agents": float(active_agents_mean),
            }
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
                resource_metrics["gpu_mem_mb"] = float(
                    torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                )
                torch.cuda.reset_peak_memory_stats(self.device)
            else:
                resource_metrics["gpu_mem_mb"] = 0.0

            train_infos.update(resource_metrics)
            train_infos["current_lr"] = float(current_lr)
            train_infos["current_entropy_coef"] = float(current_entropy)
            self.analytics.record_training_metrics(global_ep, total_num_steps, train_infos)
            self.analytics.record_resource_metrics(global_ep, total_num_steps, resource_metrics)

            if (ep + 1) % self.log_interval == 0 or ep == 0 or (ep + 1) == episodes:
                print(
                    "[Progress] Iter {}/{} | avg_reward={:.4f} | p2p_mean={:.4f} | grid_buy_mean={:.4f} | grid_sell_mean={:.4f} | carbon_resp_mean={:.4f} | carbon_price={:.4f} | avg_agents={:.2f} | lr={:.6f} | entropy_coef={:.4f}".format(
                        global_ep + 1,
                        total_target_episodes,
                        episode_stats.get("average_global_reward", 0.0),
                        episode_stats.get("average_p2p_volume", 0.0),
                        episode_stats.get("average_grid_buy", 0.0),
                        episode_stats.get("average_grid_sell", 0.0),
                        episode_stats.get("average_carbon_responsibility", 0.0),
                        episode_stats.get("average_carbon_price", 0.0),
                        episode_stats.get("average_n_agents", float(self.num_agents)),
                        float(current_lr),
                        float(current_entropy),
                    )
                )

            if (ep + 1) % self.log_interval == 0:
                self.log_train(train_infos, total_num_steps)

            if self.use_eval and self.eval_envs is not None and (ep + 1) % self.eval_interval == 0:
                self.eval(total_num_steps)

            if self.save_interval > 0 and (global_ep + 1) % self.save_interval == 0:
                self.save(episode=global_ep + 1)

            if self.export_summary_interval > 0 and (global_ep + 1) % self.export_summary_interval == 0:
                report_path = self.analytics.export_summary()
                print(f"[Progress] Summary saved to: {report_path}")
                if self.export_last_episode_summary:
                    last_path = self.analytics.export_summary(
                        summary_filename=self.last_episode_summary_filename,
                        last_episode_only=True,
                    )
                    print(f"[Progress] Last-episode summary saved to: {last_path}")

        if self.save_interval <= 0 or total_target_episodes % self.save_interval != 0:
            self.save(episode=total_target_episodes)

        report_path = self.analytics.export_summary()
        print(f"[Progress] Summary saved to: {report_path}")
        if self.export_last_episode_summary:
            last_path = self.analytics.export_summary(
                summary_filename=self.last_episode_summary_filename,
                last_episode_only=True,
            )
            print(f"[Progress] Last-episode summary saved to: {last_path}")

    @torch.no_grad()
    def eval(self, total_num_steps: int) -> None:
        if self.eval_envs is None:
            return

        obs = self.eval_envs.reset()
        rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic = np.zeros_like(rnn_states)
        masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        agent_masks = np.asarray(self.eval_envs.agent_masks, dtype=np.float32)[..., None]
        eval_rewards: List[float] = []

        if self.log_eval_summary:
            self.analytics.set_thread_count(self.n_eval_rollout_threads)

        for _ in range(self.episode_length):
            obs_batch = np.concatenate(obs, axis=0)
            if self.use_centralized_V:
                share_batch = np.repeat(
                    np.asarray(self.eval_envs.share_obs, dtype=np.float32)[:, None, :],
                    self.num_agents,
                    axis=1,
                )
                share_batch = np.concatenate(share_batch, axis=0)
            else:
                share_batch = obs_batch

            agent_mask_batch = np.concatenate(agent_masks, axis=0)
            _, actions, _, rnn_states, rnn_states_critic = self.policy.get_actions(
                share_batch,
                obs_batch,
                rnn_states,
                rnn_states_critic,
                masks,
                agent_mask=agent_mask_batch,
                deterministic=True,
            )
            action_array = np.array(np.split(_t2n(actions), self.n_eval_rollout_threads))
            obs, rewards, dones, infos = self.eval_envs.step(action_array)
            eval_rewards.append(float(np.mean(rewards)))

            if self.log_eval_summary:
                self.analytics.push_step(infos, rewards, obs, actions=action_array, phase="eval")

            masks = (1.0 - np.asarray(dones, dtype=np.float32)).reshape(
                self.n_eval_rollout_threads,
                self.num_agents,
                1,
            )
            agent_masks = np.asarray(
                [np.asarray(info.get("agent_mask", [1.0] * self.num_agents), dtype=np.float32) for info in infos],
                dtype=np.float32,
            )[..., None]

        if eval_rewards:
            self.analytics.record_eval_metrics(total_num_steps, float(np.mean(eval_rewards)))

        if self.log_eval_summary:
            self.analytics.finalize_batch(int(total_num_steps))
            report_path = self.analytics.export_summary(summary_filename=self.eval_summary_filename)
            print(f"[Eval] Summary saved to: {report_path}")
