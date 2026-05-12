"""PPO-compatible non-Transformer policies for Experiment 1 baselines."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from dp_lcrl_rl.algorithms.utils.transformer_act import _squash_action, _unsquash_action
from dp_lcrl_rl.algorithms.utils.util import check, init
from dp_lcrl_rl.utils.util import get_shape_from_act_space, get_shape_from_obs_space, update_linear_schedule


def _init_layer(module: nn.Module, gain: float = 0.01, activate: bool = False) -> nn.Module:
    if activate:
        gain = nn.init.calculate_gain("relu")
    return init(module, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0.0), gain=gain)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, layer_n: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
    last_dim = input_dim
    for _ in range(max(1, int(layer_n))):
        layers.extend(
            [
                _init_layer(nn.Linear(last_dim, hidden_dim), activate=True),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ]
        )
        last_dim = hidden_dim
    layers.append(_init_layer(nn.Linear(last_dim, output_dim)))
    return nn.Sequential(*layers)


class BaselineActorCritic(nn.Module):
    """Actor-critic model used by MLP-Pad, MAPPO-Shared, and DeepSets baselines."""

    def __init__(
        self,
        arch: str,
        obs_dim: int,
        share_obs_dim: int,
        action_dim: int,
        n_agent: int,
        hidden_dim: int,
        layer_n: int,
        action_low: Optional[np.ndarray],
        action_high: Optional[np.ndarray],
        use_id_emb: bool = True,
        log_std_min: float = -2.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.arch = str(arch).strip().lower()
        self.obs_dim = int(obs_dim)
        self.share_obs_dim = int(share_obs_dim)
        self.action_dim = int(action_dim)
        self.n_agent = int(n_agent)
        self.hidden_dim = int(hidden_dim)
        self.use_id_emb = bool(use_id_emb)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.id_emb = nn.Parameter(torch.zeros(1, self.n_agent, hidden_dim))
        nn.init.normal_(self.id_emb, std=0.02)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        if action_low is not None and action_high is not None:
            self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
            self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))
        else:
            self.action_low = None
            self.action_high = None

        if self.arch == "mlp_pad":
            global_dim = self.n_agent * self.obs_dim + self.n_agent
            self.actor = _mlp(global_dim, hidden_dim, self.n_agent * self.action_dim, layer_n)
            self.critic = _mlp(self.share_obs_dim + self.n_agent, hidden_dim, 1, layer_n)
        elif self.arch == "mappo_shared":
            actor_in = self.obs_dim + 1 + (hidden_dim if self.use_id_emb else 0)
            self.actor = _mlp(actor_in, hidden_dim, self.action_dim, layer_n)
            self.critic = _mlp(self.share_obs_dim + self.n_agent, hidden_dim, 1, layer_n)
        elif self.arch == "deepsets":
            phi_in = self.obs_dim + 1 + (hidden_dim if self.use_id_emb else 0)
            self.phi = _mlp(phi_in, hidden_dim, hidden_dim, layer_n)
            self.actor = _mlp(hidden_dim * 2 + 1, hidden_dim, self.action_dim, layer_n)
            self.critic = _mlp(hidden_dim, hidden_dim, 1, layer_n)
        else:
            raise ValueError(
                f"Unsupported baseline policy_arch={arch}. "
                "Expected one of ['mlp_pad', 'mappo_shared', 'deepsets']."
            )

    def _mask(self, obs: torch.Tensor, agent_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if agent_mask is None:
            return torch.ones((*obs.shape[:2],), dtype=obs.dtype, device=obs.device)
        return agent_mask.reshape(obs.shape[0], self.n_agent).to(device=obs.device, dtype=obs.dtype)

    def _id_features(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.use_id_emb:
            return torch.empty((batch_size, self.n_agent, 0), device=device, dtype=dtype)
        return self.id_emb.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    def _critic_input(self, cent_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        global_obs = cent_obs[:, 0, :] if cent_obs.dim() == 3 else cent_obs
        return torch.cat([global_obs, mask], dim=-1)

    def _deep_set_rep(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = obs.shape[0]
        id_feat = self._id_features(batch_size, obs.device, obs.dtype)
        phi_input = torch.cat([obs * mask.unsqueeze(-1), mask.unsqueeze(-1), id_feat], dim=-1)
        per_agent = self.phi(phi_input) * mask.unsqueeze(-1)
        denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
        pooled = per_agent.sum(dim=1) / denom
        return per_agent, pooled

    def action_means(self, cent_obs: torch.Tensor, obs: torch.Tensor, agent_mask: Optional[torch.Tensor]) -> torch.Tensor:
        mask = self._mask(obs, agent_mask)
        obs = obs * mask.unsqueeze(-1)

        if self.arch == "mlp_pad":
            global_input = torch.cat([obs.reshape(obs.shape[0], -1), mask], dim=-1)
            means = self.actor(global_input).view(obs.shape[0], self.n_agent, self.action_dim)
        elif self.arch == "mappo_shared":
            id_feat = self._id_features(obs.shape[0], obs.device, obs.dtype)
            actor_input = torch.cat([obs, mask.unsqueeze(-1), id_feat], dim=-1)
            means = self.actor(actor_input.reshape(-1, actor_input.shape[-1])).view(
                obs.shape[0],
                self.n_agent,
                self.action_dim,
            )
        else:
            per_agent, pooled = self._deep_set_rep(obs, mask)
            pooled_agents = pooled.unsqueeze(1).expand(-1, self.n_agent, -1)
            actor_input = torch.cat([per_agent, pooled_agents, mask.unsqueeze(-1)], dim=-1)
            means = self.actor(actor_input.reshape(-1, actor_input.shape[-1])).view(
                obs.shape[0],
                self.n_agent,
                self.action_dim,
            )

        return means * mask.unsqueeze(-1)

    def values(self, cent_obs: torch.Tensor, obs: torch.Tensor, agent_mask: Optional[torch.Tensor]) -> torch.Tensor:
        mask = self._mask(obs, agent_mask)
        if self.arch == "deepsets":
            _, pooled = self._deep_set_rep(obs, mask)
            scalar = self.critic(pooled)
        else:
            scalar = self.critic(self._critic_input(cent_obs, mask))
        return scalar.unsqueeze(1).expand(-1, self.n_agent, -1)

    def distribution(self, means: torch.Tensor) -> Normal:
        with torch.no_grad():
            self.log_std.data.clamp_(self.log_std_min, self.log_std_max)
        std = torch.sigmoid(self.log_std).view(1, 1, -1) * 0.5
        return Normal(means, std.expand_as(means))

    def act(
        self,
        cent_obs: torch.Tensor,
        obs: torch.Tensor,
        deterministic: bool = False,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self._mask(obs, agent_mask)
        means = self.action_means(cent_obs, obs, mask)
        dist = self.distribution(means)
        raw_action = means if deterministic else dist.sample()
        action, squashed = _squash_action(raw_action, self.action_low, self.action_high)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - squashed.pow(2) + 1e-6)
        action = action * mask.unsqueeze(-1)
        log_prob = log_prob * mask.unsqueeze(-1)
        value = self.values(cent_obs, obs, mask)
        return action, log_prob, value

    def evaluate(
        self,
        cent_obs: torch.Tensor,
        obs: torch.Tensor,
        action: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self._mask(obs, agent_mask)
        means = self.action_means(cent_obs, obs, mask)
        dist = self.distribution(means)
        raw_action, squashed = _unsquash_action(action, self.action_low, self.action_high, 1e-6)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - squashed.pow(2) + 1e-6)
        entropy = dist.entropy()
        log_prob = log_prob * mask.unsqueeze(-1)
        entropy = entropy * mask.unsqueeze(-1)
        value = self.values(cent_obs, obs, mask)
        return value, log_prob, entropy


class BaselinePolicy:
    """Policy wrapper exposing the same methods as TransformerPolicy."""

    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents, device=torch.device("cpu")):
        self.device = device
        self.lr = float(args.lr)
        self.opti_eps = float(args.opti_eps)
        self.weight_decay = float(args.weight_decay)
        self._use_policy_active_masks = bool(args.use_policy_active_masks)
        self.policy_arch = str(getattr(args, "policy_arch", "mlp_pad")).strip().lower()

        if act_space.__class__.__name__ != "Box":
            raise ValueError("BaselinePolicy currently supports continuous Box action spaces only.")

        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.share_obs_dim = get_shape_from_obs_space(cent_obs_space)[0]
        self.act_dim = get_shape_from_act_space(act_space)
        self.act_num = self.act_dim
        self.num_agents = int(num_agents)
        self.tpdv = dict(dtype=torch.float32, device=device)

        action_low = np.asarray(act_space.low, dtype=np.float32)
        action_high = np.asarray(act_space.high, dtype=np.float32)
        self.transformer = BaselineActorCritic(
            arch=self.policy_arch,
            obs_dim=self.obs_dim,
            share_obs_dim=self.share_obs_dim,
            action_dim=self.act_dim,
            n_agent=self.num_agents,
            hidden_dim=int(getattr(args, "n_embd", getattr(args, "hidden_size", 64))),
            layer_n=int(getattr(args, "layer_N", 2)),
            action_low=action_low,
            action_high=action_high,
            use_id_emb=not bool(getattr(args, "no_id_emb", False)),
            log_std_min=float(getattr(args, "log_std_min", -2.0)),
            log_std_max=float(getattr(args, "log_std_max", 2.0)),
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.transformer.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def set_lr(self, lr: float) -> None:
        target_lr = float(max(0.0, lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = target_lr

    def get_lr(self) -> float:
        if not self.optimizer.param_groups:
            return float(self.lr)
        return float(self.optimizer.param_groups[0].get("lr", self.lr))

    def _reshape_inputs(self, cent_obs, obs, agent_mask=None):
        cent_obs_t = check(cent_obs).to(**self.tpdv).reshape(-1, self.num_agents, self.share_obs_dim)
        obs_t = check(obs).to(**self.tpdv).reshape(-1, self.num_agents, self.obs_dim)
        if agent_mask is None:
            mask_t = torch.ones((obs_t.shape[0], self.num_agents), **self.tpdv)
        else:
            mask_t = check(agent_mask).to(**self.tpdv).reshape(-1, self.num_agents)
        return cent_obs_t, obs_t, mask_t

    def get_actions(
        self,
        cent_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        agent_mask=None,
        deterministic=False,
    ):
        cent_obs_t, obs_t, mask_t = self._reshape_inputs(cent_obs, obs, agent_mask)
        actions, action_log_probs, values = self.transformer.act(
            cent_obs_t,
            obs_t,
            deterministic=deterministic,
            agent_mask=mask_t,
        )
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        return (
            values.reshape(-1, 1),
            actions.reshape(-1, self.act_num),
            action_log_probs.reshape(-1, self.act_num),
            rnn_states_actor,
            rnn_states_critic,
        )

    def get_values(self, cent_obs, obs, rnn_states_critic, masks, agent_mask=None):
        cent_obs_t, obs_t, mask_t = self._reshape_inputs(cent_obs, obs, agent_mask)
        values = self.transformer.values(cent_obs_t, obs_t, mask_t)
        return values.reshape(-1, 1)

    def evaluate_actions(
        self,
        cent_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        actions,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        cent_obs_t, obs_t, mask_t = self._reshape_inputs(cent_obs, obs, active_masks)
        action_t = check(actions).to(**self.tpdv).reshape(-1, self.num_agents, self.act_num)
        values, action_log_probs, entropy = self.transformer.evaluate(
            cent_obs_t,
            obs_t,
            action_t,
            agent_mask=mask_t,
        )

        entropy = entropy.reshape(-1, self.act_num)
        if self._use_policy_active_masks and active_masks is not None:
            active_masks_t = check(active_masks).to(**self.tpdv)
            entropy_out = (entropy * active_masks_t).sum() / torch.clamp(active_masks_t.sum(), min=1.0)
        else:
            entropy_out = entropy.mean()

        return values.reshape(-1, 1), action_log_probs.reshape(-1, self.act_num), entropy_out

    def act(self, cent_obs, obs, rnn_states_actor, masks, available_actions=None, deterministic=True):
        rnn_states_critic = np.zeros_like(rnn_states_actor)
        _, actions, _, rnn_states_actor, _ = self.get_actions(
            cent_obs,
            obs,
            rnn_states_actor,
            rnn_states_critic,
            masks,
            available_actions=available_actions,
            deterministic=deterministic,
        )
        return actions, rnn_states_actor

    def save(self, save_dir, episode):
        torch.save(self.transformer.state_dict(), str(save_dir) + "/transformer_" + str(episode) + ".pt")

    def restore(self, model_dir):
        state_dict = torch.load(model_dir, map_location=self.device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        current_state = self.transformer.state_dict()
        compatible = {}
        skipped = []
        for key, value in state_dict.items():
            if key not in current_state:
                skipped.append(key)
                continue
            if current_state[key].shape != value.shape:
                skipped.append(key)
                continue
            compatible[key] = value
        current_state.update(compatible)
        self.transformer.load_state_dict(current_state)
        if skipped:
            print(
                "[Restore] Skipped {} incompatible baseline keys: {}".format(
                    len(skipped),
                    ", ".join(skipped[:5]) + ("..." if len(skipped) > 5 else ""),
                )
            )

    def train(self):
        self.transformer.train()

    def eval(self):
        self.transformer.eval()

    def pop_attn_stats(self):
        return None
