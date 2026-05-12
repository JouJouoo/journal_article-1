import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F


def _atanh(x):
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _squash_action(raw_action, action_low, action_high):
    squashed = torch.tanh(raw_action)
    if action_low is None or action_high is None:
        return squashed, squashed
    scale = (action_high - action_low) / 2.0
    bias = (action_high + action_low) / 2.0
    return squashed * scale + bias, squashed


def _unsquash_action(action, action_low, action_high, eps):
    if action_low is None or action_high is None:
        squashed = torch.clamp(action, -1.0 + eps, 1.0 - eps)
        raw_action = _atanh(squashed)
        return raw_action, squashed
    scale = (action_high - action_low) / 2.0
    bias = (action_high + action_low) / 2.0
    scale = torch.clamp(scale, min=eps)
    squashed = (action - bias) / scale
    squashed = torch.clamp(squashed, -1.0 + eps, 1.0 - eps)
    raw_action = _atanh(squashed)
    return raw_action, squashed


def discrete_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False, agent_mask=None):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)
    if agent_mask is None:
        agent_mask = torch.ones((batch_size, n_agent), **tpdv)
    else:
        agent_mask = agent_mask.to(**tpdv)

    for i in range(n_agent):
        logit = decoder(shifted_action, obs_rep, obs, agent_mask=agent_mask)[:, i, :]
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logit)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        active = agent_mask[:, i] > 0.5
        output_action[active, i, :] = action[active].unsqueeze(-1)
        output_action_log[active, i, :] = action_log[active].unsqueeze(-1)
        if i + 1 < n_agent:
            next_idx = agent_mask[:, i + 1] > 0.5
            if next_idx.any():
                shifted_action[next_idx, i + 1, 1:] = F.one_hot(action[next_idx], num_classes=action_dim)
    return output_action, output_action_log


def discrete_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None, agent_mask=None):
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (batch, n_agent, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
    if agent_mask is None:
        agent_mask = torch.ones((batch_size, n_agent), **tpdv)
    else:
        agent_mask = agent_mask.to(**tpdv)
    logit = decoder(shifted_action, obs_rep, obs, agent_mask=agent_mask)
    if available_actions is not None:
        logit[available_actions == 0] = -1e10
    if agent_mask is not None:
        logit = logit.masked_fill(agent_mask.unsqueeze(-1) <= 0, -1e10)

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    mask = agent_mask.unsqueeze(-1)
    action_log = action_log * mask
    entropy = entropy * mask
    return action_log, entropy


def continuous_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                  deterministic=False, agent_mask=None, action_low=None, action_high=None):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32).to(**tpdv)
    output_action_log = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32).to(**tpdv)
    if agent_mask is None:
        agent_mask = torch.ones((batch_size, n_agent), **tpdv)
    else:
        agent_mask = agent_mask.to(**tpdv)
    eps = 1e-6

    for i in range(n_agent):
        act_mean = decoder(shifted_action, obs_rep, obs, agent_mask=agent_mask)[:, i, :]
        active = agent_mask[:, i] > 0.5
        act_mean = torch.nan_to_num(act_mean, nan=0.0, posinf=0.0, neginf=0.0)
        if (~active).any():
            act_mean = act_mean.clone()
            act_mean[~active] = 0.0
        action_std = torch.sigmoid(decoder.log_std) * 0.5

        # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
        # distri = Normal(act_mean, log_std.exp())
        distri = Normal(act_mean, action_std)
        raw_action = act_mean if deterministic else distri.sample()
        action, squashed = _squash_action(raw_action, action_low, action_high)
        action_log = distri.log_prob(raw_action) - torch.log(1.0 - squashed.pow(2) + eps)

        output_action[active, i, :] = action[active]
        output_action_log[active, i, :] = action_log[active]
        if i + 1 < n_agent:
            next_idx = agent_mask[:, i + 1] > 0.5
            if next_idx.any():
                shifted_action[next_idx, i + 1, :] = action[next_idx]

        # print("act_mean: ", act_mean)
        # print("action: ", action)

    return output_action, output_action_log


def continuous_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,
                            agent_mask=None, action_low=None, action_high=None):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    shifted_action[:, 1:, :] = action[:, :-1, :]

    act_mean = decoder(shifted_action, obs_rep, obs, agent_mask=agent_mask)
    act_mean = torch.nan_to_num(act_mean, nan=0.0, posinf=0.0, neginf=0.0)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())

    eps = 1e-6
    raw_action, squashed = _unsquash_action(action, action_low, action_high, eps)
    action_log = distri.log_prob(raw_action) - torch.log(1.0 - squashed.pow(2) + eps)
    entropy = distri.entropy()
    if agent_mask is not None:
        mask = agent_mask.to(**tpdv).unsqueeze(-1)
        action_log = action_log * mask
        entropy = entropy * mask
    return action_log, entropy
