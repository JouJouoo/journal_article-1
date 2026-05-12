import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from dp_lcrl_rl.algorithms.utils.util import check, init
from dp_lcrl_rl.algorithms.utils.transformer_act import discrete_autoregreesive_act
from dp_lcrl_rl.algorithms.utils.transformer_act import discrete_parallel_act
from dp_lcrl_rl.algorithms.utils.transformer_act import continuous_autoregreesive_act
from dp_lcrl_rl.algorithms.utils.transformer_act import continuous_parallel_act

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))

        self.att_bp = None

    def forward(self, key, value, query, agent_mask=None, diag=None):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            causal_mask = torch.tril(torch.ones(L, L, device=att.device, dtype=torch.bool))
            att = att.masked_fill(~causal_mask.view(1, 1, L, L), float('-inf'))
        invalid_keys = None
        if agent_mask is not None:
            mask = agent_mask.to(att.device, dtype=att.dtype)
            invalid_keys = mask <= 0
            att = att.masked_fill(invalid_keys.unsqueeze(1).unsqueeze(2), float('-inf'))
        valid_rows = torch.isfinite(att).any(dim=-1, keepdim=True)
        att = torch.where(valid_rows, att, torch.zeros_like(att))
        att = F.softmax(att, dim=-1)
        att = torch.where(valid_rows, att, torch.zeros_like(att))
        if invalid_keys is not None and diag is not None and torch.any(invalid_keys):
            leak_mask = invalid_keys.unsqueeze(1).unsqueeze(2).to(att.dtype)
            leak_mass = (att * leak_mask).sum()
            diag["attn_leakage"] = diag.get("attn_leakage", 0.0) + float(leak_mass.detach().cpu().item())
            diag["attn_calls"] = diag.get("attn_calls", 0) + 1
        if diag is not None:
            log_att = torch.log(torch.clamp(att, min=1e-12))
            entropy = -(att * log_att).sum(dim=-1)  # (B, nh, L)
            head_entropy = entropy.mean(dim=-1)  # (B, nh)
            diag.setdefault("attn_entropy", []).append(head_entropy.detach().cpu())
        if agent_mask is not None:
            query_mask = agent_mask.to(att.device, dtype=att.dtype).unsqueeze(1).unsqueeze(3)
            att = att * query_mask

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, agent_mask=None, diag=None):
        x = self.ln1(x + self.attn(x, x, x, agent_mask, diag=diag))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc, agent_mask=None, diag=None):
        x = self.ln1(x + self.attn1(x, x, x, agent_mask, diag=diag))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc, agent_mask=agent_mask, diag=diag))
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, use_id_emb=True):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        self.use_id_emb = use_id_emb
        self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))
        nn.init.normal_(self.agent_id_emb, std=0.02)

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.ModuleList([EncodeBlock(n_embd, n_head) for _ in range(n_block)])
    def forward(self, state, obs, agent_mask=None, diag=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)
            x = obs_embeddings

        if self.use_id_emb:
            x = x + self.agent_id_emb
        x = self.ln(x)
        for block in self.blocks:
            x = block(x, agent_mask, diag=diag)
        return x


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=False, share_actor=False, use_id_emb=True):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type
        self.use_id_emb = use_id_emb

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

        if self.dec_actor:
            if self.share_actor:
                print("mac_dec!!!!!")
                self.mlp = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, action_dim)))
            else:
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(nn.LayerNorm(obs_dim),
                                          init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, action_dim)))
                    self.mlp.append(actor)
        else:
            self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))
            nn.init.normal_(self.agent_id_emb, std=0.02)
            if action_type == 'Discrete':
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                                    nn.GELU())
            else:
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
            self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                             init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
            self.ln = nn.LayerNorm(n_embd)
            self.blocks = nn.ModuleList([DecodeBlock(n_embd, n_head) for _ in range(n_block)])
            self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, action_dim)))
        self._diag_context = None

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, obs, agent_mask=None, diag=None):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        if diag is None:
            diag = self._diag_context
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            action_embeddings = self.action_encoder(action)
            x = action_embeddings
            if self.use_id_emb:
                x = x + self.agent_id_emb
            x = self.ln(x)
            for block in self.blocks:
                x = block(x, obs_rep, agent_mask, diag=diag)
            logit = self.head(x)
        if agent_mask is not None:
            logit = logit * agent_mask.unsqueeze(-1).to(logit.device, dtype=logit.dtype)

        return logit


class MultiAgentTransformer(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False, log_std_min=-5.0, log_std_max=2.0,
                 action_low=None, action_high=None, use_id_emb=True):
        super(MultiAgentTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self.use_id_emb = use_id_emb

        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent,
                               encode_state, use_id_emb=use_id_emb)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor,
                               use_id_emb=use_id_emb)
        self.value_head = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, 1)),
        )
        self._last_attn_stats = None
        if action_low is not None and action_high is not None:
            self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
            self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))
        else:
            self.action_low = None
            self.action_high = None
        self.to(device)
        # Keep clamp range configurable
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max


    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def _init_diag(self):
        return {"attn_leakage": 0.0, "attn_calls": 0, "attn_entropy": []}

    def _finalize_diag(self, diag):
        if not isinstance(diag, dict):
            return None
        calls = int(diag.get("attn_calls", 0))
        leak = float(diag.get("attn_leakage", 0.0))
        entropy_records = diag.get("attn_entropy")
        entropy_array = None
        if entropy_records:
            try:
                tensors = []
                for entry in entropy_records:
                    tensor = torch.as_tensor(entry, dtype=torch.float32).reshape(-1)
                    if tensor.numel() > 0:
                        tensors.append(tensor)
                if tensors:
                    entropy_array = torch.cat(tensors).cpu().numpy()
            except Exception:
                entropy_array = None
        result = {"attn_leakage": leak, "attn_calls": calls}
        if entropy_array is not None:
            result["attn_entropy"] = entropy_array
        if calls <= 0 and leak == 0.0 and "attn_entropy" not in result:
            return None
        return result

    def pop_attn_stats(self):
        stats = self._last_attn_stats
        self._last_attn_stats = None
        return stats

    def _masked_pool(self, rep, agent_mask):
        if agent_mask is None:
            return rep.mean(dim=1)
        weights = agent_mask.to(rep.dtype).unsqueeze(-1)
        denom = torch.clamp(weights.sum(dim=1), min=1.0)
        return (rep * weights).sum(dim=1) / denom

    def _value_from_rep(self, rep, agent_mask):
        pooled = self._masked_pool(rep, agent_mask)
        scalar = self.value_head(pooled)
        return scalar.unsqueeze(1).repeat(1, self.n_agent, 1)

    def forward(self, state, obs, action, available_actions=None, agent_mask=None):
        # Keep policy log_std stable
        if hasattr(self, 'decoder') and hasattr(self.decoder, 'log_std') and self.decoder.log_std is not None:
            with torch.no_grad():
                self.decoder.log_std.data.clamp_(self.log_std_min, self.log_std_max)
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        if agent_mask is None:
            agent_mask = torch.ones((*obs.shape[:2],), **self.tpdv)
        else:
            agent_mask = check(agent_mask).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        diag = self._init_diag()
        obs_rep = self.encoder(state, obs, agent_mask, diag=diag)
        self.decoder._diag_context = diag
        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions,
                                                        agent_mask=agent_mask)
        else:
            action_log, entropy = continuous_parallel_act(
                self.decoder,
                obs_rep,
                obs,
                action,
                batch_size,
                self.n_agent,
                self.action_dim,
                self.tpdv,
                agent_mask=agent_mask,
                action_low=self.action_low,
                action_high=self.action_high,
            )
        self.decoder._diag_context = None

        v_tot = self._value_from_rep(obs_rep, agent_mask)
        self._last_attn_stats = self._finalize_diag(diag)
        return action_log, v_tot, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False, agent_mask=None):
        # Keep policy log_std stable
        if hasattr(self, 'decoder') and hasattr(self.decoder, 'log_std') and self.decoder.log_std is not None:
            with torch.no_grad():
                self.decoder.log_std.data.clamp_(self.log_std_min, self.log_std_max)
        # state unused
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if agent_mask is None:
            agent_mask = torch.ones((*obs.shape[:2],), **self.tpdv)
        else:
            agent_mask = check(agent_mask).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        diag = self._init_diag()
        obs_rep = self.encoder(state, obs, agent_mask, diag=diag)
        self.decoder._diag_context = diag
        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic,
                                                                           agent_mask=agent_mask)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                             self.n_agent, self.action_dim, self.tpdv,
                                                                             deterministic, agent_mask=agent_mask,
                                                                             action_low=self.action_low,
                                                                             action_high=self.action_high)
        self.decoder._diag_context = None

        v_tot = self._value_from_rep(obs_rep, agent_mask)
        self._last_attn_stats = self._finalize_diag(diag)
        return output_action, output_action_log, v_tot

    def get_values(self, state, obs, agent_mask=None):
        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if agent_mask is None:
            agent_mask = torch.ones((*obs.shape[:2],), **self.tpdv)
        else:
            agent_mask = check(agent_mask).to(**self.tpdv)
        diag = self._init_diag()
        obs_rep = self.encoder(state, obs, agent_mask, diag=diag)
        v_tot = self._value_from_rep(obs_rep, agent_mask)
        self._last_attn_stats = self._finalize_diag(diag)
        return v_tot




