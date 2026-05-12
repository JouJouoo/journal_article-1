#!/usr/bin/env python3
"""Paper-aligned P2P low-carbon trading environment without blockchain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import random

import numpy as np

try:  # gymnasium is optional in some deployments
    from gymnasium import spaces  # type: ignore
except Exception:  # pragma: no cover
    class spaces:  # type: ignore
        class Box:
            def __init__(self, low, high, shape, dtype) -> None:
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype


TRADE_Q_IDX = 0
QUOTE_IDX = 1
ESS_POWER_IDX = 2


@dataclass
class StorageSpec:
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    e_max: float = 12.0
    p_ch_max: float = 3.0
    p_dis_max: float = 3.0
    cycle_cost: float = 0.01


@dataclass
class MarketSpec:
    base_buy: float = 1.0
    base_sell: float = 0.2
    buy_amp: float = 0.25
    sell_amp: float = 0.10
    spread_min: float = 0.08
    carbon_price: float = 0.08
    carbon_price_alpha: float = 0.20
    dynamic_carbon_price: bool = False
    grid_carbon_factor: float = 0.70
    pv_carbon_factor: float = 0.00  # 占位符，实际在 __post_init__ 中设为 -grid_carbon_factor
    price_buy: List[float] = field(init=False)
    price_sell: List[float] = field(init=False)

    def __post_init__(self) -> None:
        self.pv_carbon_factor = -self.grid_carbon_factor
        self.recompute_prices()

    def recompute_prices(self) -> None:
        self.price_buy = []
        self.price_sell = []
        for hour in range(24):
            buy = self.base_buy + self.buy_amp * math.sin(hour / 24.0 * 2.0 * math.pi)
            raw_sell = self.base_sell + self.sell_amp * math.sin((hour + 6) / 24.0 * 2.0 * math.pi)
            sell = min(raw_sell, buy - self.spread_min)
            self.price_buy.append(float(max(0.0, buy)))
            self.price_sell.append(float(max(0.0, sell)))


@dataclass
class ParticipationSpec:
    min_agents: int = 5
    step_churn_prob: float = 0.0


@dataclass
class AgentState:
    id: int
    energy: float = 0.0
    carbon_mass: float = 0.0
    pv: float = 0.0
    load: float = 0.0
    dynamic_intensity: float = 0.0
    storage_intensity: float = 0.0
    sell_intensity: float = 0.0


class DPLCRLPaperEnv:
    """Environment matching the paper's CMTM + DP-LCRL setting."""

    def __init__(
        self,
        max_agents: int = 30,
        horizon: int = 24,
        seed: int = 0,
        min_agents: Optional[int] = None,
        storage_spec: Optional[StorageSpec] = None,
        market_spec: Optional[MarketSpec] = None,
        step_churn_prob: float = 0.0,
        price_quote_range: Optional[Tuple[float, float]] = None,
        storage_capacity_variance: float = 0.25,
        storage_capacity_range: Optional[Tuple[float, float]] = (6.0, 18.0),
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
        self.max_agents = int(max(1, max_agents))
        self.n = self.max_agents
        self.horizon = int(max(1, horizon))
        self.seed(seed)

        self.storage_spec = storage_spec or StorageSpec()
        self.market_spec = market_spec or MarketSpec()
        if dynamic_carbon_price is not None:
            self.market_spec.dynamic_carbon_price = bool(dynamic_carbon_price)

        participation_min = self.max_agents if min_agents is None else int(min_agents)
        self.participation_spec = ParticipationSpec(
            min_agents=max(1, min(participation_min, self.max_agents)),
            step_churn_prob=float(max(0.0, min(1.0, step_churn_prob))),
        )

        if storage_capacity_range is None:
            cap_lo = cap_hi = float(self.storage_spec.e_max)
        else:
            cap_lo, cap_hi = sorted(map(float, storage_capacity_range))
        cap_lo = max(1.0, cap_lo)
        cap_hi = max(cap_lo, cap_hi)
        self._capacity_range = (cap_lo, cap_hi)
        self._storage_capacity_variance = float(max(0.0, storage_capacity_variance))
        self.agent_energy_caps = self._build_agent_caps()

        self.pv_peak_scale = float(max(0.5, pv_peak_scale))
        self.pv_phase_jitter = float(max(0.0, pv_phase_jitter))
        self.load_phase_jitter = float(max(0.0, load_phase_jitter))
        self.load_base = float(max(0.1, load_base))
        self.load_peak = float(max(self.load_base, load_peak))
        self.cmtm_mode = self._normalize_cmtm_mode(cmtm_mode)
        self.mask_mode = str(mask_mode).strip().lower()
        self.p2p_reward_weight = float(max(0.0, p2p_reward_weight))
        self.grid_buy_penalty_weight = float(max(0.0, grid_buy_penalty_weight))
        self.unmatched_penalty_weight = float(max(0.0, unmatched_penalty_weight))
        self.pv_profiles, self.load_profiles = self._build_profiles()

        if price_quote_range is None:
            quote_lo = min(self.market_spec.price_sell)
            quote_hi = max(self.market_spec.price_buy)
        else:
            quote_lo, quote_hi = sorted(map(float, price_quote_range))
        self.quote_price_range = (float(max(0.0, quote_lo)), float(max(quote_lo, quote_hi)))

        self.obs_dim = 12
        self.act_dim = 3
        self.cent_obs_dim = self.obs_dim * self.max_agents
        self._crfm_eps = 1e-9

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.cent_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cent_obs_dim,),
            dtype=np.float32,
        )

        self.agent_mask = np.ones(self.max_agents, dtype=np.float32)
        self.active_ids: List[int] = list(range(self.max_agents))
        self.states: List[AgentState] = []
        self.current_carbon_price = float(self.market_spec.carbon_price)
        self.last_info: Dict[str, Any] = {}
        self.last_p2p_deals: List[Tuple[int, int, float, float]] = []
        self.t = 0

    @staticmethod
    def _normalize_cmtm_mode(cmtm_mode: str) -> str:
        mode = str(cmtm_mode).strip().lower()
        if mode not in {"full", "stateless"}:
            raise ValueError(f"Unsupported cmtm_mode={cmtm_mode}. Expected one of ['full', 'stateless'].")
        return mode

    @property
    def min_agents(self) -> int:
        return int(self.participation_spec.min_agents)

    @min_agents.setter
    def min_agents(self, value: int) -> None:
        self.participation_spec.min_agents = max(1, min(int(value), self.max_agents))

    @property
    def step_churn_prob(self) -> float:
        return float(self.participation_spec.step_churn_prob)

    @step_churn_prob.setter
    def step_churn_prob(self, value: float) -> None:
        self.participation_spec.step_churn_prob = float(max(0.0, min(1.0, value)))

    def seed(self, seed: int = 0) -> None:
        self._seed = int(seed)
        self.rng = np.random.default_rng(self._seed)
        random.seed(self._seed)

    def _build_agent_caps(self) -> np.ndarray:
        lo, hi = self._capacity_range
        caps = self.rng.uniform(lo, hi, size=self.max_agents).astype(np.float32)
        if self._storage_capacity_variance > 0.0:
            scale = self.rng.uniform(
                1.0 - self._storage_capacity_variance,
                1.0 + self._storage_capacity_variance,
                size=self.max_agents,
            ).astype(np.float32)
            caps *= scale
            np.clip(caps, lo, hi, out=caps)
        return caps

    def _build_profiles(self) -> Tuple[np.ndarray, np.ndarray]:
        hours = np.arange(self.horizon, dtype=np.float32)
        pv_profiles: List[np.ndarray] = []
        load_profiles: List[np.ndarray] = []
        for agent_idx in range(self.max_agents):
            pv_phase = agent_idx * (math.pi / max(1, self.max_agents - 1))
            if self.pv_phase_jitter > 0.0:
                pv_phase += float(self.rng.uniform(-self.pv_phase_jitter, self.pv_phase_jitter))
            pv_wave = np.maximum(
                0.0,
                np.sin(((hours - 6.0) / 24.0 * 2.0 * math.pi) + pv_phase),
            )
            pv_scale = self.pv_peak_scale * (1.0 + 0.15 * math.sin(agent_idx))
            pv_profiles.append((pv_wave * pv_scale).astype(np.float32))

            load_center = 12.0 + (agent_idx - (self.max_agents - 1) / 2.0) * 0.7
            if self.load_phase_jitter > 0.0:
                load_center += float(self.rng.uniform(-self.load_phase_jitter, self.load_phase_jitter))
            width = 3.2 + 0.25 * math.sin(agent_idx)
            gaussian = np.exp(-0.5 * ((hours - load_center) / max(0.8, width)) ** 2)
            offset = self.load_base + 0.15 * agent_idx
            amplitude = self.load_peak - self.load_base + 0.25 * math.sin(agent_idx)
            load_profiles.append((offset + amplitude * gaussian).astype(np.float32))
        return np.stack(pv_profiles, axis=0), np.stack(load_profiles, axis=0)

    def _sample_active_ids(self) -> List[int]:
        if self.participation_spec.min_agents >= self.max_agents:
            return list(range(self.max_agents))
        count = int(
            self.rng.integers(self.participation_spec.min_agents, self.max_agents + 1)
        )
        ids = self.rng.choice(self.max_agents, size=count, replace=False)
        return sorted(int(x) for x in ids.tolist())

    def _refresh_agent_mask(self) -> None:
        self.active_ids = self._sample_active_ids()
        self.agent_mask.fill(0.0)
        self.agent_mask[self.active_ids] = 1.0

    def _maybe_resample_agent_mask(self) -> None:
        if self.participation_spec.step_churn_prob <= 0.0:
            return
        if float(self.rng.random()) >= self.participation_spec.step_churn_prob:
            return
        self._refresh_agent_mask()

    def _get_prices(self, step: int) -> Tuple[float, float]:
        idx = int(step % 24)
        buy = float(self.market_spec.price_buy[idx])
        sell = float(self.market_spec.price_sell[idx])
        sell = min(sell, buy - self.market_spec.spread_min)
        return buy, max(0.0, sell)

    def _map_scalar(self, raw_value: float, lo: float, hi: float) -> float:
        raw = float(np.clip(raw_value, -1.0, 1.0))
        alpha = 0.5 * (raw + 1.0)
        return float(lo + alpha * (hi - lo))

    def _zero_info(self, agent_idx: int) -> Dict[str, Any]:
        return dict(
            agent_id=int(agent_idx),
            active=False,
            pv_generation=0.0,
            load_demand=0.0,
            soc=0.0,
            E_before=0.0,
            E_after=0.0,
            storage_charge=0.0,
            storage_discharge=0.0,
            p2p_buy_planned=0.0,
            p2p_sell_planned=0.0,
            p2p_buy_executed=0.0,
            p2p_sell_executed=0.0,
            p2p_price=None,
            p2p_value=0.0,
            grid_buy=0.0,
            grid_sell=0.0,
            grid_value=0.0,
            energy_revenue=0.0,
            carbon_settlement=0.0,
            p2p_reward_bonus=0.0,
            grid_buy_penalty=0.0,
            unmatched_penalty=0.0,
            reward=0.0,
            carbon_grid_import=0.0,
            carbon_p2p_import=0.0,
            carbon_storage_discharge=0.0,
            carbon_pv_source=0.0,
            carbon_load_responsibility=0.0,
            carbon_charge_responsibility=0.0,
            carbon_grid_export=0.0,
            carbon_storage_delta=0.0,
            carbon_baseline_no_trade=0.0,
            low_carbon_contribution=0.0,
            C_dynamic=0.0,
            C_storage_avg=0.0,
            C_sell=0.0,
        )

    def reset(self) -> List[np.ndarray]:
        self.t = 0
        self.current_carbon_price = float(self.market_spec.carbon_price)
        self.last_p2p_deals = []
        self.states = []
        for agent_idx in range(self.max_agents):
            cap = float(self.agent_energy_caps[agent_idx])
            energy = float(self.rng.uniform(0.15 * cap, 0.55 * cap))
            if self.cmtm_mode == "full":
                carbon_mass = energy * float(self.market_spec.grid_carbon_factor)
                storage_intensity = float(self.market_spec.grid_carbon_factor)
            else:
                carbon_mass = 0.0
                storage_intensity = 0.0
            self.states.append(
                AgentState(
                    id=agent_idx,
                    energy=energy,
                    carbon_mass=carbon_mass,
                    storage_intensity=storage_intensity,
                    dynamic_intensity=float(self.market_spec.grid_carbon_factor),
                    sell_intensity=float(self.market_spec.grid_carbon_factor),
                )
            )
        self._refresh_agent_mask()
        self._update_profiles_for_step(0)
        obs = self._get_obs()
        self.last_info = {
            "agent_mask": self.agent_mask.astype(float).tolist(),
            "n_active_agents": int(np.sum(self.agent_mask)),
            "active_agent_ids": list(self.active_ids),
            "cent_obs": self._get_cent_obs().tolist(),
            "cmtm_mode": self.cmtm_mode,
        }
        return obs

    def _update_profiles_for_step(self, step: int) -> None:
        idx = int(step % self.horizon)
        for agent_idx, state in enumerate(self.states):
            pv = float(self.pv_profiles[agent_idx, idx] * self.rng.uniform(0.9, 1.1))
            load = float(self.load_profiles[agent_idx, idx] * self.rng.uniform(0.92, 1.08))
            state.pv = max(0.0, pv)
            state.load = max(0.01, load)

    def _get_obs(self) -> List[np.ndarray]:
        buy_price, sell_price = self._get_prices(self.t)
        obs: List[np.ndarray] = []
        for agent_idx, state in enumerate(self.states):
            if self.agent_mask[agent_idx] < 0.5 and self.mask_mode == "full":
                obs.append(np.zeros(self.obs_dim, dtype=np.float32))
                continue
            cap = float(max(1e-6, self.agent_energy_caps[agent_idx]))
            soc = float(np.clip(state.energy / cap, 0.0, 1.0))
            obs.append(
                np.array(
                    [
                        state.pv,
                        state.load,
                        cap,
                        soc,
                        float(state.dynamic_intensity),
                        float(state.storage_intensity),
                        float(state.sell_intensity),
                        float(self.current_carbon_price),
                        float(buy_price),
                        float(sell_price),
                        float(self.market_spec.pv_carbon_factor),
                        float(self.market_spec.grid_carbon_factor),
                    ],
                    dtype=np.float32,
                )
            )
        return obs

    def _get_cent_obs(self) -> np.ndarray:
        return np.concatenate(self._get_obs(), axis=0).astype(np.float32)

    def _decode_actions(
        self,
        actions_per_agent: Sequence[np.ndarray],
        grid_buy_price: float,
        grid_sell_price: float,
    ) -> List[Dict[str, float]]:
        if len(actions_per_agent) != self.max_agents:
            raise ValueError(
                f"Expected {self.max_agents} actions, got {len(actions_per_agent)}."
            )

        decoded: List[Dict[str, float]] = []
        for agent_idx in range(self.max_agents):
            raw = np.asarray(actions_per_agent[agent_idx], dtype=np.float32).reshape(-1)
            if raw.shape[0] != self.act_dim:
                raise ValueError(
                    f"Agent {agent_idx} action dim mismatch: expected {self.act_dim}, got {raw.shape[0]}."
                )
            if self.agent_mask[agent_idx] < 0.5:
                decoded.append(
                    dict(
                        trade_signal=0.0,
                        quote=0.0,
                        ess_signal=0.0,
                        trade_signed=0.0,
                        buy_qty=0.0,
                        sell_qty=0.0,
                        quote_price=0.0,
                        ess_net=0.0,
                        charge=0.0,
                        discharge=0.0,
                    )
                )
                continue

            state = self.states[agent_idx]
            cap = float(max(1e-6, self.agent_energy_caps[agent_idx]))
            eta_ch = float(self.storage_spec.eta_ch)
            eta_dis = float(max(1e-6, self.storage_spec.eta_dis))

            charge_headroom = max(0.0, (cap - state.energy) / max(eta_ch, 1e-6))
            discharge_headroom = max(0.0, state.energy * eta_dis)
            feasible_charge = min(float(self.storage_spec.p_ch_max), charge_headroom)
            feasible_discharge = min(float(self.storage_spec.p_dis_max), discharge_headroom)

            ess_signal = float(np.clip(raw[ESS_POWER_IDX], -1.0, 1.0))
            if ess_signal >= 0.0:
                discharge = ess_signal * feasible_discharge
                charge = 0.0
            else:
                charge = (-ess_signal) * feasible_charge
                discharge = 0.0
            ess_net = discharge - charge

            sell_cap = max(0.0, state.pv + discharge - state.load - charge)
            buy_cap = max(0.0, state.load + charge - state.pv - discharge)
            trade_signal = float(np.clip(raw[TRADE_Q_IDX], -1.0, 1.0))
            if trade_signal >= 0.0:
                sell_qty = trade_signal * sell_cap
                buy_qty = 0.0
                trade_signed = sell_qty
            else:
                buy_qty = (-trade_signal) * buy_cap
                sell_qty = 0.0
                trade_signed = -buy_qty

            price_lo = min(grid_sell_price, self.quote_price_range[0])
            price_hi = max(grid_buy_price, self.quote_price_range[1])
            quote_price = self._map_scalar(float(raw[QUOTE_IDX]), price_lo, price_hi)

            decoded.append(
                dict(
                    trade_signal=trade_signal,
                    quote=float(raw[QUOTE_IDX]),
                    ess_signal=ess_signal,
                    trade_signed=float(trade_signed),
                    buy_qty=float(buy_qty),
                    sell_qty=float(sell_qty),
                    quote_price=float(quote_price),
                    ess_net=float(ess_net),
                    charge=float(charge),
                    discharge=float(discharge),
                )
            )
        return decoded

    @staticmethod
    def _match_cda(
        sellers: List[Tuple[int, float, float]],
        buyers: List[Tuple[int, float, float]],
    ) -> List[Tuple[int, int, float, float]]:
        seller_book = sorted(sellers, key=lambda item: item[2])
        buyer_book = sorted(buyers, key=lambda item: -item[2])
        deals: List[Tuple[int, int, float, float]] = []
        s_idx = 0
        b_idx = 0
        while s_idx < len(seller_book) and b_idx < len(buyer_book):
            s_id, s_qty, s_price = seller_book[s_idx]
            b_id, b_qty, b_price = buyer_book[b_idx]
            if s_qty <= 1e-9:
                s_idx += 1
                continue
            if b_qty <= 1e-9:
                b_idx += 1
                continue
            if b_price < s_price:
                s_idx += 1
                continue
            qty = min(s_qty, b_qty)
            price = 0.5 * (s_price + b_price)
            deals.append((s_id, b_id, float(qty), float(price)))
            seller_book[s_idx] = (s_id, s_qty - qty, s_price)
            buyer_book[b_idx] = (b_id, b_qty - qty, b_price)
        return deals

    def _solve_output_intensity(
        self,
        p2p_matrix: np.ndarray,
        total_out_energy: np.ndarray,
        source_carbon: np.ndarray,
    ) -> np.ndarray:
        total_out = np.asarray(total_out_energy, dtype=np.float64).reshape(-1)
        rhs = np.asarray(source_carbon, dtype=np.float64).reshape(-1).copy()
        if total_out.size == 0:
            return np.zeros(0, dtype=np.float32)

        system = -np.asarray(p2p_matrix, dtype=np.float64).T
        for idx in range(total_out.size):
            system[idx, idx] += float(total_out[idx])
            if total_out[idx] <= self._crfm_eps:
                system[idx, :] = 0.0
                system[idx, idx] = 1.0
                rhs[idx] = 0.0

        try:
            rho_out = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            rho_out, *_ = np.linalg.lstsq(system, rhs, rcond=None)
        rho_out = np.asarray(rho_out, dtype=np.float64)
        rho_out[~np.isfinite(rho_out)] = 0.0
        # Negative values represent PV offset credits and must remain traceable
        # through P2P carbon-responsibility accounting.
        return rho_out.astype(np.float32)

    def step(self, actions_per_agent: Sequence[np.ndarray]):
        step_index = int(self.t)
        self._update_profiles_for_step(step_index)
        grid_buy_price, grid_sell_price = self._get_prices(step_index)
        step_agent_mask = self.agent_mask.copy()
        step_active_ids = [int(idx) for idx in np.flatnonzero(step_agent_mask > 0.5)]
        step_active_count = len(step_active_ids)

        decoded = self._decode_actions(actions_per_agent, grid_buy_price, grid_sell_price)

        sellers = [
            (agent_idx, item["sell_qty"], item["quote_price"])
            for agent_idx, item in enumerate(decoded)
            if item["sell_qty"] > 1e-9 and step_agent_mask[agent_idx] > 0.5
        ]
        buyers = [
            (agent_idx, item["buy_qty"], item["quote_price"])
            for agent_idx, item in enumerate(decoded)
            if item["buy_qty"] > 1e-9 and step_agent_mask[agent_idx] > 0.5
        ]
        deals = self._match_cda(sellers, buyers)
        self.last_p2p_deals = deals

        per_agent: List[Dict[str, float]] = [dict() for _ in range(self.max_agents)]
        for agent_idx in range(self.max_agents):
            per_agent[agent_idx].update(
                p2p_buy=0.0,
                p2p_sell=0.0,
                p2p_value=0.0,
                matched_price=0.0,
            )
        for s_id, b_id, qty, price in deals:
            per_agent[s_id]["p2p_sell"] += float(qty)
            per_agent[b_id]["p2p_buy"] += float(qty)
            per_agent[s_id]["p2p_value"] += float(qty * price)
            per_agent[b_id]["p2p_value"] -= float(qty * price)
            per_agent[s_id]["matched_price"] = float(price)
            per_agent[b_id]["matched_price"] = float(price)

        p2p_matrix = np.zeros((self.max_agents, self.max_agents), dtype=np.float64)
        for s_id, b_id, qty, _ in deals:
            p2p_matrix[s_id, b_id] += float(qty)

        grid_buy = np.zeros(self.max_agents, dtype=np.float64)
        grid_sell = np.zeros(self.max_agents, dtype=np.float64)
        source_carbon = np.zeros(self.max_agents, dtype=np.float64)
        storage_discharge_carbon = np.zeros(self.max_agents, dtype=np.float64)
        carbon_p2p_import = np.zeros(self.max_agents, dtype=np.float64)
        carbon_pv_source = np.zeros(self.max_agents, dtype=np.float64)
        carbon_grid_import = np.zeros(self.max_agents, dtype=np.float64)
        carbon_charge = np.zeros(self.max_agents, dtype=np.float64)
        carbon_load = np.zeros(self.max_agents, dtype=np.float64)
        carbon_grid_export = np.zeros(self.max_agents, dtype=np.float64)
        carbon_baseline = np.zeros(self.max_agents, dtype=np.float64)
        low_carbon_contribution = np.zeros(self.max_agents, dtype=np.float64)
        output_intensity = np.zeros(self.max_agents, dtype=np.float64)
        storage_prev_intensity = np.zeros(self.max_agents, dtype=np.float64)
        carbon_storage_delta = np.zeros(self.max_agents, dtype=np.float64)
        energy_output_basis = np.zeros(self.max_agents, dtype=np.float64)
        carbon_flow_matrix = np.zeros((self.max_agents, self.max_agents), dtype=np.float64)

        for agent_idx, state in enumerate(self.states):
            if step_agent_mask[agent_idx] < 0.5:
                continue
            action = decoded[agent_idx]
            p2p_in = float(per_agent[agent_idx]["p2p_buy"])
            p2p_out = float(per_agent[agent_idx]["p2p_sell"])
            load = float(state.load)
            charge = float(action["charge"])
            discharge = float(action["discharge"])
            residual = state.pv + discharge + p2p_in - (load + charge + p2p_out)
            if residual >= 0.0:
                grid_sell[agent_idx] = residual
            else:
                grid_buy[agent_idx] = -residual

            pv_source = float(state.pv) * float(self.market_spec.pv_carbon_factor)
            grid_source = float(grid_buy[agent_idx]) * float(self.market_spec.grid_carbon_factor)
            carbon_pv_source[agent_idx] = pv_source
            carbon_grid_import[agent_idx] = grid_source
            source_carbon[agent_idx] = pv_source + grid_source
            if self.cmtm_mode == "full":
                storage_prev_intensity[agent_idx] = (
                    float(state.carbon_mass / max(state.energy, self._crfm_eps))
                    if state.energy > self._crfm_eps
                    else 0.0
                )
                storage_discharge_carbon[agent_idx] = discharge * storage_prev_intensity[agent_idx]
            energy_output_basis[agent_idx] = load + charge + grid_sell[agent_idx] + p2p_out

        if self.cmtm_mode == "full":
            output_intensity[:] = self._solve_output_intensity(
                p2p_matrix=p2p_matrix,
                total_out_energy=energy_output_basis,
                source_carbon=source_carbon + storage_discharge_carbon,
            )
        else:
            provisional_output = self._solve_output_intensity(
                p2p_matrix=p2p_matrix,
                total_out_energy=energy_output_basis,
                source_carbon=source_carbon,
            ).astype(np.float64)
            fallback = (energy_output_basis > self._crfm_eps) & (provisional_output <= self._crfm_eps)
            provisional_output[fallback] = float(self.market_spec.grid_carbon_factor)
            for agent_idx, action in enumerate(decoded):
                if step_agent_mask[agent_idx] < 0.5:
                    continue
                storage_discharge_carbon[agent_idx] = float(action["discharge"]) * float(provisional_output[agent_idx])
            output_intensity[:] = self._solve_output_intensity(
                p2p_matrix=p2p_matrix,
                total_out_energy=energy_output_basis,
                source_carbon=source_carbon + storage_discharge_carbon,
            )
        carbon_flow_matrix[:, :] = output_intensity[:, None] * p2p_matrix
        carbon_p2p_import[:] = np.sum(carbon_flow_matrix, axis=0)

        if self.market_spec.dynamic_carbon_price:
            total_baseline = float(
                np.sum([state.load for state in self.states]) * self.market_spec.grid_carbon_factor
            )
            total_actual = float(
                np.sum(output_intensity * np.array([state.load for state in self.states], dtype=np.float64))
            )
            pressure = 0.0 if total_baseline <= self._crfm_eps else (total_actual - total_baseline) / total_baseline
            updated_price = self.market_spec.carbon_price * (
                1.0 + self.market_spec.carbon_price_alpha * pressure
            )
            self.current_carbon_price = float(max(0.0, updated_price))
        else:
            self.current_carbon_price = float(self.market_spec.carbon_price)

        rewards = np.zeros(self.max_agents, dtype=np.float32)
        info_per_agent: List[Dict[str, Any]] = []
        responsibility_updates: List[Dict[str, Any]] = []
        settlement_records: List[Dict[str, Any]] = []
        grid_buy_penalty_total = 0.0
        unmatched_penalty_total = 0.0

        for s_id, b_id, qty, price in deals:
            settlement_records.append(
                dict(
                    record_type="p2p",
                    seller_id=int(s_id),
                    buyer_id=int(b_id),
                    quantity=float(qty),
                    price=float(price),
                    carbon_intensity=float(output_intensity[s_id]),
                    carbon_responsibility=float(carbon_flow_matrix[s_id, b_id]),
                )
            )

        for agent_idx, state in enumerate(self.states):
            if step_agent_mask[agent_idx] < 0.5:
                info_per_agent.append(self._zero_info(agent_idx))
                continue

            action = decoded[agent_idx]
            cap = float(self.agent_energy_caps[agent_idx])
            load = float(state.load)
            charge = float(action["charge"])
            discharge = float(action["discharge"])
            p2p_in = float(per_agent[agent_idx]["p2p_buy"])
            p2p_out = float(per_agent[agent_idx]["p2p_sell"])

            carbon_load[agent_idx] = float(output_intensity[agent_idx]) * load
            carbon_charge[agent_idx] = float(output_intensity[agent_idx]) * charge
            carbon_grid_export[agent_idx] = float(output_intensity[agent_idx]) * float(grid_sell[agent_idx])
            carbon_baseline[agent_idx] = load * float(self.market_spec.grid_carbon_factor)
            low_carbon_contribution[agent_idx] = carbon_baseline[agent_idx] - carbon_load[agent_idx]

            energy_before = float(state.energy)
            carbon_before = float(state.carbon_mass)
            state.energy = energy_before + self.storage_spec.eta_ch * charge - (
                discharge / max(self.storage_spec.eta_dis, self._crfm_eps)
            )
            state.energy = float(np.clip(state.energy, 0.0, cap))
            if self.cmtm_mode == "full":
                state.carbon_mass = carbon_before + (
                    self.storage_spec.eta_ch * charge * float(output_intensity[agent_idx])
                ) - (
                    discharge / max(self.storage_spec.eta_dis, self._crfm_eps)
                ) * float(storage_prev_intensity[agent_idx])
                state.carbon_mass = float(max(0.0, state.carbon_mass))
                carbon_storage_delta[agent_idx] = state.carbon_mass - carbon_before
                state.storage_intensity = (
                    float(state.carbon_mass / max(state.energy, self._crfm_eps))
                    if state.energy > self._crfm_eps
                    else 0.0
                )
            else:
                state.carbon_mass = 0.0
                carbon_storage_delta[agent_idx] = 0.0
                state.storage_intensity = 0.0
            state.dynamic_intensity = (
                float(carbon_load[agent_idx] / max(load, self._crfm_eps))
                if load > self._crfm_eps
                else 0.0
            )
            state.sell_intensity = float(output_intensity[agent_idx])

            p2p_value = float(per_agent[agent_idx]["p2p_value"])
            grid_value = float(grid_sell[agent_idx] * grid_sell_price - grid_buy[agent_idx] * grid_buy_price)
            energy_revenue = p2p_value + grid_value
            carbon_settlement = float(self.current_carbon_price) * float(low_carbon_contribution[agent_idx])
            planned_p2p = float(action["buy_qty"] + action["sell_qty"])
            executed_p2p = float(p2p_in + p2p_out)
            unmatched_p2p = max(0.0, planned_p2p - executed_p2p)
            p2p_reward_bonus = self.p2p_reward_weight * (p2p_in + p2p_out)
            grid_buy_penalty = self.grid_buy_penalty_weight * float(grid_buy[agent_idx])
            unmatched_penalty = self.unmatched_penalty_weight * unmatched_p2p
            storage_cost = float(self.storage_spec.cycle_cost) * (charge + discharge)
            reward = (
                energy_revenue
                + carbon_settlement
                + p2p_reward_bonus
                - grid_buy_penalty
                - unmatched_penalty
                - storage_cost
            )
            rewards[agent_idx] = float(reward)
            grid_buy_penalty_total += float(grid_buy_penalty)
            unmatched_penalty_total += float(unmatched_penalty)

            info_per_agent.append(
                dict(
                    agent_id=int(agent_idx),
                    active=True,
                    pv_generation=float(state.pv),
                    load_demand=float(load),
                    soc=float(np.clip(state.energy / max(cap, self._crfm_eps), 0.0, 1.0)),
                    E_before=float(energy_before),
                    E_after=float(state.energy),
                    storage_charge=float(charge),
                    storage_discharge=float(discharge),
                    ess_net=float(action["ess_net"]),
                    p2p_trade_signed=float(action["trade_signed"]),
                    p2p_buy_planned=float(action["buy_qty"]),
                    p2p_sell_planned=float(action["sell_qty"]),
                    p2p_buy_executed=float(p2p_in),
                    p2p_sell_executed=float(p2p_out),
                    p2p_price=float(per_agent[agent_idx]["matched_price"]) if (p2p_in + p2p_out) > 0.0 else None,
                    p2p_quote=float(action["quote_price"]),
                    p2p_value=float(p2p_value),
                    grid_buy=float(grid_buy[agent_idx]),
                    grid_sell=float(grid_sell[agent_idx]),
                    grid_value=float(grid_value),
                    energy_revenue=float(energy_revenue),
                    carbon_settlement=float(carbon_settlement),
                    p2p_reward_bonus=float(p2p_reward_bonus),
                    grid_buy_penalty=float(grid_buy_penalty),
                    unmatched_penalty=float(unmatched_penalty),
                    reward=float(reward),
                    carbon_grid_import=float(carbon_grid_import[agent_idx]),
                    carbon_p2p_import=float(carbon_p2p_import[agent_idx]),
                    carbon_storage_discharge=float(storage_discharge_carbon[agent_idx]),
                    carbon_pv_source=float(carbon_pv_source[agent_idx]),
                    carbon_load_responsibility=float(carbon_load[agent_idx]),
                    carbon_charge_responsibility=float(carbon_charge[agent_idx]),
                    carbon_grid_export=float(carbon_grid_export[agent_idx]),
                    carbon_storage_delta=float(carbon_storage_delta[agent_idx]),
                    carbon_baseline_no_trade=float(carbon_baseline[agent_idx]),
                    low_carbon_contribution=float(low_carbon_contribution[agent_idx]),
                    C_dynamic=float(state.dynamic_intensity),
                    C_storage_avg=float(state.storage_intensity),
                    C_sell=float(state.sell_intensity),
                )
            )
            responsibility_updates.append(
                dict(
                    agent_id=int(agent_idx),
                    energy=float(state.energy),
                    carbon_mass=float(state.carbon_mass),
                    C_dynamic=float(state.dynamic_intensity),
                    C_storage_avg=float(state.storage_intensity),
                    C_sell=float(state.sell_intensity),
                )
            )

        self.t += 1
        done = bool(self.t >= self.horizon)
        if not done:
            self._maybe_resample_agent_mask()
            self._update_profiles_for_step(self.t)

        obs = self._get_obs()
        next_active_count = int(np.sum(self.agent_mask))
        p2p_total_volume = float(np.sum([deal[2] for deal in deals])) if deals else 0.0
        p2p_volume_mean_active = float(
            np.sum([item["p2p_buy"] + item["p2p_sell"] for item in per_agent]) / max(1, step_active_count)
        )
        grid_buy_total = float(np.sum(grid_buy))
        grid_sell_total = float(np.sum(grid_sell))
        load_responsibility_total = float(np.sum(carbon_load))
        global_reward = float(rewards.sum() / max(1, step_active_count))
        info: Dict[str, Any] = dict(
            agent_mask=self.agent_mask.astype(float).tolist(),
            n_active_agents=step_active_count,
            n_active_agents_next=next_active_count,
            active_agent_ids=step_active_ids,
            next_active_agent_ids=list(self.active_ids),
            global_reward=global_reward,
            cmtm_mode=self.cmtm_mode,
            cent_obs=self._get_cent_obs().tolist(),
            p2p_matrix=p2p_matrix.astype(float).tolist(),
            carbon_flow_matrix=carbon_flow_matrix.astype(float).tolist(),
            settlement_records=settlement_records,
            responsibility_state_updates=responsibility_updates,
            per_agent=info_per_agent,
            market_summary=dict(
                p2p_total_volume=p2p_total_volume,
                p2p_mean_active=p2p_volume_mean_active,
                p2p_average_price=float(np.mean([deal[3] for deal in deals])) if deals else None,
                grid_buy_total=grid_buy_total,
                grid_buy_mean_active=float(grid_buy_total / max(1, step_active_count)),
                grid_sell_total=grid_sell_total,
                grid_sell_mean_active=float(grid_sell_total / max(1, step_active_count)),
                grid_buy_penalty_total=float(grid_buy_penalty_total),
                grid_buy_penalty_mean_active=float(grid_buy_penalty_total / max(1, step_active_count)),
                unmatched_penalty_total=float(unmatched_penalty_total),
                unmatched_penalty_mean_active=float(unmatched_penalty_total / max(1, step_active_count)),
                carbon_price=float(self.current_carbon_price),
            ),
            carbon_trace=dict(
                source_injection=float(np.sum(source_carbon)),
                p2p_import=float(np.sum(carbon_p2p_import)),
                load_responsibility=load_responsibility_total,
                load_responsibility_mean_active=float(load_responsibility_total / max(1, step_active_count)),
                charge_responsibility=float(np.sum(carbon_charge)),
                grid_export=float(np.sum(carbon_grid_export)),
                storage_delta=float(np.sum(carbon_storage_delta)),
            ),
        )
        self.last_info = info
        return obs, rewards.astype(float).tolist(), done, info

    def close(self) -> None:
        return None


if __name__ == "__main__":  # pragma: no cover
    env = DPLCRLPaperEnv(max_agents=6, min_agents=3, horizon=4, seed=7)
    env.reset()
    print("obs_dim/cent_dim/act_dim:", env.observation_space.shape, env.cent_observation_space.shape, env.action_space.shape)
    print("active_mask:", env.agent_mask.tolist())
    for step in range(4):
        action_batch = [
            np.array([0.6 if env.agent_mask[i] > 0.5 else 0.0, 0.0, 0.2], dtype=np.float32)
            for i in range(env.max_agents)
        ]
        _, _, done, info = env.step(action_batch)
        print(
            f"t={step} active={info['n_active_agents']} reward={info['global_reward']:.4f} "
            f"p2p_vol={info['market_summary']['p2p_total_volume']:.4f}"
        )
        if done:
            break
