"""Training analytics for the paper-aligned MAT runner."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


class PaperTrainingAnalytics:
    """Collect episode metrics and export lightweight HTML/JSON summaries."""

    def __init__(
        self,
        run_dir: Path,
        num_agents: int,
        n_threads: int,
        episode_length: int,
        summary_filename: str = "paper_training_summary.html",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.num_agents = int(num_agents)
        self.n_threads = int(max(1, n_threads))
        self.episode_length = int(max(1, episode_length))
        self.summary_filename = str(summary_filename or "paper_training_summary.html")

        self._episode_counter = 0
        self._step_buffers: List[List[Dict[str, object]]] = [[] for _ in range(self.n_threads)]

        self.episode_summaries: List[Dict[str, object]] = []
        self.training_history: List[Dict[str, object]] = []
        self.eval_history: List[Dict[str, float]] = []
        self.resource_history: List[Dict[str, object]] = []
        self.attn_entropy_sum = 0.0
        self.attn_entropy_count = 0

    def set_thread_count(self, n_threads: int) -> None:
        n_threads = int(max(1, n_threads))
        if n_threads == self.n_threads and len(self._step_buffers) == n_threads:
            return
        self.n_threads = n_threads
        self._step_buffers = [[] for _ in range(self.n_threads)]

    def push_step(
        self,
        infos: Sequence[Dict[str, object]],
        rewards: np.ndarray,
        observations: np.ndarray,
        actions: Optional[np.ndarray] = None,
        phase: str = "train",
    ) -> None:
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        self.set_thread_count(len(infos))

        for thread_idx in range(self.n_threads):
            info = dict(infos[thread_idx] if thread_idx < len(infos) else {})
            record = {
                "phase": str(phase),
                "global_reward": float(info.get("global_reward", float(np.mean(rewards_arr[thread_idx])))),
                "n_active_agents": int(info.get("n_active_agents", self.num_agents)),
                "market_summary": dict(info.get("market_summary") or {}),
                "carbon_trace": dict(info.get("carbon_trace") or {}),
            }
            self._step_buffers[thread_idx].append(record)

    def finalize_batch(self, iteration_index: int) -> None:
        for thread_idx in range(self.n_threads):
            records = self._step_buffers[thread_idx]
            if not records:
                continue
            summary = self._summarize_episode(records)
            summary.update(
                episode_id=self._episode_counter,
                iteration_index=int(iteration_index),
                thread_index=int(thread_idx),
            )
            self.episode_summaries.append(summary)
            self._episode_counter += 1
            self._step_buffers[thread_idx] = []

    def _summarize_episode(self, records: Sequence[Dict[str, object]]) -> Dict[str, object]:
        phase = str(records[0].get("phase", "train"))
        active_counts = [int(record.get("n_active_agents", self.num_agents)) for record in records]
        global_rewards = [float(record.get("global_reward", 0.0)) for record in records]

        p2p_mean_active_values: List[float] = []
        grid_buy_mean_active_values: List[float] = []
        grid_sell_mean_active_values: List[float] = []
        carbon_price_values: List[float] = []
        carbon_responsibility_mean_values: List[float] = []
        p2p_import = 0.0
        source_injection = 0.0

        for record in records:
            market_summary = dict(record.get("market_summary") or {})
            carbon_trace = dict(record.get("carbon_trace") or {})
            active_count = max(1, int(record.get("n_active_agents", self.num_agents)))
            p2p_mean_active_values.append(
                float(
                    market_summary.get(
                        "p2p_mean_active",
                        (float(market_summary.get("p2p_total_volume", 0.0) or 0.0) / active_count),
                    )
                    or 0.0
                )
            )
            grid_buy_mean_active_values.append(
                float(
                    market_summary.get(
                        "grid_buy_mean_active",
                        (float(market_summary.get("grid_buy_total", 0.0) or 0.0) / active_count),
                    )
                    or 0.0
                )
            )
            grid_sell_mean_active_values.append(
                float(
                    market_summary.get(
                        "grid_sell_mean_active",
                        (float(market_summary.get("grid_sell_total", 0.0) or 0.0) / active_count),
                    )
                    or 0.0
                )
            )
            if market_summary.get("carbon_price") is not None:
                carbon_price_values.append(float(market_summary["carbon_price"]))
            carbon_responsibility_mean_values.append(
                float(
                    carbon_trace.get(
                        "load_responsibility_mean_active",
                        (float(carbon_trace.get("load_responsibility", 0.0) or 0.0) / active_count),
                    )
                    or 0.0
                )
            )
            p2p_import += float(carbon_trace.get("p2p_import", 0.0) or 0.0)
            source_injection += float(carbon_trace.get("source_injection", 0.0) or 0.0)

        p2p_mean_active = float(np.mean(p2p_mean_active_values)) if p2p_mean_active_values else 0.0
        grid_buy_mean_active = float(np.mean(grid_buy_mean_active_values)) if grid_buy_mean_active_values else 0.0
        grid_sell_mean_active = float(np.mean(grid_sell_mean_active_values)) if grid_sell_mean_active_values else 0.0
        carbon_responsibility_episode = (
            float(np.mean(carbon_responsibility_mean_values)) if carbon_responsibility_mean_values else 0.0
        )

        return {
            "phase": phase,
            "n_steps": len(records),
            "n_agents_mean": float(np.mean(active_counts)) if active_counts else float(self.num_agents),
            "n_agents_last": int(active_counts[-1]) if active_counts else int(self.num_agents),
            "average_global_reward": float(np.mean(global_rewards)) if global_rewards else 0.0,
            "reward_std": float(np.std(global_rewards)) if global_rewards else 0.0,
            "p2p_volume_mean_active": p2p_mean_active,
            "grid_buy_mean_active": grid_buy_mean_active,
            "grid_sell_mean_active": grid_sell_mean_active,
            "carbon_price_average": float(np.mean(carbon_price_values)) if carbon_price_values else 0.0,
            "carbon_responsibility_mean_active_episode": carbon_responsibility_episode,
            # Backward-compatible aliases kept for existing reports/scripts.
            "p2p_total_volume": p2p_mean_active,
            "grid_buy_total": grid_buy_mean_active,
            "grid_sell_total": grid_sell_mean_active,
            "load_responsibility_total": carbon_responsibility_episode,
            "p2p_import_total": float(p2p_import),
            "source_injection_total": float(source_injection),
        }

    def record_training_metrics(self, iteration_index: int, total_num_steps: int, train_infos: Dict[str, float]) -> None:
        self.training_history.append(
            {
                "iteration_index": int(iteration_index),
                "total_num_steps": int(total_num_steps),
                "metrics": {str(k): self._coerce_float(v) for k, v in (train_infos or {}).items()},
            }
        )

    def record_eval_metrics(self, total_num_steps: int, episode_reward: float) -> None:
        self.eval_history.append(
            {
                "total_num_steps": int(total_num_steps),
                "average_episode_reward": float(episode_reward),
            }
        )

    def record_resource_metrics(self, iteration_index: int, total_num_steps: int, metrics: Dict[str, float]) -> None:
        self.resource_history.append(
            {
                "iteration_index": int(iteration_index),
                "total_num_steps": int(total_num_steps),
                "metrics": {str(k): self._coerce_float(v) for k, v in (metrics or {}).items()},
            }
        )

    def record_attention_samples(self, samples: np.ndarray) -> None:
        arr = np.asarray(samples, dtype=np.float32).ravel()
        if arr.size:
            self.attn_entropy_sum += float(np.sum(arr, dtype=np.float64))
            self.attn_entropy_count += int(arr.size)

    def recent_episode_stats(self, count: int, phase: str = "train") -> Dict[str, float]:
        candidates = [item for item in self.episode_summaries if item.get("phase", "train") == phase]
        recent = candidates[-max(1, int(count)) :]
        if not recent:
            return {}
        return {
            "average_global_reward": float(np.mean([item["average_global_reward"] for item in recent])),
            "average_p2p_volume": float(
                np.mean([item.get("p2p_volume_mean_active", item["p2p_total_volume"]) for item in recent])
            ),
            "average_grid_buy": float(
                np.mean([item.get("grid_buy_mean_active", item["grid_buy_total"]) for item in recent])
            ),
            "average_grid_sell": float(
                np.mean([item.get("grid_sell_mean_active", item["grid_sell_total"]) for item in recent])
            ),
            "average_carbon_responsibility": float(
                np.mean(
                    [
                        item.get("carbon_responsibility_mean_active_episode", item["load_responsibility_total"])
                        for item in recent
                    ]
                )
            ),
            "average_n_agents": float(np.mean([item["n_agents_mean"] for item in recent])),
            "average_carbon_price": float(np.mean([item["carbon_price_average"] for item in recent])),
        }

    def export_summary(self, summary_filename: Optional[str] = None, last_episode_only: bool = False) -> Path:
        filename = str(summary_filename or self.summary_filename)
        report_path = self.run_dir / filename
        json_path = report_path.with_suffix(".json")
        payload = self._build_payload(last_episode_only=last_episode_only)

        report_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(self._to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        report_path.write_text(self._render_html(payload, json_path.name), encoding="utf-8")
        return report_path

    def _build_payload(self, last_episode_only: bool = False) -> Dict[str, object]:
        episode_summaries = self.episode_summaries[-1:] if last_episode_only and self.episode_summaries else self.episode_summaries
        train_eps = [item for item in episode_summaries if item.get("phase", "train") == "train"]
        eval_eps = [item for item in episode_summaries if item.get("phase", "train") == "eval"]
        attention_entropy_mean = (
            float(self.attn_entropy_sum / self.attn_entropy_count)
            if self.attn_entropy_count > 0
            else None
        )
        overall = {
            "episodes_total": len(episode_summaries),
            "train_episodes": len(train_eps),
            "eval_episode_summaries": len(eval_eps),
            "mean_train_reward": float(np.mean([item["average_global_reward"] for item in train_eps])) if train_eps else 0.0,
            "mean_train_p2p_volume": float(
                np.mean([item.get("p2p_volume_mean_active", item["p2p_total_volume"]) for item in train_eps])
            )
            if train_eps
            else 0.0,
            "mean_train_grid_buy": float(
                np.mean([item.get("grid_buy_mean_active", item["grid_buy_total"]) for item in train_eps])
            )
            if train_eps
            else 0.0,
            "mean_train_grid_sell": float(
                np.mean([item.get("grid_sell_mean_active", item["grid_sell_total"]) for item in train_eps])
            )
            if train_eps
            else 0.0,
            "mean_train_carbon_responsibility": float(
                np.mean(
                    [
                        item.get("carbon_responsibility_mean_active_episode", item["load_responsibility_total"])
                        for item in train_eps
                    ]
                )
            )
            if train_eps
            else 0.0,
            "mean_train_carbon_price": float(np.mean([item["carbon_price_average"] for item in train_eps])) if train_eps else 0.0,
            "latest_eval_reward": None if not self.eval_history else float(self.eval_history[-1]["average_episode_reward"]),
            "attention_entropy_mean": attention_entropy_mean,
        }
        return {
            "overall_metrics": overall,
            "episode_summaries": episode_summaries,
            "training_history": self.training_history,
            "eval_history": self.eval_history,
            "resource_history": self.resource_history,
        }

    def _render_html(self, payload: Dict[str, object], json_name: str) -> str:
        overall = dict(payload.get("overall_metrics") or {})
        episode_summaries = list(payload.get("episode_summaries") or [])
        training_history = list(payload.get("training_history") or [])
        eval_history = list(payload.get("eval_history") or [])

        def rows_from_mapping(mapping):
            return "".join(
                f"<tr><th>{escape(str(key))}</th><td>{escape(self._format_value(value))}</td></tr>"
                for key, value in mapping.items()
            )

        episode_rows = "".join(
            "<tr>"
            f"<td>{escape(str(item.get('episode_id', '')))}</td>"
            f"<td>{escape(str(item.get('phase', '')))}</td>"
            f"<td>{escape(self._format_value(item.get('n_agents_mean')))}</td>"
            f"<td>{escape(self._format_value(item.get('average_global_reward')))}</td>"
            f"<td>{escape(self._format_value(item.get('p2p_volume_mean_active', item.get('p2p_total_volume'))))}</td>"
            f"<td>{escape(self._format_value(item.get('grid_buy_mean_active', item.get('grid_buy_total'))))}</td>"
            f"<td>{escape(self._format_value(item.get('grid_sell_mean_active', item.get('grid_sell_total'))))}</td>"
            f"<td>{escape(self._format_value(item.get('carbon_responsibility_mean_active_episode', item.get('load_responsibility_total'))))}</td>"
            f"<td>{escape(self._format_value(item.get('carbon_price_average')))}</td>"
            "</tr>"
            for item in episode_summaries[-20:]
        )
        training_rows = "".join(
            "<tr>"
            f"<td>{escape(str(item.get('iteration_index', '')))}</td>"
            f"<td>{escape(str(item.get('total_num_steps', '')))}</td>"
            f"<td>{escape(self._format_value((item.get('metrics') or {}).get('policy_loss')))}</td>"
            f"<td>{escape(self._format_value((item.get('metrics') or {}).get('value_loss')))}</td>"
            f"<td>{escape(self._format_value((item.get('metrics') or {}).get('dist_entropy')))}</td>"
            f"<td>{escape(self._format_value((item.get('metrics') or {}).get('fps_policy')))}</td>"
            "</tr>"
            for item in training_history[-20:]
        )
        eval_rows = "".join(
            "<tr>"
            f"<td>{escape(str(item.get('total_num_steps', '')))}</td>"
            f"<td>{escape(self._format_value(item.get('average_episode_reward')))}</td>"
            "</tr>"
            for item in eval_history[-20:]
        )

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>DP-LCRL Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    code {{ background: #f4f4f4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>DP-LCRL Summary</h1>
  <p>JSON 数据文件: <code>{escape(json_name)}</code></p>
  <h2>Overall Metrics</h2>
  <table><tbody>{rows_from_mapping(overall)}</tbody></table>
  <h2>Recent Episode Summaries</h2>
  <table>
    <thead><tr><th>Episode</th><th>Phase</th><th>Avg Active Agents</th><th>Avg Reward</th><th>P2P Mean</th><th>Grid Buy Mean</th><th>Grid Sell Mean</th><th>Carbon Resp. (Episode Mean)</th><th>Carbon Price</th></tr></thead>
    <tbody>{episode_rows}</tbody>
  </table>
  <h2>Recent Training Updates</h2>
  <table>
    <thead><tr><th>Iter</th><th>Total Steps</th><th>Policy Loss</th><th>Value Loss</th><th>Entropy</th><th>FPS</th></tr></thead>
    <tbody>{training_rows}</tbody>
  </table>
  <h2>Evaluation History</h2>
  <table>
    <thead><tr><th>Total Steps</th><th>Average Episode Reward</th></tr></thead>
    <tbody>{eval_rows}</tbody>
  </table>
</body>
</html>
"""

    def _to_jsonable(self, value):
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_jsonable(v) for v in value]
        if isinstance(value, tuple):
            return [self._to_jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.astype(float).tolist()
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        return value

    @staticmethod
    def _coerce_float(value) -> float:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    @staticmethod
    def _format_value(value) -> str:
        if value is None:
            return "-"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value)
