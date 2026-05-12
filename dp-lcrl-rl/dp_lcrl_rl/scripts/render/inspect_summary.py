"""Inspect exported DP-LCRL summary JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a DP-LCRL summary JSON file.")
    parser.add_argument("--summary_json", type=str, required=True, help="Path to summary JSON.")
    args = parser.parse_args()

    summary_path = Path(args.summary_json).expanduser().resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    overall = payload.get("overall_metrics", {})
    print(f"summary: {summary_path}")
    print(f"episodes_total: {overall.get('episodes_total', 0)}")
    print(f"mean_train_reward: {float(overall.get('mean_train_reward', 0.0)):.4f}")
    print(f"mean_train_p2p_volume: {float(overall.get('mean_train_p2p_volume', 0.0)):.4f}")
    latest_eval = overall.get("latest_eval_reward")
    if latest_eval is not None:
        print(f"latest_eval_reward: {float(latest_eval):.4f}")


if __name__ == "__main__":
    main()
