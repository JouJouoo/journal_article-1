# dp-lcrl-rl

Python package for DP-LCRL experiments in dynamic-participation P2P low-carbon energy trading.

## Main Components

- `dp_lcrl_rl/envs/p2ptrading/`: paper-aligned P2P trading environment with PV, grid trading, storage, active-agent masks, and carbon-responsibility tracing.
- `dp_lcrl_rl/algorithms/`: policy and training components used by DP-LCRL and baseline variants.
- `dp_lcrl_rl/scripts/train/`: training entry points.
- `dp_lcrl_rl/scripts/eval/`: evaluation scripts for active-agent-count sweeps, CMTM validation, and P2P carbon-flow heatmaps.
- `dp_lcrl_rl/scripts/render/`: plotting utilities for paper figures.
- `reports/`: selected experiment outputs used in the paper.
- `tests/`: regression tests.

## Install

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .[test]
```

## Train

```powershell
python -m dp_lcrl_rl.scripts.train.train_paper_mat `
  --experiment_name demo `
  --policy_arch transformer `
  --num_agents 30 `
  --min_agents 5 `
  --episode_length 24 `
  --num_env_steps 100000 `
  --use_wandb false
```

## Evaluate Active-Agent Generalization

```powershell
python -m dp_lcrl_rl.scripts.eval.eval_agent_count_sweep `
  --run_dir runs/demo `
  --label transformer-seed0 `
  --run_policy_arch transformer `
  --agent_count_min 1 `
  --agent_count_max 30 `
  --eval_episodes 5 `
  --output_dir reports/demo_active_count_eval
```

## Notes

Runtime directories such as `runs/`, `eval_runs/`, temporary evaluation folders, cache folders, and local assistant/tooling state are not intended for version control.
