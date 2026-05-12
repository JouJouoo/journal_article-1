# DP-LCRL Dynamic Low-Carbon P2P Trading

This repository contains the experimental code and paper artifacts for **Dynamic-Participation Low-Carbon Reinforcement Learning (DP-LCRL)** in P2P energy trading. The project studies carbon-responsibility tracing, storage carbon inheritance, P2P carbon-responsibility transfer, and policy generalization under varying numbers of active agents.

## Repository Layout

```text
.
├── dp-lcrl-rl/                         # Main Python project
│   ├── dp_lcrl_rl/                      # Package source code
│   │   ├── algorithms/                  # MAT / DP-LCRL policy components
│   │   ├── envs/p2ptrading/             # P2P low-carbon trading environment
│   │   ├── runner/                      # Training runners and analytics
│   │   └── scripts/                     # Training, evaluation, and plotting scripts
│   ├── reports/                         # Selected experiment reports and figures
│   ├── tests/                           # Unit tests
│   ├── tools/                           # Utility scripts
│   ├── pyproject.toml
│   └── requirements.txt
├── AGENTS.md                            # Local agent/project instructions
└── README.md
```

Local state directories (`.claude/`, `.agents/`, etc.) are excluded from version control.

## Research Focus

The project targets four connected questions:

1. **Carbon responsibility tracing:** How can carbon responsibility be tracked across grid import, PV generation, P2P trades, load consumption, and storage operation?
2. **Storage carbon inheritance:** Can stored carbon responsibility be retained during storage holding periods and released during later discharge?
3. **P2P carbon transfer:** Can P2P trading transfer not only energy but also signed carbon responsibility between agents?
4. **Dynamic participation:** Can one learned policy adapt to changing active-agent counts rather than a fixed-size group?

## Environment Setup

From the repository root:

```powershell
Set-Location .\dp-lcrl-rl
python -m pip install -r requirements.txt
python -m pip install -e .[test]
```

The experiments were run with Python 3.10 and PyTorch. For local runs, disabling optional online logging is recommended:

```powershell
$env:WANDB_DISABLED="true"
$env:WANDB_MODE="disabled"
$env:DP_LCRL_DISABLE_TENSORBOARD="true"
```

## Typical Commands

Train the DP-LCRL method for three seeds and evaluate active-agent counts from 1 to 30:

```powershell
python -m dp_lcrl_rl.scripts.run_experiment1_sequence_modeling `
  --methods dp_lcrl `
  --seeds 0,1,2 `
  --active_counts 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 `
  --experiment_prefix dp_lcrl_100k_no_rho_clip_20260508 `
  --num_env_steps 100000 `
  --n_rollout_threads 2 `
  --n_eval_rollout_threads 1 `
  --eval_episodes 5 `
  --hidden_size 64 `
  --n_embd 32 `
  --n_block 1 `
  --n_head 1 `
  --layer_N 1 `
  --report_dir reports/dp_lcrl_100k_no_rho_clip_s012_20260508
```

Render multi-seed training curves:

```powershell
python -m dp_lcrl_rl.scripts.render.plot_multiseed_training_curves `
  --summary_json runs/dp_lcrl_100k_no_rho_clip_20260508_dp_lcrl_seed0/paper_training_summary.json `
  --summary_json runs/dp_lcrl_100k_no_rho_clip_20260508_dp_lcrl_seed1/paper_training_summary.json `
  --summary_json runs/dp_lcrl_100k_no_rho_clip_20260508_dp_lcrl_seed2/paper_training_summary.json `
  --output reports/dp_lcrl_100k_no_rho_clip_s012_20260508/training_curves_no_rho_clip_100k_s012.png `
  --smoothing_window 50 `
  --mode mean_std
```

Run tests:

```powershell
pytest
```

## Key Outputs

The final paper figures are organized under `dp-lcrl-rl/reports/`, including:

- Storage carbon-memory validation.
- DP-LCRL training curves over three random seeds.
- P2P signed carbon-responsibility flow heatmap.
- Active-agent-count generalization test from 1 to 30 agents.

Generated runtime outputs such as `runs/`, `eval_runs/`, temporary evaluation folders, and local tool state are excluded from version control.
