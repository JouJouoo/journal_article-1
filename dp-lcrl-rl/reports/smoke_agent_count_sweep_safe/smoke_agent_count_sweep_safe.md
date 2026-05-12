# Fixed Active-Agent Count Evaluation

- Runs: `Seed42`
- Checkpoint: `10000`
- Eval episodes per count: `1`
- n_eval_rollout_threads: `1`
- Active-agent sweep: `1..2`
- Fixed eval seed: `20260417`

## Aggregate (mean ± std across runs)

| Active Agents | Reward | P2P Mean | Grid Buy Mean | Grid Sell Mean | Carbon Mean |
| --- | --- | --- | --- | --- | --- |
| 1 | 37.1686 ± 0.0000 | 0.0000 ± 0.0000 | 1.7811 ± 0.0000 | 1.5621 ± 0.0000 | 1.3677 ± 0.0000 |
| 2 | 72.0162 ± 0.0000 | 0.5078 ± 0.0000 | 2.7151 ± 0.0000 | 0.4978 ± 0.0000 | 1.9816 ± 0.0000 |

## Per-Run Results

| Run | Active Agents | Reward | P2P Mean | Grid Buy Mean | Grid Sell Mean | Carbon Mean |
| --- | --- | --- | --- | --- | --- | --- |
| Seed42 | 1 | 37.1686 | 0.0000 | 1.7811 | 1.5621 | 1.3677 |
| Seed42 | 2 | 72.0162 | 0.5078 | 2.7151 | 0.4978 | 1.9816 |
