# Fixed Active-Agent Count Evaluation

- Runs: `Seed 10000`
- Checkpoint: `10000`
- Eval episodes per count: `2`
- n_eval_rollout_threads: `1`
- Active-agent sweep: `1..2`
- Fixed eval seed: `20260417`

## Aggregate (mean ± std across runs)

| Active Agents | Reward | P2P Mean | Grid Buy Mean | Grid Sell Mean | Carbon Mean |
| --- | --- | --- | --- | --- | --- |
| 1 | 54.3143 ± 0.0000 | 0.0000 ± 0.0000 | 3.1451 ± 0.0000 | 1.0247 ± 0.0000 | 2.2870 ± 0.0000 |
| 2 | 66.7790 ± 0.0000 | 0.2539 ± 0.0000 | 3.3150 ± 0.0000 | 0.4793 ± 0.0000 | 2.4075 ± 0.0000 |

## Per-Run Results

| Run | Active Agents | Reward | P2P Mean | Grid Buy Mean | Grid Sell Mean | Carbon Mean |
| --- | --- | --- | --- | --- | --- | --- |
| Seed 10000 | 1 | 54.3143 | 0.0000 | 3.1451 | 1.0247 | 2.2870 |
| Seed 10000 | 2 | 66.7790 | 0.2539 | 3.3150 | 0.4793 | 2.4075 |
