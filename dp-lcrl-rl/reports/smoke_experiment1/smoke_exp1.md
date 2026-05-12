# Fixed Active-Agent Count Evaluation

- Runs: `deepsets, dp_lcrl, mappo_shared, mlp_pad`
- Checkpoint: `2`
- Eval episodes per count: `1`
- n_eval_rollout_threads: `1`
- Active-agent sweep: `2..4`
- Fixed eval seed: `20260417`

## Aggregate (mean ± std across runs)

| Active Agents | Reward | P2P Mean | Grid Buy Mean | Grid Sell Mean | Carbon Mean |
| --- | --- | --- | --- | --- | --- |
| 2 | 42.9412 ± 0.0678 | 0.0010 ± 0.0018 | 1.0970 ± 0.0135 | 1.5810 ± 0.0096 | 0.7670 ± 0.0012 |
| 4 | 43.6492 ± 0.0184 | 0.0000 ± 0.0000 | 1.1487 ± 0.0139 | 1.9042 ± 0.0099 | 0.8046 ± 0.0000 |

## Per-Run Results

| Run | Active Agents | Reward | P2P Mean | Grid Buy Mean | Grid Sell Mean | Carbon Mean |
| --- | --- | --- | --- | --- | --- | --- |
| dp_lcrl | 2 | 42.8899 | 0.0000 | 1.1103 | 1.5797 | 0.7677 |
| dp_lcrl | 4 | 43.6321 | 0.0000 | 1.1627 | 1.8989 | 0.8046 |
| mlp_pad | 2 | 43.0549 | 0.0041 | 1.0904 | 1.5738 | 0.7648 |
| mlp_pad | 4 | 43.6598 | 0.0000 | 1.1396 | 1.9041 | 0.8046 |
| mappo_shared | 2 | 42.8889 | 0.0000 | 1.1092 | 1.5734 | 0.7677 |
| mappo_shared | 4 | 43.6311 | 0.0000 | 1.1618 | 1.8937 | 0.8046 |
| deepsets | 2 | 42.9309 | 0.0000 | 1.0782 | 1.5971 | 0.7677 |
| deepsets | 4 | 43.6740 | 0.0000 | 1.1308 | 1.9202 | 0.8046 |
