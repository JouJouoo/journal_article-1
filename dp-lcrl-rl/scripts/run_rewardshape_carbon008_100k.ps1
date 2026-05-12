$ErrorActionPreference = "Stop"

$python = "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe"
$reportDir = "reports\dp_lcrl_100k_s012_carbon008_20260506"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

$env:WANDB_DISABLED = "true"
$env:DP_LCRL_DISABLE_TENSORBOARD = "true"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:MPLBACKEND = "Agg"

foreach ($seed in 0, 1, 2) {
    Write-Host "[Carbon008-100k] train seed=$seed"
    & $python -m dp_lcrl_rl.scripts.train.train_paper_mat `
        --experiment_name "dp_lcrl_100k_s012_carbon008_20260506_seed$seed" `
        --policy_arch transformer `
        --num_agents 30 `
        --min_agents 5 `
        --curriculum_min_agents 5 `
        --scale_mode random_scale `
        --episode_length 24 `
        --n_rollout_threads 2 `
        --n_eval_rollout_threads 1 `
        --num_env_steps 100000 `
        --seed $seed `
        --ppo_epoch 2 `
        --num_mini_batch 1 `
        --hidden_size 64 `
        --n_embd 32 `
        --n_block 1 `
        --n_head 1 `
        --layer_N 1 `
        --use_eval false `
        --save_interval 0 `
        --use_wandb false `
        --carbon_price 0.08 `
        --p2p_reward_weight 1.0 `
        --grid_buy_penalty_weight 0.5 `
        --unmatched_penalty_weight 0.01 `
        --lr 0.0003 `
        --use_linear_lr_decay false
    Write-Host "[Carbon008-100k] done seed=$seed"
}

@'
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

runs = [
    Path("runs/dp_lcrl_100k_s012_carbon008_20260506_seed0/paper_training_summary.json"),
    Path("runs/dp_lcrl_100k_s012_carbon008_20260506_seed1/paper_training_summary.json"),
    Path("runs/dp_lcrl_100k_s012_carbon008_20260506_seed2/paper_training_summary.json"),
]
out_dir = Path("reports/dp_lcrl_100k_s012_carbon008_20260506")
out_dir.mkdir(parents=True, exist_ok=True)
out_png = out_dir / "training_curves_carbon008_100k.png"
out_csv = out_dir / "training_curves_carbon008_100k_stats.csv"

METRICS = {
    "reward": "average_global_reward",
    "p2p": "p2p_volume_mean_active",
    "grid_buy": "grid_buy_mean_active",
    "grid_sell": "grid_sell_mean_active",
    "carbon": "carbon_responsibility_mean_active_episode",
}

def load_run(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    buckets = defaultdict(lambda: defaultdict(list))
    for item in payload.get("episode_summaries", []):
        if item.get("phase") != "train":
            continue
        idx = int(item.get("iteration_index", 0))
        for key, src in METRICS.items():
            buckets[idx][key].append(float(item.get(src, 0.0) or 0.0))
    xs = np.array(sorted(buckets), dtype=np.int32) + 1
    data = {"iteration": xs}
    for key in METRICS:
        data[key] = np.array([np.mean(buckets[int(i - 1)][key]) for i in xs], dtype=np.float64)
    return data

def ma(values, window=50):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or window <= 1:
        return values
    window = min(window, values.size)
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")

series = [load_run(path) for path in runs]
base_x = series[0]["iteration"]
for item in series:
    if not np.array_equal(item["iteration"], base_x):
        raise RuntimeError("Iteration axes do not match across seeds.")

window = 50
stats = {}
for key in METRICS:
    stacked = np.stack([ma(item[key], window) for item in series], axis=0)
    stats[key] = {"mean": np.mean(stacked, axis=0), "std": np.std(stacked, axis=0)}

fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=180)
fig.suptitle("DP-LCRL Training Curves, carbon_price=0.08, 100k Steps, Seeds 0/1/2", fontsize=15)

def draw(ax, key, title, ylabel, color):
    for item in series:
        ax.plot(base_x, ma(item[key], window), color=color, alpha=0.18, linewidth=1.0)
    mean = stats[key]["mean"]
    std = stats[key]["std"]
    ax.plot(base_x, mean, color=color, linewidth=2.5, label="Mean")
    ax.fill_between(base_x, mean - std, mean + std, color=color, alpha=0.15, label="Std")
    ax.set_title(f"{title} (MA{window})")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

draw(axes[0, 0], "reward", "Training Reward", "Reward", "#111111")
draw(axes[0, 1], "p2p", "P2P Trading Volume", "Mean Volume per Active Agent", "#238b45")

ax = axes[1, 0]
for key, color, label in [("grid_buy", "#d94801", "Grid Buy"), ("grid_sell", "#2171b5", "Grid Sell")]:
    for item in series:
        ax.plot(base_x, ma(item[key], window), color=color, alpha=0.16, linewidth=1.0)
    mean = stats[key]["mean"]
    std = stats[key]["std"]
    ax.plot(base_x, mean, color=color, linewidth=2.5, label=label)
    ax.fill_between(base_x, mean - std, mean + std, color=color, alpha=0.12)
ax.set_title(f"Grid Buy and Grid Sell (MA{window})")
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Mean Volume per Active Agent")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)

draw(axes[1, 1], "carbon", "Carbon Responsibility", "Mean Carbon Responsibility", "#756bb1")

fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out_png, bbox_inches="tight")
plt.close(fig)

with out_csv.open("w", encoding="utf-8") as f:
    f.write("metric,first100,last100,delta,first10pct,last10pct,delta10pct\n")
    n = len(base_x)
    k = max(1, int(round(0.10 * n)))
    for key in ["reward", "p2p", "grid_buy", "grid_sell", "carbon"]:
        raw = np.stack([item[key] for item in series], axis=0)
        mean_raw = np.mean(raw, axis=0)
        first100 = float(np.mean(mean_raw[:100]))
        last100 = float(np.mean(mean_raw[-100:]))
        first10 = float(np.mean(mean_raw[:k]))
        last10 = float(np.mean(mean_raw[-k:]))
        f.write(
            f"{key},{first100:.8f},{last100:.8f},{last100-first100:.8f},"
            f"{first10:.8f},{last10:.8f},{last10-first10:.8f}\n"
        )

print(f"plot={out_png.resolve()}")
print(f"csv={out_csv.resolve()}")
'@ | & $python -

Write-Host "[Carbon008-100k] all done"
