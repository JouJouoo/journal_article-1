#!/bin/bash
# DP-LCRL Full Training Pipeline
# 6 methods × 3 seeds = 18 runs

set -e
cd /Users/joujou/Desktop/论文-1.1/dp-lcrl-rl

BASE_ARGS="--num_agents 30 --min_agents 20 --curriculum_min_agents 20 \
  --episode_length 24 --n_rollout_threads 4 --num_env_steps 240000 \
  --curriculum_warmup_episodes 2000 --mps"

SEEDS=(42 43 44)

echo "=== Starting DP-LCRL training pipeline ==="
echo "Total: 6 methods × 3 seeds = 18 runs"
echo ""

for method in \
  "full:--cmtm_mode full --mask_mode full --scale_mode curriculum" \
  "cmtm_stateless:--cmtm_mode stateless --mask_mode full --scale_mode curriculum" \
  "mask_obs_only:--cmtm_mode full --mask_mode obs_only --scale_mode curriculum" \
  "direct_max:--cmtm_mode full --mask_mode full --scale_mode direct_max" \
  "random_scale:--cmtm_mode full --mask_mode full --scale_mode random_scale" \
  "no_id_emb:--cmtm_mode full --mask_mode full --scale_mode curriculum --no_id_emb"; do

  IFS=":" read -r name extra_args <<< "$method"
  echo "===== Method: $name ====="
  for seed in "${SEEDS[@]}"; do
    exp_name="paper_${name}_seed${seed}"
    echo "[$(date '+%H:%M')] Starting: $exp_name"
    MPLCONFIGDIR="$TMPDIR/matplotlib" python3 -m dp_lcrl_rl.scripts.train.train_paper_mat \
      $BASE_ARGS $extra_args \
      --experiment_name "$exp_name" --seed "$seed" 2>&1 | tail -5
    echo "[$(date '+%H:%M')] Completed: $exp_name"
    echo ""
  done
done

echo "=== All 18 runs completed ==="
