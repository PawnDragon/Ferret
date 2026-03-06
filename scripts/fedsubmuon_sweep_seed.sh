#!/usr/bin/env bash
set -euo pipefail

# Sweep 4 seeds for FedIT on Dolly.
seeds=(495 496 497)

for seed in "${seeds[@]}"; do
  run_name="fedsubmuon_dolly_qwen_seed${seed}"
  echo "[fedsubmuon sweep] seed=${seed}, run_name=${run_name}"

  python main.py \
    --algo fedsubmuon \
    --model /root/autodl-tmp/qwen/ \
    --dataset dolly \
    --lr 0.005 \
    --log \
    --device 0 \
    --momentum 0.0 \
    --n_accum 4 \
    --equal_weight \
    --seed "${seed}" \
    --rank_r 64 \
    --seed_refresh_F 5 \
    --rounds 30 \
    --use_wandb \
    --wandb_project ferret \
    --wandb_run_name "${run_name}" \
    --batch_or_epoch epoch \
    --stop_F 10 \
    --early_stop \
    --early_stop_patience 5 \
    "$@"
done
