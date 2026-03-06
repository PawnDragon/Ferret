#!/usr/bin/env bash
set -euo pipefail

# Sweep 4 seeds for FedIT on Dolly.
seeds=(494 495 496 497)

for seed in "${seeds[@]}"; do
  run_name="fedavg_dolly_qwen_seed${seed}"
  echo "[fedavg sweep] seed=${seed}, run_name=${run_name}"

  python main.py \
    --algo fedavg \
    --model /root/autodl-tmp/qwen/ \
    --dataset dolly \
    --lr 0.00001 \
    --log \
    --n_accum 4 \
    --equal_weight \
    --seed "${seed}" \
    --rounds 40 \
    --use_wandb \
    --wandb_project ferret \
    --wandb_run_name "${run_name}" \
    --batch_or_epoch epoch \
    --optimizer adamw \
    --early_stop \
    --early_stop_patience 5 \
    "$@"
done
