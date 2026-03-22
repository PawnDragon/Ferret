#!/usr/bin/env bash
set -euo pipefail

# Sweep 4 seeds for FedIT on Dolly.
seeds=(494 495 496 497)

for seed in "${seeds[@]}"; do
  run_name="fedit_gsm8k_Qwen8B_seed${seed}"
  echo "[fedit sweep] seed=${seed}, run_name=${run_name}"

  python main.py \
    --algo fedit \
    --model /root/autodl-tmp/qwen8B/ \
    --dataset gsm8k \
    --lr 0.00005 \
    -m 0.1 \
    --num_clients 100 \
    --log \
    --device 0 \
    --momentum 0.0 \
    --n_accum 4 \
    --equal_weight \
    --seed "${seed}" \
    --lora_r 12 \
    --lora_alpha 12 \
    --rounds 20 \
    --use_wandb \
    --wandb_project ferret \
    --wandb_run_name "${run_name}" \
    --batch_or_epoch epoch \
    --early_stop \
    --early_stop_patience 5 \
    --optimizer adamw \
    "$@"
done
