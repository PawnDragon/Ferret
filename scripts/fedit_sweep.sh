#!/usr/bin/env bash
set -euo pipefail

# Sweep 4 seeds for FedIT on Dolly.
seeds=(494 495 496 497)

for seed in "${seeds[@]}"; do
  run_name="fedit_dolly_seed${seed}"
  echo "[fedit sweep] seed=${seed}, run_name=${run_name}"

  python main.py \
    --algo fedit \
    --save \
    --model /root/autodl-tmp/llama-3.2-1B/ \
    --dataset dolly \
    --lr 0.005 \
    --log \
    --device 0 \
    --momentum 0.0 \
    --n_accum 4 \
    --equal_weight \
    --seed "${seed}" \
    --lora_r 64 \
    --lora_alpha 64 \
    --rounds 60 \
    --use_wandb \
    --wandb_project ferret \
    --wandb_run_name "${run_name}" \
    --batch_or_epoch epoch \
    "$@"
done
