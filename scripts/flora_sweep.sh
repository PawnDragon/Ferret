#!/usr/bin/env bash
set -euo pipefail

# Sweep 4 seeds for FedIT on Dolly.
seeds=(494 495 496 497)

for seed in "${seeds[@]}"; do
  run_name="flora_dolly_qwen_seed${seed}"
  echo "[flora sweep] seed=${seed}, run_name=${run_name}"

  python main.py \
    --algo flora \
    --model /root/autodl-tmp/qwen/ \
    --dataset dolly \
    --lr 0.00005 \
    --log \
    --device 0 \
    --momentum 0.0 \
    --n_accum 4 \
    --equal_weight \
    --seed "${seed}" \
    --lora_r 8 \
    --lora_alpha 8 \
    --rounds 60 \
    --use_wandb \
    --wandb_project ferret \
    --wandb_run_name "${run_name}" \
    --batch_or_epoch epoch \
    --early_stop \
    --early_stop_patience 5 \
    --optimizer adamw \
    "$@"
done
