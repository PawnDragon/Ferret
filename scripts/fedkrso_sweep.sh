#!/usr/bin/env bash
set -euo pipefail

# Sweep 4 seeds for FedIT on Dolly.
seeds=(494 495)

for seed in "${seeds[@]}"; do
  run_name="fedkrso_dolly_llama1B_seed${seed}"
  echo "[fedkrso sweep] seed=${seed}, run_name=${run_name}"

  python main.py \
    --algo fedkrso \
    --model /root/autodl-tmp/llama-3.2-1B/ \
    --dataset dolly \
    --lr 0.00001 \
    --log \
    --device 0 \
    --momentum 0.0 \
    --n_accum 4 \
    --equal_weight \
    --seed "${seed}" \
    --rank_r 8 \
    --lora_alpha 8 \
    --rounds 60 \
    --use_wandb \
    --wandb_project ferret \
    --wandb_run_name "${run_name}" \
    --batch_or_epoch epoch \
    --early_stop \
    --early_stop_patience 5 \
    --optimizer adamw \
    --lora_target_modules q_proj,v_proj \
    --krso_num_seeds 4 \
    --krso_interval_len 25 \
    "$@"
done
