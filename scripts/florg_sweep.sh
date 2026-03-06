#!/usr/bin/env bash
set -euo pipefail

# Sweep 4 seeds for FedIT on Dolly.
seeds=(494 495 496 497)

for seed in "${seeds[@]}"; do
  run_name="florg_dolly_qwen_seed${seed}"
  echo "[florg sweep] seed=${seed}, run_name=${run_name}"

  python main.py \
    --algo florg \
    --model /root/autodl-tmp/qwen/ \
    --dataset dolly \
    --lr 0.00005 \
    --log \
    --device 0 \
    --momentum 0.0 \
    --n_accum 4 \
    --equal_weight \
    --seed "${seed}" \
    --rounds 60 \
    --use_wandb \
    --wandb_project ferret \
    --wandb_run_name "${run_name}" \
    --batch_or_epoch epoch \
    --florg_rank_r 8\
    --optimizer adamw \
    --early_stop \
    --early_stop_patience 5 \
    --lora_target_modules 'q_proj,k_proj,v_proj,o_proj' \
    "$@"
done
