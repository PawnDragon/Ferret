#!/usr/bin/env bash
set -euo pipefail

# Sweep 4 seeds for FedIT on Dolly.
seeds=(495)

for seed in "${seeds[@]}"; do
  run_name="fedsubmuon_gt_dolly_Llama1B_seed${seed}"
  echo "[fedsubmuon_gt sweep] seed=${seed}, run_name=${run_name}"

  python main.py \
    --algo fedsubmuon_gt \
    --model /root/autodl-tmp/llama-3.2-1B/ \
    --dataset dolly \
    --lr 0.005 \
    --log \
    --device 0 \
    --momentum 0.0 \
    --n_accum 4 \
    --equal_weight \
    --seed "${seed}" \
    --rank_r 64 \
    --rounds 30 \
    --use_wandb \
    --wandb_project ferret \
    --wandb_run_name "${run_name}" \
    --batch_or_epoch epoch \
    --early_stop \
    --early_stop_patience 5 \
    --stop_F 20 \
    --seed_refresh_F 5 \
    --gt_probe_batches 4 \
    --gt_sub_lr 0.05 \
    --lora_target_modules "q_proj,v_proj" \
    "$@"
done
