#!/usr/bin/env bash
set -euo pipefail

# Sweep seeds and basis initialization modes for FedSubMuon-GT on Dolly.
seeds=(494 495)
basis_init_modes=(svd_left svd_right svd_both)

for seed in "${seeds[@]}"; do
  for basis_init_mode in "${basis_init_modes[@]}"; do
    run_name="fedsubmuon_gt_dolly_Llama1B_seed${seed}_basis${basis_init_mode}"
    echo "[fedsubmuon_gt basis-init sweep] seed=${seed}, basis_init_mode=${basis_init_mode}, run_name=${run_name}"

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
      --gt_probe_batches 1 \
      --gt_sub_lr 0.05 \
      --basis_init_mode "${basis_init_mode}" \
      --lora_target_modules "q_proj,v_proj" \
      "$@"
  done
done
