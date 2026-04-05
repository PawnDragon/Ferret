#!/usr/bin/env bash
set -euo pipefail

# Sweep seeds and rank_r for FedSubMuon on Dolly.
seeds=(494 495)
rank_rs=(16 32 48 128)

for seed in "${seeds[@]}"; do
  for rank_r in "${rank_rs[@]}"; do
    run_name="fedsubmuon_dolly_Llama1B_seed${seed}_rank${rank_r}"
    echo "[fedsubmuon rank sweep] seed=${seed}, rank_r=${rank_r}, run_name=${run_name}"

    python main.py \
      --algo fedsubmuon \
      --model /root/autodl-tmp/llama-3.2-1B/ \
      --dataset dolly \
      --lr 0.005 \
      --log \
      --device 0 \
      --momentum 0.0 \
      --n_accum 4 \
      --equal_weight \
      --seed "${seed}" \
      --rank_r "${rank_r}" \
      --rounds 30 \
      --use_wandb \
      --wandb_project ferret \
      --wandb_run_name "${run_name}" \
      --batch_or_epoch epoch \
      --early_stop \
      --early_stop_patience 5 \
      --stop_F 10 \
      --seed_refresh_F 5 \
      --lora_target_modules "q_proj,v_proj" \
      "$@"
  done
done
