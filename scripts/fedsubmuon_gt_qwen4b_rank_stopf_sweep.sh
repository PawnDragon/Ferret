#!/usr/bin/env bash
set -euo pipefail

# Sweep FedSubMuon-GT rank/stop_F pairs on Qwen3-4B Dolly.
# With seed_refresh_F=5:
#   stop_F=6  -> one refresh at round 5
#   stop_F=11 -> two refreshes at rounds 5,10
#   stop_F=16 -> three refreshes at rounds 5,10,15
seeds=(494 495)
configs=(
  "90 6"
  "48 11"
  "36 16"
)

for seed in "${seeds[@]}"; do
  for config in "${configs[@]}"; do
    read -r rank_r stop_f <<< "${config}"
    run_name="fedsubmuon_gt_dolly_Qwen4B_seed${seed}_rank${rank_r}_stopF${stop_f}"
    echo "[fedsubmuon_gt qwen4b rank-stopF sweep] seed=${seed}, rank_r=${rank_r}, stop_F=${stop_f}, run_name=${run_name}"

    python main.py \
      --algo fedsubmuon_gt \
      --model /root/autodl-tmp/qwen/ \
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
      --stop_F "${stop_f}" \
      --seed_refresh_F 5 \
      --gt_probe_batches 1 \
      --gt_sub_lr 0.05 \
      --lora_target_modules "q_proj,v_proj" \
      "$@"
  done
done
