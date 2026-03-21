#!/usr/bin/env bash
set -euo pipefail

# Run 2 configs for FedSubMuon-GT on Dolly:
# 1) stop_F=10 (same as the provided baseline)
# 2) stop_F=20
seeds=(494)
stop_fs=(10 20)

for seed in "${seeds[@]}"; do
  for stop_f in "${stop_fs[@]}"; do
    run_name="fedsubmuon_gt_dolly_Llama1B_seed${seed}_stopF${stop_f}"
    echo "[fedsubmuon_gt sweep] seed=${seed}, stop_F=${stop_f}, run_name=${run_name}"

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
      --stop_F "${stop_f}" \
      --seed_refresh_F 5 \
      --gt_probe_batches 1 \
      --gt_sub_lr 0.01 \
      --lora_target_modules "q_proj,v_proj" \
      "$@"
  done
done
