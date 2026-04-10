#!/usr/bin/env bash
set -euo pipefail

# Sweep seeds and optimizer choices for FedSubMuon-GT on Dolly.
seeds=(494 495)
optimizers=(adamw sgd)
rank_r=64

for seed in "${seeds[@]}"; do
  for optimizer in "${optimizers[@]}"; do
    case "${optimizer}" in
      adamw)
        lr=0.00005
        ;;
      sgd)
        lr=0.001
        ;;
      *)
        echo "[fedsubmuon_gt optimizer sweep] unsupported optimizer=${optimizer}" >&2
        exit 1
        ;;
    esac

    run_name="fedsubmuon_gt_dolly_Llama1B_seed${seed}_opt${optimizer}_rank${rank_r}"
    echo "[fedsubmuon_gt optimizer sweep] seed=${seed}, optimizer=${optimizer}, lr=${lr}, rank_r=${rank_r}, run_name=${run_name}"

    python main.py \
      --algo fedsubmuon_gt \
      --model /root/autodl-tmp/llama-3.2-1B/ \
      --dataset dolly \
      --lr "${lr}" \
      --optimizer "${optimizer}" \
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
      --stop_F 20 \
      --seed_refresh_F 5 \
      --gt_probe_batches 1 \
      --gt_sub_lr 0.05 \
      --lora_target_modules "q_proj,v_proj" \
      "$@"
  done
done
