#!/usr/bin/env bash
set -euo pipefail

# Sweep basis-init/update-mode combinations for FedSubMuon-GT on Dolly.
# Baseline hyperparameters follow scripts/fedsubmuon_gt_sweep.sh.
seeds=(495)

# 1) left svd, right update
# 2) left svd, both update
# 3) right svd, left update
# 4) right svd, both update
configs=(
  "svdleft_upright:svd_left:right"
  "svdleft_upboth:svd_left:both"
  "svdright_upleft:svd_right:left"
  "svdright_upboth:svd_right:both"
)

for seed in "${seeds[@]}"; do
  for cfg in "${configs[@]}"; do
    IFS=':' read -r tag basis_init_mode gt_update_mode <<< "${cfg}"
    run_name="fedsubmuon_gt_dolly_Llama1B_seed${seed}_${tag}"
    echo "[fedsubmuon_gt basis/update sweep] seed=${seed}, basis_init_mode=${basis_init_mode}, gt_update_mode=${gt_update_mode}, run_name=${run_name}"

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
      --basis_init_mode "${basis_init_mode}" \
      --gt_update_mode "${gt_update_mode}" \
      --lora_target_modules "q_proj,v_proj" \
      "$@"
  done
done
