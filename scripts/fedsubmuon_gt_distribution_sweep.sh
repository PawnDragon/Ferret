#!/usr/bin/env bash
set -euo pipefail

# Sweep FedSubMuon-GT data heterogeneity on Dolly with fixed client count/participation.
seeds=(494 495)
iid_values=(dir0.1 dir1.0)

for seed in "${seeds[@]}"; do
  for iid in "${iid_values[@]}"; do
    iid_tag="${iid//./p}"
    run_name="fedsubmuon_gt_dolly_llama1B_${iid_tag}_seed${seed}"
    echo "[fedsubmuon_gt distribution sweep] seed=${seed}, iid=${iid}, run_name=${run_name}"

    python main.py \
      --algo fedsubmuon_gt \
      --model /root/autodl-tmp/llama-3.2-1B/ \
      --dataset dolly \
      --iid "${iid}" \
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
      --batch_or_epoch batch \
      --local_step 100 \
      --num_clients 20 \
      -m 0.1 \
      --early_stop \
      --early_stop_patience 5 \
      --seed_refresh_F 5 \
      --stop_F 20 \
      --gt_probe_batches 1 \
      --gt_sub_lr 0.05 \
      --lora_target_modules q_proj,v_proj \
      "$@"
  done
done
