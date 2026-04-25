#!/usr/bin/env bash
set -euo pipefail

# Sweep FedSubMuon data heterogeneity on Dolly with fixed client count/participation.
seeds=(494 495)
iid_values=(dir0.1 dir1.0)

for seed in "${seeds[@]}"; do
  for iid in "${iid_values[@]}"; do
    iid_tag="${iid//./p}"
    run_name="fedsubmuon_dolly_llama1B_${iid_tag}_seed${seed}"
    echo "[fedsubmuon distribution sweep] seed=${seed}, iid=${iid}, run_name=${run_name}"

    python main.py \
      --algo fedsubmuon \
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
      --rounds 20 \
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
      --stop_F 10 \
      --lora_target_modules q_proj,v_proj \
      "$@"
  done
done
