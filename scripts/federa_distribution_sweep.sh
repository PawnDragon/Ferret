#!/usr/bin/env bash
set -euo pipefail

# Sweep FeDeRA data heterogeneity on Dolly with fixed client count/participation.
seeds=(494 495)
iid_values=(dir0.1 dir1.0)

for seed in "${seeds[@]}"; do
  for iid in "${iid_values[@]}"; do
    iid_tag="${iid//./p}"
    run_name="federa_dolly_llama1B_${iid_tag}_seed${seed}"
    echo "[federa distribution sweep] seed=${seed}, iid=${iid}, run_name=${run_name}"

    python main.py \
      --algo federa \
      --model /root/autodl-tmp/llama-3.2-1B/ \
      --dataset dolly \
      --iid "${iid}" \
      --lr 0.00005 \
      --log \
      --device 0 \
      --momentum 0.0 \
      --n_accum 4 \
      --equal_weight \
      --seed "${seed}" \
      --lora_r 8 \
      --lora_alpha 8 \
      --rounds 60 \
      --use_wandb \
      --wandb_project ferret \
      --wandb_run_name "${run_name}" \
      --batch_or_epoch batch \
      --local_step 100 \
      --num_clients 20 \
      -m 0.1 \
      --early_stop \
      --early_stop_patience 5 \
      --optimizer adamw \
      --lora_target_modules q_proj,v_proj \
      "$@"
  done
done
