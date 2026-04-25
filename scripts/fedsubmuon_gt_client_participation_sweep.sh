#!/usr/bin/env bash
set -euo pipefail

# Sweep FedSubMuon-GT client participation on Dolly with fixed local batch steps.
seeds=(494 495)
m_values=(0.1 0.5 1)

for seed in "${seeds[@]}"; do
  for m in "${m_values[@]}"; do
    rounds=30
    if [[ "${m}" == "1" || "${m}" == "1.0" ]]; then
      rounds=20
    fi
    m_tag="${m//./p}"
    run_name="fedsubmuon_gt_dolly_llama1B_clients20_m${m_tag}_batch100_seed${seed}"
    echo "[fedsubmuon_gt client participation sweep] seed=${seed}, m=${m}, rounds=${rounds}, run_name=${run_name}"

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
      --rounds "${rounds}" \
      --use_wandb \
      --wandb_project ferret \
      --wandb_run_name "${run_name}" \
      --batch_or_epoch batch \
      --local_step 100 \
      --num_clients 20 \
      -m "${m}" \
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
