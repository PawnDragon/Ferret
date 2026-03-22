#!/usr/bin/env bash
set -euo pipefail

# Sweep 4 seeds for FedIT on Dolly.
seeds=(494 495 496 497)

for seed in "${seeds[@]}"; do
  run_name="ferret_NI_qwen4B_seed${seed}"
  echo "[ferret sweep] seed=${seed}, run_name=${run_name}"

  python main.py \
    --algo ferret \
    --model /root/autodl-tmp/qwen/ \
    -m 0.03 \
    --lr 0.0001 \
    --K 2048 \
    --log \
    --slr_max 3 \
    --anneal no \
    --device 0 \
    --momentum 0.0 \
    --n_accum 4 \
    --equal_weight \
    --seed "${seed}" \
    --rounds 20 \
    --use_wandb \
    --wandb_project ferret \
    --wandb_run_name "${run_name}" \
    --batch_or_epoch batch \
    --local_step 100 \
    --optimizer sgd \
    --early_stop \
    --early_stop_patience 5 \
    --ni_root /root/autodl-tmp/natural-instructions/ \
    "$@"
done
