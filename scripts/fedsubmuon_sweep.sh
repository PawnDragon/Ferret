#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --algo fedsubmuon \
  --model /root/autodl-tmp/llama-3.2-1B/ \
  --dataset dolly \
  --log \
  --device 0 \
  --momentum 0.0 \
  --n_accum 4 \
  --equal_weight \
  --seed 494 \
  --seed_refresh_F 100 \
  --rounds 20 \
  --use_wandb \
  --wandb_project ferret \
  --wandb_name sweep_lr_r_localstep \
  --batch_or_epoch epoch \
  "$@"
