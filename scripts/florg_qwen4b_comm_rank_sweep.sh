#!/usr/bin/env bash
set -euo pipefail

# Sweep FLoRG rank and communication-budget-matched rounds for Qwen3-4B Dolly.
seeds=(494 495)
ranks=(8 4 2)

rounds_for_rank() {
  case "$1" in
    8) echo 34 ;;
    4) echo 60 ;;
    2) echo 60 ;;
    *) echo "[florg qwen4b comm-rank sweep] unsupported rank=$1" >&2; return 1 ;;
  esac
}

for seed in "${seeds[@]}"; do
  for rank in "${ranks[@]}"; do
    rounds="$(rounds_for_rank "${rank}")"
    run_name="florg_dolly_Qwen4B_seed${seed}_rank${rank}_rounds${rounds}"
    echo "[florg qwen4b comm-rank sweep] seed=${seed}, rank=${rank}, rounds=${rounds}, run_name=${run_name}"

    python main.py \
      --algo florg \
      --model /root/autodl-tmp/qwen/ \
      --dataset dolly \
      --lr 0.00005 \
      --log \
      --device 0 \
      --momentum 0.0 \
      --n_accum 4 \
      --equal_weight \
      --seed "${seed}" \
      --rounds "${rounds}" \
      --use_wandb \
      --wandb_project ferret \
      --wandb_run_name "${run_name}" \
      --batch_or_epoch epoch \
      --florg_rank_r "${rank}" \
      --optimizer adamw \
      --early_stop \
      --early_stop_patience 5 \
      --lora_target_modules "q_proj,v_proj" \
      "$@"
  done
done
