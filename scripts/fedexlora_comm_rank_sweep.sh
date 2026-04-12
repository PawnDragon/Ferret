#!/usr/bin/env bash
set -euo pipefail

# Sweep LoRA rank and communication-budget-matched rounds for FedEx-LoRA on Dolly.
seeds=(494 495)
ranks=(8 4 2)

rounds_for_rank() {
  case "$1" in
    8) echo 0 ;;
    4) echo 1 ;;
    2) echo 2 ;;
    *) echo "[fedexlora comm-rank sweep] unsupported rank=$1" >&2; return 1 ;;
  esac
}

for seed in "${seeds[@]}"; do
  for rank in "${ranks[@]}"; do
    rounds="$(rounds_for_rank "${rank}")"
    if [ "${rounds}" -le 0 ]; then
      echo "[fedexlora comm-rank sweep] skip seed=${seed}, rank=${rank}: communication budget gives rounds=${rounds}"
      continue
    fi
    run_name="fedexlora_dolly_Llama1B_seed${seed}_rank${rank}_rounds${rounds}"
    echo "[fedexlora comm-rank sweep] seed=${seed}, rank=${rank}, rounds=${rounds}, run_name=${run_name}"

    python main.py \
      --algo fedexlora \
      --model /root/autodl-tmp/llama-3.2-1B/ \
      --dataset dolly \
      --lr 0.00005 \
      --log \
      --device 0 \
      --momentum 0.0 \
      --n_accum 4 \
      --equal_weight \
      --seed "${seed}" \
      --lora_r "${rank}" \
      --lora_alpha "${rank}" \
      --rounds "${rounds}" \
      --use_wandb \
      --wandb_project ferret \
      --wandb_run_name "${run_name}" \
      --batch_or_epoch epoch \
      --early_stop \
      --early_stop_patience 5 \
      --optimizer adamw \
      --lora_target_modules "q_proj,v_proj" \
      "$@"
  done
done
