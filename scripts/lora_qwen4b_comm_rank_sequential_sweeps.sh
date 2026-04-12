#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
  "${script_dir}/fedit_qwen4b_comm_rank_sweep.sh"
  "${script_dir}/flora_qwen4b_comm_rank_sweep.sh"
  "${script_dir}/federa_qwen4b_comm_rank_sweep.sh"
  "${script_dir}/fedexlora_qwen4b_comm_rank_sweep.sh"
  "${script_dir}/florg_qwen4b_comm_rank_sweep.sh"
)

failed_scripts=()

for sweep_script in "${scripts[@]}"; do
  script_name="$(basename "${sweep_script}")"
  echo "[qwen4b lora comm-rank sequential sweep] start ${script_name}"

  if bash "${sweep_script}" "$@"; then
    echo "[qwen4b lora comm-rank sequential sweep] success ${script_name}"
  else
    exit_code=$?
    echo "[qwen4b lora comm-rank sequential sweep] failed ${script_name} (exit=${exit_code})"
    failed_scripts+=("${script_name}:${exit_code}")
  fi
done

if [ "${#failed_scripts[@]}" -gt 0 ]; then
  echo "[qwen4b lora comm-rank sequential sweep] completed with failures:"
  for item in "${failed_scripts[@]}"; do
    echo "  - ${item}"
  done
  exit 1
fi

echo "[qwen4b lora comm-rank sequential sweep] all sweeps completed successfully"
