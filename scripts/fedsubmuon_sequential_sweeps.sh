#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
  "${script_dir}/fedsubmuon_gt_rank_sweep.sh"
  "${script_dir}/fedsubmuon_gt_refresh_sweep.sh"
  "${script_dir}/fedsubmuon_refresh_sweep.sh"
)

failed_scripts=()

for sweep_script in "${scripts[@]}"; do
  script_name="$(basename "${sweep_script}")"
  echo "[sequential sweep] start ${script_name}"

  if bash "${sweep_script}" "$@"; then
    echo "[sequential sweep] success ${script_name}"
  else
    exit_code=$?
    echo "[sequential sweep] failed ${script_name} (exit=${exit_code})"
    failed_scripts+=("${script_name}:${exit_code}")
  fi
done

if [ "${#failed_scripts[@]}" -gt 0 ]; then
  echo "[sequential sweep] completed with failures:"
  for item in "${failed_scripts[@]}"; do
    echo "  - ${item}"
  done
  exit 1
fi

echo "[sequential sweep] all sweeps completed successfully"
