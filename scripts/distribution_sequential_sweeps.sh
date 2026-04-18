#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
  "${script_dir}/fedkrso_distribution_sweep.sh"
  "${script_dir}/fedsubmuon_distribution_sweep.sh"
  "${script_dir}/fedsubmuon_gt_distribution_sweep.sh"
  "${script_dir}/fedit_distribution_sweep.sh"
  "${script_dir}/flora_distribution_sweep.sh"
  "${script_dir}/federa_distribution_sweep.sh"
  "${script_dir}/fedexlora_distribution_sweep.sh"
  "${script_dir}/florg_distribution_sweep.sh"
)

failed_scripts=()

for sweep_script in "${scripts[@]}"; do
  script_name="$(basename "${sweep_script}")"
  echo "[distribution sequential sweep] start ${script_name}"

  if bash "${sweep_script}" "$@"; then
    echo "[distribution sequential sweep] success ${script_name}"
  else
    exit_code=$?
    echo "[distribution sequential sweep] failed ${script_name} (exit=${exit_code}); continue"
    failed_scripts+=("${script_name}:${exit_code}")
  fi
done

if [ "${#failed_scripts[@]}" -gt 0 ]; then
  echo "[distribution sequential sweep] completed with failures:"
  for item in "${failed_scripts[@]}"; do
    echo "  - ${item}"
  done
  exit 1
fi

echo "[distribution sequential sweep] all sweeps completed successfully"
