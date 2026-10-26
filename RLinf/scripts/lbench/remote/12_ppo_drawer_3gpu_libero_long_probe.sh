#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${LBENCH_PPO_CPU_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export BLIS_NUM_THREADS="$OMP_NUM_THREADS"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/08_ppo_drawer_2gpu.sh" "$@" --layout libero_long_3gpu_probe
