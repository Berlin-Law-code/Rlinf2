#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${LBENCH_PPO_CPU_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export BLIS_NUM_THREADS="$OMP_NUM_THREADS"

# Ray otherwise starts one idle Python worker per CPU before RLinf actors.
# Each idle worker owns native thread pools, exhausting the container's PID
# budget alongside Isaac processes. Required actors still start on demand.
export RAY_prestart_worker_first_driver="${LBENCH_PPO_RAY_PRESTART_WORKERS:-0}"
export RAY_enable_worker_prestart="$RAY_prestart_worker_first_driver"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/08_ppo_drawer_2gpu.sh" "$@" --layout collocated_1gpu --task insert_the_flower_in_vase_part0
