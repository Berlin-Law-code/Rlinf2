#!/usr/bin/env bash
set -euo pipefail

# Bound early NumPy/Open3D/PyTorch pools before the 24 Isaac processes import
# native libraries. Isaac's own Kit/TBB limit is configured in the profile.
export OMP_NUM_THREADS="${LBENCH_PPO_CPU_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export BLIS_NUM_THREADS="$OMP_NUM_THREADS"

# Reuse the aligned checkpoint/runtime setup; all GPUs train and sample.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/08_ppo_drawer_2gpu.sh" "$@" --layout collocated_3gpu
