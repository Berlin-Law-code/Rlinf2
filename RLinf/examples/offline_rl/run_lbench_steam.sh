#!/usr/bin/env bash
set -euo pipefail
export REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="/root/autodl-tmp/.venvs/rlinf-openpi/bin:$PATH"
export PYTHONPATH="$REPO_PATH:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/.cache}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/root/autodl-tmp/.cache/openpi}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export JAX_PLATFORMS=cpu USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export RAY_ADDRESS=local
cd "$REPO_PATH"
stage="${1:-check}"
if (( $# > 0 )); then shift; fi
config_dir="$REPO_PATH/examples/offline_rl/config"
case "$stage" in
  check)
    exec python toolkits/lbench_steam_preflight.py "$@" ;;
  value)
    exec python examples/offline_rl/advantage_labeling/steam/train_steam.py \
      --config-path "$config_dir" --config-name lbench_steam_value "$@" ;;
  advantages)
    : "${STEAM_VALUE_CHECKPOINT:?Set STEAM_VALUE_CHECKPOINT to the trained STEAM actor directory}"
    exec torchrun --standalone --nproc_per_node=2 \
      examples/offline_rl/advantage_labeling/steam/process/compute_advantages_ensemble.py \
      --config-path "$config_dir" --config-name lbench_steam_advantages "$@" ;;
  policy)
    exec python examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py \
      --config-path "$config_dir" --config-name lbench_cfgrl "$@" ;;
  *) echo 'Usage: bash examples/offline_rl/run_lbench_steam.sh {check|value|advantages|policy} [overrides...]' >&2; exit 2 ;;
esac
