#!/usr/bin/env bash
set -euo pipefail

export RLINF_ROOT="${RLINF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export RLINF_PYTHON="${RLINF_PYTHON:-/root/autodl-tmp/.venvs/rlinf-openpi/bin/python}"
export PYTHONPATH="$RLINF_ROOT:${PYTHONPATH:-}"
export PATH="$(dirname "$RLINF_PYTHON"):$PATH"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$XDG_CACHE_HOME/openpi}"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export MALLOC_MMAP_THRESHOLD_="${LBENCH_PPO_MALLOC_MMAP_THRESHOLD:-131072}"
export MALLOC_TRIM_THRESHOLD_="${LBENCH_PPO_MALLOC_TRIM_THRESHOLD:-131072}"

# Dedicated overrides prevent stale evaluation/SFT shell exports from silently
# changing the task contract. The Python launcher selects checkpoint assets.
export LBENCH_ROOT="${LBENCH_PPO_ROOT:-/root/autodl-tmp/Berlin/LBench_multi_env}"
export LBENCH_TASK_CONFIG="${LBENCH_PPO_TASK_CONFIG:-$LBENCH_ROOT/configs/simbox/lbench_joint_8tasks.yaml}"
export LBENCH_ISAAC_PYTHON="${LBENCH_PPO_ISAAC_PYTHON:-/root/autodl-tmp/conda/envs/env_isaaclab/bin/python}"
# Metadata only: the original v3 dataset retains the eight SFT instructions.
# PPO does not open demonstration frames or replace checkpoint normalization.
export LBENCH_LEROBOT_REPO="${LBENCH_PPO_DATASET:-/root/autodl-tmp/datasets/lbench_8tasks_v30_merged_200}"

cd "$RLINF_ROOT"
# Foreground only. --dry-run validates files/configuration without Ray or Isaac.
exec "$RLINF_PYTHON" toolkits/lbench_two_task_ppo.py "$@"
