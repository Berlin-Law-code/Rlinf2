#!/usr/bin/env bash
set -euo pipefail

# Standalone four-GPU recipe. Separate override names prevent stale exports
# from an old evaluation shell from selecting old BF16 weights or old stats.
export RLINF_ROOT="${RLINF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export RLINF_PYTHON="${RLINF_PYTHON:-/root/autodl-tmp/.venvs/rlinf-openpi/bin/python}"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
export PYTHONPATH="$RLINF_ROOT:${PYTHONPATH:-}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$XDG_CACHE_HOME/openpi}"
export PI05_BASE_CHECKPOINT="${LBENCH_ALIGNED_BASE_CHECKPOINT:-/root/autodl-tmp/models/openpi/pi05_base_lerobot_fp32_001}"
export LBENCH_LEROBOT_REPO="${LBENCH_ALIGNED_DATASET:-/root/autodl-tmp/datasets/lbench_8tasks_v21_merged}"
export LBENCH_NORM_STATS="${LBENCH_ALIGNED_NORM_STATS:-/root/autodl-tmp/outputs/norm/lbench_8tasks_v30_lerobot_exact_001/norm_stats.json}"
export LBENCH_SFT_OUTPUT="${LBENCH_SFT_OUTPUT:-/root/autodl-tmp/outputs/lbench_sft_lerobot_aligned_fp32_4gpu_$(date -u +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${RLINF_GPUS:-0,1,2,3}"
export JAX_PLATFORMS=cpu
# LeRobot's float32 video-frame temporaries can raise glibc's dynamic mmap
# threshold, leaving freed buffers in each long-lived DataLoader worker heap.
# Fix both thresholds before Python/Ray/spawn start so large buffers return to
# the OS. These controls change allocation only, not decoding or preprocessing.
export MALLOC_MMAP_THRESHOLD_="${LBENCH_ALIGNED_MALLOC_MMAP_THRESHOLD:-131072}"
export MALLOC_TRIM_THRESHOLD_="${LBENCH_ALIGNED_MALLOC_TRIM_THRESHOLD:-131072}"

IFS=',' read -r -a lbench_training_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#lbench_training_gpus[@]}" -ne 4 ]]; then
  echo "This recipe requires four GPU IDs in RLINF_GPUS (default: 0,1,2,3)." >&2
  exit 2
fi
cd "$RLINF_ROOT"

# Read headers only. Refuse rounded BF16 inputs even if the model would later
# cast them to FP32; upcasting cannot restore the original source weights.
"$RLINF_PYTHON" - <<'PY'
import os
from pathlib import Path
from toolkits.lerobot.convert_pi05_base_to_openpi import read_header
from toolkits.lbench_data_dry_run import validate_norm_stats

base = Path(os.environ["PI05_BASE_CHECKPOINT"]) / "model.safetensors"
header = read_header(base)
bad = [k for k, v in header.items() if k != "__metadata__" and v["dtype"] != "F32"]
if bad:
    raise ValueError(f"Aligned training requires original FP32 base weights: {bad[:3]}")
validate_norm_stats(Path(os.environ["LBENCH_NORM_STATS"]))
print("Base weights:", base)
print("Norm stats:", os.environ["LBENCH_NORM_STATS"])
print("Training output:", os.environ["LBENCH_SFT_OUTPUT"])
print("Host allocator mmap/trim thresholds:", os.environ["MALLOC_MMAP_THRESHOLD_"], os.environ["MALLOC_TRIM_THRESHOLD_"])
PY

mkdir -p "$LBENCH_SFT_OUTPUT"
"$RLINF_PYTHON" examples/sft/train_vla_sft.py \
  --config-name lbench_sft_openpi_pi05_lerobot_aligned \
  'cluster.component_placement={actor\,env\,rollout:0-3}' \
  runner.logger.log_path="$LBENCH_SFT_OUTPUT" \
  runner.logger.experiment_name=lbench_sft_pi05_lerobot_aligned_4gpu \
  "$@" 2>&1 | tee -a "$LBENCH_SFT_OUTPUT/train.log"
