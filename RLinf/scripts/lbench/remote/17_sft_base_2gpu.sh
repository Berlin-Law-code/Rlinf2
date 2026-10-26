#!/usr/bin/env bash
set -euo pipefail

export RLINF_ROOT="${RLINF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export RLINF_PYTHON="${RLINF_PYTHON:-/root/autodl-tmp/.venvs/rlinf-openpi/bin/python}"
export PYTHONPATH="$RLINF_ROOT:${PYTHONPATH:-}"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
export PATH="$(dirname "$RLINF_PYTHON"):$PATH"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$XDG_CACHE_HOME/openpi}"
# Separate names prevent a previous SFT adaptation shell selecting old weights.
export LBENCH_SFT_BASE_CHECKPOINT="${LBENCH_SFT_BASE_CHECKPOINT:-/root/autodl-tmp/models/openpi/pi05_base_lerobot_fp32_001}"
export LBENCH_SFT_BASE_DATASET="${LBENCH_SFT_BASE_DATASET:-/root/autodl-tmp/datasets/lbench_8tasks_v21_sft_adapt_h20}"
export LBENCH_SFT_BASE_NORM_STATS="${LBENCH_SFT_BASE_NORM_STATS:-/root/autodl-tmp/outputs/lbench_norm_verify_20260917/openpi_h20_batch32/norm_stats.json}"
export LBENCH_SFT_BASE_OUTPUT="${LBENCH_SFT_BASE_OUTPUT:-/root/autodl-tmp/outputs/lbench_sft_base_h20_2gpu_$(date -u +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${LBENCH_SFT_BASE_GPUS:-0,1}"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072
export RAY_prestart_worker_first_driver=0 RAY_enable_worker_prestart=0
export RAY_ADDRESS=local

if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "Set LBENCH_SFT_BASE_GPUS to exactly two distinct GPU IDs (default: 0,1)." >&2
  exit 2
fi
IFS=',' read -r -a lbench_base_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${lbench_base_gpus[0]}" == "${lbench_base_gpus[1]}" ]]; then
  echo "LBENCH_SFT_BASE_GPUS must contain distinct GPU IDs." >&2
  exit 2
fi
cd "$RLINF_ROOT"

# CPU preflight only: read metadata and weight headers, without constructing a model.
"$RLINF_PYTHON" - <<'PY'
import json, os
from pathlib import Path
from toolkits.lbench_data_dry_run import validate_norm_stats
from toolkits.lerobot.convert_pi05_base_to_openpi import read_header

data = Path(os.environ['LBENCH_SFT_BASE_DATASET'])
info = json.loads((data / 'meta/info.json').read_text())
if info['codebase_version'] != 'v2.1' or (
    info['total_tasks'], info['total_episodes'], info['total_frames']
) != (8, 1600, 1171428):
    raise ValueError('Expected the verified eight-task LBench v2.1 dataset')
verified = json.loads((data / 'adaptation_verification.json').read_text())
if not verified['all_numeric_columns_match_original_sft']:
    raise ValueError('Dataset numeric verification failed')
base = Path(os.environ['LBENCH_SFT_BASE_CHECKPOINT'])
if (base / 'actor').exists() or (base / 'model_state_dict').exists():
    raise ValueError('This recipe starts from base weights, not an SFT checkpoint')
header = read_header(base / 'model.safetensors')
bad = [k for k, v in header.items() if k != '__metadata__' and v['dtype'] != 'F32']
if bad:
    raise ValueError(f'Original FP32 base weights are required: {bad[:3]}')
validate_norm_stats(Path(os.environ['LBENCH_SFT_BASE_NORM_STATS']))
print('Dataset:', data)
print('Base initialization:', base)
print('Norm stats:', os.environ['LBENCH_SFT_BASE_NORM_STATS'])
print('Output:', os.environ['LBENCH_SFT_BASE_OUTPUT'])
PY

if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  exec "$RLINF_PYTHON" examples/sft/train_vla_sft.py \
    --config-name lbench_sft_openpi_pi05_base_2gpu --cfg job --resolve "$@"
fi

mkdir -p "$LBENCH_SFT_BASE_OUTPUT"
"$RLINF_PYTHON" examples/sft/train_vla_sft.py \
  --config-name lbench_sft_openpi_pi05_base_2gpu "$@" \
  2>&1 | tee -a "$LBENCH_SFT_BASE_OUTPUT/train.log"
