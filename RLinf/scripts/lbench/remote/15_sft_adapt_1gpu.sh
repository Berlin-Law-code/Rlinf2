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
export LBENCH_SFT_ADAPT_DATASET="${LBENCH_SFT_ADAPT_DATASET:-/root/autodl-tmp/datasets/lbench_8tasks_v21_sft_adapt_h20}"
export LBENCH_SFT_ADAPT_INIT="${LBENCH_SFT_ADAPT_INIT:-/root/autodl-tmp/outputs/lbench_sft_adapt_h20_20260917/initialization_h20}"
export LBENCH_SFT_ADAPT_OUTPUT="${LBENCH_SFT_ADAPT_OUTPUT:-/root/autodl-tmp/outputs/lbench_sft_adapt_h20_$(date -u +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${LBENCH_SFT_ADAPT_GPU:-0}"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072
export RAY_prestart_worker_first_driver=0 RAY_enable_worker_prestart=0
# A separate local Ray cluster avoids attaching to an unrelated job.
export RAY_ADDRESS=local

if [[ -z "$CUDA_VISIBLE_DEVICES" || "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "Set LBENCH_SFT_ADAPT_GPU to exactly one GPU ID." >&2
  exit 2
fi
cd "$RLINF_ROOT"

# CPU preflight only: no policy construction, Ray, Isaac, or CUDA initialization.
"$RLINF_PYTHON" - <<'PY'
import json, os
from pathlib import Path
from rlinf.models.embodiment.openpi.checkpoint_assets import resolve_checkpoint_norm_stats

data = Path(os.environ['LBENCH_SFT_ADAPT_DATASET'])
info = json.loads((data / 'meta/info.json').read_text())
assert info['codebase_version'] == 'v2.1'
assert (info['total_tasks'], info['total_episodes'], info['total_frames']) == (8, 1600, 1171428)
assert (data / 'adaptation_verification.json').is_file(), 'Dataset verification is missing'
initial = Path(os.environ['LBENCH_SFT_ADAPT_INIT'])
assert (initial / 'actor/model_state_dict/full_weights.pt').is_file()
stats = resolve_checkpoint_norm_stats(initial, initial / 'lbench_sft_adapt_h20/norm_stats.json')
assert stats is not None
print('Dataset:', data)
print('Initialization:', initial)
print('Norm stats:', stats)
print('Output:', os.environ['LBENCH_SFT_ADAPT_OUTPUT'])
PY

if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  exec "$RLINF_PYTHON" examples/sft/train_vla_sft.py \
    --config-name lbench_sft_openpi_pi05_adapt_1gpu --cfg job --resolve "$@"
fi

mkdir -p "$LBENCH_SFT_ADAPT_OUTPUT"
# pipefail propagates OOM or any training failure to screen/the caller.
"$RLINF_PYTHON" examples/sft/train_vla_sft.py \
  --config-name lbench_sft_openpi_pi05_adapt_1gpu "$@" \
  2>&1 | tee -a "$LBENCH_SFT_ADAPT_OUTPUT/train.log"
