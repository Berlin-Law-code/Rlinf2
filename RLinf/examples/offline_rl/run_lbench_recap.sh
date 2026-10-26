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
export LBENCH_RECAP_OUTPUT="${LBENCH_RECAP_OUTPUT:-/root/autodl-tmp/outputs/lbench_recap_round1}"
cd "$REPO_PATH"
stage="${1:-check}"
if (( $# > 0 )); then shift; fi
config_dir="$REPO_PATH/examples/offline_rl/config"
case "$stage" in
  prepare)
    exec /root/autodl-tmp/conda/envs/lerobot01/bin/python \
      toolkits/lerobot/prepare_lbench_recap.py "$@" ;;
  check)
    exec python toolkits/lbench_recap_preflight.py "$@" ;;
  returns)
    exec python examples/offline_rl/advantage_labeling/recap/process/compute_returns.py \
      --config-path "$config_dir" --config-name lbench_recap_returns "$@" ;;
  value)
    # The full pipeline already owns the lock, logging and memory monitor.
    if [[ "${LBENCH_RECAP_PIPELINE:-0}" == 1 ]]; then
      exec python examples/offline_rl/advantage_labeling/recap/train_value.py \
        --config-path "$config_dir" --config-name lbench_recap_value "$@"
    fi
    mkdir -p "$LBENCH_RECAP_OUTPUT"
    exec 9>"$LBENCH_RECAP_OUTPUT/pipeline.lock"
    flock -n 9 || { echo "Training is already running in this output directory." >&2; exit 1; }
    python toolkits/monitor_training_memory.py --parent-pid "$$" \
      --output "$LBENCH_RECAP_OUTPUT/memory.jsonl" 9>&- &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT
    echo value > "$LBENCH_RECAP_OUTPUT/stage.txt"
    if python examples/offline_rl/advantage_labeling/recap/train_value.py \
      --config-path "$config_dir" --config-name lbench_recap_value "$@" \
      2>&1 | tee -a "$LBENCH_RECAP_OUTPUT/value.log"; then
      echo complete:value > "$LBENCH_RECAP_OUTPUT/stage.txt"
    else
      train_status=$?
      echo failed:value > "$LBENCH_RECAP_OUTPUT/stage.txt"
      exit "$train_status"
    fi ;;
  advantages)
    : "${RECAP_VALUE_CHECKPOINT:?Set RECAP_VALUE_CHECKPOINT to a completed value checkpoint}"
    exec torchrun --standalone --nproc_per_node=2 \
      examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py \
      --config-path "$config_dir" --config-name lbench_recap_advantages "$@" ;;
  policy)
    exec python examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py \
      --config-path "$config_dir" --config-name lbench_recap_policy "$@" ;;
  all)
    if (( $# > 0 )); then
      echo "Use individual stages to supply Hydra overrides." >&2
      exit 2
    fi
    mkdir -p "$LBENCH_RECAP_OUTPUT"
    # Hold a lock for the whole run to prevent duplicate launches.
    exec 9>"$LBENCH_RECAP_OUTPUT/pipeline.lock"
    flock -n 9 || { echo "A pipeline is already running in this output directory." >&2; exit 1; }
    python toolkits/monitor_training_memory.py --parent-pid "$$" \
      --output "$LBENCH_RECAP_OUTPUT/memory.jsonl" &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    trap 'echo "failed:${next_stage:-startup}" > "$LBENCH_RECAP_OUTPUT/stage.txt"' ERR
    for next_stage in check returns value; do
      echo "$next_stage" > "$LBENCH_RECAP_OUTPUT/stage.txt"
      LBENCH_RECAP_PIPELINE=1 bash "$0" "$next_stage" > "$LBENCH_RECAP_OUTPUT/$next_stage.log" 2>&1
    done
    export RECAP_VALUE_CHECKPOINT="$LBENCH_RECAP_OUTPUT/value/checkpoints/global_step_8000"
    test -f "$RECAP_VALUE_CHECKPOINT/.checkpoint_complete"
    for next_stage in advantages policy; do
      echo "$next_stage" > "$LBENCH_RECAP_OUTPUT/stage.txt"
      bash "$0" "$next_stage" > "$LBENCH_RECAP_OUTPUT/$next_stage.log" 2>&1
    done
    echo complete > "$LBENCH_RECAP_OUTPUT/stage.txt" ;;
  *)
    echo 'Usage: run_lbench_recap.sh {prepare|check|returns|value|advantages|policy|all} [overrides...]' >&2
    exit 2 ;;
esac
