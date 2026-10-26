#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [--dry-run] [hydra overrides...]" >&2
  exit 2
fi
checkpoint="$1"
shift
export RLINF_ROOT="${RLINF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export RLINF_PYTHON="${RLINF_PYTHON:-/root/autodl-tmp/.venvs/rlinf-openpi/bin/python}"
export PYTHONPATH="$RLINF_ROOT:${PYTHONPATH:-}"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
export LBENCH_ROOT=/root/autodl-tmp/Berlin/LBench_multi_env
export LBENCH_TASK_CONFIG="$LBENCH_ROOT/configs/simbox/lbench_joint_8tasks.yaml"
export LBENCH_ISAAC_PYTHON=/root/autodl-tmp/conda/envs/env_isaaclab/bin/python
export LBENCH_LEROBOT_REPO=/root/autodl-tmp/datasets/lbench_8tasks_v30_merged_200
export LBENCH_SFT_CHECKPOINT="$checkpoint"
export LBENCH_NORM_STATS="$checkpoint/lbench_sft_adapt_h20/norm_stats.json"
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cpu
export XDG_CACHE_HOME=/root/autodl-tmp/.cache
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export OPENPI_DATA_HOME="$XDG_CACHE_HOME/openpi"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072
export RAY_prestart_worker_first_driver=0 RAY_enable_worker_prestart=0 RAY_ADDRESS=local
output="${LBENCH_SFT_EVAL_OUTPUT:-/root/autodl-tmp/outputs/lbench_sft_eval_h20_$(date -u +%Y%m%d_%H%M%S)}"
dry_args=()
if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  dry_args=(--cfg job --resolve)
fi
cd "$RLINF_ROOT"
test -f "$LBENCH_NORM_STATS"
exec "$RLINF_PYTHON" evaluations/eval_embodied_agent.py \
  --config-path "$RLINF_ROOT/evaluations/lbench" \
  --config-name lbench_openpi_pi05_eval \
  "${dry_args[@]}" \
  "runner.logger.log_path=$output" \
  runner.logger.experiment_name=sft_h20_flower_eval \
  'cluster.component_placement={env\,rollout:0}' \
  env.eval.total_num_envs=2 env.eval.rollout_epoch=8 \
  'env.eval.task_indices=[3]' \
  'env.eval.fixed_seeds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]' \
  env.eval.seed=0 env.eval.max_episode_steps=600 \
  env.eval.max_steps_per_rollout_epoch=600 \
  env.eval.startup_timeout_s=1200 env.eval.request_timeout_s=300 \
  'env.eval.isaac_cuda_visible_devices=[0]' \
  ++env.eval.backend_config.simbox_eval_config.cpu_thread_limit=4 \
  ++env.eval.use_fixed_reset_state_ids=true \
  env.eval.data_collection.enabled=false env.eval.video_cfg.save_video=true \
  ++env.eval.episode_log.enabled=true \
  "++env.eval.episode_log.dir=$output/eval_episodes" \
  ++env.eval.episode_log.model_label=sft_adapt_h20 \
  rollout.model.num_steps=5 rollout.model.openpi.action_horizon=20 \
  ++rollout.model.openpi.num_steps=5 \
  ++rollout.model.openpi.parameter_precision=float32 \
  ++rollout.model.openpi_data.lerobot_compatible=true \
  ++rollout.eval_initial_noise_seed=42 ++rollout.eval_repeat_initial_noise=true \
  "$@"
