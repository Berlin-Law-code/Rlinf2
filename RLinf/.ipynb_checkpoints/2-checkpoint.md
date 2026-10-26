cd /root/autodl-tmp/Berlin/RLinf

  source /root/autodl-tmp/.venvs/rlinf-openpi/bin/activate

  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export EMBODIED_PATH=/root/autodl-tmp/Berlin/RLinf/examples/embodiment
  export PYTHONPATH=/root/autodl-tmp/Berlin/RLinf
  export PI05_BASE_CHECKPOINT=/root/autodl-tmp/models/openpi/pi05_base_pytorch_bf16
  export LBENCH_LEROBOT_REPO=/root/autodl-tmp/datasets/lbench_8tasks_v21_merged
  export LBENCH_NORM_STATS=/root/autodl-tmp/outputs/norm/lbench_8tasks_v21/norm_stats.json
  export HF_HOME=/root/autodl-tmp/.cache/huggingface

  # 减少无关日志
  export RAY_DEDUP_LOGS=1
  export RAY_DISABLE_DOCKER_CPU_WARNING=1
  export TF_CPP_MIN_LOG_LEVEL=2
  export PYTHONWARNINGS="ignore::FutureWarning"

  export OUTPUT=/root/autodl-tmp/outputs/lbench_sft_4gpu_mb8_gb64_003
  mkdir -p "$OUTPUT"

  python examples/sft/train_vla_sft.py \
    --config-name lbench_sft_openpi_pi05 \
    'cluster.component_placement={actor\,env\,rollout:0-3}' \
    runner.max_steps=30000 \
    runner.save_interval=10000 \
    runner.logger.log_path="$OUTPUT" \
    runner.logger.experiment_name=lbench_sft_4gpu_mb8_gb64 \
    actor.micro_batch_size=8 \
    actor.global_batch_size=64 \
    actor.model.openpi.train_expert_only=false \
    actor.fsdp_config.gradient_checkpointing=false \
    actor.fsdp_config.mixed_precision.param_dtype=null \
    2>&1 | tee "$OUTPUT/train.log"

  只查看进度和错误，避免刷屏：

  tail -F "$OUTPUT/train.log" \
    | tr '\r' '\n' \
    | rg --line-buffered 'Global Step|Exception|Traceback|RuntimeError|out of memory'

  查看显存：

  watch -n 1 nvidia-smi

  断点续训，例如从第 5000 步恢复：

  export CKPT="$OUTPUT/lbench_sft_4gpu_mb8_gb64/checkpoints/global_step_10000"

  python examples/sft/train_vla_sft.py \
    --config-name lbench_sft_openpi_pi05 \
    'cluster.component_placement={actor\,env\,rollout:0-3}' \
    runner.max_steps=30000 \
    runner.save_interval=5000 \
    runner.resume_dir="$CKPT" \
    runner.logger.log_path="$OUTPUT" \
    runner.logger.experiment_name=lbench_sft_4gpu_mb8_gb64 \
    actor.micro_batch_size=8 \
    actor.global_batch_size=64 \
    actor.model.openpi.train_expert_only=false \
    actor.fsdp_config.gradient_checkpointing=false \
    actor.fsdp_config.mixed_precision.param_dtype=null \
    2>&1 | tee "$OUTPUT/train.log"

  echo $! | tee "$OUTPUT/train.pid"