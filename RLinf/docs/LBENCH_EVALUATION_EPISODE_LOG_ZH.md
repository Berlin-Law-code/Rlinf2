# LBench 独立评估逐回合日志

`LBenchEnv` 可在每个回合第一次终止时立即保存一条 JSONL；默认关闭，仅允许在 `is_eval=true` 的环境开启。它不需要开启图像轨迹采集，也不依赖整次评估正常结束。

对独立评估入口添加以下 Hydra 参数（使用本次评估自己的输出目录）：

```bash
'+env.eval.episode_log.enabled=true' \
'+env.eval.episode_log.dir=/absolute/evaluation/output/episode_logs' \
'+env.eval.episode_log.model_label=PPO46' \
'+env.eval.episode_log.checkpoint=/absolute/checkpoints/global_step_46'
```

每个环境 rank / pipeline stage 独立写入 `rank_000.jsonl` 等文件。文件采用追加模式，每条记录写完立即 flush 并关闭；重新启动时应使用新的评估输出目录，避免把不同批次混在一起。`writer_pid`、`recorded_utc` 和 `run_id` 可追溯记录来源。模型标签和检查点字段是调用方提供的标识，不代替权重哈希校验。

记录包含：`episode_id`、任务、`scene_seed`、`success`、真实执行步数 `episode_length`、`done_reason`、`failure_reason`、`stage_id`、`terminated`、`truncated`、`policy_action_override_count`。计数和判据取自真实终止的物理步，自动 reset 之前落盘；终止之后的填充步不会产生重复记录。override 只统计适配器检测到、并真正执行的动作替换。

`done_reason_source=backend` 表示直接保存仿真器原因；缺少该信息时使用成功、已有 failure reason、适配器超时或一般终止推导，并标记 `derived`。不能仅根据 `stage_id` 或“提前结束”推断具体失败原因。成功阶段与失败阶段应分开统计。

场景 seed 不代表策略初始动作噪声已经固定；动作噪声配对需要评估采样器另行控制。本功能只保存终止标量，不改变奖励、动作、场景调度或成功判据。中止评估时，仅统计已经完整写入的回合。

CPU 验证：

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python -m pytest -q tests/unit_tests/test_lbench_episode_log.py \
  tests/unit_tests/test_lbench_env.py
```

测试使用假的向量仿真后端，覆盖不同时间的终止、填充、自动 reset、显式失败、适配器超时、动作替换计数、跨 rank 标识、追加写入及默认关闭路径，不启动 Isaac 或 GPU 模型。
