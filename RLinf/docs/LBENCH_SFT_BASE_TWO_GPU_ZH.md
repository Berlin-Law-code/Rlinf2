# 两卡从 π0.5 基础权重重新训练 SFT

入口：`scripts/lbench/remote/17_sft_base_2gpu.sh`。
从原始 FP32 π0.5 预训练权重初始化，重新建立优化器和学习率计划。
不加载旧 LBench SFT/PPO 权重；不是将模型参数随机初始化。

| 参数 | 设置 |
| --- | --- |
| GPU | 0、1，FSDP full_shard |
| 每卡 micro batch / 全局 batch | 16 / 32，梯度累积 1 次 |
| 训练预算 | 60000 次优化器更新 |
| 学习率 | 峰值 5e-5，最低 5e-6，预热 1000 步，lerobot_cosine |
| 保存 | 每 5000 步，成功保存后仅保留本次最近 3 个完整检查点 |
| 微调范围 | 全量参数，关闭 LoRA |
| 精度 | FP32 参数和优化器，BF16 AMP，梯度检查点开启 |
| 动作 | 预测 20 / 执行 20，实际 8 维 |
| 推理去噪 | 5 步；SFT 训练采样连续 flow time |

基础权重：`/root/autodl-tmp/models/openpi/pi05_base_lerobot_fp32_001`。
`conversion.json` 记录其来自 `lerobot/pi05_base` 原始 FP32 权重，仅转换存储格式。
数据读取目录：`/root/autodl-tmp/datasets/lbench_8tasks_v21_sft_adapt_h20`，
来自 `domain_random_v21_200` 的 8 个任务、1600 条示范、1171428 帧。
使用新计算的 `lbench_norm_verify_20260917/openpi_h20_batch32/norm_stats.json`。
训练保存时会把统计文件和 20 步动作约定打包到检查点。

```bash
cd /root/autodl-tmp/Berlin/RLinf
bash scripts/lbench/remote/17_sft_base_2gpu.sh --dry-run
# 真正启动训练：
bash scripts/lbench/remote/17_sft_base_2gpu.sh
```

默认输出到独立的 `outputs/lbench_sft_base_h20_2gpu_<UTC时间戳>`。
`train.log` 和 `tensorboard/` 记录过程；`--dry-run` 只做 CPU 检查和配置解析。
可通过 `LBENCH_SFT_BASE_OUTPUT` 指定本次目录。
恢复本次两卡训练时显式传入 `runner.resume_dir=/路径/global_step_N`；
新训练默认 `runner.resume_dir=null`，从 0 开始计步。

当前 SFT 入口不执行环境评测，不能根据训练 loss 单独确认任务成功率。
环境评测可在释放训练显存后使用 `16_eval_sft_adapt_flower.sh`。
CPU 验证包括启动器退出码和 GPU 参数校验、20 步数据加载、原始 FP32 转换、学习率调度和检查点保留规则。
