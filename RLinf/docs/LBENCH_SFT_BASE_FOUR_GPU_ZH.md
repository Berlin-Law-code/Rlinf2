# 四卡从 π0.5 基础权重重新训练 SFT

入口：`scripts/lbench/remote/18_sft_base_4gpu.sh`。
本次从原始 FP32 π0.5 预训练权重开始，重新创建优化器、学习率计划和数据迭代器。
不恢复两卡运行，也不加载旧 LBench SFT/PPO 权重。

| 参数 | 设置 |
| --- | --- |
| GPU | 0、1、2、3，FSDP full_shard |
| 每卡 micro batch / 全局 batch | 16 / 64，梯度累积 1 次 |
| 训练预算 | 30000 次优化器更新 |
| 学习率 | 峰值 5e-5，最低 5e-6，预热 1000 步，lerobot_cosine 衰减周期 30000 步 |
| 保存 | 每 5000 步，成功保存后仅保留本次最近 3 个完整检查点 |
| 微调范围 | 全量参数，关闭 LoRA |
| 精度 | FP32 参数和优化器、BF16 AMP、梯度检查点开启 |
| 动作 | 预测 20 / 执行 20，实际 8 维 |
| 数据加载进程 | 每卡 2 个，总共 8 个 |

基础权重：`/root/autodl-tmp/models/openpi/pi05_base_lerobot_fp32_001`。
数据：`/root/autodl-tmp/datasets/lbench_8tasks_v21_sft_adapt_h20`，来自
`domain_random_v21_200`，8 个任务、1600 条示范、1171428 帧。
统计：`/root/autodl-tmp/outputs/lbench_norm_verify_20260917/openpi_h20_batch32/norm_stats.json`。
每个检查点会保存本次使用的 norm-stats 和动作约定。

```bash
cd /root/autodl-tmp/Berlin/RLinf
bash scripts/lbench/remote/18_sft_base_4gpu.sh --dry-run
# 真正启动训练：
bash scripts/lbench/remote/18_sft_base_4gpu.sh
```

默认输出：`outputs/lbench_sft_base_h20_4gpu_<UTC时间戳>`，包含
`train.log`、`tensorboard/` 和实验子目录下的检查点。
覆盖输出、设备、统计文件分别使用 `LBENCH_SFT_BASE4_OUTPUT`、
`LBENCH_SFT_BASE4_GPUS`、`LBENCH_SFT_BASE4_NORM_STATS`；这些变量与两卡启动器隔离。
默认 `runner.resume_dir=null`，从 0 开始计步。

四卡 30000 步与两卡 60000 步处理的样本总量同为 192 万，优化器更新次数不同。
保持 1000 步预热意味着四卡预热期间处理的样本数是两卡时的两倍。
当前 SFT 入口不自动执行环境评测。推理使用 5 步去噪，SFT 本身采样连续 flow time。

验证：CPU 启动器测试覆盖 dry-run、错误退出码、卡数与重复 GPU 校验；
实际运行需要确认四个 worker 均报告 `local_batch_size: 16`，并检查 loss、梯度和步耗时。
