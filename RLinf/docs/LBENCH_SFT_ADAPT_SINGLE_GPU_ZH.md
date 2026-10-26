# 单卡 SFT 适配（20/20）

入口：`scripts/lbench/remote/15_sft_adapt_1gpu.sh`。
从原四卡 SFT 的 `global_step_30000` 完整 FP32 权重初始化，重新建立优化器和学习率计划。
本次新增 30000 次优化器更新；微批 16、全局 batch 16、单卡累积 1 次。
骨干全量微调（`train_expert_only=false`，关闭 LoRA），使用 FP32 参数、BF16 AMP、梯度检查点和 NO_SHARD。
学习率设置为 `1e-5`，预热 1000 步，使用现有 `lerobot_cosine` 衰减至 `1e-6`。

## 数据与初始化

- 来源：`/root/autodl-tmp/datasets/domain_random_v21_200`。
- 读取目录：`/root/autodl-tmp/datasets/lbench_8tasks_v21_sft_adapt_h20`。
- 8 个任务、1600 条示范、1171428 帧；所有数值列哈希与旧 SFT 数据一致。
- 4800 段视频通过硬链接复用；不要原地改写来源视频或合并目录的视频。
- 初始化：`/root/autodl-tmp/outputs/lbench_sft_adapt_h20_20260917/initialization_h20`。
- 新统计来自 `lbench_norm_verify_20260917/openpi_h20_batch32/norm_stats.json`，已复制到初始化目录。
- 初始化目录复用原 SFT 权重，`initialization.json` 明确记录其尚未适配；没有更改原模型或恢复旧 optimizer。
- SFT 加载器现在将显式 `openpi.action_horizon` 传到数据采样与变换配置，真实 batch 动作标签为 `[16,20,32]`，loss 使用其中的实际 8 维。

## 启动与检查

```bash
cd /root/autodl-tmp/Berlin/RLinf
bash scripts/lbench/remote/15_sft_adapt_1gpu.sh --dry-run
# 下面这一行会真正启动训练：
bash scripts/lbench/remote/15_sft_adapt_1gpu.sh
```

每次启动默认创建独立的带时间戳输出目录，终端打印路径，日志写入 `train.log`，TensorBoard 写入 `tensorboard/`。
可通过 `LBENCH_SFT_ADAPT_OUTPUT` 指定输出目录；恢复本次新 SFT 时传入 `runner.resume_dir=.../global_step_N`。
不要将旧四卡 checkpoint 填到 `resume_dir`：数据加载状态要求相同 world size。
每 5000 步保存完整 checkpoint，完成新保存后仅保留本次运行最近三个完成的 checkpoint。

CPU 预检查不启动 Ray、Isaac、模型前反向或 GPU 验证。单卡微批 16 的显存峰值尚未实测；遇到 OOM 将返回失败，不会自动更改 batch。

## 环境评测

本训练命令只自动保存 checkpoint，**不自动执行环境评测**。当前 VLA SFT 的内置 eval 接口未实现，不能直接开启 `val_check_interval`。
单卡需先释放训练占用的显存，再对某个新 checkpoint 独立评测：

```bash
bash scripts/lbench/remote/16_eval_sft_adapt_flower.sh /绝对路径/global_step_5000
```

插花任务评测固定种子 0–15、2 个环境分 8 批、每回合最多 600 步、20/20、5 步 ODE 推理，初始动作噪声种子 42，保留视频和 episode 日志。
评测脚本支持第二个参数 `--dry-run`，仅解析配置。该脚本尚未进行 GPU 仿真验证。
TODO(agent): 在单卡 GPU 验证后接入自动定期环境评测；避免每次重建训练进程造成大量历史数据重放。

## CPU 验证

目标单测包括 `test_lbench_sft_adaptation.py`、`test_lbench_lerobot_merge.py`、`test_lbench_aligned_pipeline.py` 和 `test_lerobot_lr_scheduler.py`。
核验日志位于 `/root/autodl-tmp/outputs/lbench_sft_adapt_h20_20260917/`，包括数据哈希、真实 batch、启动配置解析和测试结果。
