# 四卡插花 PPO 一轮验证

入口：`scripts/lbench/remote/19_ppo_flower_4gpu_validation.sh --checkpoint <SFT检查点>`。
必须显式指定本次完成的 SFT 检查点；Actor 和 Rollout 使用检查点内的 norm-stats。
包装脚本把 `--smoke-test-steps` 固定为 1，执行更新前评测、一轮 PPO、保存、更新后评测后退出。
学习率调度仍按 1000 轮，不将短验证当成完整衰减周期。

| 参数 | 设置 |
| --- | --- |
| 布局 | 4 卡共卡，Actor/Rollout/环境各 4 个 rank |
| 环境 | 每卡 4 个，总共 16 个；训练和评测环境分阶段释放、重建 |
| 训练采样 | 4 批，共 64 条轨迹 |
| 评测 | 前后各 32 条，固定 seed 0–31，初始噪声 seed 42 |
| 动作 | 20/20、5 步去噪；训练 SDE 噪声 0.25、ignore_last=true，评测 ODE |
| 时长 | 每回合最多 600 步；训练窗口 640，终止后的填充由 mask 排除 |
| PPO | micro batch 32、global batch 256、每卡累积 2 次、4 遍更新 |
| 更新上限 | 16×4×32÷256×4=32 次优化器更新 |
| 学习率 | Actor 1e-6、Critic 1e-4 |
| Critic | 隐藏层 1024/512/256；dual clip c=3 |

使用 FP32 PPO、NO_SHARD 梯度同步、Actor/Rollout offload。保留数值检查，额外 KL 停止和零奖励跳过 Actor 均关闭。
加 `--record-diagnostics` 保留全部训练和评测轨迹、三相机视频、每 20 步图像、状态/动作/奖励和 PPO 数组。

```bash
bash scripts/lbench/remote/19_ppo_flower_4gpu_validation.sh \
  --gpus 0,1,2,3 --checkpoint /绝对路径/global_step_30000 \
  --record-diagnostics --output /绝对路径/新的验证目录 --dry-run
```

去掉 `--dry-run` 才真正启动；输出目录必须不存在。
本次不改变插花成功判定或光照。单轮及 32 条评测用于检查流程和观察行为，不能证明长期训练收益。
CPU 测试检查任务选择、卡数、采样/更新预算、前后评测和真实 Hydra 命令的一致性。

## 2026-09-19 实测

从 `lbench_sft_base_h20_4gpu_20260917_190130` 的 SFT `global_step_30000` 开始：
更新前 7/32、训练采样 12/64、更新后 6/32 成功。32 次更新全部接受，
更新耗时约 224 秒，全流程约 37.1 分钟，采样到的单卡显存峰值 66.49 GiB。
退出码 0，结束后四卡显存释放。128 条轨迹、384 个视频及四份 PPO rank 数据的
数值、奖励与有效动作 mask 核验通过；134 项相关 CPU 测试通过。

本轮未观察到成功率提升；4 个种子由失败变成功，5 个反向变化。
结果目录：`/root/autodl-tmp/outputs/lbench_flower_sft30000_4gpu_validation_20260919`，
包含 `RESULT_ZH.md`、`videos.html`、原始轨迹、PPO 数据和第 1 轮完整检查点。
