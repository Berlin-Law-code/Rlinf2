# 单卡插花 PPO 一轮验证

适用单张约 96 GiB GPU，从原始 SFT 检查点开始。`14_ppo_flower_1gpu.sh` 使用 `collocated_1gpu` 布局，只选择插花任务，默认一轮后保存并退出。不会恢复 clean PPO。

也支持用 `--checkpoint` 指定 20 步适配后的 FP32 SFT 检查点，并使用 `--action-horizon 20`。启动器读取该检查点内的归一化统计，不使用原始 SFT 的旧统计。20 步来源的模型须保持预测长度为 20；原 50 步来源的实验行为保持兼容。

```bash
bash scripts/lbench/remote/14_ppo_flower_1gpu.sh --gpus 0 --record-diagnostics --dry-run
bash scripts/lbench/remote/14_ppo_flower_1gpu.sh --gpus 0 --record-diagnostics --output /root/autodl-tmp/outputs/flower_single_gpu_trial
```

输出目录必须不存在。去掉 `--dry-run` 才运行。`--smoke-test-steps` 默认为 1，学习率日程保留 1000 轮。配置文件为 `examples/embodiment/config/lbench_ppo_openpi_pi05_flower_1gpu.yaml`。

- Actor、Rollout、环境都在 GPU 0；两种模型启用 offload。
- 训练/评测各 4 个并行环境，按阶段释放切换，不同时驻留两套环境。
- 每轮训练采样 4 批，共 16 条；评测采样 2 批，共 8 条，固定种子 0–7，固定初始噪声种子 42。更新前后均评测。
- 单回合 600 个真实动作；训练窗口 640，终止填充由 mask 排除。
- 预测/执行动作 20/20，去噪 10；Actor lr=1e-6，Critic lr=1e-4。
- micro batch=32，global batch=256；单卡累积 8 次。16×32=512 个含填充动作块，4 遍更新，每轮最多 512÷256×4=8 次优化器更新。
- dual clip c=3；Critic 隐藏层 1024/512/256；保持原有数值检查，额外 KL 停止关闭，零奖励不额外跳过 Actor。
- `--record-diagnostics` 保留训练与评测的状态/动作/奖励、三个相机视频、每 20 步无损图像和更新前 PPO 数组。

本次沿用已有插花成功谓词，未修改几何判定或奖励。单轮、8 条评测只用于验证流程与观察行为，不能证明训练有效性；插花的包围盒 IoU 判定仍不等价于松手后稳定插入。

CPU 检查覆盖单卡设备映射、任务选择、串行环境池、更新预算、学习率日程和实际 Hydra 命令一致性；显存和实际模拟必须由 GPU 验证确认。

2026-09-16 实测：原始 SFT 3/8、训练采样 4/16、更新后评测 4/8；8 次优化器更新全部接受，更新耗时约 220 秒，含初始化和两次评测总耗时约 34 分钟。每 5 秒监测到的显存峰值为 65.77 GiB，无 OOM，退出码 0，结束后显存释放。32 条轨迹、96 个视频及 PPO 数组检查通过。结果保存在 `/root/autodl-tmp/outputs/lbench_flower_single_validation_20260916/RESULT_ZH.md`。样本少，成功率差异不能作为稳定提升的证据。

## 5 步去噪对照

保持原始 SFT、环境数量、种子和 PPO 参数一致，以新实验进行 5 步验证：

```bash
bash scripts/lbench/remote/14_ppo_flower_1gpu.sh --gpus 0 --denoise-steps 5 --record-diagnostics --output /root/autodl-tmp/outputs/flower_single_gpu_denoise5_trial
```

`--denoise-steps` 同时覆盖 `actor.model.num_steps` 和 `actor.model.openpi.num_steps`；训练运行中的 Rollout 从同一个 Actor 模型配置构建，因此采样和 PPO 概率重算一致。默认仍为 10。SDE 至少需要 2 步。改变去噪步数会被现有 `--resume-dir` 配置一致性检查拒绝，本对照从原始 SFT 新建一轮实验。

2026-09-16 的 5 步实测已完成：SFT 基线 2/8、训练采样 4/16、更新后评测 5/8；8 次优化器更新全部接受，更新耗时约 217 秒，监测到的显存峰值为 66.49 GiB。32 条轨迹、96 个视频及 PPO 有效步数掩码核验通过；退出码 0，结束后 GPU 显存归零。结果保存在 `/root/autodl-tmp/outputs/lbench_flower_single_denoise5_20260916/RESULT_ZH.md`。

这次启动前，实际加载的合并任务配置中的光照随机化已被关闭，验证保留了该设置（固定强度 1500、旋转为零）；上次 10 步使用随机光照。因此两次实测不是仅改变去噪步数的严格对照，不能将结果差异单独归因于 5 步去噪。本次前后评测使用相同种子 0–7，但 8 条样本仍不足以证明稳定提升，且成功条件仍未要求松手后稳定插入。

## 排除最后一步探索噪声

`--ignore-last` 将 `actor.model.openpi.ignore_last` 设为 `true`，训练的随机 SDE 步只从最后一步之前选择；评测仍为 ODE。默认不覆盖模型设置，`--no-ignore-last` 可显式设为 `false`。这是一项针对振荡的实验，不是 LIBERO-Long 的默认配置。

该参数可与 `--resume-dir` 一起使用，恢复模型、优化器和轮数，只允许这一项显式采样配置变化；其他模型、优化器及轨迹配置一致性检查仍保留。恢复后必须采集新轨迹，不重用旧采样批次。新输出目录和 `launch_plan.json` 会记录本次覆盖。单卡默认仅新增一轮，验证完成后保存并停止：

```bash
bash scripts/lbench/remote/14_ppo_flower_1gpu.sh --gpus 0 --denoise-steps 5 --ignore-last --resume-dir /path/to/global_step_N --smoke-test-steps 1 --record-diagnostics --output /root/autodl-tmp/outputs/flower_ignore_last_trial
```

2026-09-16 `ignore_last=true` 实测：从第 15 轮恢复至 16，训练采样 5/16 成功，更新前后 ODE 评测 3/8→1/8，8 次优化器更新全部完成，验证结束后停止。与旧配置同一检查点、seed 1240–1243 的四条轨迹相比，块内目标变化均值下降约 65%，实际关节反转比例下降约 74%；但初始图像和物体姿态并非完全重现，不能当成严格单因素因果对照。32 条轨迹、96 个视频及 PPO 数据完整性核验通过。结果：`/root/autodl-tmp/outputs/lbench_flower_ignore_last_20260916/RESULT_ZH.md`。

2026-09-17（目录日期）从原始SFT重新验证ignore_last=true：基线4/8、训练3/16、更新后5/8；8次优化器更新全部完成，一轮后正常停止。相较同SFT起点false配置的16条训练轨迹，块内目标变化均值约下降66%、实际反转比例约下降77%，但最终评测与旧false验证同为5/8。重建场景的初始图像仍有差异，少量样本不能确定因果或长期收益。结果：`/root/autodl-tmp/outputs/lbench_flower_sft_ignore_last_20260917/RESULT_ZH.md`。


## 较小 SDE 探索噪声对照

```bash
bash scripts/lbench/remote/14_ppo_flower_1gpu.sh --gpus 0 --denoise-steps 5 --ignore-last --noise-level 0.25 --smoke-test-steps 1 --record-diagnostics --output /root/autodl-tmp/outputs/flower_sft_noise025_trial
```

`--noise-level` 覆盖 `actor.model.openpi.noise_level`，Actor 和 Rollout 使用同一模型配置。省略时仍沿用配置中的 0.5。仅接受有限正数；零噪声不适用于当前 PPO 概率计算。此对照从原始 SFT 开始；修改噪声的 `--resume-dir` 恢复会被已有模型配置一致性检查拒绝。从本次 0.25 检查点恢复时，需要再次显式提供相同噪声、去噪步数和 ignore-last 设置。

训练采样使用 SDE；评测仍为 ODE，噪声幅度不改变评测的去噪公式。固定场景种子不保证 Isaac 场景初始化及渲染完全相同，对照时须检查初始状态和图像。

2026-09-17 噪声 0.25 实测：原始 SFT 3/8，训练采样 3/16，一轮后 4/8；8 次更新全部接受，耗时约216秒，全流程约33分钟。32条轨迹、96个视频及PPO奖励/掩码校验通过，显存峰值65.92GiB，退出码0、结束后显存释放。相较噪声0.5同SFT/ignore_last=true对照，实际关节反转比例均值降低43.5%，训练成功数相同；最终KL诊断0.03553高于对照0.01417。单轮结果不足以确认成功率优劣。报告：`/root/autodl-tmp/outputs/lbench_flower_sft_noise025_20260917/RESULT_ZH.md`。
