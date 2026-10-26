# 四卡装箱 PPO 一轮验证

任务为 `pack_in_objects_franka`，与 SFT 数据集 `pack_in_objects_franka_first200`
对应。显式传入新 SFT `global_step_30000`，读取其随检查点保存的 h20 norm-stats。
从 SFT 开始，先评测 32 次，采样 64 条训练轨迹并完成一轮 PPO，再保存和评测 32 次。
前后评测固定 seed 0–31，评测初始噪声 seed 42；结束后自动退出。

配置沿用已验证的四卡插花参数，改变任务、日志名称和任务时长：
装箱示范共 200 条，长度 669–937、中位数 796；按用户设置，训练和评测回合上限为 960。
训练窗口为 960，提前结束后的填充通过 mask 排除。
每卡 4 个环境、微批 32，全局 batch 256，梯度累积 2 次，PPO 更新 4 遍。
更新上限为 `16×4×(960÷20)÷256×4 = 48` 次；学习率调度仍按 1000 轮。
Actor 学习率 1e-6，Critic 学习率 1e-4；动作 20/20，5 步去噪，
训练 SDE 噪声 0.25、ignore_last=true，评测 ODE。未修改任务判定和光照。

```bash
bash scripts/lbench/remote/20_ppo_pack_4gpu_validation.sh \
  --gpus 0,1,2,3 --checkpoint /绝对路径/global_step_30000 \
  --record-diagnostics --output /绝对路径/新的验证目录 --dry-run
```

去掉 `--dry-run` 启动。包装脚本强制只运行一轮。
`--record-diagnostics` 保存成功与失败轨迹的三个视角视频、状态、动作、奖励和 PPO 数组。
CPU 配置测试检查任务选择、预算整除、结束时评测和实际 Hydra 命令的一致性。
单轮结果用于观察任务起点和验证训练流程，不能证明长期训练收益。

## 2026-09-19 实测

从 `lbench_sft_base_h20_4gpu_20260917_190130` 的 `global_step_30000` 开始，
960 步上限下，SFT 评测 1/32 成功、训练采样 7/64 成功、PPO 一轮后评测 2/32 成功。
48 次优化器更新全部接受，更新约 331 秒，全流程约 46.9 分钟，
每 5 秒采样的单卡显存峰值 67.32 GiB。结束时四卡显存为 0，未继续训练。
128 条轨迹、384 个视频和四份 PPO 数组完整性核验通过。

更新后新增成功种子 6、27，原来成功的种子 10 变为失败，
其余 30 条评测失败都未通过抓取阶段。样本量小，仅多成功一次，不能认定稳定提升。
现有放置判定没有显式要求稳定落入箱底；抽查成功终止画面仍在箱口附近，
因此这些是现有判定口径的成功率，需结合视频查看实际行为。

结果目录：`/root/autodl-tmp/outputs/lbench_pack_sft30000_4gpu_validation_20260919`。
详细结果为 `RESULT_ZH.md`，视频入口为 `videos.html`。

## 正式训练：500 轮，不录制

2026-09-19 正式启动时，上述装箱验证目录已不在磁盘上，因此从同一份新 SFT
`global_step_30000` 重新开始，未恢复 PPO 第 1 轮。回合和采样窗口仍为 960 步，
训练预算 500 轮，学习率调度沿用验证的 1000 轮周期，其余算法参数保持一致。
训练/评测 `data_collection.enabled=false`、`video_cfg.save_video=false`，
未启用 `actor.rollout_diagnostics_dir`。每 10 轮评测 32 次，保留 TensorBoard，
每轮保存并仅保留本次运行最新的一个检查点；保留原有数值检查和连续 3 轮零成功停止规则。

独立 screen 为 `lbench_pack`。运行目录：
`/root/autodl-tmp/outputs/lbench_pack_4gpu_500_20260919_013715`。
该目录的 `run_training.sh` 保存了完整启动命令，`console.log` 为实时日志，
`run/resolved_config.yaml` 为实际配置，`launch_checks.json` 为启动检查结果。
已有运行时请勿重复执行启动脚本。

```bash
screen -r lbench_pack
```

按 Ctrl+A，然后 D，脱离 screen 并继续运行。
