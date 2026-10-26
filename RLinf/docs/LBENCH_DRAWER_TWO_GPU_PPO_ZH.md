# 两张 PRO6000：抽屉任务 PPO 强化微调

当前三卡入口见 [三卡共卡训练与采样](LBENCH_DRAWER_THREE_GPU_COLLOCATED_ZH.md)。
本页两卡入口继续保留原设备分工。

本配置从原始 SFT step 30000 重新开始，只训练
`store_objects_in_the_drawer_0`（任务索引 6）。当前每轮采集 32 个回合，
将同一批 rollout 用于 4 遍 PPO 更新，保留严格 FP32、critic 预热和可撤销的概率变化检查。
训练期间不自动评测。2026-09-11 将 `update_epoch` 从 1 调到 4，其余训练参数保持原设置；
新配置的成功率、运行耗时和资源峰值需要实际运行验证。

## 前台运行

```bash
cd /root/autodl-tmp/Berlin/RLinf
bash scripts/lbench/remote/08_ppo_drawer_2gpu.sh \
  --gpus 0,1 \
  --task store_objects_in_the_drawer_0 \
  --max-steps 500
```

由用户执行，脚本保持前台运行；输出同时显示在终端并保存为 `train.log`。
脚本使用 `/root/autodl-tmp/.venvs/rlinf-openpi/bin/python`，
Isaac 使用 `/root/autodl-tmp/conda/envs/env_isaaclab/bin/python`，无需激活 LeRobot 环境。
在命令末尾增加 `--dry-run` 可以只验证检查点、配置、元数据和 GPU 可见性；
该模式不创建训练输出目录，也不启动 Ray、模型训练或 Isaac。

`--gpus` 依次指定物理模型卡、仿真卡。GPU 0 常驻 actor 与 rollout 两份 FP32 模型；
GPU 1 运行 8 个并行训练环境。每轮先采样，再更新参数。
`runner.val_check_interval=-1` 关闭训练期间及结束时的自动评测，因此没有评测环境池。
可用 `--gpus 1,0` 交换两张卡的职责。

## 每轮训练预算

| 项目 | 当前设置 |
|---|---|
| 并行环境 / rollout_epoch | 8 / 4 |
| 每轮回合数 | 32；训练种子从 1000 连续推进 |
| 单回合上限 | 1200 个原始动作步，即 60 个执行 chunk |
| 每轮计划样本 | 1920 个 chunk 槽位，终止后的 padding 不参与 loss |
| actor micro / global batch | 1 / 480，即每次优化器更新最多累积 480 个 microbatch |
| PPO update_epoch | 4；同一批数据训练 4 遍，每轮最多 16 次优化器更新 |
| actor / critic 学习率 | `1e-6` / `1e-4`，使用 cosine 调度 |
| critic 预热 | 前 8 次成功应用的优化器更新仅更新 critic，通常是首轮前两遍 |
| 训练上限 | 500 轮，最多 16000 个回合、8000 次优化器更新 |
| 保存 | 每 5 轮保存，保留本次运行最近 3 份完整检查点 |
| 自动停止 | 连续 3 轮没有有效正奖励时，保存后正常停止 |
| 视频 / 完整回合导出 | 关闭 |

`--max-steps` 表示完整 PPO 外层轮数。当前换算为 `32 × max_steps` 个训练回合上限，
因此 500 轮的上限是 16000 回合。旧方案的 8 回合/轮、4000 回合总预算已不适用。
实际优化器更新次数可能更少：空样本批次、数值异常、策略更新保护和提前停止都会影响次数。
按正常完成预热且后续更新全部接受计算，首轮最多有 8 次 actor 更新，
后续每轮最多 16 次，500 轮最多有 7992 次 actor 更新。

`update_epoch=4` 增加的是同一批数据的使用次数，每轮仍只有 32 个新回合。
πRL 论文附录 J 表 11 中，π0.5 的 LIBERO Spatial/Object/Goal/Long 分别使用
1/1/3/4 个 update epochs，仓库 LIBERO-Long 示例也为 4。
这里采用 Long 的更新遍数作对照，不代表抽屉任务已经复现论文效果。
来源：[πRL 训练超参数](https://arxiv.org/html/2510.25889v3)。
actor 学习率仍为 `1e-6`，KL 阈值仍为 `0.02`；优化器阶段的计算量会增加，
采样量和 microbatch 大小不变，因此不能把总耗时或峰值显存简单乘以 4。
critic warmup 按优化器步计数：原来的 8 步在 `update_epoch=1` 时通常占前两轮，
现在通常在首轮第二遍末结束，第三遍开始训练 actor。这是本次配置变化的连带影响。

8 个 Isaac 进程仍并行；每轮连续采样 4 批后才开始更新，因此轨迹缓存比旧方案增加。
CPU 预检查不证明这套新配置的显存、内存峰值或学习效果。
TensorBoard 从 step 0 记录首轮；第 500 轮对应日志 step 499 和检查点 `global_step_500`。

## 更新保护如何工作

`actor.model.openpi.strict_ppo_precision=true` 在 actor 和 rollout 中禁用 TF32，
要求最高 FP32 矩阵乘精度、FP32 参数和关闭 AMP。仅把权重存成 FP32 不足以保证这些条件。
启动预检查调用实际精度验证器；运行时也验证模型配置。
FSDP 的三个精度字段使用 `fp32`，OpenPI 的 `parameter_precision` 使用 `float32`，
分别由各自的解析器处理。
本两卡入口仅启用单个 actor；保护模式要求 `NO_SHARD`、`use_orig_params=true` 和
`micro_batch_size=1`；按有效 chunk 归一化 loss，全 padding 批次不执行优化器更新。

`algorithm.ppo_update_guard` 当前设置为：

```yaml
enabled: true
target_kl: 0.02
max_initial_logratio: 0.01
probe_batch_size: 64
skip_actor_on_zero_reward: true
```

每轮从有效 chunk 中固定选取最多 64 个作为探针；初始重算概率与采样时概率的绝对
log-ratio 超过 `0.01` 时，拒绝继续按正常 PPO 更新。每个 global batch 在更新前也会检查
有效样本的概率；策略更新后再次检查固定探针。若超出 `target_kl=0.02`，撤销这一次
actor 参数和 Adam 状态的变化，保留独立 critic 更新，并跳过本轮剩余的 actor 更新。
4 遍始终使用同一 rollout 的旧概率与固定探针；KL 参考不会每遍重置。
因此 16 次是计划上限，实际 actor 更新次数应看 `train/guard/actor_updates`。
每次尝试 actor 更新前都会在 CPU 保存参数和 Adam 状态的回滚快照，增加瞬时主机内存；
最多 64 个有效 chunk 的额外前向检查也会增加每轮耗时。

这里的 KL 使用有效 chunk 上 `exp(logratio) - 1 - logratio` 的非负采样估计，
检查的是当前 rollout 旧策略与候选新策略；它不是相对最初 SFT 的 KL，
也不是最终 ODE 动作分布的精确 KL。64 个探针不能保证所有状态都满足阈值。
TensorBoard 中的 `train/actor/approx_kl` 是平均负 log-ratio，不能与该阈值直接混用。

一整轮没有有效非零奖励时，策略更新被跳过，critic 仍可学习价值目标。
当前奖励契约只在首次成功时给 1，因此连续 3 轮没有有效正奖励对应连续
96 个训练回合无成功；runner 会保存当前检查点后正常停止，避免继续进行长时间无成功训练。
概率一致性或非有限值异常也会触发保护退出，需根据 TensorBoard 的 `train/guard/*` 指标和日志检查原因。
停止原因写入输出根目录的 `stop_reason.json`。出现成功会清零连续无成功计数；
通用 runner 的阈值默认 0 表示关闭这项计数停止规则，但不会关闭 `train/guard/abort` 对应的保存与退出。

critic 预热期间 actor 权重和 Adam 状态保持不变；预热计数只累计成功应用的优化器更新。
预热结束继续使用同一组 optimizer/scheduler，保留已经学到的 critic 状态。
critic 使用冻结 VLM 的独立、已 detach 的特征，因此单独更新 critic 不应改变策略概率。

## SFT 起点与数据契约

默认检查点为：

```text
/root/autodl-tmp/outputs/lbench_sft_lerobot_aligned_fp32_4gpu_memfix_004/lbench_sft_pi05_lerobot_aligned_4gpu/checkpoints/global_step_30000
```

加载其 `actor/model_state_dict/full_weights.pt`，按 `openpi_assets.json` 动态解析
检查点内的归一化文件；不根据历史 asset 名拼接路径。
`runner.resume_dir=null` 创建新的 PPO 优化器和训练进度，不恢复历史 PPO step 50。
如需明确换起点，可传入 `--checkpoint /检查点目录`；预检查仍要求 SFT 对齐资产契约和 FP32 权重。

默认读取现存 `/root/autodl-tmp/datasets/lbench_8tasks_v30_merged_200` 的八任务文字元数据。
PPO 不打开演示视频，不重新计算归一化统计。启动脚本覆盖旧的通用环境变量，
专用 `LBENCH_PPO_DATASET` 等变量可用于明确指定资源。

模型保留 LeRobot 兼容图像/状态变换、三路相机、8 维动作、预测 50 步/执行 20 步/
10 次去噪，以及原夹爪、成功与终止判定。训练使用 Flow-SDE，噪声系数仍为 0.5；
独立评测使用 ODE。VLM 保持冻结，更新 action expert、相关投影层和独立价值头。
`group_size=1` 配合 PPO + GAE 不会自动将优势归零；奖励过滤保持关闭。

## 输出与查看

输出目录在启动时打印，默认结构如下：

```text
/root/autodl-tmp/outputs/lbench_ppo_single_task_2gpu_<UTC时间>/
  launch_plan.json
  resolved_config.yaml
  train.log
  stop_reason.json  # 保护提前停止时生成
  tensorboard/
  lbench_ppo_single_task_2gpu/checkpoints/global_step_<轮数>/
    .checkpoint_complete
    openpi_assets.json
    <asset_id>/norm_stats.json
    actor/model_state_dict/full_weights.pt
    actor/dcp_checkpoint/
```

可以用 `--output /新的/输出目录` 指定位置；已有目录会被拒绝。
正常完成 500 轮后保留 `global_step_490`、`global_step_495`、`global_step_500`。
若提前停止，也保存对应轮次的检查点。新检查点完整写入后才删除旧检查点，
保存期间可能暂时存在第 4 份。清理只针对本 runner 实例成功保存的检查点，不清理原始 SFT。

除训练成功率外，在 TensorBoard 查看 `train/guard/initial_*`、`train/guard/final_*`、
`train/guard/actor_updates`、`train/guard/critic_updates`、
`train/guard/rejected_actor_updates` 和 `train/guard/positive_reward_count`，
区分策略正在学习、只训练 critic、更新被拒绝或因无成功而停止。
训练含探索噪声，最终效果需要与原 SFT 的固定种子 0–99、100 回合 ODE 评测比较；
历史允许 TF32 的 SFT 结果为 64/100；2026-09-11 严格 FP32 的 SFT 为 66/100，
此前 `update_epoch=1` 的 guarded PPO50 为 64/100。这些结果使用原成功标准，
还未按场景固定初始动作噪声；不能把两回合差异单独认定为显著退化。
本次 `update_epoch=4` 尚无评测结果。成功判据的物理校验问题见
`/root/autodl-tmp/outputs/lbench_ppo_audit_20260911_092859/AUDIT.md`；若更改标准，
必须让 SFT 与 PPO 使用同一个新标准重新比较。

本启动器的 37 项 CPU 回归覆盖真实配置组合、精度/guard 参数拒绝、预算整除、
GPU 职责、训练种子推进、现存元数据路径、动态归一化和 dry-run 不启动训练。
实际原 SFT 检查点预检查通过：813 个 FP32 权重张量、归一化资产、任务索引和预算均一致，
Ruff、格式与 shell 语法检查通过。
`update_epoch=4` 的新配置没有由助手启动训练；CPU 检查不代表模型成功率已改善。
