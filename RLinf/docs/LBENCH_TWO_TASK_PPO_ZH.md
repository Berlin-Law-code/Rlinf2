# 三张 PRO6000、两个任务的 PPO 试跑

最新的两卡抽屉单任务方案见 [两卡单任务命令](LBENCH_DRAWER_TWO_GPU_PPO_ZH.md)。

精度配置修复：FSDP 的 `param_dtype` / `reduce_dtype` / `buffer_dtype` 使用 `fp32`；
OpenPI 的 `parameter_precision` 使用 `float32`，实际均为 FP32。
启动预检查调用与 FSDP 相同的精度解析器，格式错误会在创建 Ray / Isaac 前报出。
expert-only PPO 同时将 `gemma_expert` 容器独立进行 FSDP 包装，避免未使用的语言输出头
在更新后变为未注册的 non-leaf 参数，导致下一轮权重同步失败。内部 decoder 仍保持原包装方式。

此配置从已经验证的 SFT `global_step_30000` 权重开始新的 PPO 训练。
默认任务为 `stack_bowls` 与 `store_objects_in_the_drawer_0`，共同更新同一个策略。
不会恢复 SFT 优化器、SFT 学习率进度或旧 PPO 训练状态。

## 运行

在能看到三张 PRO6000 的机器上前台执行：

```bash
cd /root/autodl-tmp/Berlin/RLinf
bash scripts/lbench/remote/07_ppo_two_tasks_3gpu.sh \
  --gpus 0,1,2 \
  --tasks stack_bowls store_objects_in_the_drawer_0 \
  --max-steps 20
```

脚本使用 `/root/autodl-tmp/.venvs/rlinf-openpi/bin/python`，不需要激活 LeRobot 环境。
Isaac 使用独立的 `/root/autodl-tmp/conda/envs/env_isaaclab/bin/python`。
脚本使用前台子进程，终端持续显示输出并写入 `train.log`；没有 `nohup`、`&` 或 detached session。

同一命令增加 `--dry-run` 只读取权重元数据、归一化、任务提示词并展开 Hydra 配置，
不会启动 Ray、模型训练或 Isaac，也不会创建训练输出目录。
硬件尚未到位时，dry-run 会报告 `requested_gpus_available: false`；正常运行会拒绝缺少 GPU 的配置。
此次准备时当前容器只可见 GPU 0，因此三卡真实训练尚未验证。

`--gpus` 的三个数字依次为**物理 GPU 编号**：actor、rollout、Isaac。
启动脚本会覆盖父进程的 `CUDA_VISIBLE_DEVICES` 为前两个编号，并在配置中显式固定对应物理卡。
第三张卡只供外部 Isaac 进程使用。旧 Ray 集群仍运行时，脚本报错退出，避免接入旧任务。

## 权重及 SFT 对齐

默认权重：

```text
/root/autodl-tmp/outputs/lbench_sft_lerobot_aligned_fp32_4gpu_memfix_004/
  lbench_sft_pi05_lerobot_aligned_4gpu/checkpoints/global_step_30000/
    actor/model_state_dict/full_weights.pt
```

| 项目 | 本次 PPO 设置 |
|---|---|
| 参数 | 原始 FP32 SFT 参数，读取 tensor 元数据核实精度 |
| 图像及状态处理 | 与 SFT 相同的 LeRobot 兼容变换 |
| 归一化 | 检查点内 `lbench_8tasks_v30_lerobot_exact_001/norm_stats.json` |
| 相机 | obs_camera、obs_camera_2、realsense |
| 状态/动作维度 | 8 / 8 |
| 预测 / 执行长度 | 50 / 20 |
| 去噪次数 | 10 |
| 环境长度 | 1200 个原始动作步，最多 60 次策略调用 |
| 成功、终止、夹爪规则 | 保持已验证评测设置；首次成功奖励一次，终止后不自动重置 |
| 更新范围 | 冻结 VLM，更新 action expert、相关投影层和新价值头 |
| 训练采样 | Flow-SDE，noise_level=0.5 |
| 周期评测采样 | ODE，沿用模型 eval 路径 |

SFT 使用过 BF16 autocast；本次 PPO actor 概率重算和 rollout 均使用 FP32，避免两端计算精度不同。
PPO 使用奖励和 GAE，不再使用 SFT 的动作 MSE，也不照搬 SFT 的学习率和全参数更新范围。
`runner.resume_dir=null` 表示新的 PPO 实验；`actor.model.model_path` 仍会加载上面的 SFT 权重。

预检查使用原始 `lbench_8tasks_v30_merged_200` 的元数据核对八个语言提示词。
旧的 v2.1 转换数据目录已不在本机；PPO 不读取演示视频，也不从当前数据目录重新计算归一化。

## 资源、步数与评测

| 项目 | 默认值 |
|---|---|
| GPU 0 | 1 个 actor，micro batch=1，全局 batch=32 |
| GPU 1 | 1 个 rollout 模型、环境通信 worker |
| GPU 2 | 2 个训练 Isaac 进程 + 2 个评测 Isaac 进程，合计 4 个常驻进程 |
| 每轮采样 | 2 环境 × 4 轮 = 8 回合，每个任务 4 回合 |
| 每轮策略样本 | 8 × 60 = 480 个 chunk，包含终止后的 mask padding |
| PPO update_epoch | 2；每轮最多 30 次优化器更新尝试 |
| actor / critic 学习率 | 5e-6 / 1e-4，cosine |
| 训练种子 | 从 1000 开始按任务推进 |
| 评测种子 | 两个任务各 0–9，每次周期评测重复同一组种子 |
| 保存 / 评测间隔 | 每 5 轮 PPO，并在最后一轮执行 |
| 试跑长度 | 20 轮，共 160 个训练回合，每个任务 80 回合 |
| 视频 / 演示导出 | 训练和周期评测均关闭 |

这里的 `--max-steps 20` 是完整的“采样 → 优势估计 → 参数更新”轮数，
与 SFT 的 optimizer step 不同。保存的默认轮次为 5、10、15、20。
当前 runner 日志横轴从 0 开始，因此第 5 轮的 TensorBoard 横轴为 4，而目录叫 `global_step_5`。

记录的分任务指标包括：

```text
eval/task/stack_bowls/success
eval/task/store_objects_in_the_drawer_0/success
```

当前配置不额外执行训练前评测。之前各 100 回合的 SFT 成功率为 35% 和 64%；
周期评测每任务仅 10 回合，适合看训练是否恶化或开始改善，不能直接作为与 100 回合结果等精度的比较。
完成试跑后应对候选检查点再用同一套 100 回合协议评测。

## 输出及修改内容

默认创建新的目录，路径会打印在终端：

```text
/root/autodl-tmp/outputs/lbench_ppo_two_tasks_3gpu_<UTC时间>/
  launch_plan.json
  resolved_config.yaml
  train.log
  tensorboard/
  lbench_ppo_two_tasks_3gpu/checkpoints/global_step_<轮数>/
    openpi_assets.json
    <asset_id>/norm_stats.json
    actor/
      openpi_assets.json
      <asset_id>/norm_stats.json
      model_state_dict/full_weights.pt
      dcp_checkpoint/
```

可用 `--output /新的/输出目录` 自定位置；已存在的目录会被拒绝，避免混入旧训练。
可用 `--checkpoint /另一个/检查点目录` 指定兼容权重，预检查仍要求 FP32、LeRobot 兼容以及完整的归一化资产。
环境路径可通过 `LBENCH_PPO_ROOT`、`LBENCH_PPO_TASK_CONFIG`、`LBENCH_PPO_ISAAC_PYTHON`、
`LBENCH_PPO_DATASET` 改写。旧 shell 的通用权重、归一化、GPU 导出不会覆盖此配方。

此次代码修复：

- OpenPI PPO 实际传递 dual clip=3 和 log-ratio 范围 [-20, 20]，避免有限的大 log-ratio 指数溢出。
- `critic/value_clip_ratio` 统计有效 mask 内裁剪前超阈值的比例。
- 同步 actor 在所有 update epoch 和指标统计结束后释放旧 rollout batch；异常路径也清理。
- PPO 与 SFT 的 FSDP 保存路径都导出实际生效的精度、预处理和归一化资产；资产写入失败会通知全部 rank。
- 加载优先使用检查点自带归一化，显式提供的外部统计若有数值差异则报错。
- 限制仿真环境数量，关闭视频/完整回合收集，并在 Python 启动前设置 glibc mmap/trim 阈值。

2026-09-10 联合验证：145 项 CPU 测试通过，Ruff 与 `git diff --check` 通过；
真实 SFT 权重预检查核实 813 个 tensor 的浮点精度为 FP32，两个打包归一化资产数值一致，八个提示词匹配。
这些改动已按 CPU 数学、生命周期、资产保存/重新加载和配置调度验证；
仍需用户启动后的实际三卡运行验证 CUDA/FSDP、权重同步和 Isaac 的完整链路。
本次修改针对同步 PPO，未扩展到 AsyncPPO 的独立更新实现。
