# LBench 对比评估：共用初始动作噪声

默认行为不变。单独评估 SFT 与 PPO 时，在两条评估命令中同时增加：

```text
+rollout.eval_initial_noise_seed=42
env.eval.total_num_envs=5
env.eval.rollout_epoch=10
env.eval.auto_reset=false
env.eval.task_indices=[6]
env.eval.seed=0
env.eval.max_episode_steps=900
env.eval.max_steps_per_rollout_epoch=900
```

组件放置必须产生 **1 个 env worker、1 个 rollout worker、1 个 pipeline stage**。
每个 env worker 内有 5 个环境子进程，10 个 epoch 共 50 回合；这不是 5 个 env worker。
例如在单节点选择一张可用 GPU：`cluster.component_placement.env,rollout=0`，
具体 Hydra 转义与现有启动器保持一致。启动前应检查已解析 placement，而不能只看总卡数。

以上命令是单 worker 的独立评测示例。本功能也支持训练中的周期评测，以及 env 与
rollout worker 数相同、环境总数可等分的同步多 worker 布局。仍要求 OpenPI、LBench
单任务、`balanced_round_robin` 和非自动重置；多 pipeline、异步路由、CUDA graph、
RLT、DSRL、RTC 会被拒绝。未配置此开关时，上述限制不适用。

LBench 当前给 rollout 的观测没有场景 seed。此实现按全局环境顺序恢复序号
`epoch * total_num_envs + global_slot`，多 worker 使用各自连续的 global_slot 范围，
与环境 `_episode_spec()` 同步。
不设 `fixed_seeds` 时，场景为 `env.eval.seed + ordinal`；设唯一的 `fixed_seeds`
列表时循环使用该列表，重复遍数作为 repeat。环境提前终止不触发新回合，因此
其他环境完成得早或晚，不会改变尚在运行环境的噪声序列。

周期评测可额外设置 `rollout.eval_repeat_initial_noise=true`，要求唯一的固定场景种子
列表恰好覆盖一次评测（`total_num_envs * rollout_epoch`）。每次评测都重用同一组噪声键。
clean 使用 16 个环境（每卡 4 个）× 2 批、种子 0–31。该选项默认 false，保留旧独立评测重复场景时
继续推进 repeat 序列的行为。详见 `LBENCH_CLEAN_FOUR_GPU_LIBERO_LONG_ZH.md`。

每个初始噪声由 `(noise_seed, task_index, scene_seed, repeat, chunk_index)` 的
BLAKE2b 哈希创建专用 CPU generator，抽取模型完整预测形状的 float32 `N(0,I)`，
再转换到模型设备/精度。它不是零噪声，也不缩小原有噪声分布。相同软件版本、
模型维度和上述键对应逐元素相同的初始噪声，不消耗全局 CPU RNG。评估采用原有
ODE 采样；`noise_level` 不是初始高斯幅度。训练路径不传入此参数，保持原有 RNG。

这固定了策略初始随机输入，并不保证不同 GPU/底层库上的仿真或浮点运算逐位一致。
也不使两个模型的状态轨迹相同：模型采取不同动作后，后续观测本应不同。
两模型须使用相同场景与任务、动作归一化、相机、精度、50 预测/20 执行、10 次去噪和
900 步上限。结果报告应记录 PyTorch 版本、开关值和每回合场景 seed。

CPU 验证覆盖真实环境任务/种子调度、50 回合映射、重复种子轮次、非零标准高斯、
独立 RNG、真实模型入口、真实 rollout 入口及 10×45 次调用的评估循环。完整 GPU
评估仍需要在实际运行中确认场景 JSONL 与预期序号一一对应。
