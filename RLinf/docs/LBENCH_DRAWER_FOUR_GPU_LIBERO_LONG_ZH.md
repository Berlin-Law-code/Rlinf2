# LBench 抽屉任务：当前四卡方案与 LIBERO-Long 参数对照

<!-- CURRENT_FORMAL_TRAINING -->
用户已于 2026-09-13 20:40 UTC 启动正式训练，目录 `/root/autodl-tmp/outputs/lbench_ppo_formal_20260914_044009`，screen `lbench_train`。配置与下方通过验证的方案一致，外层上限为 1000 轮，从原始 SFT 开始。首轮已完成并保存 `global_step_1`，成功 31/64，Actor/Critic 实际更新 2/48 次、1 次 Actor 候选回滚，最终 KL=0.017818；第二轮已正常开始。首轮更新阶段约 400.7 秒，全部轮内流程约 1211.4 秒（不含初始进程启动）。这些训练采样指标不能直接证明独立评测成功率提升。

独立后台 screen `lbench_monitor` 已运行，15 秒记录资源/日志、60 秒记录聚合指标；只观察，不改算法参数。异常保存在该目录 `monitor/alerts.jsonl`，最新状态见 `monitor/status.json`，首次人工检查见 `monitor/first_round_review.json`。仅在原训练 launcher 已退出并等待 90 秒后，自动清理其已确认、同出生标识的残留后代；不自动重启，也不关闭 KL 保护。监控是持久检测与留证工具，不能自动分析修复新的代码故障。下方“未开启正式长训”描述的是此前单轮验证结束时的状态。
<!-- END_CURRENT_FORMAL_TRAINING -->

<!-- CURRENT_FULL_VALIDATION -->
2026-09-13 19:57:12–20:24:47 UTC 的完整一轮验证已通过：从原始 SFT30000 开始，使用下述 micro32 / 16 环境 / global256 / Actor LR 1e-6 参数，完成 64 条采样、四卡 PPO 更新和检查点保存。总耗时 27 分 35 秒；采样 13 分 26 秒（含首次场景编译），PPO 阶段 6 分 46 秒。3 秒 NVML 采样到的显存最大值为 67.85 GiB，主机内存峰值 260.37 GiB，未发生 OOM 或新增线程触限。

实际保留 **2 次 Actor 更新、48 次 Critic 更新**；第 3 个 Actor 候选因 KL=0.022034 超过 0.02 被回滚，后续 45 个 minibatch 仅更新 Critic。最终独立 probe KL=0.014839，guard abort=0。不能把本轮称为 Actor 完整更新四遍。训练采样成功 33/64（51.56%），来自更新前 SFT 的探索策略，不能据此宣称 PPO 提升。

检查点已通过 821 个模型张量与 651 个 Adam 状态的 CPU 核验：均为有限 FP32，DCP 与完整权重一致，冻结 VLM 未变化，208 个实际动作策略张量已改变，Actor/Critic Adam 步数分别为 2/48。未使用的语言输出头保持零步与零 moment。未进行新的分布式恢复训练或更新后独立评测。

当前证据目录为 `/root/autodl-tmp/outputs/lbench_full_validation_global256_micro32_20260913_195303`，详见 `VALIDATION_REPORT_ZH.md`、`validation_summary.json`、`checkpoint_validation.json`。模型位于该目录 `run/lbench_ppo_drawer_4gpu_libero_long/checkpoints/global_step_1`。验证 screen 已退出，四卡显存已释放，**未开启正式长训**。训练期间算法参数与源代码保持不变。

下文早期校准及停止记录是历史信息；相应临时输出目录在本轮开始前已被外部清理，不应视为当前仍可读取的证据。
<!-- END_CURRENT_FULL_VALIDATION -->

当前配置为 `examples/embodiment/config/lbench_ppo_openpi_pi05_drawer_4gpu_libero_long.yaml`。按用户最新选择，保留四卡共卡，改为每卡四个 Isaac 环境、合计 16 个；每卡微批 32、每卡累积 2 次，全局批大小 256，Actor/Critic 学习率为 1e-6/1e-4。去噪 10 次、每次执行 20 个动作、每轮采样 4 批回合、每批数据更新 4 遍均保留。它从原始 SFT 权重新建 PPO 优化器。当前 Actor 学习率、global batch 和微批均与官方 LIBERO-Long 有明确差异。

当前方案已按用户选择恢复为上一次 micro32 / 16 环境 / rollout4 配置。本次恢复记录目录为 `/root/autodl-tmp/outputs/lbench_restore_global256_env16_micro32_20260913_194606`。对应的单 GPU 校准记录位于 `/root/autodl-tmp/outputs/lbench_global256_env16_micro32_20260913_192956`：当时完成两次真实 Adam 更新及卸载，NVML 峰值 37.530 GiB，52 项 CPU 测试通过。恢复时只核对配置并执行启动预检；随后已完成顶部所述完整四卡验证。共卡运行、跨 rank 更新和保存已通过，训练效果仍未由独立评测证明。

旧四卡 micro4 / global1024 / 32 环境测试目录 `/root/autodl-tmp/outputs/lbench_4gpu_micro4_threadfix_20260913_191807` 保存当时的 `preflight.json`、`rlinf_tests.log` 和 `run/launch_plan.json`：51 项 RLinf 配置与客户端测试、18 项环境分批启动测试通过；外层上限仅 1 轮，学习率调度仍为 1000 轮。这些是旧参数的记录。更早的 `/root/autodl-tmp/outputs/lbench_single_task_batch1024_20260913_183804/preflight_4gpu.json` 是当时只有三张 GPU、micro128 的历史 dry-run，外层上限为 10 轮但未执行。上述临时产物后来已被清理，数值为此前记录。

官方参照固定为 RLinf commit `25429955693013b5e547c54418917dc47681d434` 的 `examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml`，以及它继承的 model、env、FSDP、weight-sync 配置。2026-09-14 复核本地五份 YAML 与此前保存的官方文件哈希相同。这里对齐的是官方 YAML 的 1000 轮设置，不能与论文表格的 400 轮混为一谈。

上一轮 micro4 / 32 环境测试于 2026-09-13 19:23:00 UTC 按用户要求停止，阶段为环境初始化，尚未采样、执行 PPO 更新或生成新检查点。停止后四卡显存均已归零、`lbench_4gpu_validation` screen 和全部跟踪子进程已退出；汇总见该旧测试目录 `validation_summary.json`。本次修改参数及隔离校准不会自动重启完整训练，也没有开启正式长训。

## 已对齐参数

| 项目 | 官方 LIBERO-Long | 新四卡配置 |
|---|---:|---:|
| Critic 学习率 | 1e-4 | 1e-4 |
| 每批数据更新遍数 | 4 | 4 |
| Actor seed | 42 | 42 |
| Adam beta1 / beta2 / epsilon | 0.9 / 0.95 / 1e-8 | 相同 |
| weight decay / 梯度裁剪阈值 | 0.01 / 1.0 | 相同 |
| Critic 单独预热更新数 | 0 | 0 |
| LR scheduler / 总调度轮数 / 最低比例 | cosine / 1000 / 0.1 | 相同 |
| 配置外层轮数上限 | 1000 | 1000；先短测，再依据第 0/5/10 轮评估决定是否继续 |
| PPO / advantage | actor_critic / GAE | 相同 |
| 优势标准化 | true | true |
| reward / logprob 粒度 | chunk_level / chunk_level | 相同 |
| entropy 粒度 / 系数 | token_level / 0 | 相同 |
| gamma / gae_lambda | 0.99 / 0.95 | 相同 |
| clip low / high | 0.2 / 0.2 | 相同 |
| 实际 dual clip / log-ratio loss clamp | 未启用 | 未启用，三个选项显式 null，见下文 |
| value clip / Huber delta | 0.2 / 10 | 相同 |
| KL penalty 类型 / KL beta | kl / 0 | 相同 |
| reward_coef / group_size | 1 / 1 | 相同 |
| reward filtering | false | false；0.1/0.9 筛选界限均不启用 |
| 训练奖励 | 首次成功 +1，其他 0 | 相同 |
| 训练 auto_reset / ignore_terminations | false / false | 相同 |
| 探索 | flow_sde，noise_level=0.5 | 相同 |
| 训练参数范围 | 冻结 VLM，仅训练 expert/projections 和 value head | 相同 |
| value head / value_after_vlm / 聚合 | true / true / mean_token | 相同 |
| LoRA / DSRL / 独立奖励模型 / 独立 critic 模型 | 关闭 | 相同 |
| FSDP / 分片 / activation checkpointing | fsdp / no_shard / false | 相同 |
| AMP / GradScaler | false / false | 相同；这不代表模型权重精度相同 |
| rollout pipeline stage / 重算 rollout logprob | 1 / false | 相同 |
| 自动在线评估 | val_check_interval=-1 | 相同 |

噪声退火、joint_logprob、double_layer、ignore_last 均沿用模型默认关闭，未新增探索调度。复制数值不会消除机器人动作单位、场景随机化和任务难度差异。

## 明确保留的差异

| 项目 | 官方 | 四卡 LBench / 原因 |
|---|---|---|
| Actor 学习率 | 5e-6 | 1e-6，按用户最新选择降低；Critic 仍为 1e-4 |
| 全局批大小 | 2048 | 256，按用户最新选择调整 |
| 微批大小（每卡一次 forward/backward） | 128 | 32，完整四卡共卡一轮验证通过，正式长训未启动 |
| 训练环境数 | 64 | 16，每卡 4 个 Isaac；不能按 MuJoCo 环境成本估算 |
| 每轮 rollout_epoch | 8 | 4，缩短首次反馈周期，仍满足 global batch 整除 |
| 每轮完整回合上限 | 64×8=512 | 16×4=64，且这里只训练一个抽屉任务 |
| 任务 / 模型配置 | libero_10 / pi05_libero | LBench task 6 / pi05_lbench_joint |
| Critic 价值头隐藏层 | 1024 / 512 / 256 | 512 / 256 / 128；自定义 config_name 落入默认分支，2026-09-14 检查补录，尚未改动模型 |
| 去噪次数 / 执行动作长度 | 5 / 10 | 10 / 20，按用户指定保留 LBench 采样设置 |
| 预测动作 horizon | 10 | 50，保持原始 SFT 预测契约；执行前 20 个后重规划 |
| 图像 / 动作 | 2 路 / 7D 末端动作 | 3 路 / 8D 关节与夹爪动作，保持相机映射、prompt、归一化 |
| 实际回合上限 | 480 个物理步 | 900 个物理步，未延长到 960 |
| 固定采样窗口 | 480 | 960，为全局 batch 整除添加终止后的填充槽位 |
| 参数精度 | 默认 OpenPI：选定 BF16 权重与 FP32 投影等混合 | strict FP32，TF32/AMP 关闭 |
| FSDP use_orig_params | false | true，沿用已验证的冻结参数、优化器与保存路径 |
| Actor / rollout worker offload | false / false | true / true，给每卡 4 个 Isaac 留出显存 |
| 权重同步 | CPU snapshot patch / 128 MiB 初始化桶 | 512 MiB bucket、Actor CPU 导出，沿用修复后的共卡路径 |
| 检查点 | 每 40 轮 | 每轮保存、只保留最新 1 个，以符合当前磁盘容量 |
| 额外更新保护 | 无当前 rollback guard | 保留初始概率校验、KL 回滚、零奖励跳过及连续三轮零成功停止 |
| 训练视频 / 图像轨迹 | 官方训练视频开启 | 关闭，避免额外三相机帧缓存；另行做少量独立评估录像 |
| 评估环境池 | dormant 500 env，auto_reset / ignore_terminations=true | dormant 4 env 占位；实际独立评估一槽一个回合，不直接启动 500 个 Isaac |

`detach_critic_input=true` 相较官方 YAML 的 `null` 保留为 guard 契约；当前 `value_after_vlm=true` 且 VLM 已冻结，不能仅凭这个配置差异推断训练目标发生额外变化。DSRL 关闭时其状态维度、Q head 等占位参数不参与 PPO。

2026-09-14 的 GitHub issues 核查补充了一项此前未列出的模型结构差异：`openpi_action_model.py` 通过 `config_name` 选择价值头，`pi05_libero` 使用 `(1024, 512, 256)`，`pi05_lbench_joint` 使用默认 `(512, 256, 128)`。本次正式训练第 19 轮检查点的 8 个价值头张量已通过 CPU meta 模式核对，实际层宽确为 `2048 → 512 → 256 → 128 → 1`；未初始化 CUDA 或修改权重。这一差异与 [PR #623](https://github.com/RLinf/RLinf/pull/623) 扩大 LIBERO 价值头的历史修改相关，但当前 Critic EV 已达约 0.78，尚不能断定较小价值头造成成功率停滞。不能在运行中直接更换头或通过更改任务 config_name 来对齐，否则会改变模型/数据契约及检查点形状。完整核查记录见 `/root/autodl-tmp/outputs/lbench_ppo_formal_20260914_044009/monitor/github_issue_review_20260914_032936/ISSUE_REVIEW_ZH.md`。

官方 YAML 写了 `clip_ratio_c=3.0`，但 pinned 上游 worker 的 OpenPI 分支未把它传入损失；该损失实际使用的 `clip_ratio_c`、`clip_log_ratio_min`、`clip_log_ratio_max` 均为 None。新配置将三个选项明确设为 null，以对齐实际行为。额外 guard 仍会检查异常概率并拒绝更新；它不是损失中的 log-ratio clamp。LBench 的真实已执行子动作 mask 也必须保留，不能为了匹配文本删除终止尾部掩码。

更新保护与 loss 权重需要分开看：旧 guard 路径使用有效 chunk 平均，官方传入 `loss_mask_sum/max_episode_steps` 以补偿不同轨迹长度。新 profile 明确设置 `algorithm.ppo_update_guard.loss_weighting=libero_trajectory`，对应：

```text
mean_over_padded_slots(mask × per_chunk_loss × 900 / episode_executed_actions)
```

此模式不再施加旧 guard 的额外 `global_batch_size/global_valid_chunks` 补偿。分母中的轨迹长度来自 `action_mask.sum()`，排除终止后的填充和被控制器覆盖的动作。相同数据批次和相同 FP32 计算下，正确梯度累积可与官方大微批的该损失公式对应；终止于 chunk 中间时，有效动作长度和有效 chunk 数并非固定倍数，因此不能把它描述为严格的每轨迹等权。保留 guard 的概率检查与回滚，也意味着完整算法不是未经修改的官方实现。

例如一条成功轨迹执行 300 步，另一条失败轨迹执行 900 步，执行块长均为 20：前者 15 个有效块、每块长度权重 3，后者 45 个有效块、每块权重 1。单看长度因素，两者总权重相同；实际梯度还取决于优势值、概率比和 value 误差。该规则不读取成功标签，提前失败的短轨迹也会获得长度补偿。成功和失败轨迹都保留，不截去失败轨迹尾部，也不通过增大回合上限解决权重问题。

填充槽位虽然直接梯度为零，仍进入固定槽位均值的分母。在固定模型参数、有效数据相同且对整份 rollout 的所有槽位只求一次均值的比较中，将窗口从900补到960，均值相对未填充版本乘以900/960=0.9375，轨迹间相对长度权重不变。实际训练每次更新的分母固定为256；窗口改变还会影响打乱分组及批次数，不能把0.9375描述成每次优化器更新或整段训练的统一缩放比例。独立评估的窗口保持900。

仅修改 `loss_agg_func=token-mean` 不会切换旧 guard 的上述权重。新选项依赖配套 guard 实现，启动预检必须确认其存在；不能把新 YAML 放到旧 guard 代码上直接运行。

## 900、960 与批大小的含义

真实回合最多执行 900 个动作，以 20 动作为一次决策，共 45 个有效决策上限。固定窗口设为 960，形成 48 个决策槽；最后至少 3 个是终止后的填充，不执行物理动作、不产生奖励、不贡献有效 loss。提前成功或失败的轨迹有更多填充。

四卡每轮的计划量为：

```text
16 env × 4 rollout epochs = 64 回合
64 × 48 chunk slots = 3072 个填充后样本位置
3072 / 256 = 12 个全局 minibatch / pass
12 × 4 passes = 最多 48 次共享优化器更新 / 轮
4 actor ranks × micro32 × 2 次梯度累积 = global256
```

官方每轮是 512 回合、24576 个位置、最多 48 次更新；当前方案每轮回合数和样本位置数均为其八分之一，计划更新次数相同，每次更新的 global batch 为其八分之一。因此“训练 5 轮”不能理解成与官方 5 轮使用相同的数据量。当前真实决策上限为 45，官方为 48，保留 gamma=0.99、gae_lambda=0.95；决策长度接近不代表物理时间尺度、任务难度或信用分配完全相同。

micro32 表示每个 actor 一次处理 32 个样本位置，各自累积 2 次，再由 4 个 actor 共同完成一次 global256 更新。微批大小不等于整次优化器更新的批大小；256 也不是 256 条独立回合，而是包含 padding 的 chunk 位置数。真实有效数量以及每条轨迹贡献，取决于终止 mask 和已选 loss 权重。与上一版 global1024 / 32 环境 / micro4 相比，每轮回合数从 128 降至 64，样本位置从 6144 降至 3072，每轮计划更新次数从 24 增至 48，同时 Actor 学习率由 5e-6 降为 1e-6。更新次数增加不等于策略改进增加：本轮实测仅保留 2 次 Actor 更新、完成 48 次 Critic 更新；耗时见顶部，效果仍需独立评测。

不能直接使用 900 作为固定窗口而同时保留 16 env、4 rollout、执行 20、global 256：`16×4×45=2880` 无法整除 256，会被调度验证拒绝。这里继续使用 960 的显式填充窗口解决整除，不改变成功条件或真实超时。

## 单任务需要改变什么

只有一个抽屉任务，并不意味着应把 LIBERO 的 batch 或学习率按任务数除以 10。global batch 控制每次优化器更新聚合的数据量；样本是否独立、成功回合比例和场景覆盖，不能从任务数量直接推算。当前按用户选择采用 global256、Actor 学习率 1e-6、16 个环境和 micro32，保持每轮 rollout_epoch=4，让一轮先收集 64 条回合。这是已验证能完成一轮训练的起始选择，不是已经证实更适合单任务的最优值。

其余先保留 update_epoch=4、gamma=0.99、gae_lambda=0.95、noise_level=0.5、clip=0.2 和现有长度权重及更新保护。训练继续使用不同场景 seed，不能因任务相同就重复少量固定场景。如果新评估没有进步，再根据有效更新数、KL/回滚比例、训练与保留场景的成功率差距，决定是否单独减少更新遍数或学习率；不要同时改变多个算法参数后归因。

## 从原始 SFT 开始与验证顺序

配置使用 `LBENCH_SFT_CHECKPOINT` 与其对应 `LBENCH_NORM_STATS`，`runner.resume_dir=null`、`runner.ckpt_path=null`。设置新的 PPO 学习率和余弦调度，然后从外层第 0 轮建立新优化器。原 SFT 权重和已有 PPO 检查点均保留。

不要用旧 PPO5 的 `runner.resume_dir` 来伪装新的 SFT 起点：该路径还恢复 optimizer param groups、学习率 scheduler、计数和 RNG；旧学习率可能覆盖新 YAML，3→4 卡恢复也没有经过实测。若将来选择旧 PPO 权重作为新实验初始化，必须显式区分“只加载策略权重的新实验”和“完整断点续训”。

此前 global1024 方案的隔离校准使用真实 SFT 权重和执行 20 / 去噪 10 / 预测 50 协议，以重复的合成三相机观测完成两次真实 Actor/Critic Adam 更新、完整本地梯度累积、64 次 guard 前向和卸载。结果保留在 `/root/autodl-tmp/outputs/lbench_batch1024_micro128_validation_20260913_184915` 的 `micro128_probe_results.json`、`micro8_probe_results.json`、`micro4_probe_results.json`。这些测试只有一个实际 rank，未启动 Isaac/Ray，未验证跨 rank 全局更新、权重同步或检查点保存。下表保留当时每卡八环境的估算，不能作为新 micro32 / 每卡四环境方案的实测结果。

| 微批 | 本地累积 | 更新及随后卸载的 NVML 峰值 | 加 68 GiB Isaac 和 2 GiB 预留后的估算 |
|---|---:|---:|---:|
| 128 | 2 | 82.421 GiB | 152.421 GiB |
| 8 | 32 | 26.048 GiB | 96.048 GiB |
| 4 | 64 | 24.278 GiB | 94.278 GiB |

旧校准的初始化峰值另为 29.597 GiB。在旧每卡八环境方案下，micro4 的估算低于约 94.971 GiB 的 Torch 可见容量，扣除 2 GiB 预留后仅剩约 0.69 GiB；micro8 加模拟器后约剩 0.92 GiB，达不到当时的预留目标。上述合计是估算，不是已经测出的共卡峰值。

当前 micro32 的隔离校准记录为新目录 `/root/autodl-tmp/outputs/lbench_global256_env16_micro32_20260913_192956/micro32_probe_results.json`。真实 SFT 权重、执行 20 / 去噪 10 / 预测 50、strict FP32 下，单个实际 rank 按每次累积 2 个微批完成了两次真实 Actor/Critic Adam 更新、64 次 guard 前向与卸载，进程正常退出。两次本地更新分别约 6.73、6.87 秒，更新及随后卸载的 NVML 峰值为 37.530 GiB。观测为重复的合成三相机数据，未启动 Isaac/Ray，未执行跨四 rank 的 global256 更新、权重同步或保存检查点，不能据此推算完整轮次耗时或证明训练有效。

按旧八环境约 68 GiB 的占用折半，暂估当前每卡四个 Isaac 为 34 GiB；与实测模型峰值和 2 GiB 预留相加为 `37.530 + 34 + 2 = 73.530 GiB`，低于约 94.971 GiB 的 Torch 可见容量。34 GiB 及组合总量属于当时估算。本轮完整共卡验证已实测到每卡最高 67.85 GiB（3 秒 NVML 采样）；一轮结束后已退出，正式长训未启动。

当前为 NO_SHARD，每卡仍保留完整策略；四卡不会将每卡的模型和激活均分成四份。当前 micro32、2 次累积与四个 actor 共同形成 global256；相比旧 micro4，一次前向和反向的激活占用会增加，而每卡 Isaac 数从八个降为四个，本轮组合占用已通过完整运行验证。`runner.actor_first_initialization=true` 会先完成 Actor 初始化并卸载到 CPU，再启动 Isaac，避免两者初始化峰值重叠；它要求 `actor.enable_offload=true`，并不解决训练阶段的激活峰值。本次完整一轮已检查采样、真实更新、模型保存、显存和轮次时间；此前单 GPU 校准与本次四卡实测应分开引用。

旧四卡 micro4 / global1024 / 32 环境的首次测试目录为 `/root/autodl-tmp/outputs/lbench_4gpu_micro4_validation_20260913_190144`。当时 32 个 Isaac 同时 INIT 时出现 `can't start new thread`，容器 `pids.max=20480`、`pids.events` 记录触限，内存 OOM 计数为零，未采样或更新。单环境实测正常初始化后有 412 个线程，CPU affinity 限到八核仍为 412；`cpu_thread_limit` 不是进程总线程上限。后续分批启动重试又定位到 Ray 的 87 个空闲预启动 worker 合计占用 4524 个线程，故在采样前主动结束该次重试（目录 `lbench_4gpu_micro4_staged_20260913_191321`）。这些目录与其原始 32 环境配置均保留，不能改写成新 16 环境方案的结果。

当前四卡配置继续设置 `startup_batch_size=2`，每 rank 按两环境一批启动，全部完成后保留每卡四环境，reset/step 继续并行。`cpu_thread_limit=4` 缩小受控工作线程池；`startup_timeout_s=1200` 是每 rank 全部 INIT 等待共用预算。默认未指定分批选项的旧配置保留原启动行为。监督日志同时记录 cgroup 线程数量与触限计数。四卡 wrapper 设置 `RAY_prestart_worker_first_driver=0`、`RAY_enable_worker_prestart=0`，所需 actor 仍按需创建；可用 `LBENCH_PPO_RAY_PRESTART_WORKERS` 显式覆盖。选项语义与安装版本匹配，见 [Ray 2.58.0 配置源码](https://raw.githubusercontent.com/ray-project/ray/ray-2.58.0/src/ray/common/ray_config_def.h)。NVIDIA 对 SimulationApp 与 Carbonite/TBB 工作线程设置的说明见[性能手册](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/reference_material/sim_performance_optimization_handbook.html)。

四卡配置的 `save_interval=1`、`max_saved_checkpoints=1` 仅保留最新一次检查点。当前单个完整 PPO 检查点约 36.06 GiB，保存实现先写新文件再清理旧文件，因此轮换时两份约占 72.12 GiB，仍要为日志等保留空间。本次调整时磁盘可用约 103.34 GiB；若 keep3，瞬时四份约 144.24 GiB，不适合当前容量。此设置不删除已有实验的 SFT 或 PPO 检查点。第 5 轮检查点会被后续轮次淘汰，所以应在相应边界完成评估；如需另存多个候选权重，先安排额外空间。

效果判断先采用第 0、5、10 轮的同协议比较：第 0 轮是原始 SFT，第 5、10 轮分别累计最多 240、480 次优化器更新，实际接受次数还受 guard 影响。这是后续评测建议，当前脚本没有自动执行该评测计划。保留 cosine 的 `total_training_steps=1000`，不能为了先跑 5 或 10 轮就把调度长度缩成 5 或 10；短期评估通过也不等于已经决定训练满 1000 轮。

原 SFT 的 36/50（72%）结果使用执行 20 / 去噪 10、场景 seed 0–49、初始动作噪声 seed 42 和 strict FP32，与当前拟用评估协议的这些设置相符，复核的 10 个核心评估代码文件哈希也相同。它可以作为候选第 0 轮基线，不必因本次修改就一律重做；正式采用前仍需核对 backend、任务和 assets 版本，以及权重、归一化、相机映射、900 步上限等完整协议。若不能确认一致，应重评基线，并让 SFT 与 PPO 使用同一组场景和每回合、每个动作块的标准正态初始噪声。已有 seed 0–49 参与过模型比较；用于选择后续参数时，宜另设未参与选择的保留场景，并为它们建立对应 SFT 基线。

在第 5、10 轮检查点边界停止并回收训练环境，顺序完成独立评估，再从该检查点恢复；先验证恢复流程，避免把评估实例叠加到训练的 Isaac 池上。`val_check_interval=-1` 继续保持关闭，单纯打开在线评估不会复用现有环境。每次比较成功率、同场景的胜负变化、成功完成步数和失败类型，同时记录有效策略更新数与 KL/回滚比例；不能只凭训练成功率或单次 50 回合中的少量差异宣称提升。

`actor.log_optimizer_progress=true` 会让 actor rank 0 在每次优化器更新后输出 `PPO_UPDATE_PROGRESS` JSON，包含进度、接受或回滚结果、学习率和耗时。`candidate_post_kl` 是候选更新的 KL，`last_verified_policy_kl` 是最后保留且通过检查的策略 KL；发生回滚时不能把前者解读成当前模型的 KL。

## 已放弃的 micro64 / 32 环境参数试验

上一版试验使用 micro64、32 个环境、rollout_epoch=2、global256 和 Actor LR=1e-6，记录保留在 `/root/autodl-tmp/outputs/lbench_global256_env32_micro64_20260913_194004`。单 GPU 两次 Adam 更新与卸载峰值为 52.425 GiB；与历史每卡八环境约 68 GiB 占用相加，估算 120.425 GiB，超过约 94.971 GiB 容量。它没有启动完整共卡采样，也没有实际触发组合 OOM。用户随后选择恢复当前 micro32 / 16 环境 / rollout4；该试验数值仅作历史记录，临时目录已被清理，不作为当前方案的参数。

## 已停止的三卡历史短测：不是当前四卡方案

历史配置 `examples/embodiment/config/lbench_ppo_openpi_pi05_drawer_3gpu_libero_long_probe.yaml` 保持原样，用于复现旧短测。GPU 0、1 各有一个 actor，GPU 0、1、2 各有一个 env 与 rollout rank，每卡 8 个 Isaac、合计 24 个环境。actor world size 为 2，可使 global2048 整除；不能将该拓扑描述成三个 actor 共同更新。

旧配置为 LR 5e-6 / 1e-4、global2048、4 passes、8 rollout epochs、真实 900 步 / 窗口 960、执行 10 / 去噪 5、同一个 SFT 起点、`libero_trajectory` 权重及 strict FP32 / offload / guard。YAML 的 micro 默认仍为 1，历史启动通过 CLI 覆盖成 4；只运行一轮 `runner.max_steps=1`，余弦调度仍按 1000 轮。其规划量如下，实际未完成整轮：

```text
24 × 8 = 192 回合 / 轮
192 × 96 = 18432 个样本位置
18432 / 2048 × 4 = 最多 36 次优化器更新 / 轮
micro1：2048 / (2×1) = 1024 次累积 / 更新 / actor
micro4：2048 / (2×4) = 256 次累积 / 更新 / actor
```

该短测已验证 Actor 优先初始化、24 个环境启动、首次跨 2 actor / 3 rollout ranks 的权重同步并进入采样；未验证真实完整数据的合并、PPO 更新或保存。当前四卡方案的环境数、微批、去噪、执行长度和每轮采样量都已变化，不能据此外推当前吞吐、显存或成功率。

历史输出为 `/root/autodl-tmp/outputs/lbench_libero_alignment_20260913_172629`。它于 2026-09-13 18:09:23 UTC 按用户要求停止：日志确认完成 5/8 批采样，未进入 PPO 更新、未生成新检查点。`lbench_libero_probe` screen 与跟踪的测试进程均已退出，停止后已确认三卡显存释放；仓库中的旧配置保留；临时输出目录后来已被清理。`completion.json` 中的 `supervisor_received_signal_15` 表示主动停止，不能把退出码 1 归因为训练自行报错，也不能因该文件存在判定短测通过。

该历史临时目录已被清理，原 `show_progress.py` 命令当前不可用。当前验证报告与模型路径见文档顶部。旧 `12_ppo_drawer_3gpu_libero_long_probe.sh` 不应用来启动当前四卡方案。
