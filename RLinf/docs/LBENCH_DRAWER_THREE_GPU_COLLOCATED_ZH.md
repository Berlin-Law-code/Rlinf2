# 三张 PRO6000 共卡训练与采样

当前选择此入口：`scripts/lbench/remote/09_ppo_drawer_3gpu_collocated.sh`。
三张 GPU 都参与训练和采样，对齐仓库 `libero_10_ppo_openpi_pi05.yaml`
的 GPU 职责分配。每卡运行一个 actor、一个 rollout 模型和 8 个 Isaac 环境。
三个 actor 通过 FSDP `NO_SHARD` 数据并行同步梯度，共同训练一个策略。
每轮先同步权重、采样，再共同更新，属于同步 PPO。

| 设备 | 训练 | 采样 | Isaac 并行环境 |
|---|---|---|---:|
| GPU0 | actor rank 0 | rollout rank 0 | 8 |
| GPU1 | actor rank 1 | rollout rank 1 | 8 |
| GPU2 | actor rank 2 | rollout rank 2 | 8 |

当前运行于 2026-09-12 14:02:45 UTC 以 `102466.lbench_ppo` 启动，目录为
`/root/autodl-tmp/outputs/lbench_ppo_hostcache_20260912_140245`，从有效第 5 轮
检查点继续第 6..500 轮。已加载下文 CPU 锁页缓存清理及此前轨迹临时副本修正。
前一运行 `lbench_ppo_batchmem_20260912_124222` 已连续完成第 4、5 轮，
完整检查点校验通过；此次为保存后主动部署修正，并非训练异常退出。
第 6、7 轮已连续完成更新和保存，检查点校验通过；第 8 轮已通过轨迹接收并开始更新。查看当前状态时以新目录的 `screen_session.json`、
`health_status.json`、`monitor_summary.json`（若存在）及日志为准。

## 后台监控

查看当前正式训练：`screen -r lbench_ppo`。默认显示训练日志；按 `Ctrl+A`，再按 `D` 可离开会话而继续训练。

当前 `102466.lbench_ppo` 的 `health` 窗口每 30 秒检查训练进度、错误日志、更新保护、
磁盘空间和监控进程是否仍在更新。`checkpoints` 窗口每 15 秒查找新写出的完整
检查点，校验有限 FP32 权重、DCP 元数据和 OpenPI 资源，然后释放该检查点的干净
文件缓存；检查点文件不删除。这两个进程均在 screen 内，不依赖交互式终端。

`checkpoint_monitor_status.json`、`checkpointN_validation.json`、
`checkpointN_page_cache_release.json` 记录逐轮结果。检查失败会写失败记录并纳入
`health_status.json` 的告警。这些后台脚本负责检测和文件缓存清理，不自动重启
训练或修改代码；训练异常的分析和修复由当前交互中的检查继续处理。

## 启动

```bash
cd /root/autodl-tmp/Berlin/RLinf
bash scripts/lbench/remote/09_ppo_drawer_3gpu_collocated.sh --dry-run
```

启动前检查不启动模型、Ray 或 Isaac。正式前台训练使用：

```bash
bash scripts/lbench/remote/09_ppo_drawer_3gpu_collocated.sh --max-steps 500
```

从原始 SFT step30000 建立新训练，输出目录自动生成，不复用旧运行。
默认物理 GPU 为 `0,1,2`，也支持升序的三个不同物理编号。
`--max-steps` 指定外层轮数并决定 cosine 学习率日程。
如需实际短测，可附加 `--smoke-test-steps 1`，仅运行一轮但保留 500 轮日程。

## 批量与训练轮数

| 项目 | 值 |
|---|---:|
| 并行环境总数 | 24，每卡 8 |
| rollout_epoch | 3 |
| 新回合/外层轮次 | 72 |
| 每回合步数上限 | 900 |
| 每回合最大执行 chunk | 45，每 chunk 执行 20 步 |
| chunk 槽位/轮 | 3240，终止后填充不参与 loss |
| global batch | 1080，三卡合计 |
| micro batch | 每卡 1 |
| 每次更新每卡梯度累积 | 360 |
| update_epoch | 4 |
| 每轮优化器更新上限 | 12，三卡共同执行，不再乘以 3 |
| actor / critic 学习率 | 1e-6 / 1e-4 |
| critic 预热 | 前 8 次成功优化器更新 |
| 500 轮采集预算 | 36000 回合 |

原分离方案每轮 64 回合，不能被 24 个并行环境整除。此处取较接近的
72 回合，并将 global batch 从 960 调整到 1080，保持每轮 3 组批次、
更新 4 遍、最多 12 次优化器更新。每卡每轮执行 4320 次训练小批次计算，
三卡合计 12960 次；增加 12.5% 的回合量，因此不能直接按三倍速度估算。

保留抽屉任务的三相机、8 维动作、预测 50/执行 20、10 次去噪和严格 FP32。
自动评测关闭；每轮保存，保留最近 3 个完整检查点；连续 3 轮无成功时保存并停止。

## 显存与模型切换

旧分离方案的两张采集卡，8 个环境加采样模型的实测显存峰值约 81.5 GiB。
本配置启用 actor 和 rollout 的 `enable_offload`：采样阶段将训练模型和
优化器移到 CPU，更新阶段将采样模型移到 CPU，Isaac 环境仍驻留 GPU。
这与仓库 LIBERO 配置关闭 offload 的设置不同，是为本任务的 Isaac 显存开销调整。

权重同步使用 bucket 方式，保持原 FP32，不压缩；训练侧设置
`actor.rollout_state_dict_cpu_offload=true`，在 CPU 上准备同步用的 state dict，
按 bucket 发送。bucket 大小默认 512 MiB，单个参数不拆分，实际包可超过此值。
训练参数在发送前已卸载，发送完成后释放 CPU 传输副本，下次更新前再加载训练参数。
CPU state dict 暂存仅支持 bucket 同步；原分离入口继续使用原来的 patch 同步。

### 连续更新后的权重同步

`20260912_021619_305628` 的正式运行完成两轮后，在第三轮同步的同卡 IPC
接收端分配约 1006 MiB 张量时出现 OOM。此前的一轮短测及保存验证没有覆盖
这个连续切换场景。原 CPU 暂存实现仍先加载 GPU 模型，再导出 state dict；
FSDP `NO_SHARD` 导出过程会先在 GPU 克隆参数，留下的训练模型及分配器缓存
与充分加载的 8 个 Isaac 环境、发送 bucket 和接收副本重叠。

现在启用 `rollout_state_dict_cpu_offload` 时，先卸载训练优化器和模型，再从
CPU 参数视图导出 CPU state dict，发送过程中训练模型始终留在 CPU。
首次创建模型时，卸载前通过 `FSDP.summon_full_params(writeback=false)` 完成
FSDP 的 CUDA 延迟初始化；否则首次 CPU 导出会因参数不在 GPU 上而失败。
此初始化上下文不创建完整 state dict 的克隆副本。
此路径要求 actor offload、bucket 同步、FSDP `NO_SHARD` 和
`use_orig_params=true`。每卡 8 个环境、FP32 和 PPO 参数保持不变。
同卡 IPC 接收复制仍保留，以保证接收张量不依赖发送者存储的生命周期。

CPU 暂存模式下，仅 actor rank 0 导出和打包权重；其他 actor 完成卸载后返回。
`NO_SHARD` 的 rank 0 已持有完整参数，无需其他 rank 为导出参与聚合。
这避免另两份未发送的完整 CPU 权重副本（本模型合计约 31 GiB）。

68 项同步及更新保护回归通过，覆盖初始化同步及模拟更新后模型已驻留 GPU
的再次同步，验证卸载顺序、非发送者不导出、同步版本和参数值。三卡小模型
验证冷启动后连续 4 次仅 rank 0 CPU 导出、分发和下一次训练。
`20260912_035659` 完成第一轮（采样 687.24 秒，更新 1295.05 秒）后，
第二轮同步期间 actor rank 1 异常退出，Gloo 随后报连接关闭。
该运行仍在所有 rank 上导出 CPU 副本；最后一次故障前内存采样为 313.99 GiB，
容器 OOM 计数为 0，进程退出的直接原因未从现有日志确认。
修正后的连续运行验证见下文。

保存启用 `actor.fsdp_config.checkpoint_cpu_staging=true`，仅适用于
FSDP `NO_SHARD` 且 `use_orig_params=true`：仅 rank 0 在保存期间将参数和优化器
暂存到 CPU，另外两张卡保持保存前的驻留位置。仅 rank 0 导出完整模型；
所有 rank 仍参加优化器元数据通信和 DCP 保存，仍保存完整权重、优化器、
学习率日程及随机状态，之后恢复保存前的驻留位置。其他入口保持原保存行为。
这避免了 Isaac 驻留时 FSDP 在 GPU 上临时克隆整份模型。

第一次完成训练更新的共卡测试，在保存阶段出现上述 GPU 克隆内存不足警告，
随后被监控器的主机内存保护停止，未生成可用检查点。该运行
`lbench_collocated8_speed_20260911_212555` 不用于恢复，也没有完整的轮次计时指标。
修正后的保存方式已通过三卡小模型的保存、恢复及下一次更新数值一致性检查，
覆盖保存前参数在 GPU 和已卸载到 CPU 两种情况。

此配置增加主机内存和 CPU/GPU 数据搬运需求。修正后的真实三卡结果见下节。

### 轨迹修正后的连续两轮结果

`lbench_ppo_batchmem_20260912_124222` 从真实第 3 轮恢复后，连续完成第 4、5 轮。
两轮均完成 4 遍更新，actor、critic 各接受 12 次更新，回滚和中止均为 0。
第 4 轮成功 32/72，整轮 36.28 分钟；第 5 轮成功 38/72，整轮 37.74 分钟，
其中采样 11.27 分钟、更新 24.94 分钟，最终探针 KL 为 0.014575。
第 5 轮保存于 14:00:12 UTC 写出完成标记；0.2 秒采样的保存内存峰值
327.07 GiB，保存期间 memory.high 事件未增加，OOM/oom_kill 均为 0。

两份检查点均检查了完整标记、821 个有限 FP32 权重张量、1513 项 DCP 元数据
以及 OpenPI 资源一致性。第 5 轮校验后主动停止旧进程，从该检查点继续加载锁页
缓存清理，不从 SFT 重开训练。证据为原运行目录的 `round5_training_result.json`、
`checkpoint5_validation.json` 和 `planned_hostcache_restart.json`。

### CPU 锁页缓存清理

第 5 轮监控发现，actor 在跨轮切换后的 CPU 驻留内存仍高于首次采样。
`smaps_rollup` 显示主要占用为 `/dev/zero (deleted)` 共享映射，普通匿名堆仅约
2.4 GiB；因此不能把这部分增长归为 Python 堆缓存。当前 PyTorch 2.11 的实测
确认，`to("cpu", non_blocking=True)` 会使用锁页内存，删除张量并调用
`torch.cuda.empty_cache()` 后，闲置的 host 分配仍可留在缓存中。
64 MiB 小探针中，额外调用本机 PyTorch 的 `_host_emptyCache()` 才释放该缓存。

`clear_memory` 新增显式 `clear_host_cache=True`，仅在完成设备同步之后清理
闲置锁页缓存；保留已有张量，默认调用行为不变，旧 PyTorch 缺少 API 时跳过。
FSDP 参数、梯度、优化器的卸载和重新加载，以及 actor 整轮更新的清理边界使用
该选项，不放入每个 microbatch。第 5 轮已于 14:00 UTC 完成并通过检查点校验，
14:02 UTC 从该检查点启动新进程加载此改动，再实测跨轮内存和时间影响。

60 项回归通过，包含 CUDA 异步拷贝仍未完成时的清理顺序、保留活跃锁页张量、
释放闲置缓存、FSDP 保存与恢复。另在三张 GPU 上通过两种保存前驻留状态的
小模型实测，恢复后的模型参数、优化器状态、学习率及下一次更新结果完全一致。
日志为 `/tmp/lbench_host_cache_regression.log` 和
`/tmp/lbench_hostcache_checkpoint_gpu.log`。这验证了缓存释放语义和保存路径；
真实 PPO 多轮节省量仍在观察，不能把此前进程退出的直接原因据此确定为 OOM。

新进程已进入第 6 轮首批采样，场景种子为 1360–1383，与恢复到第 5 轮对应。
两次新进程恢复后的首批采样记录中，actor rank 0/1/2 的 RSS 由
32.590/32.436/32.410 GiB 降到 25.137/24.994/24.955 GiB，合计约少 22.35 GiB。
对应不可回收及脏页内存从 222.60 降到 201.29 GiB。两次采样的轮次及种子不同，
该结果用于确认实际卸载路径的内存改善，不代表稳态或同样本的性能对照。
原始记录为新运行的 `hostcache_initial_rollout_validation.json`。
加载源检查点完成后，另用 `POSIX_FADV_DONTNEED` 释放其约 36.80 GiB 干净文件缓存，
所有检查点文件保留；这部分文件缓存释放量与上述 actor RSS 改善分开记录。

14:42 UTC，第 6 轮完整更新及检查点保存通过，开始第 7 轮采样。第 6 轮成功
39/72，actor、critic 各更新 12 次，0 回滚/中止，最终探针 KL 为 0.012093；
整轮 35.83 分钟，采样 10.77 分钟，更新 23.40 分钟。实际轨迹有效块数量与
前轮不同，这些计时不作为只改变缓存清理的严格速度对照。

第 6 轮更新时主机内存约 210.6 GiB，不可回收及脏页约 184.0 GiB；策略检查
暂存时约 225 GiB。保存期间 0.2 秒采样总内存峰值 275.52 GiB，5 秒采样的
不可回收及脏页峰值 223.06 GiB；memory.high 事件没有增加，OOM/oom_kill 均为 0。
保存后权重同步叠加文件缓存时总内存曾约 316 GiB，随后回落；新检查点经校验后，
另释放约 36.26 GiB 干净文件缓存。跨入第 7 轮后，actor 显存约 950/944/944 MiB，
RSS 约 26.20/26.06/26.02 GiB，继续观察后续轮次是否保持。证据为新运行的
`round6_training_result.json`、`checkpoint6_validation.json`、`round6_continuity_validation.json`。

第 7 轮随后完成同样的完整流程，15:18 UTC 保存，后台自动校验通过并释放
约 35.95 GiB 文件缓存；15:19 UTC 第 8 轮开始采样。两轮结果如下：

| 项目 | 第 6 轮 | 第 7 轮 |
|---|---:|---:|
| 训练采样成功回合 | 39/72 | 43/72 |
| actor / critic 更新 | 12 / 12 | 12 / 12 |
| 回滚 / 中止 | 0 / 0 | 0 / 0 |
| 整轮（含保存） | 35.83 分钟 | 36.15 分钟 |
| 更新 | 23.40 分钟 | 23.44 分钟 |
| 保存时总内存峰值（0.2 秒采样） | 275.52 GiB | 275.98 GiB |
| 保存时不可回收及脏页峰值（5 秒采样） | 223.06 GiB | 223.18 GiB |

第 8 轮采样中的 actor RSS 为 26.21/26.07/26.03 GiB，相比上一轮同阶段
每个仅增加约 8–11 MiB；未复现多 GiB 的逐轮累积。两轮都未触发 OOM，
接收轨迹及保存期间 memory.high 事件数没有增加。以上是已观测到的连续运行结果，
并非 500 轮稳定性保证。原始结果保存在 `round7_training_result.json` 和
`round7_continuity_validation.json`。

15:33 UTC，第 8 轮也通过轨迹接收和更新前检查，三卡已接受第一次策略更新。
本轮有 2134 个有效动作块、38 个正奖励；接收窗口内存峰值 286.79 GiB，
更新阶段约 211.32 GiB，不可回收及脏页约 184.39 GiB，与前两轮同阶段基本一致。
此窗口 memory.high 事件无新增，OOM/oom_kill 均为 0。当前源文件哈希与启动时一致。
验证汇总为 `hostcache_validation_summary.json`，逐项数据为
`round8_trajectory_validation.json`。临时 0.2 秒诊断采样已结束，常规资源、健康及
检查点监控继续运行；第 8 轮尚未完成，训练保持 500 轮计划。


### 卸载后残留的前向参数视图

`20260912_043719` 后来完成了 4 轮，但在 2026-09-12 07:01:23 UTC 的
第 5 轮采样中再次 CUDA OOM。出错位置是 rollout 的 Gemma 前向，申请
484 MiB 失败；同卡 actor 已卸载参数，但仍占约 15.52 GiB 显存。
该运行随异常退出，原先每 5 轮保存的配置尚未生成检查点，不能恢复这 4 轮进度。

本地 PyTorch 的 FSDP `NO_SHARD` 在重绑原始 Parameter 时提前返回，没有清空
`flat_param._tensors` 中保存的前向视图。最后一次无梯度 PPO 探针之后，
这些视图仍引用旧 GPU 参数存储；仅把 Parameter 移到 CPU 和清理分配器缓存
不能释放它们。现在在完成前向/反向、切换到 CPU 的边界释放这些视图，
下一次前向由 FSDP 重建，不改变 Parameter 身份、参数值或优化器状态。

新增回归覆盖有/无激活检查点、有/无梯度卸载的 8 次连续切换，逐次与
未包装参考模型比较下一次更新结果，要求数值完全一致并检查旧视图已释放。
115 项相关回归通过；三卡 GPU 小模型的每轮卸载后张量占用由 49.38 MiB
降为 17.25 MiB，连续 3 次切换保持不变。当前配置也改为每轮保存、保留
最近 3 份，避免在下一次保存之前丢失多轮进度。

实际 OpenPI 模型随后通过三卡各 8 次更新、无梯度探针和卸载压力测试，使用
原 SFT 权重、正式模型配置和 FSDP 包装。第一轮人为保留旧缓存视图作对照，
卸载后仍有 9.01 GiB PyTorch 张量；释放视图后降为 17.25 MiB，后续 7 次
均约 17.25 MiB（仅增加一次 512 字节），保留缓存为 34 MiB。
所有前向、梯度和下一轮更新均为有限值。这个测试使用合成观测及损失，
用于验证真实模型的显存释放与重复更新，不代表真实任务训练质量或长期稳定性。

### 保存阶段的主机内存峰值

`20260912_095554` 完成首轮更新，于 2026-09-12 10:31:39 UTC 开始保存，
但约 8 秒后 actor rank 0 退出，尚未写出完整检查点。退出前各 actor 显存
已降至 914/906/906 MiB，说明前向视图清理生效。主机内存快速升至至少
326.27 GiB，`memory.high` 事件增加；cgroup OOM 与 oom_kill 均为 0，
现有日志不能确定进程被终止的最终来源。

旧保存流程会将三个 actor 的模型和优化器同时卸载到 CPU。PyTorch 的
`NO_SHARD` state_dict 钩子又临时替换 `FullStateDictConfig`，丢失
`rank0_only` 设置，导致每个 rank 都先克隆完整模型，再由 DCP 丢弃非零 rank
的副本。现在只暂存写入 rank，跳过另外两张卡的模型导出；优化器名称映射
所需的通信仍由所有 rank 参加。三卡小模型已验证保存、恢复及下一次更新
完全一致，并通过钩子断言非写入 rank 没有执行模型导出。
同一份模型导出同时供 DCP 和 `full_weights.pt` 使用，避免再次克隆模型。
新增三进程 Gloo 回归覆盖模型两种驻留状态、完整权重导出次数、恢复后的
参数和优化器状态及下一次更新；实际 OpenPI 三卡也验证了保存、恢复和
下一次更新完全一致（820 个参数张量）。这些模型测试使用合成观测与损失，
其检查点仅用于测试，不用于恢复正式 PPO。

## 从有效 PPO 检查点恢复的当前运行

2026-09-12 10:48:30 UTC 启动 `802884.lbench_ppo`，输出目录为
`/root/autodl-tmp/outputs/lbench_ppo_resume_rank0save_20260912_104830`。
从 `lbench_collocated8_speed_20260911_220705` 留下的真实 PPO
`global_step_1` 完整检查点继续第 2..500 轮。该检查点的模型、环境、算法
及 500 轮学习率日程与当前配置一致；原短测仅把运行上限设为 1。
已再次核验完成标记、DCP 元数据、821 个有限 FP32 权重张量及归一化资源。
当前每轮保存、保留最近 3 份。2026-09-12 11:28 UTC 已完成第 2 轮，
新 `global_step_2` 的完成标记、821 个有限 FP32 权重张量、1513 项 DCP
元数据及归一化资源均通过检查。此检查不包含在新进程中恢复该份优化器；
保存/恢复的下一次更新一致性另由上述三卡实际模型测试覆盖。
本轮 72 条轨迹中成功 45 条（62.5%），actor 接受 8 次更新、critic 更新
12 次；1 次候选策略更新因 KL 超限回滚，最终 KL 为 0.01752，
无非法数值或训练中止。采样 669.00 秒、更新 1398.77 秒、含保存整轮
2157.28 秒（35.95 分钟），当前已进入第 3 轮采样。
保存期间以 0.2 秒间隔记录的 cgroup 总内存峰值为 329.90 GiB；扣除
`inactive_file` 后的峰值为 307.62 GiB。出现 `memory.high` 回收压力，
但 `max`、`oom`、`oom_kill` 计数均保持 0；不能把总内存峰值全部视为
不可回收内存，也不能据一轮成功断言后续长期稳定。采样时三个 actor
显存为 950/944/944 MiB。详见运行目录的 `checkpoint_validation.json`、
`continuity_scalars.json` 及 `checkpoint_memory_summary.json`。
11:32 UTC 三卡均完成第 3 轮首批采样，下一轮权重同步后的 actor 显存仍为
950/944/944 MiB；恢复、更新、检查点保存及再次采样的验证通过，结果写入
`end_to_end_validation.json`，训练继续运行。
同一 screen 的 `health` 窗口每 30 秒检查训练退出、日志异常、非法指标、
PPO 停止标志、轮次停滞及检查点磁盘余量；`health_status.json` 为最新结果，
`health_history.jsonl` 和 `health_alerts.jsonl` 保留历史及异常。此监控只读，
不会自动重启训练或绕过 PPO 保护停止。

12:04 UTC 第 3 轮也完成更新及保存：72 条轨迹成功 37 条（51.39%），
actor/critic 均更新 12 次，无候选回滚、非法数值或中止；最终 KL 0.01642。
采样 636.11 秒、更新 1448.86 秒、含保存整轮 2181.96 秒（36.37 分钟）。
新 `global_step_3` 再次通过完成标记、有限 FP32 权重、DCP 元数据及资源
校验。保存总内存峰值 329.85 GiB，5 秒资源采样中的受保护内存峰值
293.57 GiB（anon、shmem、不可回收 slab、内核栈、页表及脏页/回写量之和）；
`max`、`oom`、`oom_kill` 均为 0。第 4 轮三卡已完成首批采样，actor 显存
仍为 950/944/944 MiB；详见 `round3_continuity_validation.json`、
`checkpoint3_validation.json`、`checkpoint3_memory_summary.json`。

12:04:53 UTC 后旧资源监控停止刷新，监控进程当时等待终端输出；
排查时尝试关闭终端流控，约 12:07 UTC 旧 screen 和监控进程退出，
screen 退出的最终原因未确定。
训练独立进程组的主进程 802888 及全部 actor/rollout/env 保持运行，
没有重新启动训练。12:09:30 UTC 新建 `948574.lbench_ppo` 接管查看与监控：
`logs` 窗口显示训练日志，`resources`、`health` 窗口的监控进程直接写文件，
避免日志终端阻塞资源采样。采集恢复且异常解除，记录于
`monitoring_recovery.json` 和 `health_alert_resolutions.jsonl`。
当前查看命令为 `screen -r 948574.lbench_ppo`，日志及检查点仍在同一输出目录。
接管后的资源监控按 PID 启动时间核对原训练进程身份；它不能获得非子进程
退出码，也不会替训练进程发送停止信号。
12:12 UTC 第 4 轮三卡均进入最后一批采样（进程内 episode=8），新的监控
持续刷新且无当前异常，见 `monitoring_current_status.json`。

### 第 4 轮接收轨迹失败及临时副本修正

上述运行于 12:16:10 UTC 在 `recv_rollout_trajectories` 中失去 actor 0，
随后通信关闭，主程序以 255 退出。第 2、3 轮检查点保留；该 screen 的
日志窗口存在不能证明训练仍在进行。接收阶段受保护内存从约 250 GiB
快速升至约 299 GiB；没有 Python/CUDA OOM 报错，cgroup `oom`/`oom_kill`
仍为 0，无法由现有证据确认进程被终止的最终来源。

接收代码存在可确认的整批临时副本：共卡布局中每个 actor 只接收一份
轨迹，原实现仍调用 `torch.cat` 克隆全部字段，然后重排各 epoch 时再次
复制，并持续持有源轨迹。现同步 actor 对单份轨迹复用存储、先释放接收
引用，再逐字段取出和重排；训练打乱数据时也逐字段释放输入。
其他调用者默认保留原复制及不修改输入的行为，多份轨迹仍按原顺序拼接。
测试覆盖字段生命周期、bootstrap 步、多份轨迹、优势前重排及打乱后的
精确值与类型。480×640 三相机、每卡 8 环境、135 个动作块的 CPU 模拟
轨迹有效载荷为 2.85 GiB，进程峰值 RSS 从 9.27 GiB 降至 4.49 GiB。
此结果证明临时副本减少，不能单独证明真实训练退出问题已彻底解决。
下一次正式运行从已校验的 `global_step_3` 继续第 4..500 轮，保持原训练
配置；资源监控直接写文件，并额外记录各进程 RSS 以继续诊断。

`20260912_124222` 已实际通过第 4 轮轨迹接收和更新：接收与进入更新的
0.2 秒采样窗口内，总内存峰值 305.36 GiB，`memory.high` 未新增，
无 OOM。72 条轨迹成功 32 条，1969 个有效动作块；actor/critic 均更新
12 次，无候选回滚、非法数值或中止，最终 KL 0.01129。采样 682.63 秒、
更新 1404.61 秒、含保存整轮 2176.93 秒（36.28 分钟）。新
`global_step_4` 的 821 个有限 FP32 权重张量、DCP 元数据和资源校验通过。
保存总内存峰值 329.64 GiB，5 秒采样的受保护内存峰值 286.00 GiB；
发生文件缓存回收压力，`oom`/`oom_kill` 保持 0。详见
`trajectory_reception_validation.json`、`checkpoint4_validation.json` 和
`checkpoint4_memory_summary.json`。
13:25 UTC 对已完成检查点执行 fsync 及 `POSIX_FADV_DONTNEED`，保留全部
文件，释放约 20.59 GiB 可回收文件缓存；记录于
`checkpoint4_page_cache_release.json`。此单轮通过不代表全部 500 轮稳定。

恢复使用 `/tmp/lbench_resume_existing_ppo.sh --layout collocated_3gpu --max-steps 500`，
包装器核对配置兼容性，并将新日志和检查点写入新目录。该脚本及完整启动
计划均保存在此次输出目录内；常规 `09` 入口仍表示从 SFT 新建训练。

## 前向视图修复后的运行记录（首轮保存失败）

2026-09-12 09:55:54 UTC 在独立 screen `718367.lbench_ppo` 启动正式 500 轮训练。
入口和上表参数保持不变，检查点改为每轮保存；从原 SFT 权重重新开始。
使用 `screen -r lbench_ppo` 查看，按 `Ctrl+A` 再按 `D` 离开并保留后台训练。
当前运行目录可读取 `/tmp/current_lbench_sync_transition.txt`；其中
`screen_session.json` 保存启动身份，`latest.json` 及 `resources.jsonl` 包括每个
进程的显存，`monitor_summary.json` 若存在则表示进程已经退出，应查看退出原因。
该 screen 已在首轮保存失败后退出，没有可恢复的本次 PPO 检查点。
不能以这份启动记录作为仍在训练的证据；保存流程修正后的运行需另行验证。

## 历史连续运行记录（随后第 5 轮失败）

运行目录：`/root/autodl-tmp/outputs/lbench_ppo_verified_20260912_043719`。
2026-09-12 05:54:15 UTC 确认连续两轮完整更新结束，三张卡都完成第三轮
第一批采样并进入 `episode=7` 重置。没有重现第三轮 IPC 接收 OOM，
两轮均无 worker 异常退出、更新中止、无效概率或策略更新回滚。

| 项目 | 第一轮 | 第二轮 |
|---|---:|---:|
| 采样回合 | 72 | 72 |
| 权重同步 | 8.85 秒 | 8.94 秒 |
| 采样 | 11.37 分钟 | 10.95 分钟 |
| 更新 4 遍 | 22.17 分钟 | 25.00 分钟 |
| 整轮 | 33.69 分钟 | 36.10 分钟 |
| actor / critic 接受更新 | 4 / 12 | 12 / 12 |
| 最终探针 KL | 0.00983 | 0.00980 |

验证期间 GPU 显存峰值为 93.06、92.96、94.27 GiB；主机内存峰值
315.40 GiB（限额 330 GiB，包含文件缓存），受监控的不可回收及脏页内存
峰值 288.32 GiB。较充分的一轮切换后显存占用高于此前单轮短测，
以上验证不能代表全部 500 轮的长期稳定性。

当时后台检查完成后，同一训练继续执行 500 轮日程，screen 会话曾更名为
`lbench_ppo`；它已在后续第 5 轮失败时退出，不能把这份历史记录视为仍在运行。
运行目录的 `RUN_STATUS_ZH.md`、`validation_summary.json`、
`continuity_validation.json` 保存结果及第三轮各 rank 的采样记录；
`launcher.log` 保存该运行日志。该运行最终未到第 5 轮检查点保存；
此前完整模型保存检查和三卡小模型恢复后继续更新的验证仍单独保留。

## 三卡实测结果

运行目录：`/root/autodl-tmp/outputs/lbench_collocated8_speed_20260911_220705`。
使用正式的 500 轮学习率日程，加 `--smoke-test-steps 1` 完成一轮并保存。
测试正常退出（返回码 0），完成标记、DCP 训练状态、完整模型权重及 OpenPI
资源文件均已写出，所有测试进程已退出，三张 GPU 已释放。

| 项目 | 原方案：一张更新、两张采集 | 当前：三张共卡 |
|---|---:|---:|
| 每卡采集环境 | 8（两张采集卡） | 8（三张卡） |
| 每轮新回合 | 64 | 72 |
| 采集 | 14.52 分钟 | 10.68 分钟 |
| 更新同一批轨迹 4 遍 | 53.25 分钟 | 21.47 分钟 |
| 权重同步 | 5.14 秒 | 14.32 秒 |
| 整轮，不含检查点保存 | 67.85 分钟 | 32.39 分钟 |
| 保存及其他轮末开销 | 对照首轮未保存 | 97.31 秒 |
| 含保存的整轮 | — | 34.01 分钟 |

当前更新耗时降至对照的约 40.3%，对应 2.48 倍速度；整轮核心流程约
2.10 倍速度。按回合数归一化的吞吐量约为原来的 2.36 倍。
这些是各方案一次首轮观测，不是稳态或仅改变 GPU 数量的严格对照：
当前采集回合更多、实际轨迹长度不同，也调整了 CPU 线程限制、offload、
global batch 和权重同步方式。首轮包含 critic 预热。

当前 critic 成功更新 12 次，actor 接受 3 次更新；最后一组触发更新前的
KL 停止保护，因此没有进行第 4 次 actor 更新。无回滚、无异常中止，
最终探针 KL 为 0.01389。原对照 actor 接受 4 次更新，两边均完成 4 遍训练计算。
本次速度测试不作为训练质量改善或 500 轮长期稳定性的证据。

每 5 秒采样的 GPU 显存峰值分别为 85.15、85.23、85.06 GiB。
主机 cgroup 内存峰值 328.00 GiB（限额 330 GiB，包含共享内存和文件缓存），
保存时余量较小；更新期间约 285 GiB。CPU 保存修正后没有 CUDA 克隆内存不足警告。
监控保护计入匿名内存、共享内存、不可回收内核内存及脏页，排除可回收的干净
文件缓存，避免检查点写入的缓存增长导致误停。

原始计时与更新指标位于 `tensorboard_scalars.json`，对比及清理结果位于
`validation_summary.json`，资源记录位于 `resources.jsonl`。
134 项相关单元测试通过；三卡小模型验证了保存、恢复和下一次更新结果一致。

## CPU 线程限制

第一次共卡实测在首批场景重置时触及容器 20480 个任务（进程/线程）上限，
OpenBLAS 和训练通信线程报 `pthread_create failed` / `can't start new thread`。
内存 OOM 计数为 0，未产生有效的更新速度数据。

共卡 shell 现在将 OMP、MKL、OpenBLAS、NumExpr、BLIS 的启动线程数设为 1
（可通过 `LBENCH_PPO_CPU_THREADS` 调整）；每个 Isaac 的
`backend_config.simbox_eval_config.cpu_thread_limit=8` 传给 SimulationApp，
限制其 Kit/TBB/PXR 线程池。SimulationApp 也会将自己的 OpenBLAS 上限设为 8。
首次尝试将 Isaac 线程池限制到 2 时，场景重置长时间停留；因此改为 8。
此设置保持每卡 8 个环境及任务参数，不修改容器系统限额。

## 路由与更新保护

`actor,env,rollout: 0-2` 让每张卡各运行三个角色。启动器为物理设备生成
显式 rank 映射；`isaac_gpu_assignment: env_rank` 将同一个环境 rank 的
8 个 Isaac 进程固定在同一张卡，保证 worker ID 和场景种子唯一。
`pipeline_stage_num=1`。

更新保护保留 `NO_SHARD`、`use_orig_params=true`、microbatch1 和严格 FP32。
每卡最多 64 个有效 chunk 做概率探针，并补齐前向次数以保持跨卡通信顺序。
KL 和有效样本统计跨卡汇总，所有卡一致接受或撤销策略更新。
三卡合计最多 192 个有效探针，与原单卡 64 个探针的采样量不同。

CPU 回归包含三个实际 Gloo/DDP 进程，验证有效样本数不同时的梯度归一化、
critic 预热、同步接受与回滚，并与同一全局批量的单 actor 结果比较。
这些检查不替代真实 OpenPI/FSDP/NCCL/Isaac 测试。

历史分离方案见 [一张训练、两张采集](LBENCH_DRAWER_THREE_GPU_SEPARATED_ZH.md)。
