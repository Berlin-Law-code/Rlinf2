# Clean 单任务四卡 PPO 配置

任务为 `clean_the_dirt_with_browncloth_lighttable_part0`，在默认八任务配置中的索引为 0。
入口为 `scripts/lbench/remote/13_ppo_clean_4gpu_libero_long.sh`。
它使用独立的 `lbench_ppo_openpi_pi05_clean_4gpu_libero_long.yaml`，从当前 drawer 四卡配置复制资源与模型设置，
覆盖任务、回合长度、实验标识、KL 限制选项和周期评测。独立主 YAML 避免 Hydra 对嵌套 `hydra.searchpath` 的限制。
普通四卡入口传入 `--task clean_the_dirt_with_browncloth_lighttable_part0` 也会选择同一个 clean 配置。

默认从原始 SFT `global_step_30000` 加载模型和归一化统计，PPO 优化器重新初始化；
`runner.resume_dir` 与 `runner.ckpt_path` 均为空，不继承 drawer PPO 权重或优化器。
默认 SFT 目录是：

```text
/root/autodl-tmp/outputs/lbench_sft_lerobot_aligned_fp32_4gpu_memfix_004/lbench_sft_pi05_lerobot_aligned_4gpu/checkpoints/global_step_30000
```

| 参数 | Clean 配置 |
| --- | --- |
| GPU 分配 | 4 张共卡，每卡 actor、rollout 和 4 个 Isaac 环境 |
| 训练环境 / rollout_epoch | 16 / 4，每轮 64 条轨迹 |
| 真实回合上限 | 600 个环境动作；训练和评测后端 `max_step` 都随其解析为 600 |
| 训练采样窗口 | 640，最后 40 步为 mask 掉的填充，不多执行动作 |
| 评测窗口 | 600；训练前基线，此后每完成 10 个 PPO 外层轮次以及训练正常结束时评测 |
| 每次评测 | 16 个环境（每卡 4 个）× 2 批 = 32 条，固定场景种子 0–31 |
| 执行动作 / 去噪 | 20 / 10；模型预测 horizon 仍为 50 |
| 每卡微批 / 全局 batch | 32 / 256，每卡梯度累积 2 次 |
| actor / critic 学习率 | `1e-6` / `1e-4` |
| PPO 数据更新遍数 | 4 |
| actor / rollout offload | 均为 `true` |
| 数值精度 | 严格 FP32，关闭 AMP 和 TF32 |

每轮有 `16 × 4 × (640 / 20) = 2048` 个动作块样本槽位，包括终止后的填充。
`2048 / 256 × 4 = 32` 是每轮优化器调用上限；实际有贡献的样本受轨迹终止 mask 影响。
600 是环境动作上限，`--max-steps` 是 PPO 外层训练轮数，两者含义不同。

## 预测20、执行20的对照实验

入口支持 `--action-horizon 20`，只覆盖 `actor.model.openpi.action_horizon`。
执行长度仍为20，去噪仍为10，采样数与每轮最多32次优化器更新不变。
不传该参数时，原始50/20配置保持不变。预测长度必须不小于执行长度。

```bash
bash scripts/lbench/remote/13_ppo_clean_4gpu_libero_long.sh \
  --gpus 0,1,2,3 --action-horizon 20 \
  --max-steps 100 --lr-schedule-steps 1000 \
  --output /root/autodl-tmp/outputs/lbench_clean_20x20_NEW_RUN
```

输出目录必须未存在。这条命令从原始SFT权重开始，Adam重新初始化；先做32条
20/20的SFT基线评测，再训练。原始SFT权重结构可复用，但缩短动作序列改变注意力
上下文，因此不能把新的基线直接假定为50/20的59.375%。PPO的新旧概率两条路径
必须都采用20步预测。本次实验仅改预测长度，未同时启用dual clipping或修改学习率。

从PPO checkpoint恢复时，预测长度必须与来源run相同；不能给旧50/20 checkpoint
加此参数直接续训。20/20实验后续恢复时仍需传 `--action-horizon 20`。

## 20/20、dual clipping 和 LIBERO Critic 的完整验证

`--clip-ratio-c 3` 启用官方 dual clipping；`--value-head-hidden-sizes 1024 512 256`
显式构造 `2048→1024→512→256→1` Critic，ReLU及输出层初始化保持官方实现。
无需把 `config_name` 改成 `pi05_libero`，LBench相机、动作、归一化和状态输入契约保留。
这两个参数默认不覆盖历史配置。Critic宽度必须为正整数，dual系数必须有限且大于1。

```bash
bash scripts/lbench/remote/13_ppo_clean_4gpu_libero_long.sh \
  --gpus 0,1,2,3 --action-horizon 20 --clip-ratio-c 3 \
  --value-head-hidden-sizes 1024 512 256 \
  --max-steps 100 --lr-schedule-steps 1000 --smoke-test-steps 1 \
  --output /root/autodl-tmp/outputs/lbench_clean_c3_wide_20x20_NEW_TEST
```

该命令从原始SFT初始化Actor，并初始化新的Critic与Adam；完成32条基线评测、
64条采样、最多32次PPO更新、checkpoint保存及32条结束评测后退出。
去噪10、执行20、600真实步、16环境×4批、micro32/global256、Actor LR1e-6、
Critic LR1e-4及offload均保持原设置。单轮试跑仅验证运行与数值链路，不能证明收益。

若验证后从此checkpoint继续，需要同时传上述三个参数和 `--resume-dir`，
删除 `--smoke-test-steps 1`，选择新的输出目录。恢复检查要求Critic架构和dual系数
与来源配置一致；旧的小Critic PPO权重不能按大Critic结构直接加载。

`algorithm.ppo_update_guard.enabled: true` 保留初始 log-prob 一致性检查、数值检查和诊断。
Clean 设置 `enforce_kl_limit: false`，不因额外 KL 阈值拒绝更新或提前停止 PPO 更新；
保留 `loss_weighting: libero_trajectory`，使成功与失败不同轨迹长度按 LIBERO 轨迹规则处理。
对有效动作块，权重为 `w = 600 / L`，其中 `L` 是该轨迹实际执行的动作数。
主 KL 记录 `sum(w * k3) / sum(w)`，同时记录普通有效动作块平均的
`unweighted_kl = sum(k3) / valid_count`。KL 与损失使用相同的相对轨迹权重；
损失分母仍是包含填充的 batch 大小，两者分母不同。最后一个动作块可能只执行部分动作，
因此也不能把这个规则描述为严格的每条轨迹等权。

Clean 的诊断模式保留初始与最终固定 probe，以及 minibatch 正向计算中已有的当前 KL；
不为每个候选更新复制 actor/Adam 快照或追加候选后 probe。有限 KL 超过 `0.02`
不会导致阈值停止、回滚或最终检查失败；初始新旧 log-prob 不一致、NaN 等异常仍会停止。
另有独立的单动作块检查：有限的 `abs(new_logprob - old_logprob) > 20` 也会停止。
关闭 `enforce_kl_limit` 不会关闭这一规则；不能把 `invalid_probability` 直接解释为 NaN。
2026-09-15 的诊断修复保持这些停止条件不变，分别记录非有限对数概率差、有限越界和
无效轨迹权重计数，并记录过滤前的有限对数概率差极值。发生概率检查停止时，
`PPO_GUARD_ABORT` 日志提供具体轮次、更新遍数、小批次以及全卡汇总诊断。
训练日志中的 `pre_update_kl`、`pre_update_unweighted_kl`、`pre_update_actor_stopped`
用于核对更新前状态。

原有 drawer 配置保持其默认 KL 限制；clean 配置不会修改正在运行的 drawer 任务。
`skip_actor_on_zero_reward: false`：与 LIBERO-Long 对齐，整轮采样奖励全零时仍进行
Actor 和 Critic 更新，不额外丢弃这轮 Actor 学习信号。连续三轮零成功保存并停止的规则仍保留。

## 训练中评测

本地 `libero_10_ppo_openpi_pi05.yaml` 提供评测流程和 500 个评测环境的配置，
但 `runner.val_check_interval=-1` 默认关闭中途评测。clean 按当前单任务和硬件容量
显式设置 `val_check_interval: 10`，不照搬 500 个并发环境。

`runner.eval_before_train: true` 在首次 PPO 更新前同步 SFT 权重并评测；之后完成第
10、20、30…个 PPO 外层轮次以及正常训练终点时，先同步最新 actor，再跑同样的 32 条评测。
这里的间隔按完成的 PPO 外层轮次计数；每轮 `env.train.rollout_epoch: 4` 保持不变。
训练环境步数与评测上限均为 600，PPO 更新过程中不会混入评测轨迹。数值异常导致
Actor abort 的那一轮跳过评测，按原有停止流程保存诊断。恢复训练时，训练前评测衡量
恢复的当前策略，不会被标记成原始 SFT baseline。

每次采用种子 0–31；`rollout.eval_initial_noise_seed: 42` 和
`eval_repeat_initial_noise: true` 为相同场景、相同动作块生成相同初始高斯噪声。
评测使用模型 `eval` 的 ODE 推理，不增加训练时的 SDE 探索噪声。私有噪声生成器不会
消耗训练的全局 RNG。固定噪声用于减少模型间比较的随机波动，不是把初始噪声设为零。

为避免共卡 OOM，两个 Isaac 池分阶段驻留：

- `train.lazy_start: true`、`train.offload_before_eval: true`：训练时每卡 4 个环境，
  到评测边界关闭训练池；普通训练轮之间不关闭。
- `eval.lazy_start: true`、`eval.enable_offload: true`：评测时每卡 4 个环境，2 批
  连续完成后关闭。下一次训练 reset 重建训练池，并保留训练任务/种子游标。
- 训练与评测阶段各有 16 个环境进程，不会同时驻留 32 个。切换池会产生 Isaac
  重启时间，加上 32 条评测本身的推理和仿真耗时，构成每次切换的实际开销；
  具体耗时和 GPU 峰值仍需真实训练验证。

评测间隔由每 5 轮改为每 10 轮后，周期评测频率约减半，每轮训练分摊的评测与环境
重启开销相应减少；SFT 基线和正常终点评测仍保留。这不代表训练总速度变为两倍，
每轮采样和 PPO 更新的耗时没有因此减半。

检查点仍每轮保存，最多保留最近 1 个；周期评测每 10 轮，两个间隔独立。先保存本轮
checkpoint 再评测，因此评测异常不会丢失刚完成的一轮更新。TensorBoard
查看 `eval/success_once`（成功率）、`eval/episode_len`（回合长度）和
`eval/num_trajectories`（应为 32）。评测横轴表示完成的 PPO 轮数：基线为 0，
之后为 10、20…，正常终点另行评测，与对应 checkpoint 的 global_step 一致。

训练前评测结束后立即消费计时记录，并独立写入 `time/eval_before_train`；
周期与终点评测仍记录为 `time/eval`。这样一轮验证中的基线和终点评测可以连续完成，
首轮训练的耗时指标也不会混入训练前评测。CPU 回归测试使用真实 `ScopedTimer`，
覆盖基线、首轮终点评测、后续周期评测和各阶段耗时隔离。

输出目录下的 `eval_history.jsonl` 逐次记录轮数、是否初始基线和指标；
`eval_episodes/rank_*.jsonl` 保留每条评测结果、场景种子和唯一回合编号。相同种子按
周期复用，但回合编号不重置。视频默认关闭，避免频繁编码带来的内存和磁盘开销。
这些指标沿用当前 clean 成功判定；本轮没有修正 XY 擦拭判定的高度/接触缺口。

先运行只读预检查；它不会创建 Ray、Isaac 或训练进程，也不代表硬件验证完成：

```bash
cd /root/autodl-tmp/Berlin/RLinf
bash scripts/lbench/remote/13_ppo_clean_4gpu_libero_long.sh --dry-run
```

GPU 空闲后，使用新的输出目录验证一整轮；本命令是准备好的入口，不会自动执行：

```bash
bash scripts/lbench/remote/13_ppo_clean_4gpu_libero_long.sh \
  --max-steps 1000 --smoke-test-steps 1 \
  --output /root/autodl-tmp/outputs/lbench_clean_validation/run
```

一轮验证包括 32 条 SFT 基线评测、1 轮完整 PPO 和 32 条更新后评测；保留 1000 轮的
学习率调度，只把实际轮数限制为 1。需要正式训练时去掉
`--smoke-test-steps 1`，并使用另一个全新的输出目录。包装脚本提供 clean 默认任务；
显式 `--task`、`--checkpoint`、`--max-steps` 等参数继续生效，`--help` 显示参数说明。

CPU 测试覆盖 clean/drawer 配置隔离、600/640 长度与后端上限、默认 SFT、32 次更新上限、
关闭 KL 强制限制但保留检查、周期/基线评测、分阶段环境生命周期、固定场景和噪声，
以及包装脚本的参数透传。这些检查不宣称 clean GPU 训练已通过或成功率已提高。


## 从已完成的 PPO 检查点恢复

`--resume-dir` 接受带有 `.checkpoint_complete` 的 `global_step_<N>` 目录。
它恢复模型、Adam、学习率调度器和 RNG，训练环境按已完成轮数重建采样游标。
外部仿真器不做完整状态快照，因此不承诺中断前后的轨迹逐位一致。
`--checkpoint` 仍指定原始 SFT 初始化及归一化资产，随后由 PPO 检查点覆盖训练状态。
源运行目录必须保留 `launch_plan.json`；恢复时校验任务、布局、模型、优化器与采样配置。
结构预检不能代替权重数值审计或实际恢复验证。

恢复始终写入新的 `--output`，不会覆盖或裁剪源检查点。`--max-steps` 是累计轮数上限；
恢复时 `--smoke-test-steps 1` 表示再执行一轮，例如从42到43，而不是把总上限设成1。
学习率衰减周期默认继承源运行；显式 `--lr-schedule-steps` 必须与源运行相同。
从 SFT 新训练时，也可以用 `--max-steps 100 --lr-schedule-steps 1000` 分开设置运行预算和衰减周期。

以第42轮恢复一整轮为例：

```bash
cd /root/autodl-tmp/Berlin/RLinf
CLEAN_RESUME="/root/autodl-tmp/outputs/lbench_clean_ppo_20260914_232854/lbench_ppo_clean_4gpu_libero_long/checkpoints/global_step_42"
CLEAN_RECOVERY="/root/autodl-tmp/outputs/lbench_clean_resume_$(date +%Y%m%d_%H%M%S)"
bash scripts/lbench/remote/13_ppo_clean_4gpu_libero_long.sh \
  --gpus 0,1,2,3 --resume-dir "$CLEAN_RESUME" \
  --max-steps 100 --lr-schedule-steps 1000 --smoke-test-steps 1 \
  --output "$CLEAN_RECOVERY"
```

追加 `--dry-run` 只检查配置，不创建输出目录或启动训练。验证通过后，去掉
`--smoke-test-steps 1` 即按累计上限继续；若验证产生了更新的检查点，应将
`--resume-dir` 指向该新检查点，避免重新执行已完成的优化器更新。
恢复前评测记录在恢复轮数（如42）且 `baseline=false`；验证完成后的评测记录在43。
保护停止跳过终点评测，具体原因记录在 `stop_reason.json` 和 `PPO_GUARD_ABORT`。
不要仅根据进程退出码或 screen 窗口消失判断是否完成了目标轮数。
