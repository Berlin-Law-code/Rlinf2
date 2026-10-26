# 三张 PRO6000：一张训练，两张采集

当前选择已切换为 [三卡共卡方案](LBENCH_DRAWER_THREE_GPU_COLLOCATED_ZH.md)。
本页保留分离方案及其历史实测；入口 `10_ppo_drawer_3gpu_separated.sh` 仍明确使用分离布局。

此分工中 GPU0 只执行 PPO 训练，GPU1/GPU2 各运行一份采样模型和
8 个 Isaac 环境，总计 16 个并行环境。共卡配置保留为备选，不是此入口的默认方案。

| 设备 | 职责 | Isaac 环境数 |
|---|---|---:|
| GPU0 | 一个 actor，执行 PPO 更新 | 0 |
| GPU1 | rollout rank 0，策略推理与环境交互 | 8 |
| GPU2 | rollout rank 1，策略推理与环境交互 | 8 |

一个 actor 向两个 rollout rank 同步同一版本权重。两张采集卡各处理自己的
环境观测，轨迹合并后交给 GPU0 更新。这仍是先采样、后更新的同步 PPO；
仅一个 actor，因此不执行多 actor 梯度同步。

## 启动入口

```bash
cd /root/autodl-tmp/Berlin/RLinf
bash scripts/lbench/remote/10_ppo_drawer_3gpu_separated.sh --dry-run
```

启动前检查不运行 Ray、Isaac 或模型训练，不创建训练输出目录。
实际前台训练使用：

```bash
bash scripts/lbench/remote/10_ppo_drawer_3gpu_separated.sh --max-steps 500
```

验证完整采样、PPO 更新和下一轮权重同步，可以先运行两轮：

```bash
bash scripts/lbench/remote/10_ppo_drawer_3gpu_separated.sh --max-steps 500 --smoke-test-steps 2
```

`--smoke-test-steps` 只缩短测试运行长度，保留 `--max-steps` 指定的学习率日程。
测试输出与之后的完整训练分别创建目录。

默认物理 GPU 为 `--gpus 0,1,2`，第一张训练，后两张采集。
采集卡编号需要升序。每次从原始 SFT step30000 建立新运行，不覆盖已有输出。
检查点路径、归一化、严格 FP32、actor/critic 学习率和 PPO 保护均沿用抽屉配置。
当前任务为 `store_objects_in_the_drawer_0`。

## 采样与更新预算

- `total_num_envs=16`：每张采集卡 8 个环境。
- `rollout_epoch=4`：每轮采集 16×4＝64 个新回合。
- 每回合最多 900 个动作步，即 45 个执行 chunk；训练和本配置的评测上限一致。
- 每轮最多 2880 个 chunk 槽位；终止后的填充不参与 loss。
- `global_batch_size=960`、`micro_batch_size=1`、`update_epoch=4`：
  每轮最多 12 次优化器更新，前 8 次成功更新仅训练 critic。
- 上限 500 轮，对应最多 32000 个新回合。
- 每 5 轮保存，保留最近 3 个完整检查点；自动评测关闭。

相比此前每张采集卡 4 个环境、采集 8 批的规划，本配置把环境并行数翻倍、
采集批次减半，保持每轮 64 个回合。之后按用户要求将每回合上限从 1200 缩短到
900；global batch 保持 960，因此每轮更新上限从 16 次降为 12 次，更新遍数仍为 4。
与更早的两卡实际配置（32 回合/轮）比较速度时，需要按样本量归一化。

### 三卡实测

运行记录：`/root/autodl-tmp/outputs/lbench_3gpu_900_smoke_20260911_192950`。
已完成一整轮 64 回合采样及 4 遍更新，随后确认两个 rollout rank 都使用
更新后的权重进入第二轮推理，再主动中断测试并释放三张卡。
第二轮未跑完，检查点保存未在本次测试中验证；日志末尾的 `KeyboardInterrupt`
来自主动停止，不是训练异常。

- 12 次 critic 更新成功，包含前 8 次预热；4 次策略更新全部接受。
- `guard/abort=0`，最终 probe KL 为 0.006994，低于 0.02 阈值。
- 首轮采样约 14.5 分钟，更新约 53.2 分钟，总计约 67.9 分钟，不含启动加载。
- 采样间隔为 5 秒的资源记录中，GPU0/1/2 显存峰值约 49.9/81.5/81.5 GiB，
  主机内存峰值约 156.7 GiB（容器限额 330 GiB）。

每卡 8 个环境在上述短测中可以容纳；该结果不证明长期训练稳定、效果改善或
比旧方案更快。完整两轮短测耗时较长。调整 `--max-steps` 会同步改变 cosine
学习率日程；使用 `--max-steps 500 --smoke-test-steps 2` 则保留正式日程。

共卡备选见 [三卡共卡说明](LBENCH_DRAWER_THREE_GPU_COLLOCATED_ZH.md)；
旧两卡入口见 [两卡说明](LBENCH_DRAWER_TWO_GPU_PPO_ZH.md)。
