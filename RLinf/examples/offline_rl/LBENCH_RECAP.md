# LBench round1 RECAP

此配置使用本机两张 GPU，按顺序执行：计算 return → 训练价值模型 → 计算 advantage → CFG 策略训练。

## 数据与权重

- 示范数据：`/root/autodl-tmp/datasets/lbench_8tasks_all_merge_simple_v21`，1600 条轨迹。
- Rollout：`/root/autodl-tmp/datasets/round1_success_failure`，2893 条轨迹。
- 合计 4493 条轨迹、3,740,411 帧，16 个分任务数据集。
- 初始策略：`/root/autodl-tmp/outputs/lbench_all_merge_simple_sft_h50_e20_lerobot_norm_20261001_120113/lbench_all_merge_simple_pi05_h50_e20_lerobot_norm/checkpoints/global_step_30000`。
- 策略预测 50 步、执行 20 步，绝对关节目标 8 维，三路相机。

准备目录是 `/root/autodl-tmp/datasets/lbench_recap_round1`。元数据独立复制，
视频链接到原数据。Rollout parquet 复制并增加 `is_success`，取自每条
`meta/rollout_provenance.jsonl` 的 `source_episode_success`；缺失、重复、
非二值或未结束的轨迹会报错。不会根据长度推测成功率。
所有 return、advantage sidecar 都写入准备目录。

## norm_stats 的精确计算方式

参考：
`/root/autodl-tmp/datasets/lbench_8tasks_all_merge_simple_v21_merged/assets/lerobot_meta_stats/norm_stats.json`。

使用本机 `/root/autodl-tmp/Berlin/lerobot/src/lerobot/datasets/compute_stats.py`
的 `get_feature_stats` 和 `aggregate_stats`：

1. 对每条 episode 的原始 `observation.state`、`action`，以 float32 读取。
2. 在每个维度上计算单条 episode 的统计；q01/q99 使用 LeRobot 的 5000-bin 直方图。
3. 汇总全部 episode。q01/q99 的公式为 `sum(n_episode * q_episode) / sum(n_episode)`；
   mean 按帧数加权，std 使用包含轨迹间均值差的合并方差。
4. 聚合之后转换为 float32，并映射 `observation.state → state`、`action → actions`。

没有将所有帧合并后调用 `np.quantile`，也没有根据 50 步动作窗口重复计数。
直接汇总现有 `episodes_stats.jsonl` 的分位数与参考有小差异，因此本配置从原始
parquet 逐条重算。准备脚本会先用原始 1600 条示范验证所有 mean/std/q01/q99
与参考**完全一致**，验证失败就终止。

共享训练统计：
`/root/autodl-tmp/datasets/lbench_recap_round1/assets/lerobot_episode_stats/norm_stats.json`。
它包含两组数据，数值自然不同于只有示范数据的参考文件。所有策略训练样本使用这
一份统计；数据采样概率与帧数成正比。准备目录还保留两组数据及各分任务数据集
单独计算的 norm，以及含源码和文件 SHA256 的 `preparation_manifest.json`。

这是使用新统计的策略微调，`allow_norm_stats_override: true` 只用于初始化。
新 checkpoint 会保存新统计；部署新策略时应使用新 checkpoint 自带的统计。

价值模型是 RECAP 的独立 SigLIP2 + Gemma3 模型，输入是三路图像和任务文本，
不输入状态或动作。状态/动作的 norm 用于 π0.5 策略，价值目标则用所有训练
数据的 return 范围归一化到 [-1, 0]。

## 默认超参数

| 阶段 | 设置 |
| --- | --- |
| Return | gamma=1，每步 -1，成功终点 0，失败终点 -300 |
| Value | 8000 steps，micro batch=32/GPU，global batch=256，双卡梯度累积=4，lr=5e-5 |
| Advantage | lookahead=20，positive_quantile=0.3，tag=`lbench_round1_fail300_n20_q30` |
| Policy | 30000 steps，micro batch=16/GPU，global batch=64，双卡梯度累积=2，lr=5e-6，cosine 调度总步数=30000 |
| Checkpoint | Value 每 1000 steps、Policy 每 5000 steps 保存，各保留最近 3 个完整 checkpoint |

Advantage 推理使用 batch=16/卡、8 个数据 worker/卡（双卡共 16 个）、
prefetch_factor=2。原 LeRobot 路径每读取一帧都会重新打开三路视频解码器，
重复从关键帧解码，CPU 视频处理会使 GPU 等待。当前配置启用
`advantage.video_decoder_cache_size: 3`：每个 worker 最多缓存三个 TorchCodec
解码器，按最近使用顺序淘汰，不缓存图像帧；worker 退出时释放其进程内资源。
缓存不会随 episode 总数增长，也不会跨进程复用原生解码器。
设置为 0 可以恢复原来的读取路径；其他视频后端仍使用 LeRobot 原实现。

本机 1024 样本、单卡、8 worker 的实际 advantage 流程对照测试中，
耗时从 29.95 秒降到 10.11 秒（约 2.96 倍），所有输出列逐值完全一致，
GPU 峰值 allocated 均约 2.01 GiB。测试时原双卡任务仍在运行，结果包含
数据加载器启动和临时结果写盘，不代表全量数据的固定加速倍数。
记录位于 `/root/autodl-tmp/outputs/lbench_recap_preparation/advantage_video_cache_benchmark.json`。
`tests/unit_tests/test_recap_video_cache.py` 检查解码器复用与淘汰、进程隔离、
时间戳校验、帧顺序和禁用缓存时的兼容路径。

没有单独验证集；训练损失不能代替环境成功率评测。推理时的 CFG scale 默认为 1，
可在策略训练完成后通过环境评测选择。

按用户要求，value 和 policy 都设 `val_check_interval: -1`，包括最后一步也不会
触发验证；checkpoint 保存仍按配置执行。价值模型数据加载每卡 8 个 worker，
双卡共 16 个；策略的数据加载仍为每卡 2 个 worker。两者都设置
prefetch_factor=2、persistent_workers=false、pin_memory=false；训练指标只保留当前优化步的
标量，反向后用 `zero_grad(set_to_none=True)` 释放梯度引用。
`value` 和 `all` 每 30 秒将进程 RSS/USS/PSS、线程数、文件描述符数、容器内存限额、
匿名内存、文件缓存、OOM 计数和 GPU 占用流式写入 `memory.jsonl`，
监控进程不会在内存中累积历史。价值训练启用 `actor.log_memory_usage: true`，
每个优化步额外记录各卡最大 CPU RSS、GPU allocated 和 GPU reserved。
GPU reserved 包含 PyTorch 缓存，不能单凭 `nvidia-smi` 的占用上升判断泄漏。

检查时本容器的 `memory.max` 为 220 GiB，`free` 显示的宿主机内存不能作为
训练可用额度。历史停止原因尚未确定；已完成的策略固定小批量 12 步测试中，
后 6 步 CPU RSS 和 GPU allocated 均稳定。修复下述 FSDP 问题后，价值模型
也用完整数据、双卡 32/256、每卡 2 个数据 worker 完成了 3 个优化步及 checkpoint 保存：
步末 GPU allocated 均为 5.5206 GiB，reserved 均为 82.5293 GiB，
各 rank 最大 CPU RSS 为 4.0775–4.0810 GiB，未新增 OOM 事件。
记录在 `/root/autodl-tmp/outputs/lbench_recap_value_fsdp_fix_smoke/validation.json`。
这属于短程功能验证，不能保证长时间运行没有内存增长。完整训练由用户自行启动。

价值模型的视觉骨干来自
[google/siglip2-so400m-patch14-224](https://huggingface.co/google/siglip2-so400m-patch14-224)，
存放于 `/root/autodl-tmp/models/siglip2-so400m-patch14-224`；
语言骨干使用本机 `/root/autodl-tmp/models/gemma-3-270m`。

Gemma3 的 `embed_tokens.weight` 与 `lm_head.weight` 共享参数。价值模型直接
调用词嵌入，不调用语言输出层；两者必须保留在同一个 FSDP 单元中。单独包装
`lm_head` 会使词嵌入在前向时读到一维分片，报 `weight must be 2-D`。
RECAP 的 `_no_split_names` 已移除 `lm_head`，保留参数共享和原 checkpoint 键。
`tests/unit_tests/test_recap_value_fsdp.py` 用两个 CPU/Gloo rank 实际执行
FULL_SHARD 前向、反向及优化更新，并与未分片模型比较。

CFG 策略中的 PaliGemma 同样共享 `lm_head.weight` 与词嵌入权重。
`OpenPi0ForCFGActionPrediction._no_split_names` 也移除了 `lm_head`，
使两个别名都由根 FSDP 单元管理，避免策略训练第一步报 `weight must be 2-D`。
`tests/unit_tests/test_openpi_cfg_fsdp.py` 使用实际 OpenPI 词嵌入调用和
双 rank FULL_SHARD，检查两次梯度累积后的参数更新与未分片模型一致。
修复后已用完整数据和实际 advantage 标签、双卡 16/64 完成 3 个优化步及
checkpoint 保存，未新增 OOM 事件。保存的词嵌入保持二维且与共享语言头
数值一致，checkpoint 自带 norm 与训练使用的共享 norm 完全一致。
短测记录：`/root/autodl-tmp/outputs/lbench_recap_policy_fsdp_fix_smoke/validation.json`。
这验证训练与保存功能，不代表已完成长时间内存稳定性测试。

CFG worker 仅在 OpenPI 外层梯度检查点仍开启时关闭它，保留 Gemma 在首次
前向中开启的逐层梯度检查点，避免每个优化步重复输出 Disabled/Forcing
提示。这里的梯度检查点通过反向时重算中间结果节省显存；模型文件的保存
仍由 `runner.save_interval` 控制。运行中的旧进程会继续使用原来的日志行为，
无需仅为减少这些提示中断训练。

## 运行

当前数据准备、return 计算和训练前检查均已完成。按当前需求，只启动价值模型，
训练到 8000 steps 后退出，不执行 advantage 计算或策略训练：

```bash
cd /root/autodl-tmp/Berlin/RLinf
bash examples/offline_rl/run_lbench_recap.sh value
```

单独的 `value` 命令也自动记录 `memory.jsonl`，终端输出同时追加到 `value.log`。
完成时 `stage.txt` 为 `complete:value`，失败时为 `failed:value`。
`data.train_num_workers` 按训练 rank 设置；修改配置后需要重新启动才会生效。

其他阶段的独立命令（仅在需要时执行）：

```bash
cd /root/autodl-tmp/Berlin/RLinf
# 数据准备只执行一次；已有 manifest 时拒绝覆盖。
bash examples/offline_rl/run_lbench_recap.sh prepare
# 检查数据、归一化及视觉/语言骨干。
bash examples/offline_rl/run_lbench_recap.sh check
# 加载指定策略权重，执行 GPU 前向和反向；不会保存 smoke 权重或标签。
bash examples/offline_rl/run_lbench_recap.sh check --policy-smoke
# 四阶段完整流水线。
bash examples/offline_rl/run_lbench_recap.sh all
```

输出默认在 `/root/autodl-tmp/outputs/lbench_recap_round1`，可用
`LBENCH_RECAP_OUTPUT` 指定独立运行目录。`stage.txt` 和各阶段 `.log` 记录进度。
`value` 和 `all` 持有同一个文件锁，防止同一输出目录被重复启动；
通过 `all` 调用 `value` 时复用外层监控，不额外启动监控进程。

分阶段运行和恢复：

```bash
bash examples/offline_rl/run_lbench_recap.sh returns
bash examples/offline_rl/run_lbench_recap.sh value
export RECAP_VALUE_CHECKPOINT=/root/autodl-tmp/outputs/lbench_recap_round1/value/checkpoints/global_step_8000
bash examples/offline_rl/run_lbench_recap.sh advantages
bash examples/offline_rl/run_lbench_recap.sh policy

# 示例：恢复到同一训练任务，保留原先的 max_steps 和调度设置。
bash examples/offline_rl/run_lbench_recap.sh value \
  runner.resume_dir=/path/to/value/checkpoints/global_step_1000
```

解码器缓存只在重新启动 `advantages` 后生效。当前 advantage 脚本在全部
数据集处理完成、算出统一阈值后才保存最终标签，不支持中间进度续算；
停止旧任务后重新运行会从头计算 advantage，无需重新训练价值模型。

完整训练之前可执行一次相同 32/256 配置的分布式 value smoke（输出单独存放）：

```bash
LBENCH_RECAP_OUTPUT=/root/autodl-tmp/outputs/lbench_recap_smoke \
bash examples/offline_rl/run_lbench_recap.sh value \
  runner.max_steps=3 runner.save_interval=3
```

主要配置是 `config/lbench_recap_{returns,value,advantages,policy}.yaml`；
共享数据列表是 `config/lbench_recap_datasets.yaml`。
