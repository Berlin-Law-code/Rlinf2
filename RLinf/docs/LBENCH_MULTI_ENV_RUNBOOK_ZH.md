# LBench 多环境 Pi0.5 SFT、评测与 PPO 运行手册

本手册对应以下本地仓库：

- RLinf：`/home/berlin/flywheel/RLinf`
- LBench：`/home/berlin/flywheel/LBench_multi_env`
- LeRobot：`/home/berlin/flywheel/lerobot`（无需修改源码）

当前实现采用两个完全隔离的 Python 环境：RLinf 进程只加载
PyTorch/OpenPI/Ray，LBench 的每个环境由独立 Isaac Python 子进程运行。二者通过
Unix domain socket 通讯，不需要在 RLinf 虚拟环境中安装 Isaac，也不应把 Isaac
的 `site-packages` 加进 `PYTHONPATH`。启动器保留 `env/bin/python` 的符号链接路径，
将该解释器的 `bin` 放到子进程 `PATH` 首位、禁用 user-site，并清理父进程的
Python/Conda 环境标记；显卡驱动等系统变量仍然共享，GPU 需另行显式隔离。
每个 Isaac worker 还会进入独立进程组；连接超时或 worker 异常时，父进程会清理
整棵 worker 进程树和私有 socket，避免残留进程继续占用 GPU。

## 1. 已固定的训练合同

- 模型：Pi0.5，模型动作 horizon 为 50。
- 环境每次只执行前 20 个动作；提前终止后的动作由 `action_mask` 排除。
- 状态和环境动作维度均为 8：7 个关节加 1 个夹爪标量。
- Robotiq 夹爪观测与采集器一致：第一个实际夹爪关节位置 `<= 0.4` 时编码为
  `1`（张开），否则为 `0`（闭合）。该观测阈值独立于动作的 `[0, 0]` / `[5, 5]`
  控制目标，不能用控制目标的中点判断实际开闭。
- 三路相机：`obs_camera`、`obs_camera_2`、`realsense`。
- 最大 episode 长度为 1200 个 primitive step。长度约 1000 的任务不需要改
  `rollout_epoch`；应保留 1200，给终止检测留下余量。
- 正式 PPO：8 个并行环境，`rollout_epoch=8`，`runner.max_steps=200`，
  `algorithm.update_epoch=4`，全局 batch 为 32。
- PPO 与低显存 SFT smoke 冻结 PaliGemma VLM，只训练 action expert/投影层
  （PPO 还训练 value head）；基础 SFT 配置仍保留全参数训练作为多卡基线。
- 评测使用 `auto_reset=false`：每个环境槽在每个 `rollout_epoch` 只贡献一条
  episode。短任务结束后只做 padding，不会因为完成得快而被重复采样。
- 设置 `env.eval.video_cfg.save_video=true` 可保存评测录像，画面取自模型收到的
  `main_images`。录像目录使用环境 worker 的固定 seed；每条 episode 的任务和
  随机种子仍按原有调度器分配，以评测指标中的 `task_index`、`seed` 为准。
- PPO 配置默认奖励为 `libero_sparse_success`，与原始 RLinf LIBERO PPO 对齐：只有
  episode 内第一次成功获得一次 `algorithm.reward_coef`（默认 1.0），普通步骤、
  progress/stage 变化、显式失败和 1200 步超时的 reward 都为 0。SimBox 的显式
  不可恢复失败仍按原生语义结束 episode，1200 步超时仍保留为 truncation；奖励
  对齐不会覆盖仿真器终止合同。`sparse_success`、`progress_delta` 和 `backend`
  保留用于对照实验。
- 独立评测配置使用后端奖励；成功率按 `success_once` 统计，以后端确认完成为准。
  录像里的阶段奖励和累计回报不等同于任务成功率。
- `reward_type=chunk_level` 会先求和每个 20-action chunk 内的 primitive
  reward，再在 chunk 时间轴上用 `gamma=0.99`、`gae_lambda=0.95` 计算
  GAE。`auto_reset=false` 时，与 LBench 同步的 1200-step horizon 被视为
  有限时域 episode 终点，不会把下一个任务的 value 接上来。

`rollout_epoch=8` 表示每个 PPO global step 收集 8 轮 rollout；它不是 episode
长度，也不是 PPO 的总训练轮数。按最坏情况 1200 步估算：

- 每个 global step：`8 env × 8 rollout × 1200 = 76,800` 个模拟器子步；
- 200 个 global step：最多 15,360,000 个模拟器子步；
- 执行 horizon 20 时，每个完整 episode 最多 60 个 policy chunk；
- 每个 global step 约 3840 个 env-chunk 样本；全局 batch 32、update epoch 4
  时，约 480 次 optimizer update，200 步约 96,000 次。

提前成功或失败的 episode 会减少实际模拟器步数。
`max_steps_per_rollout_epoch` 的单位是 primitive step，RLinf 内部会用它整除
`num_action_chunks` 得到策略调用次数；启动预检现在要求二者能够整除，避免尾部步数被
整数除法静默丢弃。预检 JSON 中会直接给出 `chunk_steps_per_rollout_epoch`、单个
global step 及全程的最大 primitive env-step 数。

## 2. 创建隔离的 RLinf 环境

在有 CUDA 工具链和网络/本地 wheel 缓存的训练机上执行：

```bash
export RLINF_ROOT=/home/berlin/flywheel/RLinf
cd "$RLINF_ROOT"
bash requirements/install.sh embodied \
  --model openpi \
  --env maniskill_libero \
  --venv .venv-lbench-rlinf \
  --install-rlinf

source "$RLINF_ROOT/.venv-lbench-rlinf/bin/activate"
export RLINF_PYTHON=$RLINF_ROOT/.venv-lbench-rlinf/bin/python
export EMBODIED_PATH=$RLINF_ROOT/examples/embodiment
export PYTHONPATH=$RLINF_ROOT:${PYTHONPATH:-}
```

Isaac 解释器必须另选一个已能启动当前 LBench/Isaac 版本的环境，例如：

```bash
export LBENCH_ISAAC_PYTHON=/home/berlin/miniconda3/envs/env_isaaclab_grpc/bin/python
```

先用该解释器执行 LBench 原有的单环境评测，确认它确实是正确的 Isaac 环境；
不要仅依据环境名判断。

远端阶段启动器会优先从 `LBENCH_ISAAC_PYTHON` 所在环境解析 pip/Conda 版
Isaac Sim 的 `ISAAC_PATH`、`EXP_PATH` 和 `CARB_APP_PATH`，避免父进程中的旧路径
或占位路径覆盖正确运行时。如果该环境含有 `lib/libgomp.so.1`，启动器还会只向
外部 Isaac worker 注入该库，以避免 Isaac 加载 pydrake 时出现 static TLS 错误。
不要在当前 shell 全局设置 `LD_PRELOAD`，否则模型 worker 也会继承它并可能产生
额外的 OpenMP 运行时冲突。

### 2.1 AutoDL 现有环境（不要重复安装）

AutoDL 上已经存在 RLinf 虚拟环境时，直接复用，并把缓存、模型、输出都放在
`/root/autodl-tmp`：

```bash
source /root/autodl-tmp/.venvs/rlinf-openpi/bin/activate
export RLINF_ROOT=/root/autodl-tmp/Berlin/RLinf
cd "$RLINF_ROOT"

export RLINF_PYTHON=/root/autodl-tmp/.venvs/rlinf-openpi/bin/python
export EMBODIED_PATH=$RLINF_ROOT/examples/embodiment
export PYTHONPATH=$RLINF_ROOT:${PYTHONPATH:-}
export XDG_CACHE_HOME=/root/autodl-tmp/.cache
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_LEROBOT_HOME=/root/autodl-tmp/.cache/huggingface/lerobot

export LBENCH_LEROBOT_REPO=/root/autodl-tmp/datasets/你的_v21_数据目录
export LBENCH_WORK=/root/autodl-tmp/outputs/rlinf_pi05_lbench
export LBENCH_NORM_DIR=$LBENCH_WORK/assets/norm/lbench_8tasks_v21
export LBENCH_NORM_STATS=$LBENCH_NORM_DIR/norm_stats.json
export OPENPI_DATA_HOME=/root/autodl-tmp/.cache/openpi
export PI05_BASE_CHECKPOINT=/root/autodl-tmp/models/openpi/pi05_base_pytorch_bf16
mkdir -p "$LBENCH_NORM_DIR" /root/autodl-tmp/models/openpi
```

本机 `/home/berlin/flywheel/RLinf` 的修改不会自动出现在
`/root/autodl-tmp/Berlin/RLinf`。远端运行前必须先通过 Git 分支或 `rsync`
同步代码；若远端 `toolkits/lbench_data_dry_run.py` 仍报
“requires LeRobot v3 metadata”，说明运行的仍是旧文件，而不是 v2.1 数据有问题。
同步后可先用以下只读检查确认关键修订已经到位：

```bash
cd /root/autodl-tmp/Berlin/RLinf
rg -n "validate_lerobot_runtime|supports LeRobot v2.1 or v3.0" \
  toolkits/lbench_data_dry_run.py toolkits/lbench_remote_stage.py
```

现有环境中 `uv pip check` 报出的 Torch/LeRobot/Transformers 版本冲突不能由
“import 成功”或 FlashAttention 单算子成功证明无害。不要在正式任务前继续随机
升级包；先保留环境快照，随后在远端按第 5 节做最小 SFT smoke。若 FSDP 或数据加载
失败，再建立一套严格满足 RLinf pin 的新 venv，与当前探索环境并存。

## 3. 生成联合多任务 YAML

双环境 PPO smoke 要求至少两个 task entry；正式 `eval-8` 和 `ppo-8` 要求恰好
八个 task entry。以下给出八个长任务示例；最终任务组合应以业务需要、资产可用性
和单任务成功率为准。

```bash
export LBENCH_ROOT=/home/berlin/flywheel/LBench_multi_env
export LBENCH_TASK_BASE=$LBENCH_ROOT/workflows/simbox/core/configs/tasks/long_horizon/franka_random
export LBENCH_TASK_CONFIG=$LBENCH_ROOT/configs/simbox/lbench_joint_8tasks.yaml

"$LBENCH_ISAAC_PYTHON" \
  "$LBENCH_ROOT/scripts/eval/build_multitask_task_config.py" \
  --output "$LBENCH_TASK_CONFIG" \
  "$LBENCH_TASK_BASE/clean_the_dirt_with_browncloth_lighttable_part0.yaml" \
  "$LBENCH_TASK_BASE/flip_package_0.yaml" \
  "$LBENCH_TASK_BASE/heat_the_food_in_the_microwave.yaml" \
  "$LBENCH_TASK_BASE/insert_the_flower_in_vase_part0.yaml" \
  "$LBENCH_TASK_BASE/put_the_trashcan_in_trashcan_part0_eval.yaml" \
  "$LBENCH_TASK_BASE/stack_two_l_test.yaml" \
  "$LBENCH_TASK_BASE/store_objects_in_the_drawer_0.yaml" \
  "$LBENCH_TASK_BASE/watering_plant_left.yaml"
```

生成器拒绝覆盖已有输出。需要重建时，应明确选择一个新的输出路径，或在确认旧文件
不再需要后自行移走旧文件。启动器还会检查任务名唯一、语言指令非空以及三路相机齐全。
M8 评测/PPO 预检还会读取 `LBENCH_LEROBOT_REPO` 的 task 表，要求其中 8 条 SFT
prompt 与该 YAML 的 8 条 `language_instruction` 集合完全一致；任务索引顺序可以不同，
但不能缺失、重复或临时换一种说法。

## 4. 准备与 OpenPI 环境匹配的 LeRobot SFT 数据和归一化统计

数据的 feature schema 与存储版本是两回事。`pi05_lbench_joint` 要求三路相机、
8 维 state/action 和任务文本；而 `meta/tasks.jsonl`（v2.1）还是
`meta/tasks.parquet`（v3.0）由安装的 LeRobot loader 决定。当前
`rlinf-openpi==0.1.1 + lerobot==0.3.3` 使用 v2.1，因此这里必须选择 v2.1
副本，不能只改 `info.json` 或重命名 metadata 文件。v3.0 副本保留给新版 LeRobot
训练环境使用。

设置数据集、基础模型和独立的统计输出目录：

```bash
export LBENCH_LEROBOT_REPO=/absolute/path/to/lbench_lerobot_v21
export PI05_BASE_CHECKPOINT=/absolute/path/to/pi05_base_torch
export LBENCH_NORM_DIR=/absolute/path/to/openpi_assets/lbench_joint
export LBENCH_NORM_STATS=$LBENCH_NORM_DIR/norm_stats.json
```

`PI05_BASE_CHECKPOINT` 必须是 **OpenPI PyTorch** 权重目录，而不是
`lerobot/pi05_base` 的 LeRobot policy 目录。若尚未准备基础权重，在 AutoDL
机器上把官方 OpenPI JAX checkpoint 下载到数据盘并转换一次：

```bash
export OPENPI_DATA_HOME=/root/autodl-tmp/.cache/openpi
export PI05_BASE_CHECKPOINT=/root/autodl-tmp/models/openpi/pi05_base_pytorch_bf16
mkdir -p /root/autodl-tmp/models/openpi

PI05_JAX_CHECKPOINT=$("$RLINF_PYTHON" - <<'PY'
from openpi.shared.download import maybe_download
print(maybe_download("gs://openpi-assets/checkpoints/pi05_base"))
PY
)
export PI05_JAX_CHECKPOINT

JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$RLINF_PYTHON" rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py \
  --checkpoint-dir "$PI05_JAX_CHECKPOINT" \
  --config-name pi05_lbench_joint \
  --output-path "$PI05_BASE_CHECKPOINT" \
  --precision bfloat16
```

转换阶段不需要 GPU，但会同时持有 JAX 参数、PyTorch 参数和输出权重，建议预留
至少 64 GB、最好 96–128 GB 系统内存，以及约 30–50 GB 可用磁盘。转换完成后
目录中至少应有 `model.safetensors`、`config.json` 和
`conversion_report.json`。转换器默认拒绝非空输出目录，避免覆盖旧模型；
确实要替换时才显式加 `--overwrite`。覆盖模式只会接受并清理该转换器自己的
四类产物（权重、配置、报告和 `assets/`），发现其他文件或输入/输出目录相互嵌套
时会拒绝执行，避免遗留权重被后续 loader 一并加载。它还会拒绝除 tied `lm_head` 以外的
missing/unexpected keys，避免把随机初始化参数静默保存进 checkpoint。不要把第三方转换权重当作
首选；若必须使用，应先比对 key/shape 并做数值一致性检查。

检查数据合同并生成统计：

```bash
cd "$RLINF_ROOT"
"$RLINF_PYTHON" toolkits/lbench_data_dry_run.py \
  --repo-id "$LBENCH_LEROBOT_REPO"

JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$RLINF_PYTHON" toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_lbench_joint \
  --repo-id "$LBENCH_LEROBOT_REPO" \
  --output-dir "$LBENCH_NORM_DIR"

"$RLINF_PYTHON" toolkits/lbench_data_dry_run.py \
  --repo-id "$LBENCH_LEROBOT_REPO" \
  --norm-stats "$LBENCH_NORM_STATS"
```

统计文件必须同时包含 8 维 `state` 和 `actions` 的 `mean/std/q01/q99`。SFT、
评测和 PPO 必须使用同一份统计文件。

如果目标是与 LeRobot 训练严格对齐，不要重新扫描数据计算全局分位数；应直接把
LeRobot `meta/stats.json` 中已经用于其训练的 float32 统计转换为 OpenPI 键布局：

```bash
"$RLINF_PYTHON" toolkits/lerobot/convert_lerobot_stats_to_openpi.py \
  --stats-path "$LBENCH_LEROBOT_REPO/meta/stats.json" \
  --output-dir "$LBENCH_NORM_DIR"
```

转换只映射 `observation.state -> state` 和 `action -> actions`，不会覆盖已有输出。
不要给已经用另一份统计训练完成的 checkpoint 更换统计；新统计必须从 SFT 开始，
一直用于该实验的评测和后续 PPO。普通 OpenPI 离线 SFT 会只在配置的 action chunk
和真实环境动作维度上计算 loss，不把模型内部的 padding 动作维度计入平均值。
当前 `rlinf-openpi==0.1.1` 环境只能读取 LeRobot v2.1 存储格式；如果同内容数据另有
v3.0 副本，可以从 v3.0 的 `meta/stats.json` 转换统计，但训练的 `repo_id` 仍应指向
经过逐 episode/逐帧校验的 v2.1 副本。不要只修改 `info.json` 来伪装格式版本。

与 LeRobot 对齐的四卡配置使用每卡 micro batch 16、global batch 64、seed 1000、
weight decay 0.01、gradient checkpointing 和 10 步推理设置。LeRobot 的
`batch_size=16` 是每个 Accelerate 进程的 batch；四个进程同步梯度，每个 optimizer
step 共处理 64 条样本，并不是四份模型各自独立更新：

该独立配置关闭 `openpi.sft_image_augmentation`，使用 `lerobot_cosine`（warmup
1000、绝对 step / 30000 的 cosine，最低 LR 2.5e-6）、每 5000 步保存。
`openpi_data.lerobot_compatible=true` 同时启用 float32 的精确 LeRobot quantile
公式和 float32 bilinear resize/padding；这会替换原生 OpenPI normalizer，训练
loader 明确跳过默认 normalizer，避免归一化两次。旧 checkpoint 默认保留原处理。

兼容模式的图像转换器输出 `float32 [3, 224, 224]`、范围 `[-1, 1]`，训练和评测
打包后都必须是 `[B, 3, 224, 224]`。OpenPI 的 `Observation.from_dict` 只会对
`torch.uint8` 图像自动交换通道，已经归一化的 float32 图像不会自动转置；若保留
HWC，会在 SigLIP 第一层卷积报 `expected ... to have 3 channels, but got 224`。
这是通道布局错误，不需要调整 batch、norm stats 或重新打开图像增强。
`tests/unit_tests/test_lbench_aligned_pipeline.py` 覆盖实际 PyTorch loader、评测
input wrapper、Observation、无增强预处理和 SigLIP patch embedding 前向/反向，
同时检查旧 uint8 路径不变。可在 CPU 上验证：

```bash
JAX_PLATFORMS=cpu "$RLINF_PYTHON" -m pytest -q \
  tests/unit_tests/test_lerobot_openpi_transforms.py \
  tests/unit_tests/test_lbench_aligned_pipeline.py
```

精度采用 `openpi.parameter_precision=float32`、FP32 参数/Adam 状态和 BF16
autocast。必须从原始 FP32 基础权重初始化，不能把旧 BF16 文件升成 FP32 代替。
当前已准备好的副本是 `/root/autodl-tmp/models/openpi/pi05_base_lerobot_fp32_001`。
可复现转换命令如下（输出目录必须不存在）：

```bash
"$RLINF_PYTHON" toolkits/lerobot/convert_pi05_base_to_openpi.py \
  --source /path/to/cached/lerobot/pi05_base \
  --reference /path/to/openpi/pi05_base_pytorch_bf16 \
  --output-dir /path/to/new/pi05_base_lerobot_fp32
```

当前节点路径已经预设在以下四卡入口，通常只需指定新的输出目录：

```bash
export LBENCH_SFT_OUTPUT=/root/autodl-tmp/outputs/lbench_sft_lerobot_aligned_fp32_4gpu_001
bash scripts/lbench/remote/06_sft_lerobot_aligned.sh
```

该入口覆盖旧 shell 中的 base/stats/data exports，使用 `LBENCH_ALIGNED_BASE_CHECKPOINT`、
`LBENCH_ALIGNED_NORM_STATS`、`LBENCH_ALIGNED_DATASET` 可显式另选输入。
入口不会自动 resume；重复实验应换输出目录。只做启动及保存验证可传入
`runner.max_steps=100 runner.save_interval=100`，仍保留正式的 30000 步 LR 日程。
代码/CPU 测试不替代四卡完整 forward/backward/save 验证。

该入口在启动 Python、Ray 和 DataLoader 子进程之前设置
`MALLOC_MMAP_THRESHOLD_=131072`、`MALLOC_TRIM_THRESHOLD_=131072`。
本机 LeRobot 0.3.3 / TorchCodec 0.11.1 的视频帧转 float32 路径会使 glibc
保留已释放的大块内存：2500 次随机帧读取的 RSS 从 901 MiB 增至 1510 MiB，
`gc.collect()` 无效，而 `malloc_trim(0)` 可降回 961 MiB。固定上述阈值后，
1200 次相同路径读取在预热后维持约 961 MiB。这只调整内存分配，保留原视频
解码、图像插值、归一化与训练参数。需要另选阈值时，使用
`LBENCH_ALIGNED_MALLOC_MMAP_THRESHOLD` 和 `LBENCH_ALIGNED_MALLOC_TRIM_THRESHOLD`；
手动运行 Python 时必须自己在启动前导出对应的 `MALLOC_*` 变量。

四卡配置默认每卡 `data.num_workers=2`，总共八个加载进程，为 checkpoint
转存到 CPU 的模型和 optimizer 状态留出空间。默认每 5000 步保存，
`runner.max_saved_checkpoints=3` 只在新 checkpoint 完整保存、发布
`.checkpoint_complete` 标记后，清理同一输出目录下更旧的已标记完整 checkpoint。
未完成的目录、无标记的历史保存以及 `best_model` 不会被自动清理；设置
`runner.max_saved_checkpoints=0` 可保留全部。完整保存实测约 58 GiB，关闭清理时需
自行预留磁盘。保存替换期间最多暂存四份，即约 232 GiB；只有新保存完整落盘后
才清理旧保存。仅有 `global_step_*` 空目录不表示保存成功。

修复或首次部署后，先做少量真实训练与保存，再恢复同一实验，例如：

```bash
bash scripts/lbench/remote/06_sft_lerobot_aligned.sh \
  runner.max_steps=5 runner.save_interval=5
bash scripts/lbench/remote/06_sft_lerobot_aligned.sh \
  runner.resume_dir="$LBENCH_SFT_OUTPUT/lbench_sft_pi05_lerobot_aligned_4gpu/checkpoints/global_step_5"
```

恢复前确认 `.checkpoint_complete`、DCP `.metadata`/分片、`full_weights.pt`、
`data.pt` 和 `rng.pt` 均已落盘。两次命令使用相同的 `LBENCH_SFT_OUTPUT` 和四卡
配置；第二次恢复 30000 步目标和默认保存间隔，学习率仍使用原 30000 步日程。

新 checkpoint 的 `openpi_assets.json` 记录参数精度和预处理兼容标记；独立评测
loader 会自动恢复这些设置，旧 checkpoint 没有标记则继续使用旧行为。评测的
`LBENCH_NORM_STATS` 仍应选择该次训练所保存的同份统计。

手动启动时同样需要设置新的 FP32 base 和 exact stats 路径：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 "$RLINF_PYTHON" examples/sft/train_vla_sft.py \
  --config-name lbench_sft_openpi_pi05_lerobot_aligned \
  'cluster.component_placement={actor\,env\,rollout:0-3}' \
  runner.logger.log_path="$LBENCH_SFT_OUTPUT"
```

基础配置 `lbench_sft_openpi_pi05` 默认是全参数 SFT。单张 32 GB RTX 5090 无法为
整个 Pi0.5 的可训练参数保留完整 Adam 状态，先用 expert-only 配置做 5-step
显存探测；它会冻结 PaliGemma，但仍训练 action expert 和动作投影层：

```bash
export LBENCH_SFT_OUTPUT=/absolute/path/to/outputs/lbench_pi05
CUDA_VISIBLE_DEVICES=0 "$RLINF_PYTHON" examples/sft/train_vla_sft.py \
  --config-name lbench_sft_openpi_pi05 \
  actor.model.openpi.train_expert_only=true \
  actor.micro_batch_size=1 \
  actor.global_batch_size=1 \
  data.num_workers=2 \
  runner.logger.log_path="$LBENCH_SFT_OUTPUT" \
  runner.logger.experiment_name=expert_only_smoke \
  runner.max_steps=5 \
  runner.save_interval=5
```

这里的 `global_batch_size=1` 只用于确定能否完成 forward/backward/save。显存通过后，
可将其提高到 8 或 16（通过梯度累积，不改变 micro batch 显存峰值），再逐步增加
`runner.max_steps`。全参数多卡训练则不要设置 `train_expert_only=true`。

## 5. M8 分阶段验收命令

所有脚本在真正启动前都会组合 Hydra 配置，并检查路径、checkpoint、任务、GPU
集合、world size 整除关系和已有 Ray 集群。建议先停止旧 Ray 集群：

```bash
ray status
ray stop
```

每次验收使用一个新的输出根目录。启动器默认拒绝复用同一阶段的非空输出，防止
checkpoint、episode 或固定 episode id 相互覆盖；中断后的训练应走明确的 Hydra
`runner.resume_dir=...`，不要把“重新运行整个阶段”当作恢复：

```bash
export LBENCH_M8_OUTPUT=/root/autodl-tmp/outputs/lbench_m8_run_001
```

先只打印计划：

```bash
cd "$RLINF_ROOT"
scripts/lbench/remote/01_sft_smoke.sh --dry-run --output "$LBENCH_M8_OUTPUT" --rlinf-gpus 0
scripts/lbench/remote/02_single_env_eval.sh --dry-run --output "$LBENCH_M8_OUTPUT" --rlinf-gpus 0 --isaac-gpus 4
scripts/lbench/remote/03_dual_env_ppo_smoke.sh --dry-run --output "$LBENCH_M8_OUTPUT" --rlinf-gpus 0,1 --isaac-gpus 4,5
scripts/lbench/remote/04_eight_env_eval.sh --dry-run --output "$LBENCH_M8_OUTPUT" --rlinf-gpus 0,1,2,3 --isaac-gpus 4,5,6,7,8,9,10,11
scripts/lbench/remote/05_eight_env_ppo.sh --dry-run --output "$LBENCH_M8_OUTPUT" --rlinf-gpus 0,1,2,3 --isaac-gpus 4,5,6,7,8,9,10,11
```

按顺序真正执行：

```bash
scripts/lbench/remote/01_sft_smoke.sh --output "$LBENCH_M8_OUTPUT" --rlinf-gpus 0

export LBENCH_SFT_CHECKPOINT=$LBENCH_M8_OUTPUT/m8_sft_smoke/checkpoints/global_step_5
scripts/lbench/remote/02_single_env_eval.sh --output "$LBENCH_M8_OUTPUT" --rlinf-gpus 0 --isaac-gpus 4
scripts/lbench/remote/03_dual_env_ppo_smoke.sh --output "$LBENCH_M8_OUTPUT" --rlinf-gpus 0,1 --isaac-gpus 4,5
scripts/lbench/remote/04_eight_env_eval.sh --output "$LBENCH_M8_OUTPUT" --rlinf-gpus 0,1,2,3 --isaac-gpus 4,5,6,7,8,9,10,11
scripts/lbench/remote/05_eight_env_ppo.sh --output "$LBENCH_M8_OUTPUT" --rlinf-gpus 0,1,2,3 --isaac-gpus 4,5,6,7,8,9,10,11
```

默认的 `eval-8` 是合同验收：8 个任务各跑 seed 0，共 8 条 episode。若要让每个
任务依次跑 seed 0..7（共 64 条，仍保持任务均衡），执行：

```bash
scripts/lbench/remote/04_eight_env_eval.sh \
  --output "$LBENCH_M8_OUTPUT" \
  --rlinf-gpus 0,1,2,3 \
  --isaac-gpus 4,5,6,7,8,9,10,11 \
  --eval-rollout-epochs 8
```

环境的 episode 计数不会在 rollout 轮次之间清零，因此任务、seed 和 episode id
会逐轮推进；这也避免采集文件重复命名或覆盖。

SFT smoke 会训练到 step 5、保存 checkpoint，再从该 checkpoint 恢复到 step 6，
用于检查模型、optimizer、数据位置和随机数状态的恢复路径。
`global_step_5/actor` 中的 `data.pt` 和 `rng.pt` 分别保存各 FSDP rank 的 loader
游标与 RNG；norm stats、资产 manifest 也随 checkpoint 导出。它们都使用临时文件
加原子 rename 发布，任一 rank 序列化失败会同步通知所有 rank，而不是让其余 rank
卡在 collective。精确恢复必须保持相同 FSDP world size。

`dual-ppo` 是系统链路验收，不是效果实验。它把 simulator `max_step`、RL
`max_episode_steps` 和每轮采样窗口同时缩到 40 个 primitive steps，确保两条
并行 episode 都会以 truncation 结束并真正触发落盘，同时让 GAE 在明确的有限时域
终点结束。由此产生的 return、成功率和 40-step 数据都不能用于评价模型能力。
正式 `ppo-8` 会把三者恢复为 1200。

脚本的默认采集格式是 pickle，它保留完整 LBench metadata。若评测或 dual PPO
smoke 要直接生成可供下一轮 SFT 使用的 LeRobot 分片，追加：

```bash
--collection-format lerobot
```

例如：

```bash
scripts/lbench/remote/04_eight_env_eval.sh \
  --output "$LBENCH_M8_OUTPUT" \
  --rlinf-gpus 0,1,2,3 \
  --isaac-gpus 4,5,6,7,8,9,10,11 \
  --collection-format lerobot
```

正式 `ppo-8` 即使指定该参数也保持训练采集关闭。三路 640×480 RGB 原始数据约
2.76 MB/primitive step；最坏 15.36M 步约 42 TB（尚未计入编码、元数据和
临时文件差异），因此只能对短窗口或抽样 episode 开启。LBench 默认设置
`max_pending_exports=1`：编码/磁盘落后时 rollout 会产生背压，避免已完成的长
episode 在内存中无限排队；这会降低采集吞吐，但不会静默丢数据。

### 5.1 从 PPO checkpoint 恢复

不要重新调用 `05_eight_env_ppo.sh` 覆盖原阶段；该启动器刻意只接受全新输出目录。
先用相同 GPU 集合做一次只读预检，然后直接给训练入口传
`runner.resume_dir`。例如从 step 40 继续到配置中的 step 200：

```bash
export LBENCH_M8_OUTPUT=/root/autodl-tmp/outputs/lbench_m8_run_001
export LBENCH_PPO_RESUME=$LBENCH_M8_OUTPUT/lbench_ppo_openpi_pi05/checkpoints/global_step_40

test -d "$LBENCH_PPO_RESUME/actor"
scripts/lbench/remote/05_eight_env_ppo.sh \
  --dry-run \
  --output "$LBENCH_M8_OUTPUT" \
  --rlinf-gpus 0,1,2,3 \
  --isaac-gpus 4,5,6,7,8,9,10,11

ray stop
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RLINF_GPUS=0,1,2,3 \
LBENCH_ISAAC_GPUS=4,5,6,7,8,9,10,11 \
"$RLINF_PYTHON" examples/embodiment/train_embodied_agent.py \
  --config-name lbench_ppo_openpi_pi05 \
  runner.resume_dir="$LBENCH_PPO_RESUME" \
  runner.logger.log_path="$LBENCH_M8_OUTPUT" \
  runner.logger.experiment_name=lbench_ppo_openpi_pi05 \
  'env.train.isaac_cuda_visible_devices=[4,5,6,7,8,9,10,11]' \
  'env.eval.isaac_cuda_visible_devices=[4,5,6,7,8,9,10,11]'
```

恢复必须使用同一份代码、`LBENCH_SFT_CHECKPOINT`、norm stats、任务 YAML、任务
顺序、seed 配置、`total_num_envs`、`rollout_epoch`、action horizon 和模型/FSDP
world size；`runner.max_steps` 是恢复后的总目标 step，不是“再训练多少步”。Actor
checkpoint 会恢复模型、optimizer、LR scheduler 和 actor 侧 RNG。LBench 无需保存
Isaac 内部快照：`global_step_N` 在 `auto_reset=false` 合同下确定性重建为每个环境槽
已经开始过 `N × rollout_epoch` 条 episode，下一次 reset 因而继续正确的 task、seed
和 episode id。若启用了 pickle 收集，文件 episode id 也会同步推进；若启用了
LeRobot 收集，恢复时会扫描 rank 目录并跳过所有已有和半写的 `id_*` shard。

这保证 checkpoint 边界上的训练/任务游标一致，但不承诺中断前后逐 bit 相同：Ray
调度、Isaac 数值轨迹以及 rollout worker 自身的采样 RNG 都可能改变。因此恢复后先看
一个 global step 的 task/seed 分布、loss、KL、value loss 和成功率，再继续长跑。

## 6. 合并并复查并行 LeRobot 分片

每个 RLinf rank 写到独立的 `rank_<rank>/id_<id>` 数据集，避免并发写同一个
LeRobot metadata。当前合并器面向 OpenPI 环境所用的 v2.1 物理布局，会重编号
episode/frame/task 及其统计。LBench collector 默认把三路 `dtype=image` 图像直接
写在 Parquet 中，合并时会随表保留；如果某个来源实际声明为 `dtype=video`，
合并器才会复制对应 MP4。它同时重建 `episodes_stats.jsonl` 和 `stats.json`，
并要求每条 episode 恰好对应一个已声明任务、所有分片的 Parquet 列与 Arrow 类型
一致；不可读分片不会被静默跳过。如果误传 v3.0 数据会直接拒绝。
训练前将分片合并到一个全新的输出目录：

```bash
"$RLINF_PYTHON" toolkits/lerobot/merge_lerobot_datasets.py \
  --source-dir /absolute/path/to/episodes/eval \
  --output-dir /absolute/path/to/lbench_eval_merged_v21 \
  --dry-run

"$RLINF_PYTHON" toolkits/lerobot/merge_lerobot_datasets.py \
  --source-dir /absolute/path/to/episodes/eval \
  --output-dir /absolute/path/to/lbench_eval_merged_v21

"$RLINF_PYTHON" toolkits/lbench_data_dry_run.py \
  --repo-id /absolute/path/to/lbench_eval_merged_v21
```

合并工具不会在源分片上原地写入，也拒绝非空输出目录；输出目录应当不存在或为空。
实际内容先写入目标目录同级的临时目录，所有 Parquet、视频、metadata 和统计都成功后
才原子发布到目标路径，因此中途坏帧或磁盘写入异常不会留下一个看似可用的半成品目标。

## 7. 资源规划

以下是工程验收的保守建议，不是最低显存保证；5090 路线仍需以上述 5-step
smoke 的实测峰值为准：

| 阶段 | 模型侧 GPU | Isaac 侧 GPU | 其他资源建议 |
|---|---:|---:|---|
| 静态检查/数据 dry-run | 0 | 0 | 16 GB RAM |
| expert-only SFT smoke | 1×32 GB（待实测） | 0 | 64 GB RAM，0.5 TB NVMe |
| 全参数 SFT | 2×80 GB，推荐 4×80 GB | 0 | 128–256 GB RAM，0.5–1 TB NVMe |
| 单环境评测 | 1×48–80 GB | 1×24 GB | 16 CPU 核，64–128 GB RAM |
| 双环境 PPO smoke | 2×80 GB | 2×24 GB | 24–32 CPU 核，128–256 GB RAM |
| 8 环境并行评测 | 1–4×80 GB | 推荐 8×24 GB | 32–64 CPU 核，256 GB RAM，1–2 TB NVMe |
| 8 环境正式 PPO | 至少按 4×H100 80 GB 起步，推荐 4–8 张 | 推荐 8×24/48 GB | 48–64 CPU 核，256–512 GB RAM，2 TB+ NVMe |

RLinf 官方 Pi0.5 快速配置说明中，缩减后的 LIBERO 方案是在 4×H100 上验证的，
因此 24 GB 4090 不应被当作端到端验收基线；32 GB 5090 这里只作为 expert-only
SFT 的探索性入口，不代表可承载并行 PPO。若测得一个 Isaac worker 显存占用
足够低，可让两个 worker round-robin 共用一张大显存 GPU；第一次验收仍建议一
worker 一 GPU。

RLinf GPU 数必须整除环境数：单环境只能用 1 个 rollout rank；双环境可用 1 或
2 个；8 环境可用 1、2、4 或 8 个。模型 GPU 和 Isaac GPU 的物理编号必须完全
不重叠。

## 8. 结果、监控和通过标准

默认结果根目录为：

```text
/home/berlin/flywheel/results/lbench_m8
```

TensorBoard：

```bash
tensorboard --logdir /home/berlin/flywheel/results/lbench_m8 --port 6006
```

正式训练前至少确认：

1. SFT step 5 -> 6 恢复后的 loss、数据 offset 和 checkpoint 资产无异常；
2. 单环境评测能完成 reset、20-action chunk、终止和下一 episode reset；
3. 双环境中一个环境提前终止时，另一个继续推进，padding action 不进入 PPO loss
   或采集数据；
4. 八环境任务分配和 per-task 成功率计数均衡，失败不会拖死其他 worker；
5. Ray 只看到模型 GPU，Isaac 子进程只看到明确分配的模拟器 GPU；
6. LeRobot 分片可 finalize、merge，并通过真实 50-action 时间窗口 dry-run；
7. 先根据双环境 smoke 的实测吞吐量估算正式 200 global steps 的墙钟时间，再决定
   是否降低 `runner.max_steps` 或 `rollout_epoch` 做消融。

当前 4090 主机和远端 5090 均尚未完成上述 GPU/Isaac/模型验收；本轮按要求只做了
静态源码审查，没有在本机执行 Python、CUDA、Isaac、数据解码或端到端训练。静态
检查通过也不等价于真实训练通过。
