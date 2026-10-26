# 环境记录

本仓库收录的是实际工作环境的版本清单和关键依赖源码，不是已经在空白机器测试过的通用安装锁文件。

## 训练环境

- 原解释器：`/root/autodl-tmp/.venvs/rlinf-openpi/bin/python`，Python 3.11。
- 完整包版本：[training_packages.txt](training_packages.txt)。
- 主要版本：PyTorch 2.11.0+cu130、TorchVision 0.26.0+cu130、TorchCodec 0.11.1、Transformers 4.57.6、LeRobot 0.3.3、rlinf-openpi 0.1.1、openpi-client 0.1.2、NumPy 1.26.4、Ray 2.58.0。
- 原机器已经安装了 CUDA 对应 wheel、系统视频库及训练依赖。迁移时可参考 `RLinf/requirements/install.sh` 和版本清单；不能把所有包的元数据约束当作一个可直接由 pip 自动求解的组合。

特别注意，环境同时记录 `transformers==4.57.6` 和 `rlinf-transformer-openpi==4.53.2`，安装元数据名称不能反映最终被覆盖过的源码文件。OpenPI 使用带 AdaRMS 的 Gemma 实现，本仓库保存了实际安装目录中的以下文件：

```text
third_party/transformers_overlay/transformers/models/gemma/configuration_gemma.py
third_party/transformers_overlay/transformers/models/gemma/modeling_gemma.py
third_party/transformers_overlay/transformers/models/paligemma/modeling_paligemma.py
third_party/transformers_overlay/transformers/models/siglip/modeling_siglip.py
third_party/transformers_overlay/transformers/models/siglip/check.py
```

新建训练环境时应在匹配的 Transformers 4.57.6 基础上备份并恢复这些补丁。归档过程没有改动原环境。仅把该 overlay 的父目录放入 `PYTHONPATH` 不会可靠覆盖已经安装的普通 `transformers` 包。

`third_party/training/` 中的 `openpi`、`openpi_client`、`lerobot` 是原训练环境的完整 Python 包源码，去除了缓存和编译文件。`scripts/run_stage.sh` 将该目录和本仓库 `RLinf/` 放在 `PYTHONPATH` 前部，避免误用外部 RLinf checkout。其余依赖仍由训练环境提供。

如果在新环境安装 RLinf 本身，可在依赖准备完成后使用：

```bash
python -m pip install --no-deps -e ./RLinf
```

## 数据准备环境

- 原解释器：`/root/autodl-tmp/conda/envs/lerobot01/bin/python`。
- 完整包版本：[preparation_packages.txt](preparation_packages.txt)。
- LeRobot 0.5.1、NumPy 2.2.6、PyArrow 23.0.1、PyTorch 2.10.0。
- LeRobot 源码基础 commit：`15934d8d08f3c62ec730fb814087619e6c987995`。
- `third_party/lerobot_preparation/src/` 保存了本地源码，包括准备 manifest 指定的 `lerobot/datasets/compute_stats.py`。完整文件来源及 SHA256 见 `provenance/copied_files.json`。

这套环境仅用于数据准备与逐 episode norm 统计。不要将它和训练用 LeRobot 0.3.3 的包路径混在同一个 `PYTHONPATH` 中。

## 只看 loss

无需上述训练环境：普通 Python 环境安装 `numpy matplotlib tensorboard` 即可运行 `scripts/plot_losses.py`。`scripts/verify_snapshot.py` 只依赖 Python 标准库。

