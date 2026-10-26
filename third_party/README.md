# 依赖源码快照

这些目录保留本机实际使用的依赖代码，不包含模型权重。

| 目录 | 来源 | 用途 |
| --- | --- | --- |
| `training/openpi` | 已安装 `rlinf-openpi==0.1.1` | π0.5 模型、数据变换与 OpenPI 辅助代码 |
| `training/openpi_client` | 已安装 `openpi-client==0.1.2` | OpenPI 客户端辅助代码 |
| `training/lerobot` | 已安装 `lerobot==0.3.3` | 训练时读取 LeRobot v2.1 数据 |
| `transformers_overlay/transformers` | 训练环境中实际安装的五个补丁文件 | OpenPI Gemma/PaliGemma/SigLIP 兼容实现 |
| `lerobot_preparation/src` | 本地 `/root/autodl-tmp/Berlin/lerobot` | LeRobot 0.5.1 数据准备与统计实现 |

每个复制文件的原路径、SHA256 都列在 `../provenance/copied_files.json`。安装环境和补丁使用方式见 `../environment/README.md`。许可证位于 `licenses/` 和 `lerobot_preparation/LICENSE`。
