# 保存 PPO 轨迹与 SFT 对照

单任务 PPO 启动器增加 `--record-diagnostics`，默认关闭。开启后，训练和评测的
成功、失败回合均保存到本次输出目录的 `recordings/train`、`recordings/eval`；
不修改奖励、损失、模型或采样数量。`recordings/ppo` 保存更新前每卡的优势、回报、
旧价值、旧 log-prob 和 mask 等数值张量。

每条轨迹包含：

- `transitions.npz`：T+1 个八维状态、T 个原始策略动作与实际执行动作、奖励、终止/截断标记。
- `step_info.jsonl`：逐状态的阶段、seed、episode_id、终止原因、执行覆盖信息；第 0 条为 reset 空信息。
- `obs_camera.mp4`、`obs_camera_2.mp4`、`realsense.mp4`：原分辨率的三路视频，共 T+1 帧。
- `raw_observation_frames.npz`：每 20 步的原始 RGB，无损压缩，用于检查送入预处理前的图像。
- `metadata.json` 和 `.complete`：只有三个编码器全部正常关闭才标记完成；中途失败留下 `.partial` 目录。

视频第 i 帧/状态 i 对应动作 i 执行前，reward[i] 对应该动作之后的奖励；第 T 帧为终止状态。
只记录真实执行的动作，600 步上限后的 padding 不导出；成功回合提前终止后的 padding 也不导出。
MP4 是有损编码，不能作为像素级重放输入；每 20 步的无损图像用于当前 20/20 配置的决策观测检查。
播放速率按 SFT 元数据使用 30 fps，并不表示训练墙钟速度或额外改变仿真控制频率。

每个环境进程组只使用一个后台写入线程，待写队列限制为 1。回合结束到落盘仍可能暂存原始三路图像，
开启录制会增加 CPU 内存、存储和耗时。每批 rollout 结束等待导出完成，避免验证退出后丢失尾部视频。
运行长训练前应另行估算数据量；本次用于单轮恢复验证。

本次验证从已有第 41 轮检查点恢复，只新增一轮到第 42 轮；仍保持 1000 轮学习率衰减配置：

```bash
bash scripts/lbench/remote/13_ppo_clean_4gpu_libero_long.sh \
  --gpus 0,1,2,3 --action-horizon 20 --clip-ratio-c 3 \
  --value-head-hidden-sizes 1024 512 256 \
  --resume-dir /path/to/checkpoints/global_step_41 \
  --max-steps 42 --lr-schedule-steps 1000 --smoke-test-steps 1 \
  --record-diagnostics --output /path/to/NEW_RECORDING_RUN
```

对照时只使用 SFT 中 clean 的 200 条轨迹，分别比较 PPO 成功/失败，区分训练 SDE 与评测 ODE；
不能将 clean 数据与整个八任务数据直接混合，也不能将 SFT 示范和 PPO 场景视为同 seed 的配对实验。
重点检查状态/动作对齐、关节目标、夹爪编码、动作跳变、超时长度、图像视角与奖励/阶段一致性。
