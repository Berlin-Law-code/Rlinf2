LBench RECAP pi0.5 推理包，global_step_30000

来源：lbench_recap_round1/policy_h50_e20/checkpoints/global_step_30000
导出前已确认 .checkpoint_complete。权重保持训练保存的 float32 格式。

文件说明
  model_state_dict/full_weights.pt：完整策略权重，可由 RLinf CFG loader 直接读取。
  openpi_assets.json：H50/E20、动作维数、预处理和归一化文件索引。
  lerobot_episode_stats/norm_stats.json：两组训练数据共同计算的共享 norm。
  lbench/lbench_multitask_joint/norm_stats.json：相同 norm 的规范资产路径。
  inference_config.yaml：可迁移的策略配置，不依赖原训练数据目录。
  assets/openpi_cache/big_vision/paligemma_tokenizer.model：本地 tokenizer。
  load_policy.py：加载示例及使用合成输入的推理检查命令。
  inference_validation.json：打包前的真实权重加载和推理检查结果。
  export_manifest.json：来源、框架版本和设置。
  runtime_reference/：本机 OpenPI/CFG 接入代码快照，供迁移时核对。
  SHA256SUMS：文件校验清单。

本包需要兼容的 RLinf/OpenPI Python 环境；不是自带 Python/CUDA 的容器。
本机环境：/root/autodl-tmp/.venvs/rlinf-openpi
本机 RLinf：/root/autodl-tmp/Berlin/RLinf
另一台机器需要安装匹配的 RLinf/OpenPI 和 RLinf 所需的 Transformers 补丁。
具体版本见 export_manifest.json；RLinf 的相关本地修改保存在 runtime_reference/。
runtime_reference/ 是接入代码快照，不会由加载脚本自动覆盖已安装代码。

解包与校验（先进入压缩包所在目录）
  sha256sum -c lbench_recap_pi05_h50_e20_step30000_inference.zip.sha256
  unzip lbench_recap_pi05_h50_e20_step30000_inference.zip
  cd lbench_recap_pi05_h50_e20_step30000_inference
  sha256sum -c SHA256SUMS

使用本机环境检查模型（不会连接机器人）
  USE_TF=0 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  PYTHONPATH=/root/autodl-tmp/Berlin/RLinf \
  /root/autodl-tmp/.venvs/rlinf-openpi/bin/python load_policy.py --smoke

在现有程序中加载
  from load_policy import load_policy
  model = load_policy(device="cuda:0")

调用 model.predict_action_batch(observation, mode="eval")，推荐与训练一致，
在 torch.inference_mode() 和 torch.autocast("cuda", dtype=torch.bfloat16)
上下文中运行。load_policy 自动将 model_path 设置为本包所在绝对路径，
并使用包内 tokenizer 缓存和 norm，无需手动指定原服务器路径。

observation 输入契约
  main_images：obs_camera 的 RGB uint8 图像，形状 [B,H,W,3]。
  wrist_images：obs_camera_2，格式同上。
  extra_view_images：realsense，格式同上。三路相机顺序必须与训练一致。
  states：float32 [B,8]，按训练定义输入原始关节/夹爪状态，不预先归一化；
          夹爪状态为 0=closed、1=open。
  task_descriptions：长度 B 的任务文本列表。
  返回 actions 为 numpy [B,20,8]，已按包内 norm 反归一化的绝对动作目标。
  网络预测 50 步，接口返回前 20 步，8 个真实动作维度。

RECAP 推理设置
  model_type=cfg_model
  config_name=pi05_lbench_joint
  guidance_type=positive
  positive_only_conditional=true
  cfgrl_guidance_scale=1.0
  num_steps=10（流匹配积分步数）
  lerobot_compatible=true
使用 CFG loader 保留 Advantage: positive 条件，并使用本包 norm。
norm 已在训练时变更为两组数据的共享统计，不能换回原 SFT 的旧 norm。
本次不打包优化器状态、分布式训练分片、价值模型、数据集和训练日志。
inference_validation.json 的合成输入检查验证加载与输出形状，不代表任务成功率。
