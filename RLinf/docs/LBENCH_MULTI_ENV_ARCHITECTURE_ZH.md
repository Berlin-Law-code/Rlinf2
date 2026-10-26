# LBench × LeRobot × RLinf 当前架构图

本文描述当前代码中的两条并存链路：

- LeRobot checkpoint 的原生并行评测链路使用 gRPC；
- RLinf Pi0.5 的 SFT、并行评测和 PPO 链路由 RLinf `EnvWorker` 通过 Unix
  domain socket 直接管理隔离的 Isaac/LBench 子进程。

LeRobot 源码不属于本次改动范围。RLinf 和 Isaac 使用不同 Python 环境，模型侧
GPU 与仿真侧 GPU 也必须显式隔离。

## 华丽版总览

![LBench、LeRobot 与 RLinf 多环境架构](assets/lbench_multi_env_architecture_neon.svg)

该 SVG 使用不同颜色表示不同数据流：青色为 observation/数据，紫色为模型，
橙色为动作与仿真，绿色为 PPO/数据回流，粉色为 LeRobot 原有 gRPC 评测链路。

## 1. 整体架构

```mermaid
flowchart LR
  subgraph DATA[数据与离线训练]
    LMDB[LBench 专家/回流 LMDB]
    CONVERT[LMDB 转换、筛选、合并]
    DS[(LeRobot 8-task data<br/>RLinf 用 v2.1；新版 LeRobot 用 v3 副本)]
    NORM[(8维 state/action<br/>norm_stats.json)]
    LRSFT[LeRobot 原生 SFT]
    RISFT[RLinf OpenPI Pi0.5 SFT<br/>FSDP]
    LRCKPT[(LeRobot checkpoint)]
    RICKPT[(RLinf Pi0.5 SFT checkpoint)]

    LMDB --> CONVERT --> DS
    DS --> LRSFT --> LRCKPT
    DS --> RISFT --> RICKPT
    DS --> NORM
    NORM --> RISFT
  end

  subgraph LEVAL[链路 A：LeRobot 原有并行评测]
    LSERVER[LeRobot/ResRL Policy Server<br/>独立推理环境]
    GRPC[gRPC]
    LECOORD[LBench 多任务评测调度器]
    LVEC[SimBoxSubprocVectorEnv]
    LENV1[Isaac worker 0]
    LENVN[Isaac worker N-1]
    LOUT[(LMDB、视频、manifest、指标)]

    LRCKPT --> LSERVER --> GRPC --> LECOORD --> LVEC
    LVEC --> LENV1
    LVEC --> LENVN
    LENV1 --> LOUT
    LENVN --> LOUT
  end

  subgraph RLINF[链路 B：RLinf 并行评测与强化微调]
    DRIVER[RLinf/Ray Driver]
    MODEL[Actor/Rollout GPU ranks<br/>OpenPI Pi0.5]
    ENVGROUP[EnvGroup / EnvWorker ranks]
    ADAPTER[LBenchEnv<br/>观测、动作、奖励适配]
    UVEC[SimBoxSubprocVectorEnv]
    SOCK[每环境独立 Unix Socket<br/>独立 worker 进程组]
    IENV1[Isaac/LBench worker 0<br/>独立 Python + SimulationApp]
    IENVN[Isaac/LBench worker N-1<br/>独立 Python + SimulationApp]
    COLLECT[CollectEpisode<br/>按 rank 分片、有界异步落盘]
    SHARDS[(pickle 或 LeRobot<br/>rank_r/id_n)]
    MERGE[离线合并与合同检查]
    PPO[PPO/GAE 更新<br/>FSDP Actor]
    SYNC[Patch Weight Sync]

    RICKPT --> MODEL
    DRIVER --> MODEL
    DRIVER --> ENVGROUP --> ADAPTER --> UVEC --> SOCK
    SOCK --> IENV1
    SOCK --> IENVN
    MODEL -- 20-action executed chunk --> ENVGROUP
    IENV1 -- 三相机、8维状态、reward、done --> ENVGROUP
    IENVN -- 三相机、8维状态、reward、done --> ENVGROUP
    ENVGROUP -- rollout + action_mask --> PPO
    PPO --> SYNC --> MODEL
    ENVGROUP --> COLLECT --> SHARDS --> MERGE --> DS
  end

  NORM --> MODEL
```

## 2. PPO 一个 global step 的闭环

```mermaid
sequenceDiagram
  participant D as Ray Driver
  participant R as Actor/Rollout ranks
  participant E as EnvWorker ranks
  participant S as Isaac subprocesses
  participant P as PPO Actor update

  D->>E: 开始 global_step=k
  loop rollout_epoch = 8
    E->>R: 三路图像 + 8维 state + instruction
    R-->>E: Pi0.5 生成 horizon=50
    Note over E,S: 只执行前 20 个动作
    loop 最多 20 个 primitive actions
      E->>S: 8维 joint/gripper action
      S-->>E: observation, progress, success/failure, done
    end
    Note over E: 提前终止部分通过 action_mask 排除
    E->>E: LIBERO-compatible first-success-only reward
    E->>E: 继续 chunk，直到 episode 结束或 1200 steps
  end
  E->>P: rollout batch + rewards + dones + action_mask
  P->>P: 在完整 rollout 上计算一次 GAE / returns
  loop update_epoch = 4
    P->>P: clipped PPO、value loss、backward / optimizer step
  end
  P-->>R: Patch weight synchronization
  R-->>D: global_step=k 完成
```

这里的两个计数不是一回事：

- `rollout_epoch=8`：每个 PPO `global_step` 收集 8 轮 rollout；
- `runner.max_steps=200`：最多进行 200 个“收集后再更新”的全局闭环。

## 3. 正式 8 环境、4 模型 GPU 的逻辑部署

```mermaid
flowchart TB
  subgraph MODELHOST[RLinf 模型环境：Torch、OpenPI、Ray]
    D[Ray Driver]
    R0[GPU/model rank 0<br/>Actor + Rollout + EnvWorker<br/>local env 0,1]
    R1[GPU/model rank 1<br/>Actor + Rollout + EnvWorker<br/>local env 2,3]
    R2[GPU/model rank 2<br/>Actor + Rollout + EnvWorker<br/>local env 4,5]
    R3[GPU/model rank 3<br/>Actor + Rollout + EnvWorker<br/>local env 6,7]
    D --> R0
    D --> R1
    D --> R2
    D --> R3
  end

  subgraph IPC[进程/虚拟环境边界]
    U0[Unix socket 0,1]
    U1[Unix socket 2,3]
    U2[Unix socket 4,5]
    U3[Unix socket 6,7]
  end

  subgraph SIMHOST[Isaac/LBench 环境：每个 worker 一个 SimulationApp]
    S0[Isaac worker 0]
    S1[Isaac worker 1]
    S2[Isaac worker 2]
    S3[Isaac worker 3]
    S4[Isaac worker 4]
    S5[Isaac worker 5]
    S6[Isaac worker 6]
    S7[Isaac worker 7]
  end

  R0 --> U0 --> S0
  U0 --> S1
  R1 --> U1 --> S2
  U1 --> S3
  R2 --> U2 --> S4
  U2 --> S5
  R3 --> U3 --> S6
  U3 --> S7
```

`GPU/model rank 0` 没有特殊的仿真职责；它只是分布式模型组的第一个 rank，通常
还承担日志、checkpoint 或集合通信中的协调工作。每个 `EnvWorker rank` 只负责
自己分到的局部环境，并为每个局部环境启动一个 Isaac 子进程。

每个外部 Isaac worker 使用独立进程组；超时或异常时会连同辅助进程一起清理。
长 episode 的异步导出队列也设置了上限，编码或磁盘变慢时通过背压限制内存增长。

模型 GPU 集合与 Isaac GPU 集合必须不重叠。例如模型侧使用 4 张卡、仿真侧使用
8 张卡时，可分别通过 RLinf 的 `CUDA_VISIBLE_DEVICES` 和
`LBENCH_ISAAC_GPUS` 显式指定。上图是逻辑进程映射，不代表已经在目标硬件上完成
显存和吞吐量验证。

## 4. 两种通信方式的边界

| 场景 | 模型在哪里 | 环境通信 | 是否经过 LeRobot gRPC |
|---|---|---|---|
| LeRobot checkpoint 并行评测 | LeRobot/ResRL policy server | LBench client 到 policy server 使用 gRPC；仿真 worker 由 LBench 多进程管理 | 是 |
| RLinf checkpoint 并行评测 | RLinf Actor/Rollout ranks | `EnvWorker -> LBenchEnv -> Unix Socket -> Isaac worker` | 否 |
| RLinf Pi0.5 PPO | RLinf Actor/Rollout ranks | 同上，额外回传 reward、done、`action_mask` | 否 |
| RLinf Pi0.5 SFT | RLinf FSDP Actor | 读取与当前 loader 匹配的 LeRobot 数据（当前为 v2.1），不启动 Isaac | 否 |

较早里程碑曾完成静态编译和 CPU/mock 合同检查；本轮 v2.1 数据、转换器及超时语义
加固后，按要求没有在当前机器重新运行测试。真实 Pi0.5、Ray 多 GPU、Isaac
SimulationApp 与端到端 checkpoint 仍需在资源足够的机器上分阶段验证。
