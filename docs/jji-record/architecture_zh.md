# 架构概览

本文档从高层视角解释 AReaL 的工作原理，以及各个核心组件的作用 —— 特别是不同的**训练引擎**、**推理引擎**和**权重同步机制**。

适合希望在选择后端或排查分布式 RL 训练问题前，先理解系统设计的用户阅读。

## 1. 高层架构

AReaL 采用 **单控制器（single-controller）** 设计：

- 你的训练脚本（例如 `examples/math/gsm8k_rl.py`）运行在控制器进程上。
- 控制器通过 RPC 和 PyTorch 分布式原语协调两大子系统：
  - **Rollout（推理）侧**：使用 SGLang 或 vLLM 生成轨迹。
  - **训练侧**：使用训练引擎运行 actor（以及可选的 critic/ref）。
- 数据通过 **RTensor** 在两侧流动，这是一种分布式张量抽象：控制器只持有元数据，实际张量分片留在工作进程上。

典型训练步骤：

1. **Rollout 阶段**：控制器调用 `actor.prepare_batch(...)` → `RolloutController` 将 prompt 发送给推理
   worker → 返回 `RTensor` batch。
1. **优势函数 / 损失计算**：可选的 reference、teacher 或 recompute-logp 过程。
1. **训练阶段**：`TrainController` 通过 `data_parallel_dispatch()` 将 batch 分发给训练 worker。
1. 训练引擎执行优化器步骤。
1. **权重更新**：`actor.update_weights(meta)` 将最新权重同步到 rollout 引擎。

**训练引擎**、**推理后端**和**权重更新模式**（`weight_update_mode`）的选择会对性能、可扩展性和功能可用性（LoRA、MoE、流水线并行等）产生巨大影响。

## 2. 训练引擎

AReaL 目前提供三种训练后端，通过 `actor.backend` 指定（例如 `fsdp:d8`、`megatron:d8`、`archon:d8`）。

### FSDPEngine（大多数用户的默认选择）

- 基于 Hugging Face Transformers + PyTorch **FSDP2**。
- 使用 `DTensor` 实现张量并行、上下文并行（Ulysses）和专家并行。
- **优点**：易于使用任意 Hugging Face 模型，适合中等规模训练，调试相对简单。
- **局限**：没有原生流水线并行；在超大模型上的扩展性不如 5D 系统强。
- 推荐场景：大多数新项目、快速实验、Transformers 支持良好的模型。

### MegatronEngine

- 基于 **Megatron-Core**（NVIDIA），通过 `mbridge` 或 `megatron-bridge` 适配层接入。
- 支持完整的 **5D 并行**：DP、TP、PP（含 Virtual PP）、CP、EP/ETP，并支持 MoE 的混合折叠（hybrid folding）。
- **优点**：在大模型（尤其是 MoE）上扩展性和效率最佳；成熟的流水线和专家并行实现。
- **权衡**：需要通过 bridge 层提供特定模型实现；配置更复杂；部分功能（LoRA、树注意力）受 bridge 限制。
- 参考：
  - [Megatron 教程](../tutorial/megatron.md)
  - [Bridge Backend 参考文档](bridge_backend.md)
  - [Allocation Mode 参考文档](alloc_mode.md)

### ArchonEngine（实验性）

- 纯 PyTorch 原生实现，灵感来自 torchtitan。
- 使用 FSDP2 + DTensor 实现大部分维度 + 自定义流水线调度（1F1B、交错、Zero-Bubble 等变体）。
- 支持完整的 5D 并行（DP/TP/PP/CP/EP/ETP），无需 Megatron-Core。
- **优点**：默认开启 `torch.compile`，对 RL 研究者更易扩展，无需复杂的 C++ 编译依赖。
- **当前状态**：实验性 / 部分工作负载已可用于生产。模型支持正在快速增加（Qwen2/3 dense + MoE）。
- 参考：[Archon 教程](../tutorial/archon.md)

### 快速对比

| 维度               | FSDPEngine           | MegatronEngine           | ArchonEngine                      |
| ------------------ | -------------------- | ------------------------ | --------------------------------- |
| 基础技术           | HF + FSDP2 + DTensor | Megatron-Core + bridge   | 纯 PyTorch（DTensor + 自定义）    |
| 流水线并行         | 无                   | 有（VPP）                | 有（多种调度）                    |
| 专家并行（MoE）    | 有限                 | 完整 + folding           | 完整 + folding                    |
| torch.compile      | 有限                 | 无                       | 默认开启                          |
| 添加模型的容易程度 | 极佳（任意 HF 模型） | 中等（需 bridge）        | 成长中（自定义 spec 系统）        |
| 最佳适用场景       | 大多数用户、中小模型 | 大规模 MoE、追求极致性能 | 希望使用纯 PyTorch 灵活性的研究者 |

## 3. 推理 / Rollout 引擎

Rollout worker 负责在 RL 过程中生成 token。

AReaL 支持两大主流推理框架：

- **SGLang**（主要推荐，集成度最高）
- **vLLM**（良好备选，功能略有差异）

### Remote 引擎（最常用）

`RemoteSGLangEngine` / `RemoteVLLMEngine` 会启动独立的 server 进程（或 pod），通过 HTTP +
分布式集合通信进行权重更新。

支持高级特性，例如：

- 生成侧的流水线并行（配合每 PP rank 的权重更新 group）。
- 自定义调度和前缀缓存。

### 进程内 / 实验性选项

`areal/experimental/inference_service/` 下存在一些更紧耦合的实验路径，但大多数用户仍使用 remote 引擎。

## 4. 权重同步机制

这是异步 RL 训练中最重要（也最微妙）的一部分。

Actor 更新权重后，rollout 引擎必须在下一轮生成前看到新权重。AReaL 提供三种策略，由 `actor.weight_update_mode` 和
`WeightUpdateMeta` 控制。

### 1. `disk`（最简单、最鲁棒）

- Actor 将 checkpoint（HF 或 DCP 格式）保存到共享存储。
- Rollout 引擎加载该 checkpoint。
- **优点**：极其可靠，跨任意 scheduler 都能工作，易于调试。
- **缺点**：延迟高、I/O 开销大，不适合高频权重更新。
- 以下场景必须使用：许多配置下的 LoRA、actor 与 rollout 共址（colocated）调度。

### 2. `xccl`（当前默认 —— 传统 NCCL Broadcast）

- 训练引擎（rank 0 / PP head）创建包含推理 worker 的自定义 NCCL 进程组。
- 参数被物化为完整张量后通过 NCCL **broadcast**（`dist.broadcast`）。
- 支持精细分块（`weight_chunked_mem_mb`）和 CUDA stream 流水线（FSDP 路径）。
- 当生成侧使用流水线并行时，会创建 per-PP-rank group（避免与 SGLang PP scheduler 死锁）。
- **优点**：一旦 group 建立，延迟极低；GPU 直连。
- **缺点**：FSDP/Megatron/Archon 三处存在重复的复杂代码；对 group 初始化顺序和 rank-0 语义有微妙要求；长期维护困难。

> **注意**：AReaL 正在积极现代化这条路径。迁移计划详见
> `docs/plans/legacy_xccl_broadcast_weight_update_migration.md`。

### 3. `awex`（新 —— Async Weight Exchange）

- 位于 `areal/experimental/weight_update/` 下的新一代架构。
- 使用 controller + gateway + 各 worker 适配器，通过丰富的参数元数据（`awex` transfer plan）协商 P2P
  `batch_isend_irecv` 传输。
- 支持共址和分离部署模式，并有更好的显存控制能力。
- **优点**：职责分离更清晰，更易于未来演进，对高级特性（共址、精细显存释放）支持更好。
- **缺点**：仍在成熟过程中；尚未成为所有模型/并行组合的默认选择。

### 如何选择权重更新模式

| 场景                            | 推荐模式           | 原因                                     |
| ------------------------------- | ------------------ | ---------------------------------------- |
| 首次运行 / 调试                 | `disk`             | 最可靠                                   |
| 高吞吐训练（不使用 LoRA）       | `xccl`（当前默认） | 最低延迟                                 |
| 大规模 MoE + 高频更新           | `xccl` 或 `awex`   | 视稳定性需求而定                         |
| 使用 LoRA                       | `disk`（常为必需） | 许多推理引擎对 LoRA + 分布式更新支持有限 |
| Actor 与 rollout 共址在同一 GPU | `disk`             | NCCL group 冲突常见                      |
| 希望使用现代化架构              | `awex`             | 项目未来发展方向                         |

在配置中选择模式：

```yaml
actor:
  weight_update_mode: xccl   # 或 "disk" / "awex"
```

`PPOTrainer` 等 Trainer 会据此构造对应的
`WeightUpdateMeta`（`from_fsdp_xccl`、`from_megatron_xccl`、`from_awex` 或 `from_disk`），并传递给
`connect_engine`。

## 5. 其他主要子系统

- **Allocation
  Mode**（`alloc_mode`）：声明式语法（`fsdp:d8t2`、`megatron:(attn:d1p4t2|ffn:d1p4t1e4)`
  等），决定每个引擎组件的 GPU 数量和并行策略。详见 [Allocation Mode 参考文档](alloc_mode.md)。
- **Scheduler / Launcher**：`local`、`ray` 或 `slurm`，负责进程放置和资源获取。
- **Workflows**：`RolloutWorkflow` 实现（例如 `RLVRWorkflow`）定义单次 episode
  的生成逻辑，包括多轮对话、工具调用或自定义奖励塑造。
- **Reward Functions**：可插拔的奖励函数（数学、几何、代码等），按名称注册，在 rollout 完成后由控制器执行。

## 6. 把所有部分串起来

一个典型的 GRPO 生产运行通常使用：

- `rollout.backend: sglang:...`
- `actor.backend: fsdp:...` 或 `megatron:...`
- `actor.weight_update_mode: xccl`
- 自定义或内置的 `RolloutWorkflow`
- 集群上使用 `scheduler.type: ray` 或 `slurm`

Trainer 负责整体协调，选定的训练引擎拥有优化器和反向传播，推理引擎负责生成，权重更新路径保持两侧一致。

随着 AReaL 演进，`awex` 权重更新系统以及引擎接口的进一步统一，预计将减少当前 `xccl` broadcast 路径的重复和复杂性。

## 延伸阅读

- [GSM8K GRPO 代码 walkthrough](../tutorial/gsm8k_grpo.md)
- [Allocation Mode 参考](alloc_mode.md)
- [Rollout Workflow 参考](rollout_workflow.md)
- 引擎专项教程：[Megatron](../tutorial/megatron.md)、[Archon](../tutorial/archon.md)
- 最佳实践：[Handling OOM](handling_oom.md)、[Performance Profiling](perf_profiling.md)

如果你正在贡献新的引擎或权重更新策略，请同时阅读面向开发者的文档以及 `docs/plans/` 下的已批准迁移计划。
