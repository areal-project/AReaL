# Archon：PyTorch 原生训练引擎

## 概述

Archon 是 AReaL 的 PyTorch 原生训练后端，为 RL 研究人员提供最大的灵活性，无需 Megatron-Core 依赖。它使用 PyTorch
原生分布式原语支持完整的 5D 并行性（DP、TP、PP、CP、EP），使添加 RL 特定的优化和调试分布式训练问题变得更加容易。

**易于入门**：只需运行 `uv sync` 即可安装所有依赖项。与需要 C++ 编译包（如 `transformer_engine`）的 MegatronEngine
不同，Archon 仅使用纯 Python 包，无需复杂的构建步骤。

Archon 的设计和核心实现灵感来自 [torchtitan](https://github.com/pytorch/torchtitan)，这是 PyTorch
官方的大规模 LLM 训练参考实现。我们感谢 torchtitan 团队在通过纯 PyTorch API 使分布式训练易于访问方面的出色工作。

## 引擎对比

| 特性          | FSDPEngine          | MegatronEngine                     | ArchonEngine                |
| ------------- | ------------------- | ---------------------------------- | --------------------------- |
| 后端          | HuggingFace + FSDP2 | Megatron-Core                      | PyTorch 原生                |
| 模型来源      | 任意 HF 模型        | Megatron 模型                      | 自定义 Archon 模型          |
| torch.compile | 有限                | 否                                 | 是（默认）                  |
| 数据并行      | FSDP2               | Megatron DP                        | FSDP2                       |
| 张量并行      | PyTorch DTensor     | Megatron TP                        | PyTorch DTensor             |
| 流水线并行    | 否                  | 是（VPP）                          | 是（1F1B、I1F1B、IZB、ZBV） |
| 专家并行      | 否                  | 完整 EP/ETP                        | 完整 EP/ETP                 |
| 上下文并行    | Ulysses SP          | Megatron CP                        | Ulysses SP                  |
| 支持的模型    | 任意 HF             | 通过 bridge 后端（默认 `mbridge`） | 内置 + 用户自定义           |
| 状态          | 生产就绪            | 生产就绪                           | 实验性                      |

## 关键特性

- **PyTorch 原生实现**：无需 Megatron-Core 依赖，仅使用 PyTorch 分布式原语（DTensor、DeviceMesh、FSDP2）
- **完整的并行性支持**：DP、TP、PP、CP、EP 和 ETP，配置灵活
- **默认启用 torch.compile**：通过 Inductor 编译优化性能
- **灵活的激活检查点**：支持 `none`、`full`、`selective` 和 `memory_budget` 模式
- **原生 RL 训练支持**：内置 PPO Actor/Critic 实现
- **流水线并行调度**：1F1B、Interleaved1F1B、InterleavedZeroBubble（ZB1P）和 ZBVZeroBubble 调度

## 启用 Archon

要使用 Archon 作为训练后端，请在 `actor.backend` 字段中指定：

```bash
rollout.backend=sglang:d4 actor.backend=archon:d4
```

### 支持的模型

Archon 为以下模型类型提供内置支持：

- `qwen2` - Qwen2 密集模型
- `qwen3` - Qwen3 密集模型
- `qwen3_moe` - Qwen3 MoE 模型

对于没有自定义实现的不支持模型，请改用 FSDPEngine 或 MegatronEngine。

### 添加自定义模型

用户可以通过创建新的模型规范来添加自定义模型实现。关键组件包括：

1. **模型类** (`nn.Module`)：模型架构实现
1. **ModelArgs 类**：模型配置的 dataclass，带有 `from_hf_config()` 方法用于从 HuggingFace 配置转换
1. **StateDictAdapter 类**：在 HuggingFace 和 Archon 权重格式之间转换
1. **Parallelize 函数**：应用 TP、CP、EP、FSDP 和激活检查点
1. **ModelSpec**：将所有组件注册在一起

示例结构（请参阅 `areal/experimental/models/archon/qwen3/` 作为参考）：

```
areal/experimental/models/archon/your_model/
├── __init__.py
├── spec.py                    # ModelSpec 注册
├── model/
│   ├── model.py               # 模型类
│   ├── args.py                # ModelArgs dataclass
│   └── state_dict_adapter.py  # 权重转换
└── infra/
    └── parallelize.py         # 并行化逻辑
```

在 `areal/experimental/models/archon/__init__.py` 中注册您的模型规范：

```python
from areal.experimental.models.archon.your_model import spec  # noqa: F401
```

> **提示**：AI 驱动的编码工具（如 Claude Code、OpenCode）可以帮助加速这一过程。使用 `/add-archon-model`
> 技能获取半自动化指南，分析 HuggingFace
> 源代码并生成实现框架。请参阅[AI 辅助开发指南](../reference/ai_assisted_dev.md)了解设置和使用方法。

## 并行性配置

Archon 使用与 Megatron 相同的并行语法。请参阅[分配模式参考](../reference/alloc_mode.md)获取完整的语法指南。

基本示例：

```bash
# 密集模型：4 DP × 2 PP × 2 TP = 16 GPU
rollout.backend=sglang:d4t2 actor.backend=archon:d4p2t2
```

### MoE 支持

与 FSDPEngine 不同，Archon 提供完整的 MoE 支持，包括专家并行（EP）和专家张量并行（ETP）。对于 MoE 模型，您可以使用混合并行，为注意力模块和
FFN（专家）模块使用单独的配置：

```bash
# 带混合并行的 MoE 模型
rollout.backend=sglang:d4t4 actor.backend=archon:(attn:d1p4t2c2|ffn:d1p4t1e4)
```

这启用了[MoE 并行折叠](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding)，降低了组合上下文和专家并行性的
GPU 要求。

## 高级配置

Archon 特定的选项在 `actor.archon.*` 下配置：

| 选项                           | 默认值            | 描述                                                    |
| ------------------------------ | ----------------- | ------------------------------------------------------- |
| `pp_schedule`                  | `Interleaved1F1B` | PP 调度：`1F1B`、`I1F1B`、`IZB` 或 `ZBV`                |
| `enable_compile`               | `True`            | 启用 torch.compile                                      |
| `ac_mode`                      | `selective`       | 激活检查点模式                                          |
| `offload_params`               | `False`           | 将 FSDP 参数卸载到 CPU                                  |
| `use_deterministic_algorithms` | `False`           | 可重现性确定性训练（请参阅[下文](#deterministic-mode)） |

请参阅[性能调优](#performance-tuning)获取这些选项的详细指南。

(performance-tuning)=

## 性能调优

### torch.compile

Archon 默认启用 `torch.compile` 以优化性能。启用编译后，会自动设置 `pad_to_maximum=True` 以避免 Inductor
的动态形状问题。

禁用编译（用于调试或不支持的操作）：

```bash
+actor.archon.enable_compile=False
```

### 激活检查点选择

根据您的内存约束选择合适的 AC 模式：

| 模式            | 内存使用 | 重计算   | 使用场景               |
| --------------- | -------- | -------- | ---------------------- |
| `none`          | 最高     | 无       | 小模型，内存充足       |
| `selective`     | 中等     | 部分     | 默认，平衡权衡         |
| `full`          | 最低     | 所有层   | 大模型，内存受限       |
| `memory_budget` | 可配置   | 自动调优 | 细粒度控制（需要编译） |

对于 `memory_budget` 模式，调整 `ac_memory_budget`（0.0 = 最大重计算，1.0 = 无重计算）：

```bash
+actor.archon.ac_mode=memory_budget +actor.archon.ac_memory_budget=0.5
```

## 限制

Archon 引擎的当前限制：

- **PP 不支持权重绑定**：具有 `tie_word_embeddings=True` 的模型无法使用流水线并行（PP > 1），因为嵌入层和输出层在不同 GPU 上
- **树训练**：尚不支持（`enable_tree_training` 将显示警告）
- **实验性状态**：API 可能在未来版本中更改

## 调试技巧

### 查看并行配置

Archon 在初始化时记录并行维度：

```
Initialized Archon engine with parallel dims: pp=2, dp_shard=4, tp=2, cp=1, ep=1, etp=1
```

### 常见问题

| 问题                       | 可能原因               | 解决方案                                               |
| -------------------------- | ---------------------- | ------------------------------------------------------ |
| 微批次间形状不匹配         | PP 下序列长度可变      | 设置 `pad_to_maximum=True`                             |
| 编译期间 OOM               | torch.compile 内存开销 | 尝试 `+actor.archon.enable_compile=False`              |
| "tie_word_embeddings" 错误 | PP 与权重绑定模型      | 使用 PP=1 或更换模型                                   |
| 第一次迭代慢               | torch.compile 预热     | 预期行为，后续迭代会更快                               |
| 运行间非确定性损失         | MoE 中 GPU 级非确定性  | 设置 `+actor.archon.use_deterministic_algorithms=True` |

### 激活检查点调试

启用 AC 调试以捕获详细信息（较慢）：

```bash
+actor.archon.ac_debug=True
```

(deterministic-mode)=

### 确定性模式

由于 GPU 级非确定性（matmul、NCCL 集体约简和 torch.compile
代码生成），模型可能在训练运行之间表现出非确定性行为。这使得调试训练不稳定变得困难——您无法判断损失峰值是来自算法更改还是随机的硬件噪声。

启用确定性模式以消除这些变化源：

```bash
+actor.archon.use_deterministic_algorithms=True
```

这将设置：

- `torch.use_deterministic_algorithms(True, warn_only=True)` — 强制 PyTorch
  在可能的情况下使用确定性算法变体
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` — 确定性的 cuBLAS matmul 工作区
- `NCCL_ALGO=Ring` — 确定性的 NCCL 集体约简
- `TORCH_COMPILE_DETERMINISTIC=1` — 确定性的 Inductor 代码生成（启用编译时）
- `ac_config.preserve_rng_state=True` — 确定性的激活检查点重计算

## 从 FSDPEngine 迁移

从 FSDPEngine 迁移到 Archon：

### 1. 更新 actor.backend

```bash
# 之前（FSDPEngine）
rollout.backend=sglang:d4t2 actor.backend=fsdp:d8t2

# 之后（Archon）
rollout.backend=sglang:d4t2 actor.backend=archon:d8t2
```

### 2. 配置映射

| FSDPEngine 选项          | Archon 对应选项               |
| ------------------------ | ----------------------------- |
| `gradient_checkpointing` | 相同（全局控制 AC）           |
| 无                       | `actor.archon.ac_mode`        |
| 无                       | `actor.archon.enable_compile` |
| 无                       | `actor.archon.pp_schedule`    |

### 3. 模型兼容性

确保您的模型受 Archon 支持（qwen2、qwen3、qwen3_moe）或实现自定义模型规范。

### 4. 新功能

使用 Archon，您可以访问：

- 流水线并行（`p` 维度）
- MoE 的专家并行（`e` 维度）
- torch.compile 优化
- 灵活的激活检查点模式

## 示例

### 密集模型（Qwen3-8B）

创建配置文件 `archon_qwen3_8b.yaml`：

```yaml
# Archon config for Qwen3-8B on 3 nodes (24 GPUs)
# SGLang: 4 replicas × 2 TP = 8 GPUs
# Archon: 4 DP × 2 PP × 2 TP = 16 GPUs

experiment_name: archon-gsm8k-grpo
trial_name: trial-0

cluster:
  n_nodes: 3
  n_gpus_per_node: 8

rollout:
  backend: "sglang:d4t2"
actor:
  backend: "archon:d4p2t2"

scheduler:
  type: ray

actor:
  path: Qwen/Qwen3-8B
  gradient_checkpointing: true
  archon:
    pp_schedule: Interleaved1F1B
    enable_compile: true
    ac_mode: selective
```

运行实验：

```bash
python3 examples/math/gsm8k_rl.py --config archon_qwen3_8b.yaml
```

### MoE 模型（Qwen3-30B-A3B）

创建配置文件 `archon_qwen3_moe.yaml`：

```yaml
# Archon config for Qwen3-30B-A3B MoE on 4 nodes (32 GPUs)
# SGLang: 4 replicas × 4 TP = 16 GPUs
# Archon: 1 DP × 4 PP × (attn: TP2×CP2, ffn: TP1×EP4) = 16 GPUs

experiment_name: archon-moe-gsm8k-grpo
trial_name: trial-0

cluster:
  n_nodes: 4
  n_gpus_per_node: 8

rollout:
  backend: "sglang:d4t4"
actor:
  backend: "archon:(attn:d1p4t2c2|ffn:d1p4t1e4)"

scheduler:
  type: ray

actor:
  path: Qwen/Qwen3-30B-A3B
  gradient_checkpointing: true
  archon:
    pp_schedule: Interleaved1F1B
    enable_compile: true
    ac_mode: selective
```

运行实验：

```bash
python3 examples/math/gsm8k_rl.py --config archon_qwen3_moe.yaml
```

## 另请参阅

- [分配模式参考](../reference/alloc_mode.md) — 分配模式语法的完整指南
- [大型 MoE 模型微调](megatron.md) — MoE 模型的 MegatronEngine 替代方案
