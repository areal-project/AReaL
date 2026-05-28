# 分配模式

本文档描述 AReaL 的分配模式系统，该系统控制分布式 RL 训练期间 GPU 在推理和训练后端之间的分配方式。

## 概述

每个引擎组件（actor、critic、rollout、ref、teacher）都有独立的 `backend` 配置字段，用于指定：

- 使用哪个后端（SGLang、vLLM 用于推理；FSDP、Megatron、Archon 用于训练）
- 并行化策略
- 所需的 GPU 数量

AReaL 将每个 `backend` 字符串解析为 `ModelAllocation` 对象，驱动该特定引擎的资源分配。

## 配置

### 每引擎 Backend 字段

YAML 配置中每个引擎都有独立的 `backend` 字段：

```yaml
# Rollout（推理）引擎
rollout:
  backend: "sglang:d4t2"

# Actor（训练）引擎
actor:
  backend: "fsdp:d8"

# Critic 引擎（为空时回退到 actor.backend）
critic:
  backend: ""

# Ref 引擎（为空时回退到 actor.backend）
ref:
  backend: ""
```

当 `critic.backend` 或 `ref.backend` 为空时，会自动继承 `actor.backend` 的值。

> **注意：** 顶层的 `allocation_mode` 配置字段已弃用，仅为旧版 SPMD
> 启动器（local/ray/slurm）保留向后兼容性。此字段被单控制器调度器忽略。请使用上述各引擎的 `backend` 字段。

### Backend 字符串语法

```
<backend>:<parallelism_dims>
```

例如，`fsdp:d4t2` 表示：使用 FSDP 后端，数据并行大小为 4，tensor 并行大小为 2。

### 并行维度

| 维度     | 缩写 | 描述                 | 适用于           |
| -------- | ---- | -------------------- | ---------------- |
| Data     | `d`  | 模型副本数量         | 所有后端         |
| Tensor   | `t`  | 跨 GPU 分割操作      | 所有后端         |
| Pipeline | `p`  | 跨 GPU 阶段分割层    | Megatron、Archon |
| Context  | `c`  | 跨 GPU 分割序列长度  | 所有后端         |
| Expert   | `e`  | 跨 GPU 分割 MoE 专家 | Megatron、Archon |

维度指定为 `<缩写><大小>`，例如 `d4t2` 表示数据并行大小为 4，tensor 并行大小为 2。

## 计算 GPU 需求

组件的 GPU 总数计算如下：

```
world_size = dp × tp × pp × cp
```

专家并行（`e`）不会增加 world size —— 它在现有 GPU 网格内重新分配专家的放置位置。

### 示例

| Backend 字符串      | 每引擎 GPU 数 | 说明                   |
| ------------------- | ------------- | ---------------------- |
| `fsdp:d8`           | 8             | 8 个数据并行副本       |
| `sglang:d2t4`       | 8             | 2 个实例 × 4 TP GPU    |
| `megatron:d2p2t4`   | 16            | 2 DP × 2 PP × 4 TP     |
| `megatron:d2p2t4e4` | 16            | 同一网格，4 路专家并行 |

### 完整配置示例

```yaml
# 16 GPU 配置：8 推理 + 8 训练
rollout:
  backend: "sglang:d2t4"    # 2 × 4 = 8 GPU
actor:
  backend: "fsdp:d4t2"      # 4 × 2 = 8 GPU
```

## 后端选择

### 推理后端

| 后端     | 支持的维度    |
| -------- | ------------- |
| `sglang` | `d`, `t`, `p` |
| `vllm`   | `d`, `t`, `p` |

对于推理，`d` 表示独立服务器实例的数量，每个实例使用 `t × p` 个 GPU。

请注意，内部后端配置不影响 AReaL 如何分配 GPU。给定 `rollout.backend: "sglang:d4t4"`，你还可以配置
`sglang.dp_size=4`、`sglang.ep_size=4` 和 `sglang.enable_dp_attention=True`。在这种情况下，我们启动 4
个模型副本，每个副本 4 个 GPU。在每个实例内，SGLang 仍然使用 DP attention 和专家并行来分配注意力层和专家层中的计算。

### 训练后端

| 后端       | 支持的维度              | 使用场景                      |
| ---------- | ----------------------- | ----------------------------- |
| `fsdp`     | `d`, `t`, `c`           | 简单并行的默认选择            |
| `megatron` | `d`, `t`, `p`, `c`, `e` | 流水线或专家并行必需          |
| `archon`   | `d`, `t`, `p`, `c`, `e` | Megatron 的替代方案（实验性） |

> **重要**：所有分配字符串都必须显式指定后端前缀。 裸维度字符串（如
> `d4t2`）不再被接受。请始终显式指定后端：`fsdp:d4t2`、`megatron:d2p2t4`、`sglang:d4t2`。

## MoE 混合并行

对于 Mixture-of-Experts 模型，Megatron/Archon 支持使用混合语法为注意力和 FFN（专家）模块使用不同的并行策略：

```
megatron:(attn:<attn_dims>|ffn:<ffn_dims>)
```

这启用了
[MoE Parallel Folding](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding)，可以降低组合上下文和专家并行的最低
GPU 要求。

### 约束

- 流水线并行大小（`p`）对 `attn` 和 `ffn` 必须相同
- World size 必须匹配（如果 `ffn` 中省略 `d`，则自动派生）
- 专家并行（`e`）仅在 `ffn` 部分有效

### 示例

```yaml
actor:
  backend: "megatron:(attn:d4p2t2c2|ffn:d2p2t4e2)"
```

| 模块 | dp  | pp  | tp  | cp  | ep  | World Size |
| ---- | --- | --- | --- | --- | --- | ---------- |
| attn | 4   | 2   | 2   | 2   | -   | 32         |
| ffn  | 2   | 2   | 4   | -   | 2   | 32         |

## 另见

- [Fine-tuning Large MoE Models](../tutorial/megatron.md) - Megatron 后端教程
- [Archon: PyTorch-Native Training Engine](../tutorial/archon.md) - Archon 后端教程
- [Megatron Performance Best Practice](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#performance-best-practice)
