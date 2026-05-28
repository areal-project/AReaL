# 树训练

本文档描述 AReaL 的树训练功能，该功能通过在具有公共前缀的序列之间共享前缀计算来实现高效的 RL 训练。

## 概述

树训练是一种优化技术，利用批次中多个序列之间的前缀共享。与独立处理每个序列不同，具有共享前缀的序列被打包到树结构中，公共前缀 token 仅计算一次。

这对于代理 RL 训练特别有益，因为：

- 从同一提示采样多个回复（例如 `n_samples > 1`）
- 提示共享公共系统提示或 few-shot 示例
- 多轮交互共享对话历史前缀

通过仅计算一次共享前缀，树训练减少了 FLOPs 并提高了计算效率。在 tau2 示例中（参见 `examples/tau2/`），树训练将整体 FLOPs 降低高达 **10
倍**，并实现高达 **7 倍**的加速。

### 支持的后端

| 后端     | 状态 | 备注                                       |
| -------- | ---- | ------------------------------------------ |
| FSDP     | 支持 | 通过 FlexAttention 和块掩码                |
| Megatron | 支持 | 仅 `mbridge`（暂不支持 `megatron-bridge`） |
| Archon   | 支持 | 通过 TreeAttentionWrapper                  |

## 配置

### 启用树训练

通过 `TrainEngineConfig` 中的 `enable_tree_training` 选项启用树训练：

```yaml
actor:
  enable_tree_training: true
  pad_to_maximum: true # 树训练必须设为 true
  mb_spec:
    max_tokens_per_mb: 8192  # 树训练必须设置
```

### 必需配置

| 参数                        | 类型 | 必需 | 描述                              |
| --------------------------- | ---- | ---- | --------------------------------- |
| `enable_tree_training`      | bool | 是   | 启用基于树的序列打包              |
| `pad_to_maximum`            | bool | 是   | 树训练必须设为 `true`             |
| `mb_spec.max_tokens_per_mb` | int  | 是   | 每棵树的最大 token 数（必须设置） |

注意：启用树训练时，`max_tokens_per_mb` 必须是 `BLOCK_SIZE`（128）的倍数。

## 实现

### 树构建过程

```
输入序列              打包树                 注意力掩码

Seq0: [A, B, C, D]               [A]                  因果掩码，带
Seq1: [A, B, E, F]               / \                  树结构：
Seq2: [A, G, H]                [B] [G]                token 只能关注
                              / \    \                  其祖先
                           [C] [E]   [H]
                            |    |
                           [D]  [F]
```

树构建过程：

1. **提取序列**：使用 attention_mask 解析 input_ids 获取实际 token
1. **贪心打包**：使用首次适应递减策略将序列插入 trie
1. **Trie 压缩**：将线性链合并为单个压缩节点
1. **掩码生成**：为高效 FlexAttention 计算构建块掩码

### 数据结构

**关键文件：** `areal/models/tree_attn/tree.py`

#### TrieNode

`TrieNode` 数据类表示压缩前缀树中的节点：

```python
@dataclass
class TrieNode:
    tree_id: int           # 该节点所属的树标识符
    start_idx: int         # 扁平化表示中的起始索引
    end_idx: int           # 结束索引（包含）
    tokens: list[int]      # 存储在该节点中的 token ID
    sequence_ids: list[int]  # 经过的序列 ID
    children: dict[int, TrieNode]  # 按分叉 token 的子节点
    ancestors: list[TrieNode]      # 从根开始的祖先节点
    nodes: list[TrieNode]  # 所有后代节点（仅根节点）
```

对于根节点，`start_idx` 和 `end_idx` 为 -1，`nodes` 列表按前序遍历跟踪所有后代节点。

### 对数概率计算

**关键文件：** `areal/models/tree_attn/functional.py`

计算打包树的对数概率需要特殊处理，因为：

1. 不能通过简单滚动 `input_ids` 获取标签（序列共享位置）
1. 每个序列必须从树结构中恢复其原始 token 顺序
1. 使用缓存避免共享前缀的冗余计算

`gather_packed_tree_logprobs_entropy` 函数：

1. 遍历树中的每个序列
1. 对于序列经过的每个节点：
   - 计算内部对数概率（节点内的预测）
   - 计算转换对数概率（到子节点的预测）
1. 在节点级别缓存结果，供共享相同前缀的序列使用
1. 连接所有对数概率以重建每序列结果

### 注意力机制

树训练有两种注意力实现选择：

#### 带块掩码的 FlexAttention（默认）

使用 PyTorch 的 `torch.nn.attention.flex_attention` 和 `BlockMask`：

- 块大小：128 token（通过 `BLOCK_SIZE` 可配置）
- 对稀疏注意力模式的 GPU 计算高效
- 需要序列填充到块大小倍数

#### Triton 树注意力（实验性）

一种实验性的 Triton 实现，用于树注意力，更加节省内存和计算。通过 `AREAL_USE_TRITON_TREE_ATTN=1`
环境变量启用。请注意，此实现未经过充分测试。

## 引擎集成

### FSDP 引擎

**关键文件：** `areal/engine/fsdp_engine.py`

FSDP 集成使用猴子补丁将标准注意力替换为树注意力：

```python
# 在 FSDPEngine.initialize() 中
patch_fsdp_for_tree_training(enable=self.enable_tree_training)
```

在前向传播期间，使用 `build_tree_attn_kwargs()` 构建树注意力 kwargs：

```python
tree_attn_keys: list[str] = []
if self.enable_tree_training and ctx.trie_node is not None:
    padded_size = mb_item.padded_to_length
    assert padded_size is not None
    tree_kwargs = build_tree_attn_kwargs(
        ctx.trie_node, padded_size, self.device
    )
    inputs.update(tree_kwargs)
    tree_attn_keys = list(tree_kwargs.keys())
```

字典键为 `tree_block_mask` 或 `tree_triton_data`，取决于后端。

### Megatron 引擎

**关键文件：** `areal/engine/megatron_engine.py`

> **当前限制**：`MegatronEngine` 的树训练路径目前仅支持 `mbridge` 后端， 暂不支持 `megatron-bridge`。

Megatron 在模型创建期间使用 `patch_bridge_for_tree_training` 上下文管理器：

```python
with patch_bridge_for_tree_training(self.enable_tree_training):
    self.bridge = mbridge.AutoBridge.from_pretrained(self.config.path)
```

为兼容梯度检查点，使用密集注意力掩码（张量）而不是 BlockMask 对象，因为 `save_for_backward()` 只能序列化张量。

### Archon 引擎

**关键文件：** `areal/experimental/engine/archon_engine.py`、`archon_runner.py`

Archon 使用 `TreeAttentionMeta`，它在内部包装后端选择：

```python
# 在 SequentialRunner.run() 中
tree_attn_meta = None
if ctx.trie_node is not None:
    padded_size = mb_item.padded_to_length
    assert padded_size is not None
    tree_attn_meta = TreeAttentionMeta.from_trie(
        ctx.trie_node, padded_size, inputs["input_ids"].device
    )

logits = self.model(
    inputs["input_ids"],
    inputs["position_ids"],
    cu_seqlens=cu_seqlens,
    max_seqlen=max_seqlen,
    tree_attn_meta=tree_attn_meta,
)
```

## 指标和监控

树训练跟踪以下指标：

| 指标               | 描述                                   |
| ------------------ | -------------------------------------- |
| `tree_token_ratio` | 树 token 与原始 token 的比率（\< 1.0） |

较低的 `tree_token_ratio` 表示更多前缀共享和更高的效率增益。例如，比率为 0.6 意味着通过前缀共享节省了 40% 的 token。

## 约束和限制

### 当前限制

| 约束              | 描述                                           |
| ----------------- | ---------------------------------------------- |
| FSDP/Archon 无 PP | 流水线并行不支持树模式（FSDP 和 Archon）       |
| 树无 CP           | 上下文并行（CP > 1）与树模式不兼容（所有引擎） |
| Critic 不支持     | 尚不支持带 Critic 模型的树训练                 |

### 数值精度

FlexAttention 与标准注意力实现相比可能引入数值精度差异。这可能导致启用树训练时 Mixture of Experts（MoE）模型的训练不稳定。如果 MoE
架构训练不稳定，请考虑禁用树训练。
