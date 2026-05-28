# Tree Training

This document describes AReaL's tree training feature, which enables efficient RL
training by sharing prefix computations across sequences with common prefixes.

## Overview

Tree training is an optimization technique that exploits prefix sharing among multiple
sequences in a batch. Instead of processing each sequence independently, sequences with
shared prefixes are packed into a tree structure where common prefix tokens are computed
only once.

This is particularly beneficial for agentic RL training where:

- Multiple responses are sampled from the same prompt (e.g., `n_samples > 1`)
- Prompts share common system prompts or few-shot examples
- Multi-turn interactions share conversation history prefixes

By computing shared prefixes only once, tree training reduces FLOPs and improves compute
efficiency. In the tau2 example (see `examples/tau2/`), tree training reduces overall
FLOPs by up to **10x** and achieves up to **7x** acceleration.

### Supported Backends

| Backend  | Status    | Notes                                                   |
| -------- | --------- | ------------------------------------------------------- |
| FSDP     | Supported | Via FlexAttention with block masks                      |
| Megatron | Supported | `mbridge` only (`megatron-bridge` is not supported yet) |
| Archon   | Supported | Via TreeAttentionWrapper                                |

## Configuration

### Enabling Tree Training

Enable tree training via the `enable_tree_training` option in `TrainEngineConfig`:

```yaml
actor:
  enable_tree_training: true
  pad_to_maximum: true # Must be set to true for tree training
  mb_spec:
    max_tokens_per_mb: 8192  # Must be set for tree training
```

### Required Configuration

| Parameter                   | Type | Required | Description                        |
| --------------------------- | ---- | -------- | ---------------------------------- |
| `enable_tree_training`      | bool | Yes      | Enable tree-based sequence packing |
| `pad_to_maximum`            | bool | Yes      | Must be `true` for tree training   |
| `mb_spec.max_tokens_per_mb` | int  | Yes      | Max tokens per tree (must be set)  |

NOTE: When tree training is enabled `max_tokens_per_mb` must be a multiple of
`BLOCK_SIZE` (128).

## Implementation

### Tree Building Process

```
Input Sequences              Packed Tree               Attention Mask

Seq0: [A, B, C, D]               [A]                    Causal mask with
Seq1: [A, B, E, F]               / \                    tree structure:
Seq2: [A, G, H]                [B] [G]                  tokens can attend to
                              / \    \                  their ancestors only
                           [C] [E]   [H]
                            |    |
                           [D]  [F]
```

The tree building process:

1. **Extract sequences**: Parse input_ids using attention_mask to get actual tokens
1. **Greedy packing**: Insert sequences into tries using first-fit decreasing strategy
1. **Trie compression**: Merge linear chains into single compressed nodes
1. **Mask generation**: Build block masks for efficient FlexAttention computation

### Data Structures

**Key file:** `areal/models/tree_attn/tree.py`

#### TrieNode

The `TrieNode` dataclass represents a node in the compressed prefix tree:

```python
@dataclass
class TrieNode:
    tree_id: int           # Identifier of the tree this node belongs to
    start_idx: int         # Starting index in flattened representation
    end_idx: int           # Ending index (inclusive)
    tokens: list[int]      # Token IDs stored in this node
    sequence_ids: list[int]  # IDs of sequences passing through
    children: dict[int, TrieNode]  # Child nodes by diverging token
    ancestors: list[TrieNode]      # Ancestor nodes from root
    nodes: list[TrieNode]  # All descendant nodes in pre-order (root only)
```

For root nodes, `start_idx` and `end_idx` are -1, and the `nodes` list tracks all
descendant nodes in pre-order traversal.

### Log Probability Computation

**Key file:** `areal/models/tree_attn/functional.py`

Computing log probabilities for packed trees requires special handling because:

1. Labels cannot be obtained by simply rolling `input_ids` (sequences share positions)
1. Each sequence must recover its original token order from the tree structure
1. Caching is used to avoid redundant computation for shared prefixes

The `gather_packed_tree_logprobs_entropy` function:

1. Iterates over each sequence in the tree
1. For each node the sequence passes through:
   - Computes internal logprobs (predictions within the node)
   - Computes transition logprobs (predictions to child nodes)
1. Caches results at node level for sequences sharing the same prefix
1. Concatenates all logprobs to reconstruct per-sequence results

### Attention Mechanisms

Tree training has two choices for attention implementation:

#### FlexAttention with Block Masks (Default)

Uses PyTorch's `torch.nn.attention.flex_attention` with `BlockMask`:

- Block size: 128 tokens (configurable via `BLOCK_SIZE`)
- Efficient for GPU computation with sparse attention patterns
- Requires sequences padded to block size multiples

#### Triton Tree Attention (Experimental)

An experimental Triton implementation for tree attention that is more memory and
computationally efficient. Enable via the `AREAL_USE_TRITON_TREE_ATTN=1` environment
variable. Note that this implementation is not thoroughly tested.

## Engine Integration

### FSDP Engine

**Key file:** `areal/engine/fsdp_engine.py`

FSDP integration uses monkey patching to replace standard attention with tree attention:

```python
# In FSDPEngine.initialize()
patch_fsdp_for_tree_training(enable=self.enable_tree_training)
```

During forward pass, tree attention kwargs are built using `build_tree_attn_kwargs()`:

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

The dict keys are `tree_block_mask` or `tree_triton_data` depending on the backend.

### Megatron Engine

**Key file:** `areal/engine/megatron_engine.py`

> **Current limitation**: Tree training in `MegatronEngine` currently supports only the
> `mbridge` backend. `megatron-bridge` is not supported in this path yet.

Megatron uses `patch_bridge_for_tree_training` context manager during model creation:

```python
with patch_bridge_for_tree_training(self.enable_tree_training):
    self.bridge = mbridge.AutoBridge.from_pretrained(self.config.path)
```

For gradient checkpointing compatibility, dense attention masks (tensors) are used
instead of BlockMask objects, since `save_for_backward()` can only serialize tensors.

### Archon Engine

**Key files:** `areal/experimental/engine/archon_engine.py`, `archon_runner.py`

Archon uses `TreeAttentionMeta` which wraps the backend selection internally:

```python
# In SequentialRunner.run()
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

## Metrics and Monitoring

Tree training tracks the following metrics:

| Metric             | Description                                      |
| ------------------ | ------------------------------------------------ |
| `tree_token_ratio` | Ratio of tree tokens to original tokens (\< 1.0) |

A lower `tree_token_ratio` indicates more prefix sharing and greater efficiency gains.
For example, a ratio of 0.6 means 40% of tokens were saved through prefix sharing.

## Constraints and Limitations

### Current Limitations

| Constraint             | Description                                                            |
| ---------------------- | ---------------------------------------------------------------------- |
| No PP with FSDP/Archon | Pipeline parallelism not supported with tree mode (FSDP and Archon)    |
| No CP with tree        | Context parallelism (CP > 1) incompatible with tree mode (all engines) |
| Critic not supported   | Tree training with critic models not yet implemented                   |

### Numerical Accuracy

FlexAttention may introduce numerical precision differences compared to standard
attention implementations. This can cause training instability with Mixture of Experts
(MoE) models when tree training is enabled. If you experience unstable training with MoE
architectures, consider disabling tree training.
