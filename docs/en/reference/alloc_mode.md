# Allocation Mode

This document describes AReaL's allocation mode system, which controls how GPUs are
distributed between inference and training backends during distributed RL training.

## Overview

Each engine component (actor, critic, rollout, ref, teacher) has its own `backend`
configuration field that specifies:

- Which backend to use (SGLang, vLLM for inference; FSDP, Megatron, Archon for training)
- The parallelization strategy
- The total number of GPUs required

AReaL parses each `backend` string into a `ModelAllocation` object that drives resource
allocation for that specific engine.

## Configuration

### Per-Engine Backend Fields

Each engine in the YAML config has its own `backend` field:

```yaml
# Rollout (inference) engine
rollout:
  backend: "sglang:d4t2"

# Actor (training) engine
actor:
  backend: "fsdp:d8"

# Critic engine (falls back to actor.backend if empty)
critic:
  backend: ""

# Ref engine (falls back to actor.backend if empty)
ref:
  backend: ""
```

When `critic.backend` or `ref.backend` is empty, it automatically inherits from
`actor.backend`.

> **Note:** The top-level `allocation_mode` config field is deprecated and only retained
> for backward compatibility with legacy SPMD launchers (local/ray/slurm). It is ignored
> by the single-controller scheduler. Use the per-engine `backend` fields shown above
> instead.

### Backend String Syntax

```
<backend>:<parallelism_dims>
```

For example, `fsdp:d4t2` means: use the FSDP backend with data parallelism 4 and tensor
parallelism 2.

### Parallelism Dimensions

| Dimension | Abbreviation | Description                        | Valid For        |
| --------- | ------------ | ---------------------------------- | ---------------- |
| Data      | `d`          | Number of model replicas           | All backends     |
| Tensor    | `t`          | Split operations across GPUs       | All backends     |
| Pipeline  | `p`          | Split layers across GPUs in stages | Megatron, Archon |
| Context   | `c`          | Split sequence length across GPUs  | All backends     |
| Expert    | `e`          | Split MoE experts across GPUs      | Megatron, Archon |

Dimensions are specified as `<abbrev><size>`, e.g., `d4t2` means data parallel size 4
and tensor parallel size 2.

## Calculating GPU Requirements

The total GPUs for a component is computed as:

```
world_size = dp × tp × pp × cp
```

Expert parallelism (`e`) does not increase world size—it redistributes how experts are
placed within the existing GPU mesh.

### Examples

| Backend String      | GPUs per Engine | Notes                       |
| ------------------- | --------------- | --------------------------- |
| `fsdp:d8`           | 8               | 8 data-parallel replicas    |
| `sglang:d2t4`       | 8               | 2 instances × 4 TP GPUs     |
| `megatron:d2p2t4`   | 16              | 2 DP × 2 PP × 4 TP          |
| `megatron:d2p2t4e4` | 16              | Same mesh, 4-way expert par |

### Full Config Example

```yaml
# 16-GPU setup: 8 inference + 8 training
rollout:
  backend: "sglang:d2t4"    # 2 × 4 = 8 GPUs
actor:
  backend: "fsdp:d4t2"      # 4 × 2 = 8 GPUs
```

## Backend Selection

### Inference Backends

| Backend  | Supported Dimensions |
| -------- | -------------------- |
| `sglang` | `d`, `t`, `p`        |
| `vllm`   | `d`, `t`, `p`        |

For inference, `d` represents the number of independent server instances, and each
instance uses `t × p` GPUs.

Note that the internal backend configurations do not affect how AReaL allocates GPUs.
Given `rollout.backend: "sglang:d4t4"`, you can also configure `sglang.dp_size=4`,
`sglang.ep_size=4`, and `sglang.enable_dp_attention=True`. In this case, we launch 4
model replicas each with 4 GPUs. Within each instance, SGLang will still use DP
attention and expert parallelism to distribute computations in attention and expert
layers.

### Training Backends

| Backend    | Supported Dimensions    | Use Case                                 |
| ---------- | ----------------------- | ---------------------------------------- |
| `fsdp`     | `d`, `t`, `c`           | Default for simple parallelism           |
| `megatron` | `d`, `t`, `p`, `c`, `e` | Required for pipeline or expert parallel |
| `archon`   | `d`, `t`, `p`, `c`, `e` | Alternative to Megatron (experimental)   |

> **Important**: An explicit backend prefix is **required** in all allocation strings.
> Bare dimension strings (e.g., `d4t2`) are no longer accepted. Always specify the
> backend explicitly: `fsdp:d4t2`, `megatron:d2p2t4`, `sglang:d4t2`.

## MoE Hybrid Parallelism

For Mixture-of-Experts models, Megatron/Archon supports different parallelism strategies
for attention and FFN (expert) modules using the hybrid syntax:

```
megatron:(attn:<attn_dims>|ffn:<ffn_dims>)
```

This enables
[MoE Parallel Folding](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding),
which reduces the minimum GPU requirement for combined context and expert parallelism.

### Constraints

- Pipeline parallel size (`p`) must be identical for `attn` and `ffn`
- World size must match (if `d` is omitted in `ffn`, it is derived automatically)
- Expert parallel (`e`) is only valid in the `ffn` section

### Example

```yaml
actor:
  backend: "megatron:(attn:d4p2t2c2|ffn:d2p2t4e2)"
```

| Module | dp  | pp  | tp  | cp  | ep  | World Size |
| ------ | --- | --- | --- | --- | --- | ---------- |
| attn   | 4   | 2   | 2   | 2   | -   | 32         |
| ffn    | 2   | 2   | 4   | -   | 2   | 32         |

## See Also

- [Fine-tuning Large MoE Models](../tutorial/megatron.md) - Tutorial for Megatron
  backend
- [Archon: PyTorch-Native Training Engine](../tutorial/archon.md) - Tutorial for Archon
  backend
- [Megatron Performance Best Practice](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#performance-best-practice)
