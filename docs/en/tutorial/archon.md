# Archon: PyTorch-Native Training Engine

## Overview

Archon is AReaL's PyTorch-native training backend that provides maximum flexibility for
RL researchers without Megatron-Core dependencies. It supports full 5D parallelism (DP,
TP, PP, CP, EP) using PyTorch's native distributed primitives, making it easier to add
RL-specific optimizations and debug distributed training issues.

**Easy to get started**: Simply run `uv sync` to install all dependencies. Unlike
MegatronEngine which requires C++ compiled packages like `transformer_engine`, Archon
uses only pure Python packages with no complex build steps.

The design and core implementation of Archon are inspired by
[torchtitan](https://github.com/pytorch/torchtitan), PyTorch's official reference
implementation for large-scale LLM training. We thank the torchtitan team for their
excellent work in making distributed training accessible through pure PyTorch APIs.

## Engine Comparison

| Feature           | FSDPEngine          | MegatronEngine           | ArchonEngine                |
| ----------------- | ------------------- | ------------------------ | --------------------------- |
| Backend           | HuggingFace + FSDP2 | Megatron-Core            | PyTorch-native              |
| Model Source      | Any HF model        | Megatron models          | Custom Archon models        |
| torch.compile     | Limited             | No                       | Yes (default)               |
| Data Parallel     | FSDP2               | Megatron DP              | FSDP2                       |
| Tensor Parallel   | PyTorch DTensor     | Megatron TP              | PyTorch DTensor             |
| Pipeline Parallel | No                  | Yes (VPP)                | Yes (1F1B, I1F1B, IZB, ZBV) |
| Expert Parallel   | No                  | Full EP/ETP              | Full EP/ETP                 |
| Context Parallel  | Ulysses SP          | Megatron CP              | Ulysses SP                  |
| Supported Models  | Any HF              | mbridge/ megatron-bridge | Built-in + User-defined     |
| Status            | Production          | Production               | Experimental                |

## Key Features

- **PyTorch-native implementation**: No Megatron-Core dependency, using only PyTorch
  distributed primitives (DTensor, DeviceMesh, FSDP2)
- **Full parallelism support**: DP, TP, PP, CP, EP, and ETP with flexible configuration
- **torch.compile by default**: Optimized performance with Inductor compilation
- **Flexible activation checkpointing**: Supports `none`, `full`, `selective`, and
  `memory_budget` modes
- **Native RL training support**: Built-in PPO Actor/Critic implementations
- **Pipeline parallel schedules**: 1F1B, Interleaved1F1B, InterleavedZeroBubble (ZB1P),
  and ZBVZeroBubble schedules

## Enabling Archon

To use Archon as your training backend, specify it in the `actor.backend` field:

```bash
rollout.backend=sglang:d4 actor.backend=archon:d4
```

### Supported Models

Archon provides built-in support for the following model types:

- `qwen2` - Qwen2 dense models
- `qwen3` - Qwen3 dense models
- `qwen3_moe` - Qwen3 MoE models

For unsupported models without custom implementations, use FSDPEngine or MegatronEngine
instead.

### Adding Custom Models

Users can add custom model implementations by creating a new model spec. The key
components are:

1. **Model class** (`nn.Module`): The model architecture implementation
1. **ModelArgs class**: Dataclass for model configuration, with `from_hf_config()`
   method to convert from HuggingFace config
1. **StateDictAdapter class**: Converts between HuggingFace and Archon weight formats
1. **Parallelize function**: Applies TP, CP, EP, FSDP, and activation checkpointing
1. **ModelSpec**: Registers all components together

Example structure (see `areal/experimental/models/archon/qwen3/` for reference):

```
areal/experimental/models/archon/your_model/
├── __init__.py
├── spec.py                    # ModelSpec registration
├── model/
│   ├── model.py               # Model class
│   ├── args.py                # ModelArgs dataclass
│   └── state_dict_adapter.py  # Weight conversion
└── infra/
    └── parallelize.py         # Parallelization logic
```

Register your model spec in `areal/experimental/models/archon/__init__.py`:

```python
from areal.experimental.models.archon.your_model import spec  # noqa: F401
```

> **Tip**: AI-powered coding tools (e.g., Claude Code, OpenCode) can help accelerate the
> process. Use the `/add-archon-model` skill for a semi-automated guide that analyzes
> HuggingFace source code and generates implementation scaffolding. See the
> [AI-Assisted Development Guide](../reference/ai_assisted_dev.md) for setup and usage.

## Parallelism Configuration

Archon uses the same parallelism syntax as Megatron. See
[Allocation Mode Reference](../reference/alloc_mode.md) for the complete syntax guide.

Basic example:

```bash
# Dense model: 4 DP × 2 PP × 2 TP = 16 GPUs
rollout.backend=sglang:d4t2 actor.backend=archon:d4p2t2
```

### MoE Support

Unlike FSDPEngine, Archon provides full MoE support with Expert Parallelism (EP) and
Expert Tensor Parallelism (ETP). For MoE models, you can use hybrid parallelism with
separate configurations for attention and FFN (expert) modules:

```bash
# MoE model with hybrid parallelism
rollout.backend=sglang:d4t4 actor.backend=archon:(attn:d1p4t2c2|ffn:d1p4t1e4)
```

This enables
[MoE Parallel Folding](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding),
reducing GPU requirements for combined context and expert parallelism.

## Advanced Configuration

Archon-specific options are configured under `actor.archon.*`:

| Option                         | Default           | Description                                                                   |
| ------------------------------ | ----------------- | ----------------------------------------------------------------------------- |
| `pp_schedule`                  | `Interleaved1F1B` | PP schedule: `1F1B`, `I1F1B`, `IZB`, or `ZBV`                                 |
| `enable_compile`               | `True`            | Enable torch.compile                                                          |
| `ac_mode`                      | `selective`       | Activation checkpointing mode                                                 |
| `offload_params`               | `False`           | Offload FSDP parameters to CPU                                                |
| `reshard_after_forward_policy` | `default`         | FSDP reshard after forward (`default`/`always`/`never`)                       |
| `use_deterministic_algorithms` | `False`           | Deterministic training for reproducibility (see [below](#deterministic-mode)) |

See [Performance Tuning](#performance-tuning) for detailed guidance on these options.

(performance-tuning)=

## Performance Tuning

### torch.compile

Archon enables `torch.compile` by default for optimized performance. When compile is
enabled, `pad_to_maximum=True` is automatically set to avoid dynamic shape issues with
Inductor.

To disable compilation (useful for debugging or unsupported operations):

```bash
+actor.archon.enable_compile=False
```

### Activation Checkpointing Selection

Choose the appropriate AC mode based on your memory constraints:

| Mode            | Memory Usage | Recomputation | Use Case                                |
| --------------- | ------------ | ------------- | --------------------------------------- |
| `none`          | Highest      | None          | Small models, sufficient memory         |
| `selective`     | Medium       | Partial       | Default, balanced trade-off             |
| `full`          | Lowest       | All layers    | Large models, memory constrained        |
| `memory_budget` | Configurable | Auto-tuned    | Fine-grained control (requires compile) |

For `memory_budget` mode, adjust `ac_memory_budget` (0.0 = max recompute, 1.0 = no
recompute):

```bash
+actor.archon.ac_mode=memory_budget +actor.archon.ac_memory_budget=0.5
```

## Limitations

Current limitations of Archon Engine:

- **Weight tying not supported with PP**: Models with `tie_word_embeddings=True` cannot
  use Pipeline Parallelism (PP > 1) because embeddings and output layers are on
  different GPUs
- **Tree training**: Not yet supported (`enable_tree_training` will show a warning)
- **Experimental status**: APIs may change in future releases

## Debugging Tips

### Viewing Parallel Configuration

Archon logs parallel dimensions at initialization:

```
Initialized Archon engine with parallel dims: pp=2, dp_shard=4, tp=2, cp=1, ep=1, etp=1
```

### Common Issues

| Issue                              | Possible Cause                    | Solution                                              |
| ---------------------------------- | --------------------------------- | ----------------------------------------------------- |
| Shape mismatch across microbatches | Variable sequence lengths with PP | Set `pad_to_maximum=True`                             |
| OOM during compilation             | torch.compile memory overhead     | Try `+actor.archon.enable_compile=False`              |
| "tie_word_embeddings" error        | PP with weight-tied model         | Use PP=1 or different model                           |
| Slow first iteration               | torch.compile warmup              | Expected behavior, subsequent iterations faster       |
| Non-deterministic loss across runs | GPU-level non-determinism in MoE  | Set `+actor.archon.use_deterministic_algorithms=True` |

### Activation Checkpointing Debug

Enable AC debugging to capture detailed information (slower):

```bash
+actor.archon.ac_debug=True
```

(deterministic-mode)=

### Deterministic Mode

Models can exhibit non-deterministic behavior across training runs due to GPU-level
non-determinism in matmuls, NCCL collective reductions, and torch.compile code
generation. This makes debugging training instability difficult — you cannot tell
whether a loss spike is from your algorithm change or random hardware noise.

Enable deterministic mode to eliminate these sources of variance:

```bash
+actor.archon.use_deterministic_algorithms=True
```

This sets:

- `torch.use_deterministic_algorithms(True, warn_only=True)` — forces PyTorch to use
  deterministic algorithm variants where available
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` — deterministic cuBLAS matmul workspace
- `NCCL_ALGO=Ring` — deterministic NCCL collective reductions
- `TORCH_COMPILE_DETERMINISTIC=1` — deterministic Inductor code generation (when compile
  is enabled)
- `ac_config.preserve_rng_state=True` — deterministic activation checkpointing recompute

## Migration from FSDPEngine

To migrate from FSDPEngine to Archon:

### 1. Update actor.backend

```bash
# Before (FSDPEngine)
rollout.backend=sglang:d4t2 actor.backend=fsdp:d8t2

# After (Archon)
rollout.backend=sglang:d4t2 actor.backend=archon:d8t2
```

### 2. Configuration Mapping

| FSDPEngine Option        | Archon Equivalent             |
| ------------------------ | ----------------------------- |
| `gradient_checkpointing` | Same (controls AC globally)   |
| N/A                      | `actor.archon.ac_mode`        |
| N/A                      | `actor.archon.enable_compile` |
| N/A                      | `actor.archon.pp_schedule`    |

### 3. Model Compatibility

Ensure your model is supported by Archon (qwen2, qwen3, qwen3_moe) or implement a custom
model spec.

### 4. New Capabilities

With Archon, you gain access to:

- Pipeline Parallelism (`p` dimension)
- Expert Parallelism for MoE (`e` dimension)
- torch.compile optimization
- Flexible activation checkpointing modes

## Examples

### Dense Model (Qwen3-8B)

Create a config file `archon_qwen3_8b.yaml`:

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

scheduler:
  type: ray

actor:
  backend: "archon:d4p2t2"
  path: Qwen/Qwen3-8B
  gradient_checkpointing: true
  archon:
    pp_schedule: Interleaved1F1B
    enable_compile: true
    ac_mode: selective
```

Run the experiment:

```bash
python3 examples/math/gsm8k_rl.py --config archon_qwen3_8b.yaml
```

### MoE Model (Qwen3-30B-A3B)

Create a config file `archon_qwen3_moe.yaml`:

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

scheduler:
  type: ray

actor:
  backend: "archon:(attn:d1p4t2c2|ffn:d1p4t1e4)"
  path: Qwen/Qwen3-30B-A3B
  gradient_checkpointing: true
  archon:
    pp_schedule: Interleaved1F1B
    enable_compile: true
    ac_mode: selective
```

Run the experiment:

```bash
python3 examples/math/gsm8k_rl.py --config archon_qwen3_moe.yaml
```

## See Also

- [Allocation Mode Reference](../reference/alloc_mode.md) - Complete guide on allocation
  mode syntax
- [Fine-tuning Large MoE Models](megatron.md) - MegatronEngine alternative for MoE
  models

(appendix-pipeline-parallelism-memory-guide)=

## Appendix: Pipeline Parallelism Memory Guide

Pipeline parallelism (PP) in the Archon engine introduces unique memory challenges
compared to pure data parallelism. This appendix explains the root causes and practical
mitigations.

### A.1 Microbatch Count and Warmup Accumulation

Interleaved PP schedules (e.g., `Interleaved1F1B`, `InterleavedZeroBubble`) have a
**warmup phase** that accumulates multiple forward passes before any backward pass runs.
When `n_microbatches < num_total_stages`, most or all forward passes pile up before the
first backward, causing peak GPU memory to spike far beyond what the steady-state 1F1B
phase requires.

For example, with `pp_size=2` and `stages_per_rank=2` (`num_total_stages=4`):

| `mb_spec.n_mbs` | Actual microbatches          | Warmup forwards (rank 0) | Peak activation sets    | Per-set size |
| --------------- | ---------------------------- | ------------------------ | ----------------------- | ------------ |
| 1 (default)     | 2 (auto-raised to `pp_size`) | 3                        | 4 (all before backward) | `batch / 2`  |
| 4 (recommended) | 4                            | 3                        | 4 (transient)           | `batch / 4`  |
| 8               | 8                            | 3                        | 4 (transient)           | `batch / 8`  |

While the peak count of in-flight activation sets stays the same (~`num_total_stages`),
each set shrinks proportionally with more microbatches.

**Fix:** Set `mb_spec.n_mbs` to at least `num_total_stages`:

```yaml
actor:
  mb_spec:
    n_mbs: 4  # >= pp_size * stages_per_rank
```

```{note}
AReaL automatically raises `n_mbs` to `num_total_stages` when it is too low and logs a
warning. To silence the warning and ensure optimal splitting, set `n_mbs` explicitly.
```

### A.2 Zero Bubble Schedules and `retain_graph`

Zero bubble schedules (`InterleavedZeroBubble`, `ZBVZeroBubble`, `DualPipeV`) split each
backward pass into two phases:

- **I step** (`stage_backward_input`): computes input gradients with `retain_graph=True`
- **W step** (`stage_backward_weight`): computes weight gradients, then releases the
  graph

The I step must keep the forward computation graph alive (`retain_graph=True`) because
the W step still needs it. This single design choice cascades into several memory
penalties:

| Consequence                    | Why                                                                   | Memory impact                                |
| ------------------------------ | --------------------------------------------------------------------- | -------------------------------------------- |
| Activations live longer        | Graph between I->W cannot be freed                                    | +15--20 GB (model-dependent)                 |
| `donated_buffer` disabled      | Donated buffers are freed after backward, conflicts with retain_graph | Backward temp buffers cannot be reused       |
| `torch.compile` disabled       | Compile's donated buffer optimization has the same conflict           | Lose Inductor memory optimizations           |
| Op-level selective AC unusable | Per-op cache is consumed by I step, nothing left for W step           | Must use full AC or layer-level selective AC |

Non-zero-bubble schedules (`1F1B`, `Interleaved1F1B`) perform backward in a single pass
without `retain_graph=True`, so **none of these penalties apply**. If memory is tight
and you do not need zero-bubble throughput, switching to `Interleaved1F1B` is the
simplest mitigation:

```yaml
actor:
  archon:
    pp_schedule: Interleaved1F1B  # no split backward, no retain_graph overhead
```

If you need zero-bubble throughput but IZB causes OOM, **try `ZBVZeroBubble` first**.
ZBV uses a V-shape stage assignment that is significantly more memory-friendly than
IZB's interleaved assignment:

|                         | IZB (interleaved)    | ZBV (V-shape)            |
| ----------------------- | -------------------- | ------------------------ |
| Rank 0 stages (4 total) | \[0, 2\] (same side) | \[0, 3\] (opposite ends) |
| Rank 1 stages (4 total) | \[1, 3\] (same side) | \[1, 2\] (opposite ends) |

The V-shape co-locates the first and last pipeline stages on the same rank. This matters
because the last stage produces the loss directly -- its backward can start
**immediately** after forward with no cross-rank communication. In ZBV's warmup, chunk1
activations follow an `F->I->W` pattern where each activation is created and freed
locally, never piling up.

IZB's interleaved assignment places all of a rank's stages on the same side of the
pipeline. Backward requires gradient propagation from downstream ranks, creating a real
bubble where warmup activations sit in memory waiting. This difference -- typically a
few GB -- can be decisive at the OOM boundary.

```yaml
actor:
  archon:
    pp_schedule: ZBVZeroBubble  # V-shape: less warmup memory than IZB
```

```{note}
`ZBVZeroBubble` requires exactly 2 stages per rank (`stages_per_rank=2`).
```

### A.3 FSDP Parameter Resharding

With PP enabled, FSDP defaults to keeping parameters unsharded after forward
(`reshard_after_forward=False`) to avoid redundant all-gather communication per
microbatch. This trades memory for speed -- each rank holds the full (unsharded)
parameters of its assigned layers, adding ~`model_params_per_rank * (1 - 1/dp_shard)` in
bf16.

Override with `reshard_after_forward_policy: always` if communication overhead is
acceptable:

```yaml
actor:
  archon:
    reshard_after_forward_policy: always  # reshard after each forward, saves memory
```

### A.4 Gradient Accumulation Overhead (FSDP + PP)

This is an inherent cost of combining FSDP with PP and applies to **all** PP schedules
(not just zero bubble).

PyTorch's PP scheduler disables gradient synchronization
(`set_requires_gradient_sync(False)`) and parameter resharding
(`set_reshard_after_backward(False)`) for all backward microbatches except the last one.
This means gradients accumulate in **unsharded fp32** form across microbatches rather
than being reduce-scattered immediately.

For a model with `P` parameters per rank, this adds up to `P * 4 bytes` (fp32) of
gradient memory. For example, a 30B MoE model with PP=2 holds ~13.5B parameters per
rank, resulting in ~54 GB of unsharded gradient buffers during the backward phase.

This overhead **cannot be reduced by AReaL configuration alone** -- the only mitigation
is to reduce parameters per rank via TP or EP.

### A.5 When to Add TP/EP

If OOM persists after tuning `n_mbs`, `reshard_after_forward_policy`, and activation
checkpointing, the model likely exceeds the per-rank memory budget. Add tensor
parallelism (`t2` or `t4`) or expert parallelism (`e2`, `e4`) to reduce parameters per
rank. For MoE models, EP is preferred because expert weights typically dominate model
size:

```yaml
# Before: archon:d2p2 (PP only, OOM)
# After: archon:d1p2e2 (PP + EP, fits in memory)
rollout:
  backend: "sglang:d4"
actor:
  backend: "archon:d1p2e2"
```

### A.6 Activation Checkpointing with PP

Full AC (`gradient_checkpointing: true`) is strongly recommended with PP since the
warmup phase holds activations from multiple forward passes simultaneously.

For zero bubble schedules, the AC mode is further constrained:

| AC mode                               | Zero bubble compatible | Notes                                       |
| ------------------------------------- | ---------------------- | ------------------------------------------- |
| `full`                                | Yes                    | Recommended for maximum memory savings      |
| `selective` (layer-level, e.g. `"2"`) | Yes                    | Good balance of speed and memory            |
| `selective` (`"op"`)                  | No                     | Per-op cache conflicts with split backward  |
| `memory_budget`                       | No                     | Depends on torch.compile, which is disabled |

### A.7 Memory Budget Rule of Thumb

```{tip}
Each rank needs memory for:
1. Sharded model parameters: `model_size / (dp_shard * tp * ep)` in bf16
2. Unsharded gradients during backward: `model_size / (tp * ep * pp)` in fp32
3. Optimizer states: `2 * model_size / (dp_shard * tp * ep)` in fp32 (AdamW)
4. Activations: ~`num_total_stages * (batch_tokens / n_mbs) * hidden_dim` in bf16

If the sum exceeds GPU memory, increase TP, EP, or PP to reduce per-rank load.
```
