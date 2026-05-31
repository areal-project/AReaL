# Architecture Overview

This document provides a high-level explanation of how AReaL works and the role of each
major component — especially the different **training engines**, **inference engines**,
and **weight synchronization mechanisms**.

It is intended for users who want to understand the system design before choosing a
backend or debugging distributed RL training.

## 1. High-Level Architecture

AReaL follows a **single-controller** design:

- Your training script (e.g. `examples/math/gsm8k_rl.py`) runs on a controller process.
- The controller orchestrates two main subsystems via RPC and PyTorch distributed
  primitives:
  - **Rollout (Inference) side** — generates trajectories using SGLang or vLLM.
  - **Training side** — runs the actor (and optionally critic/ref) using a training
    engine.
- Data moves between sides using **RTensor**, a distributed tensor abstraction where the
  controller only holds metadata while the actual tensor shards stay on the workers.

Typical training step:

1. **Rollout phase**: Controller calls `actor.prepare_batch(...)` → `RolloutController`
   sends prompts to inference workers → returns an `RTensor` batch.
1. **Advantage / loss computation**: Optional reference, teacher, or recompute-logp
   passes.
1. **Training phase**: `TrainController` dispatches the batch to training workers via
   `data_parallel_dispatch()`.
1. **Optimizer step** on the training engine.
1. **Weight update**: `actor.update_weights(meta)` synchronizes the latest weights to
   the rollout engines.

The choice of **training engine**, **inference backend**, and **weight update mode**
(`weight_update_mode`) dramatically affects performance, scalability, and feature
availability (LoRA, MoE, pipeline parallelism, etc.).

## 2. Training Engines

AReaL currently offers three training backends, selected via `actor.backend` (e.g.
`fsdp:d8`, `megatron:d8`, `archon:d8`).

### FSDPEngine (Default for most users)

- Built on Hugging Face Transformers + PyTorch **FSDP2** (Fully Sharded Data Parallel).
- Uses `DTensor` for tensor, context (Ulysses sequence parallel), and expert
  parallelism.
- **Strengths**: Easy to use with any Hugging Face model, good for moderate-scale
  training, straightforward debugging.
- **Limitations**: No native pipeline parallelism; weaker scaling on very large models
  compared to 5D systems.
- Recommended for: Most new projects, quick experimentation, models that are
  well-supported in Transformers.

### MegatronEngine

- Built on **Megatron-Core** (NVIDIA) with the `mbridge` or `megatron-bridge` adapter
  layer.
- Full **5D parallelism**: Data Parallel (DP), Tensor Parallel (TP), Pipeline Parallel
  (PP with Virtual PP), Context Parallel (CP), Expert Parallel (EP/ETP) with hybrid
  folding support for MoE.
- **Strengths**: Best scaling and efficiency for large (especially MoE) models; mature
  pipeline and expert parallelism.
- **Trade-offs**: Requires specific model implementations via bridge layers; more
  complex configuration; some features (LoRA, tree attention) have bridge-specific
  limitations.
- See:
  - [Megatron Tutorial](../tutorial/megatron.md)
  - [Bridge Backend Reference](bridge_backend.md)
  - [Allocation Mode Reference](alloc_mode.md)

### ArchonEngine (Experimental)

- Pure PyTorch-native implementation inspired by torchtitan.
- Uses FSDP2 + DTensor for most dimensions + custom pipeline schedules (1F1B,
  Interleaved, Zero-Bubble variants).
- Supports full 5D (DP/TP/PP/CP/EP/ETP) without Megatron-Core.
- **Strengths**: `torch.compile` by default, easier to extend for RL researchers, no
  heavy C++ build dependencies.
- **Current status**: Experimental / production for some workloads. Model support is
  growing (Qwen2/3 dense + MoE).
- See: [Archon Tutorial](../tutorial/archon.md)

### Quick Comparison

| Aspect                | FSDPEngine                        | MegatronEngine           | ArchonEngine                                   |
| --------------------- | --------------------------------- | ------------------------ | ---------------------------------------------- |
| Base technology       | HF + FSDP2 + DTensor              | Megatron-Core + bridge   | Pure PyTorch (DTensor + custom)                |
| Pipeline Parallel     | No                                | Yes (VPP)                | Yes (multiple schedules)                       |
| Expert Parallel (MoE) | Limited                           | Full + folding           | Full + folding                                 |
| torch.compile         | Limited                           | No                       | Yes (default)                                  |
| Ease of adding models | Excellent (any HF)                | Medium (via bridge)      | Growing (custom spec system)                   |
| Best for              | Most users, smaller-medium models | Large MoE, maximum scale | Researchers wanting PyTorch-native flexibility |

## 3. Inference / Rollout Engines

Rollout workers are responsible for generating tokens during RL.

AReaL supports two major inference frameworks:

- **SGLang** (primary, best integration)
- **vLLM** (good fallback, slightly different feature set)

### Remote Engines (most common)

`RemoteSGLangEngine` / `RemoteVLLMEngine` launch separate server processes (or pods) and
communicate over HTTP + distributed collectives for weight updates.

They support advanced features such as:

- Pipeline parallelism on the generation side (with corresponding per-PP-rank weight
  update groups).
- Custom scheduling and prefix caching.

### In-process / Experimental Options

Some experimental paths (e.g. inside `areal/experimental/inference_service/`) allow
tighter integration, but most users stay with the remote engines.

## 4. Weight Synchronization Mechanisms

This is one of the most important (and subtle) parts of async RL training.

After the actor updates its weights, the rollout engines must see the new weights before
the next generation round. AReaL supports three strategies, controlled by
`actor.weight_update_mode` and the `WeightUpdateMeta` object.

### 1. `disk` (Simplest, most robust)

- Actor saves a checkpoint (HF or DCP format) to shared storage.
- Rollout engines load the checkpoint.
- **Pros**: Extremely reliable, works across any scheduler, easy to debug.
- **Cons**: High latency and I/O cost; not suitable for high-frequency weight updates.
- Required for: LoRA in many configurations, colocated actor+rollout scheduling.

### 2. `xccl` (Current default — Legacy NCCL Broadcast)

- Training engine (rank 0 / PP heads) creates custom NCCL process groups that include
  the inference workers.
- Parameters are materialized to full tensors and **broadcast** over NCCL
  (`dist.broadcast`).
- Supports sophisticated chunking (`weight_chunked_mem_mb`) and CUDA-stream pipelining
  (FSDP path).
- Handles per-PP-rank groups when the generation side uses pipeline parallelism (to
  avoid deadlocks with SGLang's PP scheduler).
- **Pros**: Very low latency once groups are established; GPU-direct.
- **Cons**: Complex code duplicated across FSDP/Megatron/Archon engines; subtle
  correctness requirements around group initialization order and rank-0 semantics;
  historically hard to maintain.

> **Note**: AReaL is actively modernizing this path. See the migration plan in
> `docs/plans/legacy_xccl_broadcast_weight_update_migration.md`.

### 3. `awex` (New — Async Weight Exchange)

- Newer architecture under `areal/experimental/weight_update/`.
- Uses a controller + gateway + per-worker adapters that negotiate P2P
  `batch_isend_irecv` transfers using rich parameter metadata (`awex` transfer plans).
- Supports colocated and disaggregated modes with better memory control.
- **Pros**: Cleaner separation of concerns, more flexible future evolution, better
  support for advanced features (colocation, fine-grained memory release).
- **Cons**: Still maturing; not yet the default for all model/parallelism combinations.

### Choosing a Weight Update Mode

| Scenario                               | Recommended Mode         | Reason                                                                |
| -------------------------------------- | ------------------------ | --------------------------------------------------------------------- |
| First-time / debugging                 | `disk`                   | Most reliable                                                         |
| High-throughput training (no LoRA)     | `xccl` (current default) | Lowest latency                                                        |
| Large MoE + frequent updates           | `xccl` or `awex`         | Depends on stability needs                                            |
| Using LoRA                             | `disk` (often required)  | Many inference engines have limited LoRA + distributed update support |
| Actor + rollout colocated on same GPUs | `disk`                   | NCCL group conflicts are common                                       |
| Wanting the modern architecture        | `awex`                   | Future direction of the project                                       |

You select the mode in the config:

```yaml
actor:
  weight_update_mode: xccl   # or "disk" or "awex"
```

The `PPOTrainer` (and similar) then constructs the appropriate `WeightUpdateMeta`
(`from_fsdp_xccl`, `from_megatron_xccl`, `from_awex`, or `from_disk`) and passes it to
`connect_engine`.

## 5. Other Major Subsystems

- **Allocation Mode** (`alloc_mode`): Declarative syntax (`fsdp:d8t2`,
  `megatron:(attn:d1p4t2|ffn:d1p4t1e4)`, etc.) that determines GPU counts and
  parallelism for every engine component. See the
  [Allocation Mode Reference](alloc_mode.md).
- **Schedulers / Launchers**: `local`, `ray`, or `slurm`. They handle process placement
  and resource acquisition.
- **Workflows**: `RolloutWorkflow` implementations (e.g. `RLVRWorkflow`) define how a
  single episode is generated, including multi-turn logic, tool use, or custom reward
  shaping.
- **Reward Functions**: Pluggable functions (math, geometry, code, etc.) registered by
  name and executed on the controller after rollout.

## 6. Putting It All Together

A typical production GRPO run uses:

- `rollout.backend: sglang:...`
- `actor.backend: fsdp:...` or `megatron:...`
- `actor.weight_update_mode: xccl`
- A custom or built-in `RolloutWorkflow`
- `scheduler.type: ray` or `slurm` on a cluster

The trainer coordinates everything, the chosen training engine owns the optimizer and
backward pass, the inference engine owns generation, and the weight update path keeps
the two sides consistent.

As AReaL evolves, the `awex` weight update system and further unification of the engine
interfaces are expected to reduce the current duplication and complexity around the
`xccl` broadcast path.

## Further Reading

- [GSM8K GRPO Walkthrough](../tutorial/gsm8k_grpo.md) — detailed code walkthrough
- [Allocation Mode Reference](alloc_mode.md)
- [Rollout Workflow Reference](rollout_workflow.md)
- Engine-specific tutorials: [Megatron](../tutorial/megatron.md),
  [Archon](../tutorial/archon.md)
- Best practices: [Handling OOM](handling_oom.md),
  [Performance Profiling](perf_profiling.md)

If you are contributing new engines or weight update strategies, please also read the
developer-oriented documents and the approved migration plans under `docs/plans/`.
