# Handling OOM Issues

OOM errors are common in large-scale RL training. This guide covers how to resolve them
across generation, training, and weight updates in AReaL.

## Understanding Memory Usage

Before applying fixes, understand which parameters affect memory usage:

### Core Parameters

- **Per-engine `backend` fields (`actor.backend`, `rollout.backend`)**: How inference
  and training are distributed across GPUs. For large models, tensor parallelism
  typically uses less memory per GPU than data parallelism.

- **`train_dataset.max_length`**: Maximum prompt length. Longer prompts require more
  memory.

- **`gconfig.max_new_tokens`**: Tokens to generate per prompt. Combined with
  `max_length`, this determines the total sequence length.

- **`actor.mb_spec.max_tokens_per_mb`**: Tokens per micro-batch during forward/backward
  passes. This is the primary parameter for controlling training memory. Cannot be set
  below `max_length + max_new_tokens`.

- **`max_concurrent_rollouts`**: Number of parallel generation requests. More requests
  improve throughput but increase memory usage.

### Engine-Specific Parameters

- **Inference Engine**: `sglang.mem_fraction_static` controls how much GPU memory SGLang
  uses. Check the [SGLang docs](https://docs.sglang.io/) for more tuning options.

- **Training Engine**: FSDP sharding and other PyTorch settings also impact memory
  usage. The [FSDP docs](https://docs.pytorch.org/docs/stable/fsdp.html) have more
  details.

> Note: `train_dataset.batch_size` does not affect peak memory usage. Focus on the
> parameters above when troubleshooting OOM issues.

## Resolving Generation OOM Errors

When generation OOM errors occur, try the following solutions:

### 1. Reduce Concurrent Rollouts (Most Effective)

Lower the number of parallel generation requests:

```yaml
max_concurrent_rollouts: 200  # Try reducing from default values like 256
```

This directly reduces memory pressure on the inference servers and is often the most
effective solution.

### 2. Adjust Parallelism Strategy

Increase tensor parallelism to distribute model weights across more GPUs:

```yaml
# Before: 4 data parallel processes for rollout
# After: 2 data parallel, 2 tensor parallel for rollout
rollout:
  backend: "sglang:d2t2"
actor:
  backend: "fsdp:d4"
```

Note that higher tensor parallelism reduces generation throughput.

### 3. Tune SGLang Parameters

Adjust SGLang memory allocation:

```yaml
sglang:
  mem_fraction_static: 0.8  # Reduce from 0.9 to leave more memory headroom
```

See the [SGLang docs](https://docs.sglang.io/) for additional tuning options.

## Resolving Training OOM Errors

Training OOM errors require reducing the memory footprint of gradient computation and
model updates.

### 1. Optimize Micro-batch Size

Set `max_tokens_per_mb` as low as possible:

```yaml
actor:
  mb_spec:
    max_tokens_per_mb: 4096  # train_dataset.max_length + gconfig.max_new_tokens
```

For multi-turn conversations, calculate it like this:

```
max_tokens_per_mb = <longest_conversation_length> + gconfig.max_new_tokens
```

The exact value depends on your `RolloutWorkflow` implementation.

### 2. Enable Gradient Checkpointing

```yaml
actor:
  gradient_checkpointing: true
```

### 3. Enable 5D Parallelism

For long contexts where `max_tokens_per_mb` cannot be reduced further, use Ulysses
sequence parallelism to distribute sequences across multiple GPUs:

```yaml
# Before: 4 data parallel processes for training
# After: 2 data parallel, 2 ulysses context parallel for training
rollout:
  backend: "sglang:d4"
actor:
  backend: "fsdp:d2c2"
```

> The Ulysses context parallel size must evenly divide the model's attention head count.
>
> For example, with 40 attention heads:
>
> - Valid: `1, 2, 4, 8`
> - Invalid: `16, 32`

You can also enable tensor parallelism with FSDP:

```yaml
# Before: 4 data parallel processes for training
# After: 2 data parallel, 2 tensor parallel for training
rollout:
  backend: "sglang:d4"
actor:
  backend: "fsdp:d2t2"
```

For the Megatron and Archon backends, you can also enable pipeline and expert
parallelism:

```yaml
# Before: 4 data parallel processes for training
# After: 2 data parallel with 2 overlaid expert parallel, 2 pipeline parallel, still 4 GPUs
rollout:
  backend: "sglang:d4"
actor:
  backend: "archon:d2p2e2"
```

We recommend pipeline and expert parallelism over tensor/context parallelism. Check
[Allocation Mode Reference](../reference/alloc_mode.md) for more details.

```{seealso}
Pipeline parallelism introduces unique memory challenges (microbatch warmup accumulation,
zero-bubble `retain_graph` overhead, FSDP resharding trade-offs, gradient accumulation
costs, and per-rank memory budgeting). See
{ref}`Archon PP Memory Guide <appendix-pipeline-parallelism-memory-guide>` for a
comprehensive walkthrough.
```

### 4. Enable Per-Layer Optim Step

When using FSDP CPU offloading (`offload_params: true`), the default CPU Adam step can
be very slow. Enable per-layer optim step to stream optimizer states per-layer to device
for a significant speedup:

```yaml
actor:
  fsdp:
    per_layer_optim_step: true
    optim_step_prefetch_layers: 1  # Number of layers to prefetch (default: 1)
```

This streams one layer at a time to device instead of running Adam on CPU, keeping
device memory usage low while achieving much faster optimizer updates.

**Requirements:**

- Compatible with both `offload_params: true` and `false` (optimizer states are
  automatically managed on CPU by the per-layer wrapper; when `offload_params` is also
  enabled, params/grads are streamed per-layer to device as well)

### 5. Switch to a Lightweight Optimizer

AReaL supports different optimizers depending on the training engine.

| Optimizer       | FSDP | Megatron | Name      |
| --------------- | ---- | -------- | --------- |
| AdamW (default) | ✅   | ✅       | adam      |
| SGD             | ✅   | ✅       | sgd       |
| AdamW_bf16      | ✅   | ✅       | adam_bf16 |

`SGD` and `AdamW_bf16` use less memory than the default `AdamW`. Switch by setting
`actor.optimizer.type: <name>` in your YAML configuration file (e.g.,
`actor.optimizer.type: sgd`).

### 6. Use Memory-Efficient Model Loading

If OOM occurs during model initialization (before training starts), enable
memory-efficient loading:

```yaml
actor:
  fsdp:
    memory_efficient_load: true
```

This is useful for large models where loading full weights directly onto each GPU would
exceed memory. When enabled:

1. All ranks create model structure on CPU (without loading weights for LLM)
1. FSDP parallelization is applied
1. Rank 0 loads pretrained weights and broadcasts to all ranks
1. Weights are transferred to GPU

This approach trades some initialization time for significantly lower peak GPU memory
during model loading.

**Note for Vision-Language Models (VLMs):** VLMs do not use the rank 0 broadcast
optimization. When `memory_efficient_load: true` is set for VLMs, weights are loaded on
CPU instead of GPU, but each rank loads weights independently. This still reduces GPU
memory usage during initialization but does not reduce CPU memory or disk/network I/O.

## Resolving Weight Update OOM Errors

Weight updates consume significant memory, especially with NCCL synchronization (the
default).

### 1. Switch to Disk-Based Updates

Switch from NCCL to disk-based weight synchronization:

```yaml
actor:
  weight_update_mode: disk
```

Make sure that `cluster.fileroot` is a shared directory across the cluster.

### 2. Reduce Memory Buffer Size

To continue using NCCL, reduce the memory buffer size for weight chunking:

```python
# In WeightUpdateMeta.from_fsdp_xccl() calls
WeightUpdateMeta.from_fsdp_xccl(
    ...,
    weight_chunked_mem_mb = 512,  # Reduce from default (typically 1024+)
)
```

## Reducing optimizer state memory

By default, the FSDP backend keeps fp32 master weights and fp32 AdamW optimizer states
(`exp_avg`, `exp_avg_sq`), matching DeepSpeed ZeRO-3 and Megatron precision-aware
optimizer behavior. For an `N`-billion parameter model, this costs roughly `12N` GB
across all GPUs:

| Component                      | Bytes/param |
| ------------------------------ | ----------- |
| fp32 master weights (storage)  | 4           |
| fp32 `exp_avg` (1st moment)    | 4           |
| fp32 `exp_avg_sq` (2nd moment) | 4           |

**CPU memory note:** when `actor.fsdp.memory_efficient_load=true`, rank 0 loads the full
model on CPU before broadcasting. fp32 storage doubles this peak — for an 8B model, rank
0 CPU usage rises from ~16 GB to ~32 GB. Provision the head node accordingly, or set
`memory_efficient_load=false` to spread CPU load across ranks.

**DCP checkpoint note:** training checkpoints (Distributed Checkpoint) preserve storage
dtype (fp32) — this is required for correct resume of master weights. HF export and
rollout weight sync **always** cast back to compute dtype, so deployment artefacts stay
bf16.

If you hit OOM and cannot increase parallelism, switch to bf16 optimizer states with
Kahan-summation updates:

```yaml
actor:
  dtype: bfloat16
  optimizer_dtype: bfloat16   # storage in bf16 (matches AdamW state)
  optimizer:
    type: adam_bf16           # uses AnyPrecisionAdamW + Kahan summation
```

This recovers approximately `8N` GB:

| Component                      | Bytes/param |
| ------------------------------ | ----------- |
| bf16 master weights            | 2           |
| bf16 `exp_avg`                 | 2           |
| bf16 `exp_avg_sq`              | 2           |
| bf16 Kahan compensation buffer | 2           |

Per the AnyPrecision paper, Kahan summation recovers fp32-equivalent update precision.
Step time is ~5-10% slower than fused fp32 AdamW; quality is comparable on dense and MoE
models.

**Do not use `optimizer.type: adam` together with `optimizer_dtype: bfloat16`** —
`torch.optim.AdamW` will silently create bf16 optimizer states and the late-stage
convergence will plateau ~3× higher than fp32 master weights (see issue #1292). The
runtime emits a warning when this combination is detected.
