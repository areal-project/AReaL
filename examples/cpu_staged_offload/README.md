# CPU-staged Megatron optimizers

This example keeps FP32 AdamW master parameters and moments in pinned CPU memory.
Optimizer steps stream bounded chunks through reusable GPU buffers, freeing the full
optimizer state from the colocated rollout GPU.

The feature is configured directly through AReaL's Megatron configuration:

```yaml
actor:
  megatron:
    cpu_staged_offload:
      enabled: true
      buffer_count: 2
      bucket_size_mb: 128
```

No example-specific actor, trainer, or worker environment variables are required.
`buffer_count` controls the number of reusable GPU staging slots; `bucket_size_mb`
bounds one slot's master/moment/gradient tensors.

The AdamW backend requires Megatron-Core 0.17.0, BF16 training, distributed optimizer,
precision-aware optimizer semantics, and FP32 master/moment state. Enabling it selects
precision-aware mode automatically. Staged optimizer checkpoint saves are synchronous;
`megatron.async_save=true` is rejected.

The staged Muon backend remains available through the same core configuration:

```yaml
actor:
  optimizer:
    type: dist_muon
    muon:
      momentum: 0.95
      num_ns_steps: 5
      tp_mode: duplicated
  megatron:
    ddp:
      use_distributed_optimizer: false
    cpu_staged_offload:
      enabled: true
      buffer_count: 1
      bucket_size_mb: 128
```

The optimizer algorithm is independent from CPU staging: set `type: dist_muon` with
`cpu_staged_offload.enabled: false` to use native layer-wise Muon, or enable CPU staging
without changing any Muon hyperparameters. Muon retains MCore's official LayerWise
ownership and the staged variant's synchronous DCP schema. It requires Megatron-Core
0.17.0, emerging-optimizers 0.3.0, BF16, and synchronous parameter gather; TP or
expert-TP greater than one requires `buffer_count: 1`.

Checkpoint loading is fail-stop. DCP writes optimizer state into the authoritative CPU
slabs in place. If loading fails, the process must terminate and AReaL recovery starts a
new process from the last complete checkpoint; no in-process disk snapshot, rollback, or
recovery retry is attempted.

AWEX colocation itself does not require CPU staging. However, the current AWEX weight
exchange explicitly releases optimizer memory before restoring actor weights. That
release uses the managed CPU slabs for staged AdamW and staged Muon. Ordinary Megatron
optimizers retain AWEX's original phase-boundary GPU-to-CPU migration and are copied
back before training resumes. The optional HybridDeviceOptimizer compatibility path is
not supported.

Run with:

```bash
uv run python examples/cpu_staged_offload/dapo-math_rl_cpu_staged.py \
  --config examples/cpu_staged_offload/dapo-math_grpo_cpu_staged.yaml
```
