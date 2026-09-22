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

Set `QWEN3_30B_A3B_BASE_MODEL_PATH` and `DAPO_MATH_17K_PATH` to your model and dataset
locations. The agent proxy requires a unique admin key when binding to a non-loopback
address. Run on a local eight-GPU node with:

```bash
export AREAL_PROXY_ADMIN_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run python examples/cpu_staged_offload/dapo-math_rl_cpu_staged.py \
  --config examples/cpu_staged_offload/dapo-math_grpo_cpu_staged.yaml \
  scheduler.type=local
```

## Qwen3.5-35B-A3B-Base

`dapo-math_grpo_qwen3_5_cpu_staged.yaml` reuses the configuration above and defaults to
three training steps on eight GPUs. It uses attention TP2/DP4, MoE EP8, and four TP2
SGLang servers. Set `QWEN3_5_35B_A3B_BASE_MODEL_PATH` and `DAPO_MATH_17K_PATH` to local
model and dataset directories, then run:

```bash
export AREAL_PROXY_ADMIN_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python examples/cpu_staged_offload/dapo-math_rl_cpu_staged.py \
  --config examples/cpu_staged_offload/dapo-math_grpo_qwen3_5_cpu_staged.yaml \
  trial_name=staged
```

Use the activated training environment. Qwen3.5 additionally requires AWEX's Qwen3.5
converter, available in AWEX 0.8.1; the repository's pinned AWEX 0.8.0 lacks it. The
validated environment used Megatron Bridge 0.4.0, Megatron-Core 0.17.0, SGLang
0.5.10.post1, and the AWEX 0.8.1 source checkout on `PYTHONPATH`. When using a source
checkout, export `PYTHONPATH="${AWEX_SOURCE_DIR}:$PWD${PYTHONPATH:+:$PYTHONPATH}"`
before launching so the workers inherit it.

The configuration enables fused FLA Gated DeltaNet kernels and disables gradient
reduction overlap because text batches leave the vision parameters unused. Vision
parameters remain allocated. Precision-aware optimizer mode is explicitly enabled in
both comparison runs. To run the baseline, repeat the command with
`trial_name=baseline actor.megatron.cpu_staged_offload.enabled=false`.

Both runs completed three PPO updates and three AWEX weight updates on a single node
with eight L20X GPUs. They used the same fixed 512-record DAPO subset, seed, batch size
8, two samples per prompt, and 128 generated tokens per sample. The local Base model was
downloaded at revision `0f0813072d2358973511097385626f21fcb6d422`.

| Peak memory per GPU (GiB) | Staged on | Staged off |
| ------------------------- | --------: | ---------: |
| Actor allocated           |     35.38 |      74.30 |
| Actor reserved            |     38.31 |      77.03 |
| Whole GPU, all phases     |     99.51 |      99.01 |

Actor peaks are the maximum across eight ranks and global steps 1 and 2, reconstructed
from PyTorch allocator snapshots. Actor allocated memory fell by 52.4%. Whole-GPU peaks
come from one-second `nvidia-smi` samples and include rollout and weight exchange; their
peaks were essentially unchanged, with the staged run peaking during AWEX exchange.
These short runs validate execution and memory use, not convergence or multi-node
behavior. Optimizer staging also requires substantial pinned host memory; the validation
node had 1.5 TiB of RAM.
