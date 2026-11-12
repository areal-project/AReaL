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

The AdamW backend uses the repository-pinned Megatron-Core, BF16 training, distributed
optimizer, precision-aware optimizer semantics, and FP32 master/moment state. Enabling
it selects precision-aware mode automatically. Staged optimizer checkpoint saves are
synchronous; `megatron.async_save=true` is rejected.

The Muon optimizer can use the same staging infrastructure:

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

Set `cpu_staged_offload.enabled: false` for native layer-wise Muon. Both modes use
Megatron-Core 0.19.0 and emerging-optimizers 0.3.0 and require BF16 and synchronous
parameter gather. CPU staging supports `tp_mode: duplicated`; TP or expert-TP greater
than one requires `buffer_count: 1`. Staged checkpoints support DP ownership resharding
with fixed model-parallel topology; native checkpoints require fixed DP topology.

Checkpoint loading is fail-stop. DCP writes optimizer state into the authoritative CPU
slabs in place. If loading fails, the process must terminate and AReaL recovery starts a
new process from the last complete checkpoint; no in-process disk snapshot, rollback, or
recovery retry is attempted.

AWEX colocation itself does not require CPU staging. However, the current AWEX weight
exchange explicitly releases optimizer memory before restoring actor weights. That
release uses the managed CPU slabs for staged AdamW. Ordinary Megatron optimizers retain
AWEX's original phase-boundary GPU-to-CPU migration and are copied back before training
resumes. The optional HybridDeviceOptimizer compatibility path is not supported.

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

Use the activated repository environment, currently pinned to AWEX 0.9.0, Megatron-Core
0.19.0, and Megatron Bridge 0.6.0. Qwen3.5 requires AWEX's Qwen3.5 converter, introduced
in AWEX 0.8.1. The historical memory comparison below used Megatron Bridge 0.4.0,
Megatron-Core 0.17.0, SGLang 0.5.10.post1, and the AWEX 0.8.1 source checkout on
`PYTHONPATH`. When using a source checkout, export
`PYTHONPATH="${AWEX_SOURCE_DIR}:$PWD${PYTHONPATH:+:$PYTHONPATH}"` before launching so
the workers inherit it.

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

## Qwen3.5-122B-A10B with Muon

`dapo-math_grpo_qwen3_5_122b_muon.yaml` provides a three-step, eight-GPU smoke recipe
with CPU-staged Muon, attention TP2/DP4, expert EP8, and rollout TP8. Set
`QWEN3_5_122B_A10B_MODEL_PATH`, `DAPO_MATH_17K_PATH`, and `AREAL_EXPERIMENT_ROOT`, then
use the same entry point with this configuration. This uses Qwen3.5-122B-A10B, not a
Base checkpoint. It uses eight prompts with four samples each and an 8191-token
prompt-plus-completion budget.

The examples reserve server context headroom independently of the generation budget:
2064 for a 2047-token budget, 8208 for 8191, and 32784 for 32767. The SGLang version
used in these experiments limits request inputs to at most `context_length - 6` (and
potentially less if the token pool is smaller). A resumed rollout sends its prompt and
partial completion as a new input, so setting context to only one token above the
generation budget can reject an otherwise valid continuation. The extra context does not
increase the generation budget or the training microbatch limit; 16 tokens is
conservative headroom, not an MTP draft-token requirement.

## WeftMoE compatibility

The WeftMoE engine/AWEX/SGLang integration is maintained in a separate PR targeting
`main`. This branch supplies only the CPU-staged Muon adaptation: optional expert FP8
writeback callbacks, complete-master access for BF16 offload/restore, and an engine
binding hook. Without a WeftMoE adapter, ordinary Muon behavior is preserved.

`GPUStagedMuon.get_full_parameter_master(param)` returns the existing CPU FP32 master
for a complete locally owned parameter, or `None` otherwise. Call `drain()` before
consuming it for residency transitions; checkpoint restoration retains its identity. FP8
writer callbacks run on the update stream before staging-slot reuse. Expert DP=1 avoids
gathering released BF16 expert storage.

After combining both PRs, select `actor.optimizer.type=dist_muon`,
`actor.megatron.use_precision_aware_optimizer=false`, and
`actor.megatron.ddp.use_distributed_optimizer=false` in the WeftMoE recipe.
