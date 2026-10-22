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

## Qwen3.5-122B-A10B: real DAPO RL smoke

`dapo-math_grpo_qwen3_5_122b_muon.yaml` runs three GRPO updates on DAPO-Math with 8
prompts and 4 samples per prompt. Set `QWEN3_5_122B_A10B_MODEL_PATH`,
`DAPO_MATH_17K_PATH`, `AREAL_EXPERIMENT_ROOT`, and the proxy admin key as above:

```bash
python examples/cpu_staged_offload/dapo-math_rl_cpu_staged.py \
  --config examples/cpu_staged_offload/dapo-math_grpo_qwen3_5_122b_muon.yaml
```

Training uses TP2/DP4/EP8, FP32 gradients, CPU-staged Muon, and full one-layer
recomputation. Rollout uses one TP8/EP8 SGLang server on the same eight GPUs; four TP2
replicas of the 122B model would not fit the validated memory budget. The 8K limit
includes the prompt and completion. Checkpoints, recovery, and memory snapshots are
disabled; W&B is online and takes credentials from the environment.

This is a real reward-producing RL test, unlike the synthetic SFT capacity test in
`examples/profile`. Monitor reward, nonzero advantages, policy logprob differences,
finite loss, and the **pre-clipping** gradient norm. Three updates do not establish
convergence. The local 122B checkpoint used for validation is the post-trained model,
not a Base checkpoint.

### Qwen3.5 THD attention backward regression

The initial 122B Muon smoke's large or NaN gradient norms were a correctness failure,
not a property of Muon or the SFT data. Real 8K DAPO runs also reproduced the failure
with 35B AdamW (pre-clipping norm about `5.34e10`) and with 35B Muon after disabling
activation-storage release (about `2.25e10`). The gradients were already wrong before
the optimizer update. `update_successful=1` alone did not detect this failure.

The newer Megatron stack enables model-owned THD packing for Qwen3.5. A standalone
Transformer Engine attention test reproduced incorrect gradients in the second half of
the sequence with the default cuDNN fused backend. It required no optimizer, distributed
communication, activation checkpointing, AWEX, or activation tracker. Filling freed
allocator blocks with NaNs made the invalid gradients reproducibly NaN. Both BSHD with
cuDNN and THD with FlashAttention remained correct. The behavior was observed at 4096,
8190, and 8192 tokens; it was not just an unaligned sequence tail.

Replaying captured training Q/K/V and output gradients confirmed the diagnosis: forward
relative L2 error was only `0.00093`, but the faulty dV maximum was `31.75` versus the
reference `5.94e-8`. FlashAttention dQ/dK/dV relative L2 errors were
`0.00080 / 0.00298 / 0.00301`. The precise vendor-kernel defect has not been
established.

The isolated reproducer used PyTorch `2.9.1+cu129`, Transformer Engine
`2.14.1+27de67e1`, cuDNN `9.16.0`, BF16 Q/K/V, 8 local query heads, 1 KV head, and head
dimension 256 on L20X. This identifies the validated failing stack; it does not imply
that every cuDNN version or THD configuration is affected.

Qwen3.5 providers now select `AttnBackend.flash` before model construction, including
non-deterministic training. This avoids the faulty fused-attention path without changing
GDN, Muon, optimizer offload, activation release, or rollout. Do not force
`NVTE_FUSED_ATTN=0` externally: MCore validates these variables against its provider's
backend setting. Temporary per-layer diagnostics are not part of the training path.

Post-fix validation uses real DAPO rollouts at the same 8K context limit, BF16, TP2/EP8
on 8 GPUs, full recomputation, activation release, CPU-staged optimizer offload, and
AWEX 0.9.0. Native Megatron MoE is used (`AREAL_USE_WEFTMOE=0`) to isolate the
training-stack regression. Checkpoint and recover saving are disabled. Each update
consumes 8 prompts with 4 samples each. Gradient norms below are **before clipping**;
different rollouts are not a paired performance comparison.

| Model / optimizer | Trial                             | Validated updates | Pre-clipping gradient norms  |
| ----------------- | --------------------------------- | ----------------: | ---------------------------- |
| 35B Base / AdamW  | `grad-debug-35b-adam / fixed-r3`  |                 3 | 0.336787, 0.389280, 0.341655 |
| 35B Base / Muon   | `grad-debug-35b-muon / flash-r4`  |                 1 | 0.385123                     |
| 122B / Muon       | `grad-debug-122b-muon / fixed-r1` |                 3 | 0.310677, 0.284341, 0.282264 |

The 35B AdamW and 122B Muon runs use the clean production path and completed all three
AWEX weight versions. The 35B Muon control still had temporary gradient diagnostics and
is not a timing baseline. All listed updates have finite loss and `update_successful=1`;
these short tests do not establish reward convergence. Metrics are online in the
`sct-test/qwen3-5-grad-debug` W&B project.

The separate 122B AdamW attempt exceeded the node's host-memory capacity while
initializing CPU optimizer state, before backward. It therefore provides no numerical
validation for that model/optimizer combination.
