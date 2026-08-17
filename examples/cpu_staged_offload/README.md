# Qwen3-30B-A3B Multi-Turn GSM8K with CPU-Staged AdamW

This example keeps the multi-turn GSM8K workflow, reward verification, `concat`
trajectory export, and reward discounting from `examples/multi_turn_math`. The only
training change is that the primary actor is a Megatron actor whose AdamW master weights
and moments live in pinned CPU slabs. The default model is Qwen3-30B-A3B.

The train and validation datasets both use `openai/gsm8k`.

## CPU-staged AdamW versus `enable_offload`

`enable_offload` is the top-level engine lifecycle switch. It frees or restores model
weights, gradient buffers, and ordinary optimizer allocations when actor and rollout
phases share devices.

CPU-staged AdamW is an optimizer implementation. Its FP32 master weights, `exp_avg`, and
`exp_avg_sq` remain in pinned CPU slabs during forward and backward. Each optimizer step
streams bounded units through fixed GPU slots and returns to `CPU_RESIDENT` after D2H
completion. It does not require `enable_offload`; this example sets
`enable_offload: false` so the two mechanisms are easy to distinguish.

The example-local `cpu_staged.enabled` switch selects the optimizer implementation:

- `true` uses `CPUStagedMegatronPPOActor` and CPU-staged AdamW;
- `false` delegates actor construction to the standard `PPOTrainer` Megatron path.

Both branches retain the same precision-aware FP32 master weights and moments, so an A/B
run isolates optimizer residency and staging rather than changing MCore optimizer mode.

## Compatibility and resource behavior

- The implementation is strictly guarded for `megatron-core==0.17.0`.
- The actor must use the Megatron backend and AdamW optimizer.
- Each GPU slot contains master-weight, first-moment, second-moment, and gradient FP32
  buffers. GPU optimizer memory therefore grows approximately as
  `4 * buffer_count * bucket_size_mb`, rather than with the full optimizer state.
- The example config uses two 128 MiB slots, for approximately 1 GiB of fixed staging
  tensors per optimizer leaf. Qwen3 MoE normally has separate dense and expert leaves,
  so up to about 2 GiB may remain resident per actor rank. Tune these example-local
  settings under `cpu_staged`; no AReaL CLI schema is added.
- With Attention DP8 and FFN EP8, Qwen3-30B-A3B owns approximately 5.16 billion
  parameters per actor rank. Its three authoritative FP32 CPU slabs require about 57.7
  GiB of pinned host memory per rank (about 462 GiB across eight ranks), excluding model
  backups and runtime overhead.
- Managed asynchronous save and synchronous transactional load use the core staged
  optimizer implementation; this example does not copy or patch that code.

## Rollback snapshot root

Synchronous managed load writes bounded, chunked rollback snapshots before DCP may
mutate the CPU slabs. For production, set `AREAL_CPU_STAGED_SNAPSHOT_ROOT` to a
job-owned scratch directory. The directory must:

- already exist and be writable by every participating rank;
- be a real directory, not a symlink;
- provide stable directory inode and regular-file semantics, `fsync`, non-zero
  filesystem identity, and enough aggregate free space;
- not be renamed or replaced while the job is running.

Do not point it at a shared user home or an untrusted path. For example:

```bash
export AREAL_CPU_STAGED_SNAPSHOT_ROOT="${JOB_TMPDIR:?}/areal-rollback"
install -d -m 0700 "${AREAL_CPU_STAGED_SNAPSHOT_ROOT}"
```

The YAML leaves `checkpoint_snapshot_root: null`. An environment variable overrides the
example-local YAML value and is forwarded to remote actor workers. The related optional
variables are:

```text
AREAL_CPU_STAGED_BUFFER_COUNT
AREAL_CPU_STAGED_BUCKET_SIZE_MB
AREAL_CPU_STAGED_SNAPSHOT_CHUNK_MB
```

## Launch

The provided allocation assumes one node with eight GPUs. Eight Megatron actor ranks use
`megatron:(attn:d8p1t1|ffn:d1p1t1e8)`: dense Attention parameters are replicated with
DP8, while 128 MoE experts are sharded with EP8. One eight-GPU SGLang server uses DP
Attention and EP8 MoE on those same physical GPUs through AWEX. The ref configuration
uses the standard Megatron engine path and is not given the staged optimizer subclass.

The local scheduler intentionally does not use an explicit rollout
`scheduling_strategy`. The actor allocation has eight one-GPU workers, whereas the
rollout allocation has one eight-GPU worker, so explicit worker-for-worker colocation
would be invalid. After actor GPUs 0--7 are allocated, local round-robin placement wraps
the rollout allocation back onto GPUs 0--7.

Set `QWEN3_30B_A3B_MODEL_PATH` when using a local checkpoint. If it is unset, the YAML
uses `Qwen/Qwen3-30B-A3B`.

Single-controller local mode:

```bash
QWEN3_30B_A3B_MODEL_PATH=/path/to/Qwen3-30B-A3B \
uv run python examples/cpu_staged_offload/gsm8k_rl_cpu_staged.py \
  --config examples/cpu_staged_offload/gsm8k_grpo_cpu_staged.yaml \
  scheduler.type=local \
  experiment_name=gsm8k-grpo-qwen3-30b-a3b-cpu-staged \
  trial_name=trial0
```

Override `cpu_staged.enabled=false` to run the ordinary Megatron optimizer baseline.

## Step-2 CUDA memory snapshot

The default `memory_profiler.profile_steps: [1]` captures the actor PPO update in the
second visible training step because global steps are zero-based. It produces one
PyTorch allocator snapshot per actor rank at:

```text
${AREAL_FILEROOT}/logs/${USER}/${experiment_name}/${trial_name}/
  memory_snapshots/step_1/snapshot_rank*.pickle
```

Each `.pickle` file can be opened directly in the PyTorch CUDA Memory Visualizer at
<https://pytorch.org/memory_viz>. The snapshot covers `actor.ppo_update()` and the LR
scheduler step; rollout, log-probability recomputation, and weight transfer are outside
the recorded window. Set `total_train_steps=2` for a minimal capture, and use distinct
`experiment_name`, `trial_name`, `AREAL_FILEROOT`, and name-resolve roots for enabled
and disabled runs.

The custom trainer also retains AReaL's non-single-controller construction branch: the
custom actor is instantiated in-process, while ref/critic engines continue through the
standard `PPOTrainer` factory. When using a cluster scheduler, ensure the repository is
importable on every worker so the stable actor path
`examples.cpu_staged_offload.engine.CPUStagedMegatronPPOActor` resolves remotely.
