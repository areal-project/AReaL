# Qwen3-30B-A3B DAPO-Math with CPU-Staged AdamW

This example runs the multi-turn math agent through AReaL's OpenAI-compatible proxy. The
agent uses the standard OpenAI `AsyncOpenAI` client, while the proxy captures the
interactions for `concat` trajectory export. Turn-level reward discounting is disabled
with `turn_discount: 1.0`. The primary actor is a Megatron actor whose AdamW master
weights and moments live in pinned CPU slabs. The default model is
`Qwen/Qwen3-30B-A3B-Base`.

The training dataset is a local DAPO-Math-17k checkout. Its loader first reads
`data/*.parquet`, then falls back to JSON or JSONL files in the dataset root. Existing
OpenAI-style `prompt` messages become `messages`; `label` is used as the answer, with
`reward_model.ground_truth` as a fallback. Prompts longer than
`train_dataset.max_length` after applying the tokenizer's chat template are filtered.

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
- Authoritative CPU memory scales with each rank's owned optimizer shard. Account for
  three FP32 tensors per owned parameter (master weight and two Adam moments), excluding
  model backups and runtime overhead.
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

The provided allocation assumes one node with eight GPUs. Megatron uses dense TP4/DP2
and MoE EP8, while rollout uses four SGLang TP2 engines. Both components request all
eight physical GPUs and AWEX time-multiplexes actor training and rollout residency. The
actor workers set `NCCL_NVLS_ENABLE=0` for the conservative collective path used by this
recipe.

Set `DAPO_MATH_17K_PATH` to a local dataset checkout. Set
`QWEN3_30B_A3B_BASE_MODEL_PATH` to use a local checkpoint; otherwise the YAML uses
`Qwen/Qwen3-30B-A3B-Base`. The proxy total-token limit is `gconfig.max_tokens: 32767`,
linked through `rollout.agent.engine_max_tokens`, while the completion budget is 20480
tokens and SGLang's context length is 32768. Set a unique `AREAL_PROXY_ADMIN_API_KEY`
whenever proxy workers are reachable beyond localhost.

Single-controller local mode:

```bash
AREAL_PROXY_ADMIN_API_KEY=/replace-with-a-unique-secret \
DAPO_MATH_17K_PATH=/path/to/DAPO-Math-17k \
QWEN3_30B_A3B_BASE_MODEL_PATH=/path/to/Qwen3-30B-A3B-Base \
uv run python examples/cpu_staged_offload/dapo-math_rl_cpu_staged.py \
  --config examples/cpu_staged_offload/dapo-math_grpo_cpu_staged.yaml \
  scheduler.type=local \
  experiment_name=dapo-math-grpo-qwen3-30b-a3b-base-cpu-staged \
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

The custom trainer also retains AReaL's non-single-controller construction branch. When
using a cluster scheduler, ensure the repository is importable on every worker so the
stable actor path `examples.cpu_staged_offload.engine.CPUStagedMegatronPPOActor`
resolves remotely.
