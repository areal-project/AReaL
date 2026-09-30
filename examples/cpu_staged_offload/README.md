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

## HARTS tree training for Qwen3.5

`dapo-math_grpo_qwen3_5_harts.yaml` enables trie sharing during actor training. It keeps
the same Qwen3.5 model, DAPO Math data, and Megatron/SGLang parallel layout as the
configuration above. Full-attention layers use the tree mask. Gated DeltaNet layers use
ancestor-aware convolution and a compact state-replay plan. The actor's chunked LM-head
loss remains enabled (`enable_chunked_logits=true`, `lm_head_loss_chunk_size=1024`);
distinct rollout labels can share a compact logit row. The current Megatron Bridge path
supports text-only Qwen3.5 batches and no context parallelism. A global planner groups
trajectories by shared prefixes, assigns synchronized tree slots to DP replicas, and
exchanges the CPU rows needed by each replica before Megatron packing. Public FLA
exports only the final state of each packed sequence. For nested forks that need an
intermediate state, this adapter packs an additional state-only prefix in the same call.
It preserves the planner's call depth but may repeat linear-attention core work beyond
the paper's selective-state kernel bound.

With the model and dataset environment variables set as above, run:

```bash
export AREAL_PROXY_ADMIN_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python examples/cpu_staged_offload/dapo-math_rl_cpu_staged.py \
  --config examples/cpu_staged_offload/dapo-math_grpo_qwen3_5_harts.yaml \
  trial_name=harts actor.megatron.cpu_staged_offload.enabled=false
```

For the comparison baseline, use `dapo-math_grpo_qwen3_5_cpu_staged.yaml` with the same
command-line override and a different `trial_name`. CPU staged offload was disabled in
both comparison runs: the installed Megatron-Core 0.17.0 does not meet this branch's
Qwen3.5 staged optimizer compatibility guard. Both runs used seed 1, one eight-L20X
node, three PPO updates, eight prompts per step, two samples per prompt, and at most 128
generated tokens. All three PPO and AWEX weight updates completed in both runs. W&B
project: `wht_tree_training`
([baseline](http://8.150.1.98:8080/sct-test/wht_tree_training/runs/wht_tree_training_baseline-qwen35-20260929_train),
[HARTS with global scheduling](http://8.150.1.98:8080/sct-test/wht_tree_training/runs/wht_tree_training_harts-qwen35-global-dp-20260930_train)).

| Measure                                                    |  Baseline |    HARTS |
| ---------------------------------------------------------- | --------: | -------: |
| Mean actor training step, steps 2–3                        |    7.81 s |   4.44 s |
| Actor input tokens / actor training time, steps 2–3        |     611/s |    996/s |
| Total training time, three steps                           |  883.18 s | 899.21 s |
| AWEX weight-update time, sum of three steps                |  708.67 s | 800.57 s |
| GPU used at PPO-update checkpoint, highest of three steps  | 100.51 GB | 97.20 GB |
| Compact tree tokens / original tokens, mean of three steps |         — |    0.687 |

The actor training step is 1.76× faster on the two post-compilation steps, and actor
input-token throughput is 1.63× higher. Total training time is 1.8% higher; AWEX weight
exchange took 91.90 seconds longer. An earlier HARTS run without global scheduling
measured 3.76 seconds per actor step and 936.20 seconds end to end, so this short-run
comparison is sensitive to the sampled trajectories and weight-exchange variation. The
memory row is an update-checkpoint reading, not a peak across all phases. Generated
samples and rewards differed between runs despite the shared seed; these measurements
validate execution and give a short-run performance comparison, not convergence or
statistical significance.
