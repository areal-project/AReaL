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

### Twenty-step long-output comparison

The longer comparison used Qwen3.5-35B-A3B-Base and DAPO Math 17K on one node with eight
L20X GPUs. Both runs completed 20 PPO and AWEX updates with seed 1, 32 prompts per step,
16 samples per prompt, rollout concurrency 32, rollout queue size 512, and consumer
batch size 32. The input limit was 2048 tokens, the generation limit was 20480 tokens,
and the request token limit was 32767. SGLang used a context limit of 32768 tokens and
`max_running_requests=32` per TP2 server; the actor used `max_tokens_per_mb=32768`. The
request cap per server was needed after an earlier uncapped baseline attempt ran out of
SGLang memory at step 4. CPU staged offload was disabled in both completed runs. Chunked
LM-head loss stayed enabled in both (`enable_chunked_logits=true`,
`lm_head_loss_chunk_size=1024`). HARTS enabled tree training and used
`pad_to_maximum=true`, while the baseline disabled tree training and used
`pad_to_maximum=false`.

W&B project: `wht_tree_training`
([baseline](http://8.150.1.98:8080/sct-test/wht_tree_training/runs/wht_tree_training_baseline-qwen35-long-sg32-c32-b32-s16-20step-20260930_train),
[HARTS](http://8.150.1.98:8080/sct-test/wht_tree_training/runs/wht_tree_training_harts-qwen35-long-sg32-c32-b32-s16-20step-20260930_train)).

| Measure                                                |   Baseline |      HARTS |
| ------------------------------------------------------ | ---------: | ---------: |
| Completed updates; trajectories per update             |    20; 512 |    20; 512 |
| Mean prompt tokens                                     |        154 |        154 |
| Mean generated tokens                                  |       9258 |       9240 |
| No-EOS ratio, mean                                     |      18.9% |      19.0% |
| Mean actor training step                               |   156.94 s |   470.92 s |
| Actor input tokens / actor training time, steps 2–20   |   30,876/s |   10,263/s |
| Mean rollout time                                      |   395.16 s |   401.79 s |
| AWEX weight-update time, sum of 20 steps               |  5722.09 s |  5814.93 s |
| Total training time                                    | 17011.21 s | 23511.31 s |
| GPU used at PPO-update checkpoint, highest of 20 steps |  123.65 GB |  133.33 GB |
| Compact tree tokens / original tokens, mean            |          — |      0.989 |
| Rollout controller timeout events (3600 s)             |          0 |          6 |

The baseline processed actor input tokens 3.01 times faster, and HARTS took 38.2% longer
end to end. In these sampled long answers, the compact tree retained 98.9% of original
tokens, so prefix sharing was small. HARTS also requires the different padding setting
above; this run does not isolate its cost from the tree execution cost. Six
rollout-controller timeout events occurred on HARTS; affected trajectories were
rejected, while every completed training batch still contained 512 trajectories. Both
runs exited successfully and synced to W&B. Mean generated tokens are mean sequence
length minus mean prompt length. The GPU readings are PPO-update checkpoints, not
all-phase peaks. These runs measure execution and throughput on one node, not
convergence or multi-node behavior.

### Fixed-input actor benchmark for shared-prefix workloads

`benchmark_harts_actor.py` compares Megatron actor training on identical synthetic
Qwen3.5 trajectories, excluding rollout and AWEX. Each of 16 trajectories has the same
6,000-token prefix and a distinct 1,000-token suffix; only suffix tokens contribute to
the loss. HARTS packs 22,000 of the original 112,000 tokens (ratio 0.196). The actor
uses the eight-GPU `megatron:(attn:d4p1t2c1|ffn:d1e8)` allocation, 32,768-token
microbatch limit, and chunked LM-head loss with 1,024-token chunks in both modes.

```bash
OMP_NUM_THREADS=8 torchrun --nproc_per_node=8 \
  examples/cpu_staged_offload/benchmark_harts_actor.py \
  --model-path "$MODEL_PATH" --warmup 1 --repeats 3
OMP_NUM_THREADS=8 torchrun --nproc_per_node=8 \
  examples/cpu_staged_offload/benchmark_harts_actor.py \
  --model-path "$MODEL_PATH" --tree --warmup 1 --repeats 3
```

The benchmark reports the maximum rank time after GPU synchronization. Its first step
warms up model kernels and compilation; measured steps are steady-state training
updates. On one eight-L20X node with Qwen3.5-35B-A3B-Base, the three measured updates on
2026-10-02 gave these median times after the tree convolution was fused:

| Median time per update | Baseline |   HARTS | Baseline / HARTS |
| ---------------------- | -------: | ------: | ---------------: |
| Full actor train step  |  8.855 s | 4.579 s |            1.93× |
| Forward and backward   |  8.710 s | 3.940 s |            2.21× |

Shared prefixes in real rollouts may be shorter or less frequent. The earlier 20-step
DAPO Math comparison above retained 98.9% of tokens after packing, so it measures a
different workload and predates the fused tree convolution. This fixed-input benchmark
does not measure rollout, AWEX weight exchange, or end-to-end training throughput.
