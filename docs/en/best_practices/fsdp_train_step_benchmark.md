# Benchmarking FSDP training steps

`areal.tools.benchmark_fsdp` measures the real FSDP training engine with a fixed
synthetic SFT workload. It is a small, reproducible measurement tool and a foundation
for future performance guidance; it does not select settings for a running experiment.

## Run a sweep

Run from the repository root in an environment with the project's CUDA dependencies,
FlashAttention 2, and access to the selected Hugging Face model:

```bash
python -m areal.tools.benchmark_fsdp \
  --model-path Qwen/Qwen2.5-0.5B \
  --seq-lens 512,512,512,512 \
  --token-budgets 512,1024 \
  --gradient-checkpointing both \
  --nproc-per-node 1 \
  --warmup-steps 3 \
  --steps 10 \
  --output-dir /tmp/areal-fsdp-benchmark-qwen025
```

Each sequence length represents one row of the same global batch for every trial. The
tool resolves a remote model revision once, then starts a fresh `torchrun` process for
each token-budget/checkpointing combination. A trial timeout or OOM is recorded and the
next combination runs in a new process group. Output directories must not already exist
and must be outside the local model directory. Use `--dry-run` to validate the sweep and
save its plan without downloading a model or starting CUDA. The default sweep covers
both checkpointing settings; use `off` or `on` to run one setting. `--init-from-scratch`
is intended for a local-config smoke test, not for representative performance numbers.

The first version supports single-node Linux CUDA FSDP data parallelism for dense Qwen2,
Qwen3, and Llama text models using packed FlashAttention 2. TP, context/sequence
parallelism, MoE, vision-language models, CPU execution and multi-node runs are outside
its scope. Each global batch must divide evenly across ranks. Use model context lengths
and token budgets that fit the available GPU memory.

## Interpret the result

Each trial performs real `FSDPEngine.train_batch()` calls with
`compute_packed_sft_loss`, including microbatch preparation, forward/backward, gradient
clipping and optimizer step. Inputs are generated deterministically from `--seed`, then
moved to the GPU before timing. The synchronized wall-clock interval excludes model
loading, data creation, input staging, and metric export. It includes AReaL's loss
monitoring. A skipped optimizer update caused by a non-finite gradient fails the trial
instead of counting as a training step.

The JSON artifact records the resolved checkpoint manifest, effective engine config,
software and GPU versions, per-rank measurements, latency samples, memory peaks and
failure details. The Markdown table is a compact comparison. Throughput uses the global
valid input-token or loss-token count divided by the sum of per-step slowest-rank wall
times. Prepared and executed microbatch counts are reported separately because FSDP may
make a faster rank participate in dummy microbatches to align collectives.

Allocated memory is sampled after each timed step. Reserved memory includes caching
allocator blocks retained from warmup and metric export. GPU contention changes timings,
so compare small deltas only under controlled conditions. Results characterize this
exact workload, checkpoint, software stack and hardware; they do not identify a
universal best setting and do not predict RL convergence or rollout performance.

For a shorter installation check, use a local Qwen2 config/tokenizer and
`--init-from-scratch`; this still runs real FSDP steps but uses random weights and is
not a substitute for a checkpoint benchmark.
