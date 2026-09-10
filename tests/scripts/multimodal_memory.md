# Multimodal memory isolation checks

Run these scripts in the existing target environment. No test was run on the local
development machine. The minimal check uses CPU/Gloo; the end-to-end check requires free
GPUs and the model dependencies already installed.

```bash
bash tests/scripts/check_multimodal_memory_minimal.sh
VLM_MODEL_PATH=/path/to/supported/vlm BACKEND=megatron:d1p1t2 \
  bash tests/scripts/check_multimodal_memory_e2e.sh
```

Expected: CPU tests pass; the end-to-end runner produces `Passed`. It exercises
source-selected CPU alias broadcast, repeat forward without source mutation, text-only
VLM forward, and an optimizer update. It does not launch SGLang or claim rollout
throughput gains. Use a supported model/parallel layout; the current main explicitly
rejects VLM CP>1; this branch retains that guard. The runner supplies at least `2 * PP`
real samples when testing a supported pipeline layout. GPU equivalence and memory
reduction still require a target hardware run; the CPU checks establish data-path
contracts only.

The broadcast opt-in is restricted to CPU-staged vision engines. Its source rank further
requires nonempty tensors below `multi_modal_input*`; receivers follow the source
metadata. Text-only payloads retain the old metadata sequence and per-reference buffers.
Only multimodal calls add an internal mode marker. Lazy vision preparation affects only
microbatches with multimodal tensors; ordinary text, empty-image VLM batches, and
top-level-only vision inputs retain the old path. Batch partitioning and loss
normalization are unchanged.
