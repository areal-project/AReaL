# Qwen3-30B-A3B SFT Profile

This directory provides an end-to-end SFT profiling example. It uses Qwen3-30B-A3B and
deterministic fake 128K-token SFT data to collect kernel and memory profiles separately.

## Files

- `train_sft_profile.py`: SFT entry point. Instead of reading an external JSONL file, it
  uses the real tokenizer to encode a structured SWE/code-repair conversation, then
  repeats and truncates it to 131072 tokens.
- `qwen3_30b_a3b_sft_profile.yaml`: Megatron MoE profiling configuration for one node
  with eight GPUs. By default it uses `megatron:(attn:d2c4|ffn:d1e8)`: DP=2 and CP=4 for
  attention, and EP=8 for the MoE FFN.
- `run_qwen3_30b_a3b_sft_profile.sh`: Runs kernel and memory profiles.
- `postprocess_profile.py`: Generates kernel Chrome trace views and profile summaries.

## Quick start

Run from the repository root:

```bash
MODEL_PATH=/path/to/Qwen3-30B-A3B \
FILEROOT=/path/to/shared/experiments \
bash examples/profile/run_qwen3_30b_a3b_sft_profile.sh
```

By default, the script profiles steps 0 and 1. With `PROFILE_KINDS=kernel,memory`, it
runs four trials in total:

- `kernel`: enables `perf_tracer.profile_steps`, starts the PyTorch profiler in the SFT
  training step, and produces CPU, CUDA, and GPU kernel traces.
- `memory`: enables `memory_profiler.profile_steps` and produces PyTorch CUDA allocator
  snapshots.

Default settings:

```bash
PROFILE_STEPS=0,1
# TOTAL_STEPS is unset by default; each trial uses profile_step + 1.
# PROFILE_RANKS is unset by default; the script computes each PP stage's rank0.
PROFILE_KINDS=kernel,memory
PROFILE_FAKE_SEQ_LEN=131072
PROFILE_FAKE_DATASET_SIZE=8
PROFILE_FAKE_LOSS_START_RATIO=0.5
TRAIN_BATCH_SIZE=4
PROFILE_N_MBS=2
LOGPROBS_CHUNK_SIZE=128
LM_HEAD_LOSS_CHUNK_SIZE=0
USE_PRECISION_AWARE_OPTIMIZER=true
MAIN_GRADS_DTYPE=bfloat16
USE_DETERMINISTIC_ALGORITHMS=false
STOP_ON_FAILURE=1
```

Run only the kernel profile:

```bash
PROFILE_KINDS=kernel \
MODEL_PATH=/path/to/Qwen3-30B-A3B \
bash examples/profile/run_qwen3_30b_a3b_sft_profile.sh
```

Run only the memory profile and collect multiple ranks:

```bash
PROFILE_KINDS=memory \
PROFILE_RANKS=0,1,4-7 \
MODEL_PATH=/path/to/Qwen3-30B-A3B \
bash examples/profile/run_qwen3_30b_a3b_sft_profile.sh
```

When `PROFILE_RANKS` is unset, the script derives rank 0 of every pipeline stage from
`actor.backend`. The default `megatron:(attn:d2c4|ffn:d1e8)` configuration has PP=1, so
it resolves to `0`. An explicit `PROFILE_RANKS` value overrides the computed default;
set it to an empty string to collect all ranks.

## Fake 128K data

The fake-data generator in `train_sft_profile.py` follows three rules:

1. It uses the target model's tokenizer to encode a structured conversation instead of
   filling the sequence with one token.
1. It repeats the same token sequence to the fixed `PROFILE_FAKE_SEQ_LEN=131072`,
   ensuring that every parallel layout sees identical token content.
1. By default, `loss_mask` is 1 from the 50% position onward. The first half is the
   prompt/context and the second half is the assistant target.

This avoids profile noise from real-data I/O and dynamic filtering while retaining a
tokenizer distribution that includes long context, code blocks, JSON fragments, and
natural-language targets.

## Outputs

The script stores run-side outputs under `examples/profile/` by default:

```text
examples/profile/profile_data/<timestamp>_qwen3-30b-a3b_fake128k_sft_profile/
  profile_settings.log
  summary.tsv
  qwen3_30b_a3b_fake128k_kernel_step0_<timestamp>/
    launcher.log
    nvidia_smi.csv
    profile_summary.json
    profile_summary.md
    kernel_traces/master/
      traces-r0.chrome.json
      traces-r0.split_clean.chrome.json
      traces-r0.gpu_only.chrome.json
      traces-r0.cpu_only.chrome.json
      traces-r0.cuda_api_only.chrome.json
  qwen3_30b_a3b_fake128k_memory_step0_<timestamp>/
    launcher.log
    nvidia_smi.csv
    profile_summary.json
    profile_summary.md
    memory_snapshots/step_0/
      snapshot_*.pickle
  qwen3_30b_a3b_fake128k_kernel_step1_<timestamp>/
    launcher.log
    nvidia_smi.csv
    profile_summary.json
    profile_summary.md
    kernel_traces/master/
      traces-r0.chrome.json
      traces-r0.split_clean.chrome.json
      traces-r0.gpu_only.chrome.json
      traces-r0.cpu_only.chrome.json
      traces-r0.cuda_api_only.chrome.json
  qwen3_30b_a3b_fake128k_memory_step1_<timestamp>/
    launcher.log
    nvidia_smi.csv
    profile_summary.json
    profile_summary.md
    memory_snapshots/step_1/
      snapshot_*.pickle
```

The original AReaL logs remain under `FILEROOT`:

```text
${FILEROOT}/logs/<user>/qwen3-30b-a3b-sft-profile/<trial_name>/
  trainer.log
  perf_tracer/<role>/traces-r*.jsonl
  memory_snapshots/step_<profile_step>/snapshot_*.pickle
```

Kernel-profile postprocessing generates Chrome trace views next to each trace file and
copies them to `profile_data/.../kernel_traces/<role>/`:

```text
traces-r0.chrome.json
traces-r0.split_clean.chrome.json
traces-r0.gpu_only.chrome.json
traces-r0.cpu_only.chrome.json
traces-r0.cuda_api_only.chrome.json
```

Open these `.chrome.json` files in Chrome at `chrome://tracing` or in Perfetto. With the
default unset `PROFILE_RANKS` and PP=1 configuration, only rank 0 is collected. To
compare GPU memory across DP, CP, and EP ranks, explicitly set `PROFILE_RANKS=` to
collect all ranks.

The default 128K configuration with DP=2 and CP=4 disables deterministic algorithms. Set
`USE_DETERMINISTIC_ALGORITHMS=true` to enable them.

## Standalone postprocessing

To regenerate summaries and kernel trace views for an existing profile:

```bash
python examples/profile/postprocess_profile.py \
  --profile-kind kernel \
  --profile-step 1 \
  --log-dir /path/to/FILEROOT/logs/<user>/qwen3-30b-a3b-sft-profile/<trial_name> \
  --run-dir examples/profile/profile_data/reprocess/<trial_name> \
  --trainer-log /path/to/trainer.log \
  --nvidia-smi-csv /path/to/nvidia_smi.csv
```

## Notes

- Run kernel and memory profiles separately. PyTorch profiler memory recording disturbs
  the kernel timeline, so kernel trials set `AREAL_TORCH_PROFILER_PROFILE_MEMORY=false`
  by default.
- This 128K example sets `actor.megatron.ddp.grad_reduce_in_fp32=false` to reduce the
  steady-state gradient-reduction memory footprint in step 1.
- `LOGPROBS_CHUNK_SIZE` sets `actor.logprobs_chunk_size`, which controls the sequence
  chunk used for logprob and entropy computation. It defaults to 128 to reduce the
  steady-state temporary memory peak of the vocab-parallel path in step 1.
- The Megatron precision-aware optimizer is enabled by default, and TE Adam consumes the
  BF16 distributed gradient buffer directly. Main parameters and both Adam moments
  remain FP32. This avoids allocating another FP32 gradient for every parameter at the
  end of the optimizer step. Set
  `USE_PRECISION_AWARE_OPTIMIZER=false MAIN_GRADS_DTYPE=float32` to restore the original
  path.
- `PROFILE_RANKS` controls which ranks are collected. When unset, the script computes
  rank 0 of every pipeline stage from `actor.backend`; an empty value selects all ranks;
  `0,2-4` selects ranks 0, 2, 3, and 4.
- Qwen3-30B-A3B requires an eight-GPU profiling environment. Without suitable GPUs and
  model weights, only the scripts and postprocessing can be validated; CUDA traces and
  memory snapshots cannot be produced.

## GPU time by launching CPU operator

For original per-rank, per-capture Kineto Chrome JSON traces, run:

```bash
python -m areal.tools.kernel_operator_stats \
  /path/to/traces/ppo_update-step1-rank*.trace.json \
  --output-dir /path/to/cpu_operator_stats
```

The postprocessor also accepts these files via `--operator-traces TRACE...`.
Gzip-compressed JSON is supported. Use original traces containing CPU operators and CUDA
APIs, not GPU-only, track-rewritten, or merged multi-capture views.

Each GPU event is matched through its correlation ID to a CUDA API, then to the shortest
enclosing `cpu_op` interval on that API's original `(pid, tid)` at launch time. This
supports asynchronous kernels that execute after the CPU operator returns. Nested
runtime/driver APIs count as one launch; reused IDs with incompatible launch intervals
remain unattributed. CPU user annotations are not treated as operators.

Outputs:

- `operators-per-trace.csv`: direct GPU time by CPU operator within each input trace.
- `operators-all-traces.csv`: aggregates across the selected traces.
- `validation.json`: input and aggregated event counts/durations, plus attribution
  counts (`attributed`, `no_launch`, `ambiguous_launch`, `no_cpu_operator`).

`gpu_time_us` sums each GPU event exactly once. It is **not** inclusive parent time or
wall time: concurrent streams and ranks may overlap. `gpu_time_share_pct` uses all GPU
events of the same category as its denominator, including unattributed events. Kernel,
memcpy and memset statistics are separate. `cpu_calls_with_gpu` counts distinct CPU
operator occurrences that own an event of that category; it is not the total number of
calls to the operator. No tensor shapes or Python/C++ stacks are required.

This offline path avoids `torch.profiler.key_averages()` and its inclusive aggregation;
large captures should export the raw trace first and perform this aggregation outside
the training process. Files are processed sequentially, retaining only one raw trace at
a time. No GPU is needed.

Communication rows also retain `collective`, `process_group` (the recorded group
description), `group_name` (ID), `group_size`, and `group_ranks` from the owning CPU
operator. Thus `record_param_comms` is split by collective and actual process group,
including separate TP subgroups with the same description. The all-traces CSV keeps
these dimensions; it does not silently merge different group IDs or memberships. Missing
metadata on `record_param_comms` is labeled `unknown`; non-communication operators have
empty communication fields. Group IDs are run-local, so group-type comparisons should
use the description and size, retaining IDs/memberships when inspecting individual
groups. This changes grouping only, not GPU event attribution or the percentage
denominator.

### Coarse components for Qwen3.5

```bash
python -m areal.tools.kernel_operator_stats \
  /path/to/traces/ppo_update-step1-rank*.trace.json \
  --component-rules examples/profile/qwen3_5_profile_components.json \
  --output-dir /path/to/components
```

The postprocessor also accepts `--component-rules`. Rules are opt-in and
model/run-specific: this file assumes text-only Qwen3.5-35B-A3B, 256 routed experts, a
shared expert, all-MoE blocks, MTP disabled, and the recorded Megatron group names.
Inspect the rules before using another model or configuration.

The `component` column inherits the nearest enclosing *recognized* CPU scope on the CUDA
launch thread, while `cpu_operator` retains the direct innermost operator. Recognizable
normalization kernels override that scope so fused Norm/Linear scopes are not counted
twice. Communication metadata overrides enclosing compute scopes; Memcpy is a separate
component. Unrecognized scopes remain `OTHER`. Plain Linear projections remain
`LINEAR_UNRESOLVED` because shared-expert FC and attention output projections cannot be
distinguished from these CPU names. Recognizable shared-expert SwiGLU is labeled
separately; there are no independent dense MLP blocks in this model.

These are exclusive GPU-event categories, **not** complete module wall times. They
include recomputation and any captured autotuning kernels. Generic compiled graphs and
missing module scopes can leave work unclassified. One CPU operator call can own kernels
in multiple components, so its per-component call counts must not be summed as distinct
CPU calls. GPU event counts and durations still sum exactly once.

With component rules enabled, `component-summary.csv` sums GPU events by component and
event category. It provides both the share of all GPU event time (including
Memcpy/Memset) and the share within that event category. The selected rules are saved as
`component-rules.json` alongside the results.
