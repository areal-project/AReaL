# Qwen Flash Next examples

Two configurations cover the supported entry points:

| Configuration                | Purpose                                                                                                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `sft_qwen38_flash_next.yaml` | SFT smoke run; select the dataset with `SFT_DATASET`.                                                            |
| `swe_mm_rl.yaml`             | Multimodal Arena RL; use a SWE-bench Verified stream for text-only tasks or a multimodal stream for image tasks. |

The MM recipe uses 64 GPUs with TP8 / PP4 / CP2 / EP16, batch size 2, four samples per
group and ten training steps. Its context limit is 262144 tokens; generation is limited
to 65536 tokens. Images, mounts, resource placement, model paths and Arena
stream/harness/reward references are supplied through environment variables. No fixed
benchmark selection is bundled.

## Launch

For SFT, set the model, dataset, output, image and resource variables required by
`sbatch_sft_qwen38_flash_next.sh`, then run that script. Use the YAML's environment
variables or CLI overrides for smaller batches and shorter records; a separate math
configuration is unnecessary.

For RL, set `QWEN_LAUNCH_ENV` to your launch environment file and run:

```bash
bash examples/swe/qwen38_flash_next/submit_rl.sh swe
```

The script validates required paths and the bridge revision before submission.
`QWEN_PRIVATE_ENV` supplies credentials to the controller and workers;
`QWEN_ARENA_STREAMS_FILE` uses the standard Arena stream configuration format.
`QWEN_ARENA_TASK_IDS_FILE`, when supplied, selects an ordered subset with explicit
`env:key@version` references from one stream. Match the stream's harness and reward to
the benchmark being measured.

Evaluation reuses the same MM configuration:

```bash
bash examples/swe/qwen38_flash_next/submit_rl.sh swe-eval
```

This mode requires `QWEN_ARENA_TASK_IDS_FILE`. It sets zero training steps, disables
recovery, evaluates each selected task once without shuffling or dropping tasks, and
uses a 32768-token response budget. Task count is derived from the manifest. It performs
no optimizer update or asynchronous training prefetch. Match sampling and harness
settings before comparing scores; one sampled run does not guarantee identical results.

Training defaults `DSH_LLM_REQUEST_TIMEOUT_SECONDS` to 7200 to allow weight-update
pauses. Evaluation leaves this unset, using stream/server defaults. Explicit
`econfig.arena_task_envs` values are preserved; per-stream task envs take precedence.

## Required runtime support

Stable QSA top-k preserves score ordering and resolves exact ties by lower relative
index. Row bounds and finite scores within those bounds are always validated. On CUDA,
device assertions avoid per-call host synchronization; invalid inputs invalidate the
CUDA context and require terminating the worker, not retrying the request. Errors may
surface at a subsequent CUDA operation. CPU inputs retain immediate `ValueError`s.

Qwen4Exp vision needs a Transformers runtime exposing `Qwen4ExpModel.get_rope_index` and
`get_vision_position_ids`, validated with Transformers 5.16.1. The standard locked
SGLang/vLLM environments (Transformers 5.3.0/5.7.0) do not provide this model. Supply a
compatible training runtime; initialization checks these capabilities before model
allocation. The standard CI image does not validate full Qwen4Exp vision training.

For CP training, the adapter repairs the pinned bridge's PLE hidden-state gather with an
autograd collective: backward sums gradients from all consuming CP ranks. The bridge's
packed zigzag order and sequence-parallel handling are preserved.

Both dependency declarations and lockfiles pin the official ModelScope bridge to
`bc58ea9cf9b1dd2314637703973904f359e67c75` and the released `awex==0.8.2`, which
includes [AWEX #121](https://github.com/inclusionAI/Awex/pull/121). Actor and rollout
images must use these versions. AWEX 0.8.1 lacks the required APIs.

The small Python helpers in this directory are runtime support, not additional examples:

- `actor_worker.py`, `gdn_cp_compat.py`, `ple_chunked.py` and `grad_norm_guard.py`
  preserve the GDN CP path, causal PLE history/gradients, and rejection of nonfinite
  optimizer updates. Chunked LM-head loss uses 1024-token chunks; PLE uses 8192.
- The SGLang patch helpers validate their source revision before applying the QSA
  compatibility changes. `proxy.py` and `template_defaults.py` provide model request
  defaults. Shell helpers launch the controller and workers.
- `batch_snapshot.py` supports opt-in diagnosis through `QWEN_BATCH_SNAPSHOT_DIR` and
  replay through `QWEN_BATCH_REPLAY_PATH` or `QWEN_BATCH_REPLAY_PATHS`. Replay consumes
  each supplied batch once and requires disabled recovery and matching step count; it
  does not restore RNG or optimizer state and is not an on-policy RL run.

Vision requires `language_model_only: false`, multimodal SGLang, and processor-produced
modality IDs. AWEX derives frozen-weight exclusions from the model configuration and
checkpoint automatically; no separate manifest is needed. Before the first transfer,
every training rank verifies its frozen PLE/visual weights against the checkpoint. Each
inference rank verifies its local weights and the matching checkpoint content
fingerprints before excluding them from transfer. Only the validated BF16 PLE layout
with absent/unit scale is supported. Visual parameters are preserved across
offload/resume. Checkpoint loading invalidates cached verification, including in-place
parameter updates; changed frozen weights are rejected before the next transfer. Initial
verification reads frozen weights in bounded chunks and can add startup I/O for large
PLE tables.

Training allocations retain expandable segments; AWEX disables them only for IPC staging
allocation and restores allocator settings. Validate CUDA IPC compatibility, repeated
weight equality and checkpoint recovery with the actual image pair. CPU tests and
zero-gradient replay do not establish full-model numerical parity or RL learning.

The experiment runtime pins bridge `557aaf93b16d083fdec4f82a8251d47d47c76ccb` from
`dingzhiqiang/mcore-bridge` for modality-aware visual embedding scatter.

RL output defaults to `/storage/openpsi/experiments/qwen38-flash-next/<profile>`. Set
`QWEN_OUTPUT_ROOT` for a specific experiment directory, or `QWEN_EXPERIMENTS_ROOT` to
change the shared experiments root.
