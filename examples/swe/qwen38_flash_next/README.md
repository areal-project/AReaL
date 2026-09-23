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

The script prepares a fresh shared uv environment inside `QWEN_ACTOR_IMAGE` before
calling `sbatch`. The image must provide Python 3.12, uv >=0.12.6, and the training CUDA
extensions (including Transformer Engine and FlashAttention). Environment preparation
needs package-index and Git access. An installation or import-check failure stops
submission.

Each submission creates `QWEN_OUTPUT_ROOT/uv-runtime-XXXXXXXX`. Bind that shared output
directory and the repository at the **same absolute paths** through both
`QWEN_CONTROLLER_MOUNTS` and `QWEN_MOUNTS` on every training node. The controller and
all actor workers use this environment's Python; workers do not install packages. The
environment inherits CUDA extensions from the actor image and installs the locked Qwen
dependency group. Keep it until the job finishes; a later submission creates a new
environment without modifying an active job's runtime.

RL no longer needs `MCORE_BRIDGE_ROOT` or `QWEN_TRAIN_EXTRA_PYTHONPATH`: it imports the
installed bridge and Transformers. The rollout image and its inference `PYTHONPATH`
remain separate. `QWEN_PRIVATE_ENV` supplies credentials to the controller and workers;
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

Qwen4Exp vision uses Transformers 5.16.1, including `Qwen4ExpModel.get_rope_index` and
`get_vision_position_ids`. The opt-in group below installs that version; default
SGLang/vLLM installs retain Transformers 5.3.0/5.7.0. Initialization checks the vision
capabilities before model allocation. The standard CI image does not validate full
Qwen4Exp vision training.

For CP training, the adapter repairs the pinned bridge's PLE hidden-state gather with an
autograd collective: backward sums gradients from all consuming CP ranks. The bridge's
packed zigzag order and sequence-parallel handling are preserved.

In a Python 3.12 Linux x86_64 environment with uv 0.12.6 or newer, install the Qwen
runtime from an AReaL source checkout:

```bash
uv sync --locked --extra cuda --no-group transformers-default --group qwen-flash-next
```

The group pins Transformers 5.16.1, Tokenizers 0.23.1, and
`dingzhiqiang/mcore-bridge@557aaf93b16d083fdec4f82a8251d47d47c76ccb`, based on
ModelScope `bc58ea9` with the generated-vision-token embedding fix. The default
Transformers group is explicitly excluded because the two versions cannot be installed
together. Dependency groups are not published extras, so the bridge Git URL stays out of
the package's published requirements. Stop environment preparation if installation
fails; do not launch workers with a stale bridge. The RL submission script performs this
preparation automatically and selects the shared interpreter for the controller and
every training worker. Do not also prepend another Transformers or bridge checkout to
`PYTHONPATH`, which would override these installed versions. Existing experiments with
separate launch scripts can keep their source-directory setup; the supplied
`submit_rl.sh` now uses the installed runtime. Selecting the group configures
dependencies; full GPU training and inference compatibility still requires validation
with the actual images.

The released `awex==0.8.2` includes
[AWEX #121](https://github.com/inclusionAI/Awex/pull/121). Actor and rollout images must
use this version; AWEX 0.8.1 lacks the required APIs.

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

RL output defaults to `/storage/openpsi/experiments/qwen38-flash-next/<profile>`. Set
`QWEN_OUTPUT_ROOT` for a specific experiment directory, or `QWEN_EXPERIMENTS_ROOT` to
change the shared experiments root.
