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

The script verifies the bridge checkout before calling `sbatch`. Prepare the training
runtime first: the actor image supplies Python and CUDA extensions, while
`MCORE_BRIDGE_ROOT/src` and `QWEN_TRAIN_EXTRA_PYTHONPATH` supply the bridge and
Transformers runtime. Submission does not install packages or create a virtualenv.

Bind these runtime directories and the repository at the **same absolute paths** through
`QWEN_CONTROLLER_MOUNTS` and `QWEN_MOUNTS` on every training node. The script passes the
same training `PYTHONPATH` to the controller and all actor workers. Keep the runtime
directories unchanged until the job finishes. The rollout image and its
`QWEN_INFER_EXTRA_PYTHONPATH` remain separate; training paths are not added to rollout.
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

Qwen4Exp vision uses Transformers 5.16.1, including `Qwen4ExpModel.get_rope_index` and
`get_vision_position_ids`. Supply the compatible runtime through `PYTHONPATH`; the
project dependency declarations retain main's Transformers constraints. Initialization
checks the vision capabilities before model allocation. The standard CI image does not
validate full Qwen4Exp vision training.

For CP training, the adapter repairs the pinned bridge's PLE hidden-state gather with an
autograd collective: backward sums gradients from all consuming CP ranks. The bridge's
packed zigzag order and sequence-parallel handling are preserved.

Prepare Transformers 5.16.1 with compatible Tokenizers 0.23.1 and Safetensors 0.8.0 in a
shared runtime directory, and a clean checkout of
`dingzhiqiang/mcore-bridge@7ad1b2d0e1ebc1b11b87c469719f6225e645d2fe` (based on
ModelScope `bc58ea9`, including the generated-vision-token and PLE fixes). Set these
paths in the launch environment:

```bash
export MCORE_BRIDGE_ROOT=/path/to/mcore-bridge
export QWEN_TRAIN_EXTRA_PYTHONPATH=/path/to/transformers-runtime
```

The runtime path must contain importable packages (or use a Transformers checkout's
`src` directory when its compatible dependencies are already installed in the image).
`PYTHONPATH` selects modules; it does not install or resolve their dependencies. Verify
imports inside the actor image before submission and stop if preparation fails. The
submit script checks the bridge revision and constructs the shared training path as
`QWEN_TRAIN_EXTRA_PYTHONPATH:MCORE_BRIDGE_ROOT/src:QWEN_REPO`.

For SFT, export the same training `PYTHONPATH` in the training environment before
launching the recipe. Full GPU training and inference compatibility still requires
validation with the actual images.

The released `awex==0.8.2` includes
[AWEX #121](https://github.com/inclusionAI/Awex/pull/121). Actor and rollout images must
use this version; AWEX 0.8.1 lacks the required APIs.

The small Python helpers in this directory are runtime support, not additional examples:

- `actor_worker.py`, `gdn_cp_compat.py`, `ple_chunked.py` and `grad_norm_guard.py`
  preserve the GDN CP path, causal PLE history/gradients, and rejection of nonfinite
  optimizer updates. Chunked LM-head loss uses 1024-token chunks; PLE uses 8192.
- The SGLang patch helpers validate their source revision before applying the QSA
  compress-gather bounds and image-preprocessing compatibility fixes. `proxy.py` and
  `template_defaults.py` provide model request defaults. Shell helpers launch the
  controller and workers.

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
weight equality and checkpoint recovery with the actual image pair. CPU tests do not
establish full-model numerical parity or RL learning.

RL output defaults to `/storage/openpsi/experiments/qwen38-flash-next/<profile>`. Set
`QWEN_OUTPUT_ROOT` for a specific experiment directory, or `QWEN_EXPERIMENTS_ROOT` to
change the shared experiments root.

Normal RL uses SGLang's native QSA top-k and prefix cache. The recipe does not install
stable top-k or random cache-salt wrappers, replay saved batches, or capture batch
snapshots. Full trajectory/result persistence and verbose rollout tracing are disabled
by default. Health metrics, checkpoint recovery and model compatibility patches remain
enabled. NCCL selects its algorithm and protocol unless explicitly overridden in the
launch environment.

## Production fixes retained in this recipe

| Path                                                                                                         | Problem addressed                                                                                             | Runtime behavior                                                                                                    |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `areal/models/mcore/mcore_bridge_adapter.py`, `registry.py`, `mcore_bridge_checkpoint.py`                    | The standard bridge does not register or export Qwen4Exp's full multimodal model.                             | Register the model, configure frozen PLE/visual parameters, and load/export the matching HF layout.                 |
| `areal/engine/megatron_utils/qwen4_exp_cp.py`                                                                | A non-autograd PLE gather loses gradients from remote CP consumers.                                           | Gather with a backward collective while preserving packed zigzag ordering.                                          |
| `areal/engine/megatron_utils/qwen4_exp_mrope.py`, `packed_context_parallel.py`, `areal/trainer/ppo/actor.py` | Generated special token IDs can be mistaken for image placeholders.                                           | Use the original input loss mask to identify visual prompt tokens and construct three-axis positions.               |
| `areal/engine/sglang_remote.py`, `patch_sglang_qwen4_vl.py`                                                  | Expanded image tokens can trigger whole-prompt retokenization; processor merging must match the model.        | Compact image placeholders only for inference transport, preserve training IDs, and align image preprocessing.      |
| `areal/models/mcore/qwen4_exp_awex_*.py`, `qwen4_exp_frozen_state.py`                                        | Frozen PLE/visual weights are excluded from live transfer and can be lost during inference offload.           | Verify checkpoint contents, preserve visual parameters across offload, and release PLE caches in the correct order. |
| `areal/models/mcore/vision_checkpoint.py`                                                                    | Visual encoder activations increase training memory use.                                                      | Checkpoint visual blocks when gradient checkpointing is enabled.                                                    |
| `gdn_cp_compat.py`, `ple_chunked.py`, `patch_sglang_qsa_compress_gather.py`, `actor_worker.py`               | The 256K recipe needs compatible GDN CP, bounded PLE/LM-head work, QSA bounds, and optimizer memory.          | Install those compatibility helpers and CPU optimizer offload.                                                      |
| `areal/engine/megatron_utils/transport.py`, `areal/engine/megatron_engine.py`                                | Filtered RL groups can leave empty microbatches; the first gradient collective competes with allocator cache. | Choose a real microbatch schedule when possible and reclaim cached memory before the first gradient finalization.   |
| `grad_norm_guard.py`                                                                                         | A nonfinite gradient norm would contaminate optimizer state.                                                  | Reject the update before clipping and stepping, without scanning or dumping full gradients.                         |

The bridge's PLE host-allocation, packed-memory and TP-gradient fixes live in the pinned
external `mcore-bridge` checkout, not in this repository. `examples/swe/config.py` keeps
the OmegaConf-compatible SFT split-mode field, and `areal/utils/stats_logger.py` loads
the optional Trackio backend only when selected; both are needed by the existing
image-based launch path. OpenAI image transport preserves data URIs and shares CPU
vision tensors across completions.

These fixes support model execution and data correctness. This recipe does not enable
MoE routing replay; they do not establish that the remaining training/inference logp
difference is resolved. Validation of the full 64-GPU configuration requires a separate
run with the selected training and inference images.

Arena failure classification, terminal receipts, gateway retry/lease renewal and context
budget recovery remain part of the production workflow. They distinguish model-origin
failures from infrastructure failures and preserve usable training samples. NUMA
placement, asynchronous checkpoint saving and existing runtime reliability fixes are
also retained. The cleanup removes diagnostic helpers and diagnostic defaults only; it
does not split production fixes by whether they are model-specific.
