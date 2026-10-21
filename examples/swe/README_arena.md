# Arena single- and multi-Stream training

Arena supplies datasets, runs the external Harness, and returns terminal rewards. AReaL
serves the policy through its rollout proxy and trains on the recorded tokens. This
integration does not require AReaL-SWEAgent, PRM, or RewardSystem.

## Configuration

`arena_grpo.yaml` is a two-node FSDP/SGLang smoke profile: eight training GPUs, eight
inference GPUs, two samples per prompt, and one training step. The request budget is
32767 tokens against a 32768-token SGLang context, leaving its required one-token
margin. Set these environment variables to your deployment's values:

```bash
export AREAL_DIR=/path/to/AReaL
export AREAL_IMAGE=/path/to/compatible-image.sif
export AREAL_PYTHON=/path/to/python/in/container
export AREAL_FILEROOT=/shared/path/to/experiments
export MODEL_PATH=/shared/path/to/Qwen3-4B-Instruct
export ARENA_CREDENTIALS_FILE=/shared/private/arena.env
```

Keep only credentials and the Arena API base in the private credentials file. It must
export `ARENA_OPENAPI_BASE`, `ARENA_OPENAPI_TOKEN`, `ARENA_LLM_API_KEY`, and
`SWE_RL_ADMIN_API_KEY`. Load it in the controller environment as well. Workers load the
same file; credentials are not embedded in worker commands or committed YAML.

`ARENA_LLM_API_KEY` must authenticate to Arena's model gateway. It is not an upstream
model-provider key. If your deployment uses the same Arena bearer token for both APIs,
set `export ARENA_LLM_API_KEY="$ARENA_OPENAPI_TOKEN"` in the credentials file.
Control-plane preflight success alone does not validate model-gateway authentication.

The example assumes a compatible container with Slurm available to the controller,
shared storage, and network reachability from Arena to rollout proxies. Use the included
submission script after setting the site-specific mounts:

```bash
# Comma-separated Apptainer binds supplied by your cluster setup.
# Workers need the shared checkout, model, credentials and output paths.
export AREAL_WORKER_MOUNTS="$SITE_SHARED_MOUNTS"
# Controller additionally needs working Slurm commands and authentication.
export AREAL_CONTROLLER_MOUNTS="$SITE_SHARED_MOUNTS,$SITE_SLURM_MOUNTS"
export AREAL_CONTAINER_BIN=apptainer  # or singularity, available on compute nodes
export SBATCH_PARTITION="$SITE_PARTITION"
export SBATCH_TIMELIMIT=01:00:00      # worker limit
export ARENA_CONTROLLER_TIME=01:30:00
export ARENA_STREAM_ID=your-stream

bash examples/swe/submit_arena.sh --check-arena
bash examples/swe/submit_arena.sh
```

The script prints the submitted Slurm job ID. It allocates a **CPU-only controller** (4
CPUs, 16 GB); config validation and Arena preflight run inside the container before
AReaL submits any GPU workers. `--check-arena` stops after preflight. Check
`$AREAL_FILEROOT/submissions/$TRIAL_NAME/controller-<job-id>.log` and `preflight.json`;
set `TRIAL_NAME` explicitly to identify a run, otherwise a unique name is generated and
used as the controller job name. Use `squeue` to find the job and its name.

`CONFIG_PATH` defaults to `examples/swe/arena_grpo.yaml`. For multi-stream training:

```bash
export CONFIG_PATH=examples/swe/arena_multi_stream.yaml
# Fill in the Stream/Harness/reward entries before submitting.
bash examples/swe/submit_arena.sh
```

Use a custom YAML to change the GPU topology or training settings. Both preflight and
training resolve that same file, including its Hydra defaults. The supplied profile uses
two GPU nodes; the controller does not reserve those nodes itself. Optional Slurm site
settings use the usual `SBATCH_ACCOUNT`, `SBATCH_RESERVATION`, and related environment
variables. The worker scheduler retains its existing default shared-storage bind;
`AREAL_WORKER_MOUNTS` adds the mounts required by your site.

Keep the checkout and configuration unchanged while the job is queued or running; use a
dedicated checkout for each revision. Scripts do not copy or snapshot code. All paths
must be visible at the same locations inside the containers. The credentials file is
sourced only in the controller container and workers, never printed or embedded in
generated commands. In the controller, explicit launch paths, Stream ID and API base
take precedence over older credential files that also contain deployment settings.
Workers source the file again, so remove unrelated deployment settings from it.

For one Stream, set `ARENA_STREAM_ID`. For multiple Streams, edit the literal
`econfig.arena_streams` entries in `arena_multi_stream.yaml` and select that config. Pin
each Stream's actual Harness and reward key/version. Every group of `n_samples` uses one
Stream. The default epoch interleaves all source rows without replacement; a positive
`arena_mixture_epoch_size` selects a weighted subset, must not exceed the union, and
must be divisible by the training batch size. Rejection can change the mixture that
actually reaches training.

Before submission, extract the Stream list and validate it without launching tasks:

```bash
python -m examples.swe.arena_stream_config_projection \
  examples/swe/arena_multi_stream.yaml | base64 --decode > /tmp/arena-streams.yaml
python -m examples.swe.arena_preflight \
  --streams-file /tmp/arena-streams.yaml --base-url "$ARENA_OPENAPI_BASE"
```

The preflight checks Stream/Harness state, reward version, dataset availability, and
recent task health. A new Stream with no history can use `--min-terminal-tasks 0`; this
does not bypass the other checks.

## Routing and rewards

`session_gateway` reuses a registration for each rollout worker and binds every request
to an individual proxy session. The registered key only permits generation; it cannot
end sessions, assign rewards, or export trajectories. Registrations are probed while
tasks are active and cleaned up when the worker is destroyed. `gateway` retains the
older per-rollout registration mode; `direct` requires private network access from the
Harness to the proxy.

Only the result envelope's top-level `score` is used as reward. Per-Stream thresholds
and optional `reward_transform_fn` hooks can transform it. Heterogeneous `raw` payloads
are retained for audit, not parsed for training rewards. Model-attributed failures can
retain interactions with zero reward; system and ambiguous failures are rejected. Result
shards are written under `arena_result_dump_dir` when configured.

### Reward components in W&B

The client preserves the result envelope's `sub_scores`. Finite named entries are logged
as `rollout/reward_sub_score/<name>` and
`rollout/stream/<stream>/reward_sub_score/<name>`. These metrics describe the verifier
score before any training reward transform. They do not change the reward used by GRPO.
Sub-scores are also included in private result audit shards. Both `raw` and `sub_scores`
are independently omitted from audit shards when their encoded size exceeds
`arena_result_dump_max_bytes`; presence, size, and truncation flags remain available for
each field.

For a complete decomposition, the verifier can return a private `raw.reward_breakdown`
object with schema `arena-reward-breakdown/v1`. Arena's existing opaque `raw` field
carries it without a control-plane schema change. Build it after the final aggregation,
before writing `reward_result.json`:

```python
from arena_reward_breakdown import build_fine64_reward_breakdown

# Bundle examples/swe/arena_reward_breakdown.py with the verifier runtime.
# result is the full measured verifier result, not the reduced task API reply.
result.setdefault("raw", {})["reward_breakdown"] = build_fine64_reward_breakdown(result)
```

The helper validates the current fine64 `mean80-min20-v1` measurements. Its tree
contains these operations:

```text
total = 0.9 * rule + 0.1 * vlm
  rule = quality * factor
    quality = 0.8 * mean + 0.2 * minimum
      mean = sum(normalized_part_weight * part_quality)
        part_quality = 0.25 * layout + 0.35 * surface + 0.40 * view
      minimum = min(part_quality); selected_part_ids records all exact ties
    factor = retention * scene_excess
      retention = 1 - 0.15 * anomaly_A - 0.20 * x / (0.20 + x)
      x = max(0, local_E - 0.25)
      scene_excess = 1 / (1 + full_scene_excess_ratio)
  vlm = 0.25 * sum(four majority-voted visual items)
```

Each node has `name`, `operator`, and `value`; aggregation nodes have `children`. Names
are bounded identifiers, unique among siblings. Operations are `weighted_sum`,
`product`, `leaf`, and `terminal`. The top-level object also declares `score_mode`. An
additive child additionally carries:

| Field                    | Meaning                                                      |
| ------------------------ | ------------------------------------------------------------ |
| `value`                  | Measured score of this component                             |
| `weight`                 | Effective coefficient relative to its immediate parent       |
| `declared_weight`        | Original coefficient, or original unnormalized part weight   |
| `contribution_to_parent` | `weight * value`                                             |
| `share_of_parent`        | Contribution divided by the parent score; null if it is zero |

Product children are factors, so they have no additive weight, contribution, or share.
The old `semantic_h` is retained as a sub-score diagnostic; it does not participate in
the current mean80/min20 score. Do not add parent contributions and their descendants
together.

For `RULE_ONLY_FALLBACK`, the effective top-level weights are rule=1 and VLM=0; the
declared weights remain 0.9 and 0.1. VLM `value` is null and `excluded` is true.
`HARD_INVALID` carries a valid terminal zero without invented component zeros;
`RULE_UNSCORABLE` carries a terminal null. The training adapter's model-failure zero
policy remains separate from these verifier outcomes.

The existing stats/W&B pipeline records the tree under these paths, also with a
`rollout/stream/<stream>/` prefix:

```text
rollout/reward_breakdown/present
rollout/reward_breakdown/valid
rollout/reward_breakdown/mode/MIXED_90_10
rollout/reward_breakdown/total/score
rollout/reward_breakdown/total/rule/weight
rollout/reward_breakdown/total/rule/contribution_to_parent
rollout/reward_breakdown/total/vlm/share_of_parent_mean
rollout/reward_breakdown/total/rule/quality/mean/<part>/surface/score
rollout/reward_breakdown/total/rule/quality/mean/<part>/surface/weight
rollout/reward_breakdown/total/rule/factor/score
```

Weights and shares use fractions, not percentages. Scalars are averaged over observed
episodes in the consumed-rollout metric window; each component's `computed` and
`share_computed` metrics expose missing measurements and undefined shares.
`share_of_parent_mean` is the mean of episode shares, not the ratio of batch
contribution sums. For aggregate contribution comparisons, compare the contribution and
parent score metrics over the same observations. Do not mix different score modes.
Malformed optional telemetry is omitted and counted with validity metrics; it does not
reject an otherwise valid reward. Missing older breakdowns are reported as `present=0`
and their existing sub-scores still log.

The numeric verifier fixture in `tests/fixtures/arena_fine64_reward.json` has
total=0.1482429, rule=0.0536032, and VLM=1. The rule's weight is 0.9 but its actual
contribution is 0.0482429 (32.54% of this total); VLM contributes 0.1 (67.46%). This
illustrates why formula weights and observed contribution shares are separate fields.

Keep the breakdown private to grading and training. A published verifier needs a new
immutable version to emit it; existing frozen experiments do not gain the field merely
by updating this checkout.

Structured `LLM_CONTEXT_WINDOW_EXCEEDED` agent failures are recognized by code across
producers, including `provider_stream` and `provider_http`, with a matching native
Harness receipt. Local context-budget outcomes additionally validate the legacy details
or current `estimatedTokens / inputLimit / reason` format. For
`whitebox-unified-b-areal`, the training adapter additionally reads
`whitebox-lifecycle.json`: an explicit `BRIDGE_EXECUTION_DEADLINE` with completed
cleanup and no existing terminal receipt or conflicting failure outcome retains valid
interactions with reward zero. External termination, incomplete cleanup, generic process
kills, and missing or invalid receipts remain rejected. Both recovery paths still
require usable interactions and are vetoed by proxy system errors; they do not turn
setup or evaluation failures into zero rewards.

## Per-task reward mappings

For a Stream whose tasks use different original graders, set `reward_mapping_file` in
its Stream entry instead of `expected_reward_ref`:

```yaml
arena_streams:
  - name: whitebox-b
    stream_id: whitebox-b-overfit64-20261013
    harness: whitebox-unified-b-areal@0.1.0
    llm_protocol: chat_completions
    reward_mapping_file: ${oc.env:EXPERIMENT_ROOT}/reward-mapping.json
```

The file must be on persistent storage visible at the same path to the controller and
rollout workers. During preflight or training dataset setup, AReaL creates it if absent:
it resolves each `env:<key>@<version>` using Stream `metadata.per_task_bindings` and
published Env metadata (`source_reward_ref` or `reward_ref`, plus optional
`harness_ref`). Conflicting sources fail before rollout. Each distinct reward and
Harness version must be published, and references must include an exact version.

When Harness bindings are present, they must cover every dataset row. Each task uses its
pinned Harness in preference to the Stream-level `harness`; its LLM protocol is inferred
from that Harness using the existing protocol inference rules. Snapshots without Harness
bindings retain the Stream-level Harness and protocol behavior.

Subsequent starts reuse the snapshot, check exact dataset coverage, verify the pinned
versions remain published, and check declared Stream bindings for drift. An invalid or
incomplete existing file is rejected, never regenerated automatically. Keep the same
file when recovering; use a new run file for a changed dataset or to add Harness pins to
an older reward-only snapshot. Each rollout validates its row against the snapshot and
sends the task-specific `reward` and `harness` to Arena. Result audit records contain
`expected_reward_ref` and `expected_harness`; terminal `score` processing is unchanged.
Without `reward_mapping_file`, the existing Stream-default behavior is unchanged.
Harness routing does not certify that a Harness forwards Session Gateway headers
correctly; that still requires Harness-specific validation.

## Port provenance and scope

This port applies selected changes from `swe-dev` commits `da1da65c7`, `1d0fc7ba0`,
`714f733a0`, `586d7cc06`, `84f7bb341`, `ddb7d7f3c`, `2b994c10c`, and `9ecbde97c`. It
preserves main's processor cache, sample identity, and cancellation cleanup. Training
engine APIs, mean-only reward normalization, internal Astra launch scripts, AWEX/Qwen3.8
changes, and PRM/RewardSystem are outside this port.

## Flash V3 Megatron profile

`arena_flash_v3.yaml` and `submit_arena_flash_v3.sh` adapt the model, resource and
sampling settings from `swe-dev@5478d6384`'s `submit_arena_astra_multi_stream.sh`. The
default is eight shared training/inference nodes, 128K context, batch 32, 12 samples per
prompt, 1500 concurrent rollouts and 500 training steps. Generation leaves one token of
context headroom. `N_NODES` selects the following 8-GPU-node profiles; the CPU
controller is additional:

| Nodes | Megatron actor                | SGLang rollout |
| ----- | ----------------------------- | -------------- |
| 2     | `(attn:d1p2t4c2\|ffn:d1p2e8)` | `d4t4p1`       |
| 4     | `(attn:d2p2t4c2\|ffn:d2p2e8)` | `d8t4p1`       |
| 8     | `(attn:d2p2t2c8\|ffn:d4p2e8)` | `d16t4p1`      |
| 16    | `(attn:d4p2t2c8\|ffn:d8p2e8)` | `d32t4p1`      |

Use a checkpoint with architecture `BailingMoeV3ForCausalLM`. The profile uses the
existing `mbridge` bridge, CP-capable KDA, Adam at `3e-6`, FP32 LM head, and AWEX weight
updates. MTP is disabled and virtual pipeline size is one. Supply the common submission
environment above, then set deployment-specific paths:

```bash
export MODEL_PATH="$FLASH_V3_CHECKPOINT"
export ROLLOUT_IMAGE="$FLASH_V3_SGLANG_IMAGE"
export AWEX_ROOT="$FLASH_V3_AWEX_SOURCE"
export FLASH_LINEAR_ATTENTION_ROOT="$FLASH_V3_FLA_SOURCE"
# Optional colon-separated module directories, visible inside worker containers:
export ACTOR_RUNTIME_PYTHONPATH="$FLASH_V3_ACTOR_MODULES"
export ROLLOUT_RUNTIME_PYTHONPATH="$FLASH_V3_ROLLOUT_MODULES"

N_NODES=2 bash examples/swe/submit_arena_flash_v3.sh --print-profile
N_NODES=2 bash examples/swe/submit_arena_flash_v3.sh --check-arena
N_NODES=2 TRAIN_BATCH_SIZE=4 N_SAMPLES=2 TOTAL_TRAIN_STEPS=1 \
  MAX_CONCURRENT_ROLLOUTS=4 bash examples/swe/submit_arena_flash_v3.sh
```

The small trial retains the 128K context. Set `CONTEXT_LENGTH` to change it together
with the default generation and microbatch budgets. Adjust worker/controller time limits
for long runs. This is a resource profile, not a guarantee that the selected checkpoint
and sequence lengths fit available GPU memory.

The default dataset file pins the two source streams, their Harnesses and reward
versions at equal sampling weights. Set `DATASET_CONFIG` to another shared streams YAML
when needed. Submission runs the existing Arena preflight before creating GPU workers;
`--print-profile` neither contacts Arena nor submits a job.

The runtime must supply Ling3 tool/reasoning parsers and Flash V3 inference support. The
current AWEX plugin accepts SGLang `0.5.9` or `0.5.10.post1`; the external AWEX checkout
must register the Flash V3 converter. FLA must support the checkpoint's KDA `safe_gate`
and `lower_bound` arguments. The script does not install dependencies. GPU workers
explicitly disable expandable-segment allocation for AWEX compatibility.

This profile keeps main's outcome-only GRPO: rollout group normalization includes
standard deviation, and incomplete groups are dropped. It does not reproduce `swe-dev`'s
PRM/RewardSystem rewards, mean-only group normalization, minimum-eight partial groups,
or adaptive thinking controls. Chat rendering uses the checkpoint's template defaults
and any Harness request overrides. These differences affect training semantics; the
profile is intended for Arena integration validation, not a claim of full recipe
equivalence.
