# Elastic Actor Lifecycle Demo

Run real SWE-Gym code-repair tasks on **8×H20 GPUs**. SGLang rollout uses two TP2
replicas on four GPUs. A Megatron TP4 Actor uses the other four GPUs and can release and
restore its entire worker group while rollout services and the CPU coordinator remain
online. Weight publication briefly pauses generation.

**Status:** both resident and elastic runs completed eight updates, consumed 64 training
trajectories, and passed final cleanup checks. The elastic run released and restored the
Actor once, then continued training. **This run showed no resource savings** and did not
satisfy the stricter rollout-progress-during-offline criterion. A clean-machine rebuild
remains unverified. See [reference results](RESULTS.md) and
[environment setup](ENVIRONMENT.md).

## Fixed recipe

Both policies use the same [config.yaml](config.yaml) and
[inputs.lock.json](reproduction/inputs.lock.json). Only `resident` / `agent` changes.

| Item                 | Configuration                                                                                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model                | `Qwen/Qwen3-4B-Instruct-2507`, with a pinned revision and file SHA256 hashes                                                                      |
| Training / inference | Megatron `d1t4` on GPUs 0–3 / SGLang `d2t2` on GPUs 4–7                                                                                           |
| Concurrency / batch  | Two task groups / two task groups, with `n_samples=4` per task                                                                                    |
| Updates              | Eight updates consume 16 groups and 64 training trajectories; retries can increase the actual episode count                                       |
| Tasks                | Eight uniformly sampled SWE-Gym training tasks, seed 1, repeated twice in a fixed 16-row order                                                    |
| Generation           | Staleness 0, concat export, 16K context, at most 4096 new tokens per turn and 24 turns                                                            |
| Reward               | SWE test-based grading; grading exceptions receive zero reward and retain diagnostics, so these results do not establish strict SWE success rates |
| Controller           | OpenRouter `stepfun/step-3.7-flash`, used only by the elastic policy                                                                              |

## Control and execution

`resident` starts the Actor when initial collection begins and keeps it online. `agent`
makes decisions on initial collection, missing batches after updates, release,
significant progress changes, or a scheduled wake-up. It can restore the Actor before a
batch is complete. Agent decisions do not wait for a fixed 40-second idle threshold.
Progress notifications are coalesced over at least five seconds; per-token or per-second
ETA changes do not trigger calls. Missing history produces an unknown ETA, with a prompt
to prefer starting or retaining the Actor. An early restore keeps the Actor online until
the next update. Native batch-readiness rules still determine when training can run.

The controller receives only actions valid for the current Actor state. The coordinator
records invalid proposals as `outcome=rejected`, returns fresh status, and retries once
immediately. A second invalid proposal retains the current state and retries after five
seconds or a new event. API and infrastructure failures still propagate. Only accepted
proposals appear in decision plots; `events.jsonl` records actual execution.

Before release, the runtime saves full training state and confirms GPU-process exit.
Training resumes only after restoration completes. Initial startup, checkpoint and
release estimates are 90, 40 and 15 seconds, with a 15-second restore margin.
Measurements update these estimates. A workload may provide no useful release window.

## Prepare and preflight

Run commands from the **AReaL repository root in the training Python environment**.
Follow [ENVIRONMENT.md](ENVIRONMENT.md) to prepare the training container, model, task
data, separate SWE SDK, AEnvironment services and task images. Keep private paths and
credentials outside the repository.

```bash
export DEMO_MODEL=/demo/model
export DEMO_DATASET=/demo/data/tasks.jsonl
export DEMO_SWE_RUNTIME=/demo/private/runtime.json
export DEMO_RUNS=/demo/runs
mkdir -p "$DEMO_RUNS"

# Check input hashes, model integrity and configuration parsing.
# This does not start GPUs, sandboxes or the control LLM.
python -m examples.elastic.reproduce --policy resident \
  --model "$DEMO_MODEL" --dataset "$DEMO_DATASET" \
  --swe-runtime "$DEMO_SWE_RUNTIME" --output "$DEMO_RUNS/preflight" \
  --prepare-only --verify-model
```

The entry point checks the fixed YAML, task data, image manifest and model metadata.
`--verify-model` also hashes approximately 8 GB of model weights. Every output
directory, including preflight, must be new. Preflight does not validate CUDA, NCCL,
sandbox operation or grading. A first deployment must also pass the real empty-patch /
gold-patch checks in the environment guide.

## Run both policies

```bash
# Eight-update resident baseline; no OpenRouter configuration required.
python -m examples.elastic.reproduce --policy resident \
  --model "$DEMO_MODEL" --dataset "$DEMO_DATASET" \
  --swe-runtime "$DEMO_SWE_RUNTIME" --output "$DEMO_RUNS/baseline"

# Skip if already configured. The key prompt hides input and writes private config.
python -m examples.elastic.configure_openrouter

# Eight-update elastic run with the same model, tasks and SDK.
python -m examples.elastic.reproduce --policy agent \
  --model "$DEMO_MODEL" --dataset "$DEMO_DATASET" \
  --swe-runtime "$DEMO_SWE_RUNTIME" --output "$DEMO_RUNS/elastic"

python -m examples.elastic.plot "$DEMO_RUNS/baseline"
python -m examples.elastic.plot "$DEMO_RUNS/elastic"
```

OpenRouter configuration defaults to `~/.config/areal-elastic/openrouter.json`. Set
`AREAL_CONTROL_LLM_CONFIG` to use another private file, or use the
[environment-variable template](openrouter.env.example). Do not commit credentials.

The output filesystem needs at least **58 GiB free for the baseline** or **114 GiB for
elastic**, including space for old and new checkpoints during replacement. Reserve
additional space for logs and task images. The measured full 4B checkpoint was about 45
GiB; `52 GiB` is a capacity estimate. Run the policies sequentially. To explicitly
retire a completed baseline checkpoint:

```bash
python -m examples.elastic.retire_checkpoint --run "$DEMO_RUNS/baseline"
```

This preserves metadata and logs but deletes weights, so the retired checkpoint cannot
resume training. Recheck free space before starting elastic.

## Inspect results

- `invocation.json`: subprocess duration, exit code and model-hash verification status.
- `plan.json`, `config.json`, `tasks.*`: actual configuration and inputs.
- `result.json`: completed updates, serving versions, consumed data, grading exceptions
  and the observed elastic-cycle result. Final shutdown does not count as mid-run
  release.
- `state/events.jsonl`, `state/decisions.jsonl`, `observations.jsonl`: executed
  lifecycle events, LLM proposals, task progress and ETA.
- `episodes.jsonl`, `swe-episodes/*/report.json`: task results and diagnostics.
- `gpu-memory.jsonl`, `timeline.svg`, `timeline.png`: GPU samples and plots from real
  logs.

See [RESULTS.md](RESULTS.md) for reference measurements and metric definitions, and
[VALIDATION.md](VALIDATION.md) for CPU and eight-GPU checks. Compare policies from the
same published source revision; older four-step runs are not a substitute. Model
sampling and controller decisions vary, so exact timings and Actor release are not
guaranteed.

## Code entry points

- `reproduce.py` / `run_demo.py`: fixed-input checks, policy launch and result
  collection.
- `agent.py` / `swe/`: control LLM, real task execution, grading and progress
  observations.
- `areal/elastic/`: CPU coordination, Actor lifecycle, data consumption and checkpoints.
  This demo supports planned release with a resident coordinator; it does not replay
  failed PPO updates automatically.
- `areal/v2/`: full training-state support and AWEX reconnection, including shared
  fixes.
- `tests/test_elastic*.py` / `tests/elastic_tools/`: CPU regression and explicit GPU
  checks. The old `tests.elastic_gpu_restart` and `tests.elastic_gpu_weight_sync`
  commands remain forwarding entry points.
