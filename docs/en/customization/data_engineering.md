# Data engineering recipes

`areal.v2.data_engineering` runs bounded foreground data jobs. It provides recipe
execution, public-task calibration, deduplication, provenance, and verified dataset
snapshots. This is a default task-data workflow; task projects can compose multiple jobs
or use other workflows when their requirements differ. A research controller chooses the
objective and recipe; the recipe supplies domain-specific generation, cleaning, scoring,
selection, and training projection.

## Run the TIR example

From the repository root with AReaL's normal dependencies installed:

```bash
python3 -m areal.v2.data_engineering \
  --config examples/data_engineering/tir/data_generation.yaml \
  --output /tmp/tir-data-job \
  --export-hf /tmp/tir-training-data
```

For a new job, both output directories must be new. Use `--resume` to continue an
existing data job. This example generates exact-answer math tasks without a model, GPU,
or inference service. AReaL's package imports still require its normal Python
dependencies. The final stdout line contains `dataset_id`, `dataset_path`, and, when
requested, `training_dataset_path`.

The TIR recipe has eight families and two structures per family. Its training view
contains `task_id`, `messages`, and an exact boxed `answer`. It produces task data;
training rollouts, tokenization, and RL/SFT/OPD tensors are created downstream.

## Python API and execution protocol

```python
import asyncio
from areal.v2.data_engineering import DataJobConfig, run_data_job

config = DataJobConfig(
    recipe="examples.data_engineering.tir.data_recipe:create_recipe",
    target_count=64,
    purpose="joint",
    seed=42,
    params={"candidate_count": 128},
)
snapshot = asyncio.run(run_data_job(config, "/tmp/tir-python-job"))
rows = snapshot.training_rows()
```

Inside an existing async controller, use `await run_data_job(...)`. A factory is an
importable `module:callable`, invoked as `factory(**params)`, returning a `DataRecipe`.
The public Python protocols are exported from `areal.v2.data_engineering`:

| Type or method                     | Contract                                                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `PublicTask`                       | Stable `task_id`, domain `input`, and public `metadata`; never put an oracle in these public fields                                        |
| `DataSample`                       | `PublicTask` plus a private `reference`; retained samples must serialize to JSON                                                           |
| `generate(context)`                | Yield a bounded batch of samples; the context contains seed, batch index, target, purpose, and input file paths                            |
| `clean(sample)`                    | Return a cleaned sample or `None` to reject it                                                                                             |
| `score(sample, response)`          | Use the private reference to return a finite score in `[0, 1]`                                                                             |
| `select(samples, scores, context)` | Return `Selection` containing exactly `target_count` distinct existing task IDs, or raise `InsufficientDataError` to request another batch |
| `training_row(sample)`             | Project one selected task to one domain-specific JSON training row                                                                         |
| `source_files()`                   | Optional recipe/calibrator method declaring helper code and external files that affect the result                                          |

The executor runs generate → clean → deduplicate/exclude → calibrate → select. Cleaned
candidates and their scores accumulate across batches. Only `InsufficientDataError`
requests another candidate batch; other errors fail the job after any configured
transient retries. Generic `max_batches` and `max_candidates` bound the search; domain
parameters stay in `params`. A recipe may read raw data through `context.inputs` instead
of synthesizing it. Inputs and exclusions are files, resolved relative to the config
file by the module entry point. Paths inside plugin `params` remain plugin-owned; the
supplied calibration adapters resolve them against the process working directory.

`purpose` is `harness`, `model`, or `joint`. It is recorded and passed to the recipe;
recipes decide how it affects generation and selection. It does not automatically infer
whether a failure came from model capability or Harness behavior. Independent dev/test
sets and the decision to adopt a Harness/model remain with the caller.

Identical task IDs must identify identical cleaned content. Duplicate public inputs with
conflicting references fail the job. Exact JSON input duplicates are removed; semantic
or domain-specific normalization belongs in `clean()`. Exclusion JSONL files need
top-level `task_id` fields; use prior snapshots' `public.jsonl` or a previous job's
`work/batch-N-public.jsonl` to exclude previously selected or probed tasks.

## Parallel execution and recovery

Add an `execution` section to a job configuration:

```yaml
execution:
  max_workers: 4
  batch_size: 256
  backend: process
  max_retries: 2
  retry_delay: 1.0
```

`run_data_job` starts a dedicated coordinator thread with its own event loop. Global
coordination, source hashing, SQLite operations, and snapshot I/O execute there, leaving
the caller's event loop responsive. Factories create calibrators inside that job loop;
they must not reuse async resources tied to the caller's loop. The SQLite connection is
created and used in its owning coordinator thread.

Cancelling the caller task signals the job and cancels pending async calibration. The
call waits for worker shutdown, database closure, and final state persistence; repeated
cancellation does not abandon cleanup. Synchronous plugin/storage calls cannot be
forcibly stopped: cancellation takes effect when they return, before starting the next
stage or chunk. A snapshot already atomically published is retained and verified on
resume. This does not make a non-returning plugin interruptible.

`max_workers` defaults to 1, `batch_size` to 256, `backend` to `thread`, and
`max_retries` to 0. `thread` suits blocking I/O; `process` uses spawned Python workers
for CPU-bound generation, cleaning, scoring, ranking, and training projection. Process
workers require importable factories and picklable raw samples. Scripts that use process
workers must call `asyncio.run(...)` under an `if __name__ == "__main__":` guard.
Deduplication and the final global constraint merge run in the coordinator. Async
calibration is split into at most `max_workers` concurrent calls, each with at most
`batch_size` public tasks; repeats stay inside the calibrator protocol.

```bash
python3 -m areal.v2.data_engineering \
  --config examples/data_engineering/tir/data_generation.yaml \
  --output /tmp/tir-data-job --resume
```

Python callers use `await run_data_job(config, output, resume=True)`. Recovery requires
the same configuration, declared input/exclusion files, recipe/calibrator source files,
and executor code. Changed identities fail before execution starts. A completed job
returns its verified existing snapshot. HF export still requires a new destination; omit
`--export-hf` when that dataset has already been exported.

The local `work/checkpoints.sqlite3` database atomically commits each cleaning,
calibration, and projection chunk. Candidate payloads, deduplication indexes, and probe
results and ranked candidates stay on disk; only bounded worker chunks are prefetched.
Snapshot publication and verification stream files. Selection receives a disk-backed
`Sequence[DataSample]` and score `Mapping`: use iteration for large pools, since
repeated indexed lookups/slicing can be expensive. Recipe code can still materialize its
own pool; `training_rows()` and HF export also intentionally materialize rows. The
executor ranks TIR candidates in parallel and selects through disk indexes; calling its
standalone `select()` method still materializes a sorted list.

Use a local filesystem for the SQLite WAL database and process lock. This is single-host
execution, not a Ray/Slurm distributed job. The lock is released on process death, so
SIGKILL does not require deleting a stale lock file. Derived
`work/batch-N-{candidates,public}.jsonl` files are rebuilt from checkpoints.

Recipe methods must be deterministic from their arguments and declared inputs;
cleaning/projection workers create their own instances. No mutable instance state is
saved. If generation was interrupted, it replays from the start of that batch, skipping
committed cleaning chunks; completed generation batches are not replayed. Calibration
must support concurrent calls when more than one worker is configured.

Only `OSError` (including connection failures) and `TimeoutError` retry, using
exponential backoff capped at 60 seconds. Validation errors and cancellation do not
retry. An `Attempt.failure_reason` is a completed calibration result, not an exception
to retry. Persisted chunks are reused, but failed or uncommitted chunks may execute
again. External services should use task/repeat identities for idempotency; the executor
does not promise exactly-once external side effects.

## Parallel generation, scoring, and global selection

Existing recipes remain valid: without the optional hooks, generation and selection use
their original coordinator implementation. Scoring always runs in worker chunks. To
parallelize generation, implement both methods:

| Hook                                     | Contract                                                                                                       |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `generation_shards(context, shard_size)` | Yield deterministic `GenerationShard(start, stop)` ranges, contiguous from zero; `shard_size` is a sizing hint |
| `generate_shard(context, shard)`         | Yield at most `stop-start` raw samples; the complete plan must fit the remaining `max_candidates` budget       |

Each generation worker constructs its own recipe, generates and cleans a shard, and
streams the retained/null rows to `work/generation-B-S.jsonl`. The adjacent JSON
metadata records its range, row count and digest. Complete verified files are reused on
resume, even if the database commit was interrupted. The coordinator imports rows in
bounded `batch_size` chunks and applies global deduplication in shard order. Recipes
choose shard sizes; larger shards use fewer files but have a coarser retry boundary.
`generate(context)` should concatenate the same deterministic shard streams.

Calibration responses are validated and checkpointed before scoring. Grading failures
retry/recover from the stored responses without repeating completed model requests.
Workers instantiate their own scoring recipes, so scores must not depend on mutable
state established by generation or calibration.

To parallelize ranking, implement all three methods:

| Hook                                                    | Contract                                                                                           |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `selection_policy(context, calibrated)`                 | Return `SelectionPolicy(minimum_per_group, maximum_per_bucket)`                                    |
| `rank(sample, scores, context)`                         | Return `None` for ineligible data or a `RankedCandidate` for that same task ID                     |
| `finalize_selection(selected, eligible_count, context)` | Return final task IDs/statistics, or raise `InsufficientDataError` for domain quality requirements |

A `RankedCandidate` contains finite floating-point `priority`, string `tie_breaker`,
`group`, `bucket`, and JSON `metadata`. Workers independently check eligibility and sort
local chunks by `(priority, tie_breaker, task_id)`. SQLite indexes retain the global
ordering; already committed ranking chunks are reused. A candidate's ranking must stay
fixed as later batches arrive; it cannot depend on the evolving candidate pool.
Implementations with adaptive global policies can retain the original `select`.

The coordinator first fills group minima in the policy's insertion order, then fills
remaining slots globally by rank, checking bucket caps and duplicate IDs throughout.
This final merge is serial and deterministic; it is a greedy policy, not a solver for
arbitrary overlapping constraints. Recipes can call `select_ranked_candidates` with an
in-memory ordered-row provider to share exactly the same merge algorithm. The TIR
example uses 16 independently seeded recipe streams, family minima, recipe caps and a
final borderline-case check. Its candidates are invariant to worker count, but this
generator version does not reproduce the previous generator's candidates; recorded
calibration reports must match the actual candidate/source identities.

## Calibration protocol

A calibration factory returns an async `Calibrator`:

```python
async def __call__(self, tasks, context):
    # tasks: Sequence[PublicTask]; context: CalibrationContext
    # Process public tasks and report implementation-specific provenance.
    return CalibrationBatch(tuple(attempts), provenance={"method": "my-calibrator-v1"})
```

An `Attempt` has `task_id`, zero-based `repeat`, JSON-serializable `response`, and
optional `failure_reason`. Every requested task/repeat pair must be returned exactly
once. The generic protocol requires no model or Harness binding. A nonempty failure
reason produces a `None` score; an incorrect completed response receives the recipe's
ordinary score. Backend mutation of public request objects does not mutate retained
candidates.

The calibration backend receives no `DataSample.reference`. The executor grades raw
responses with the recipe's private reference. This is a data contract, not a security
sandbox: recipe and calibrator code is trusted. Implementation-specific dependencies and
version checks belong in the calibrator. Optional JSON provenance is preserved per
batch; for example a rule version, source revision, or evaluated model/Harness identity.
The TIR adapter validates the latter inside its own implementation.

For TIR, use two attempts per task. Its numeric verifier requires a final `\boxed{...}`
and supports integers, decimal values, rational strings, and numeric LaTeX fractions.
Expressions such as `1+2` are not evaluated. Selection drops incomplete and
twice-correct tasks, prioritizes once-correct tasks, and then uses twice-wrong tasks. It
enforces family coverage and the configured recipe cap. Without calibration, it selects
generated tasks without claiming model difficulty calibration.

### Reuse existing TIR evaluation reports

```yaml
recipe: examples.data_engineering.tir.data_recipe:create_recipe
purpose: joint
seed: 42
target_count: 512
max_batches: 8
max_candidates: 8192
params:
  candidate_count: 1024
  minimum_one_pass: 128
  maximum_per_recipe: 48
calibration:
  implementation: examples.data_engineering.tir.data_calibration:recorded_responses
  repeats: 2
  params:
    model_ref: /path/to/the/evaluated/checkpoint
    harness_ref: actual-harness-id
    reports:
      - [/path/to/batch0-repeat0.json, /path/to/batch0-repeat1.json]
    backend_model_digest: actual-model-digest
    verifier_digest: actual-verifier-digest
```

Supply reports for every batch that may be needed. Reports must cover the generated task
IDs and match the requested model, Harness, model digest, and verifier digest. Repeats
must have distinct trial IDs and the same task set/executable identity. Stored scores
are ignored and responses are graded again. Report and source digests enter the snapshot
provenance. Reuse requires matching the original generator seed and candidate count;
this adapter does not run an evaluation or start SGLang.

## Artifacts and training integration

`job.json` records running/succeeded/failed/cancelled state. `config.json`,
`progress.json`, and per-batch files under `work/` retain execution evidence.
Publication occurs only after successful selection and validation, by renaming a staging
directory on the same filesystem. Existing output directories require `--resume`. Jobs
run in the foreground; interrupting the module cancels it. Recovery continues from
committed chunks without a background daemon.

The dataset directory contains:

- `public.jsonl`: selected public tasks.
- `references.jsonl`: aligned private references.
- `train.jsonl`: the recipe's training view.
- `probes.jsonl`: raw calibration responses, failure reasons, and recomputed scores.
- `report.json`: generation, cleaning, exclusion, and selection counts.
- `manifest.json`: content identity, file digests, config, recipe/backend code digests,
  declared source digests, and implementation-specific calibration provenance.

Job directories and dataset files are owner-restricted. `load_snapshot(path)` and
`snapshot.training_rows()` verify file digests and row alignment. Snapshots are
immutable by convention and identified by content; permissions do not make them
write-proof. Factories should declare imported helper code and non-config input files
with `source_files()`. Declared inputs are checked again before publication.

Use the explicit materialization adapter for existing training loaders:

```python
from areal.dataset import get_custom_dataset
from areal.v2.data_engineering import export_hf_dataset, load_snapshot

snapshot = load_snapshot("/tmp/tir-data-job/dataset")
path = export_hf_dataset(snapshot, "/tmp/another-tir-training-data")
dataset = get_custom_dataset(path=str(path), type="rl")
```

The export includes `areal-source.json` with the source dataset identity. Choose a
workflow that consumes the recipe's columns. For SFT or OPD, provide the appropriate
projection/tokenization/teacher-processing adapter; TIR's task-only view is not an SFT
conversation or a tokenized RL trajectory.
