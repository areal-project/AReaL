# Hermes and DAPO on Arena

This adapter runs one native `hermes-agent==0.19.0` conversation in an Arena agent
sandbox. The model remains behind an external OpenAI-compatible endpoint. A separate
Reward sandbox uses the existing DAPO `score_answer` implementation. Training and
inference workers remain owned by AReaL.

## Components

| File             | Purpose                                                                           |
| ---------------- | --------------------------------------------------------------------------------- |
| `runner.py`      | Read a task, bind model/episode identity, run Hermes, write artifacts             |
| `python_tool.py` | Execute `math_python` in an offline bubblewrap service                            |
| `reward.py`      | Read the answer and private oracle, write the Arena reward envelope               |
| `dataset.py`     | Export an exact row from an existing manifest-checked DAPO split                  |
| `smoke.py`       | Exercise real Hermes against a scripted HTTP model without GPUs                   |
| `package.py`     | Build an immutable, self-extracting executable with optional offline dependencies |
| `config.py`      | Load the pinned component references, sampling and task index                     |
| `backend.py`     | Check the registry, submit tasks, download artifacts and checkpoint identities    |
| `cli.py`         | Run the shared episode client with Arena as its task executor                     |

The adapter shares tool registration, SDK identity binding and timing helpers with Agent
Service through `harnesses/hermes_runtime.py`. It shares the original Python tool
validation, output limits, call accounting and process collector.

## Run the published combination

The client uses the legacy Arena CLI's existing authenticated profile. It does not read
or copy the platform token. Install `httpx` and `tqdm` from the project's existing
dependencies; the client needs no Hermes installation or training stack. Hermes runs
inside the published sandbox. A model endpoint reachable **from Arena** is required for
real evaluation; access from the caller's workstation alone does not establish this.

Keep deployment configuration and frozen datasets outside the checkout. A configuration
has this shape (replace placeholders with published references and actual SHA256
values):

```json
{
  "format_version": 1,
  "name": "hermes-dapo",
  "benchmark": "dapo_math",
  "model_mode": "external",
  "arena": {
    "binary": "/absolute/path/to/arena",
    "base_url": "https://arena.example.invalid/openapi/v1",
    "profile": "prod",
    "stream_id": "my-dapo-stream"
  },
  "harness_ref": {"key": "my-hermes-dapo", "version": "0.1.3"},
  "reward_ref": {"key": "my-dapo-answer", "version": "0.1.3"},
  "component_spec_sha256": {
    "my-hermes-dapo@0.1.3": "SHA256_OF_CANONICAL_HARNESS_SPEC",
    "my-dapo-answer@0.1.3": "SHA256_OF_CANONICAL_REWARD_SPEC"
  },
  "oracle_schema": "dapo.final-answer.v1",
  "sampling": {"max_tokens": 4096, "temperature": 1.0, "top_p": 1.0},
  "dataset": {
    "source": {
      "dataset": "open-r1/DAPO-Math-17k-Processed",
      "revision": "IMMUTABLE_DATASET_REVISION",
      "config": "en",
      "split": "train"
    },
    "count": 14116,
    "task_index": "task-index.jsonl",
    "task_index_sha256": "SHA256_OF_INDEX_FILE"
  }
}
```

Each index line contains `task_id`, `data_id` (`env:key@version`) and `prompt_sha256`.
It contains no answer. Component spec digests use `config.digest`: UTF-8 JSON with
sorted keys, no extra whitespace and `ensure_ascii=False`. The index digest covers the
exact file bytes. A Stream has no version; the index pins its complete membership. The
frozen task file and sibling manifest still choose the experiment's subset/order.

```bash
python -m examples.task_workflows.arena.cli check \
  --config "$ARENA_CONFIG" --tasks "$FROZEN_DAPO_SPLIT" --limit 1

# AREAL_INFERENCE_ADMIN_KEY must already be available in the environment.
python -m examples.task_workflows.arena.cli run \
  --config "$ARENA_CONFIG" --tasks "$FROZEN_DAPO_SPLIT" --limit 1 \
  --output "$NEW_RUN_DIR" --inference-url "$AREAL_INFERENCE_URL" \
  --model "$AREAL_MODEL_PATH" --mode eval --concurrency 1

python -m examples.task_workflows.arena.cli resume --run-dir "$NEW_RUN_DIR"
```

`check` is read-only: it compares published spec hashes, Stream defaults and full
membership, and selected prompts/schema. It does not prove model connectivity or RL
correctness. `run` repeats that check before submitting. `--inference-url` is the origin
without `/v1`; `--inference-key-env` can select another environment variable. Removing
`--limit` submits the supplied frozen file, not every question in the Stream.

For online RL, start the existing v2 Rollout and Actor/Trainer and use
`--mode resident`. The shared `EpisodeClient` creates one session, submits Arena with
its session key and `AUTOPILOT_AREAL_SESSION_ID`, reads the separate Reward, and calls
AReaL's `set_reward`. Native inference captures the session's tokens and hands the
sealed trajectory to Trainer. No Agent Service is started. Configure the reward finish
timeout to zero. The client checks the reward receipt; actual Actor consumption and
updates require independent verification.

The old `on_demand` mode, `--state-root`, `--handoff-timeout` and SQLite inbox audit
were removed with the Elastic runtime refactor. Use a new run directory for the new
session protocol. Rebuild and publish an immutable runner version that emits
`areal_session_id`, then update the pinned Harness reference and spec digest; the
previously published `0.1.3` runner does not implement this session provenance.

The run directory snapshots configuration, selected tasks (including private labels),
the index and stable request IDs. Per-episode `state.json` holds session credentials
with mode `0600` under private directories; keep the entire run private. Artifact files,
execution references, reward receipts and `summary.json` support inspection.
`episodes.jsonl` is an observation journal and can contain replayed observations after
resume; stable request IDs identify the same episode. A directory lock prevents two CLI
processes resuming the same run simultaneously.

Resume reuses saved identities and completed results. It never automatically resubmits
an Arena launch whose acceptance is unknown. A session-creation intent with no saved
response also requires operator inspection. With the native session protocol, an
unconfirmed reward submission requires inspection before retrying; a saved reward
receipt is reused. Resume cannot repair expired sessions, lost inference workers, or
uncertain generations. Mock configurations are refused for training.

Explicit Job IDs follow the legacy `short-name-YYMMDDHHMM` constraint. The client
combines a request digest with a UTC timestamp and checkpoints the selected ID before
submission; resume reuses that stored ID even in a later minute.

## Export one frozen task

From the checkout root, supply a split with its existing sibling `manifest.json`:

```bash
python -m examples.task_workflows.arena.dataset \
  --task-file "$FROZEN_DAPO_SPLIT" --index 0 --output "$TASK_EXPORT"
```

Only `public/query.txt` is agent input. `reward_only/oracle.json` is uploaded separately
and bound as the Env oracle. Preserve `selection.json` for the exact source revision,
split digest and prompt digest. Exporting does not select a new training/evaluation
split or change the original instruction.

## Run and package

The agent interpreter needs Hermes 0.19.0. The separate reward interpreter needs
`math-verify==0.8.0` for non-rational answers. The image must provide bubblewrap and
allow rootless user, network, mount and PID namespaces. `Dockerfile` is a
dependency-image recipe; its supplied base must already satisfy those conditions.

For images with dependencies preinstalled:

```bash
python -m examples.task_workflows.arena.package --mode runner --output "$RUNNER"
python -m examples.task_workflows.arena.package --mode reward --output "$REWARD"
```

For offline delivery, `--vendor` accepts a site-packages directory built for the
**probed target interpreter and platform**, not the development machine. Include
`.target.json`, for example `{"python":[3,14],"machine":"x86_64","system":"Linux"}`.
Record exact dependency versions when preparing it. Target mismatch fails before Hermes
is imported. The package carries only the explicit source whitelist and supplied
dependencies, with deterministic archive metadata.

`--mode bundle --vendor "$VENDOR" --recipe "$RECIPE"` combines the three entrypoints.
The Harness must fix `AUTOPILOT_ENTRYPOINT=runner` or `smoke`; the Reward command fixes
it to `reward`. There is no automatic fallback to a scripted model. `--bootstrap-deps`
is an explicit alternative for development networks that permit package downloads;
offline delivery is preferable for repeatable task startup.

Upload the executable as a **single file** and use its returned URI as `runtime.uri`,
with `runtime.type=runner` and `agent.env.RUNNER_BIN=/envarena/harness/bin/runner`. For
Reward, publish it as one `files` entry with a filename destination and execute it with
`sh`. Select `evaluation_mode=separate`; never put the oracle in the runtime bundle.

## Runtime contract

The external-model runner reads `ARENA_QUERY_PATH`, `ARENA_AGENT_OUTPUT_DIR`,
`ARENA_OUTPUT_DIR`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `MODEL_NAME`.
`AUTOPILOT_SAMPLING_JSON` must contain explicit `max_tokens`, `temperature` and `top_p`
from the experiment configuration. `AUTOPILOT_AREAL_SESSION_ID` records the native
training session in the answer. All SDK generation clients use that session's
`OPENAI_API_KEY`, including recreated/summary clients. The session ID is provenance
metadata; authentication and token capture use the key. Optional
`AUTOPILOT_AREAL_EPISODE_JSON` remains supported for legacy bindings and the scripted
smoke. Arena's own `session_id` alone does not bind model calls to AReaL.

The agent writes `answer.json`, `environment.json`, `call_trace.json`, and per-call
`tools/.../tools.jsonl`. Declare those paths in `io_contract.artifact_globs`.
`harness_result.json` describes platform completion. Reward reads `answer.json` and
`ARENA_ORACLE_PATH` (a file, or a directory containing `oracle.json`), then writes
`ARENA_REWARD_OUTPUT_FILE`; normalize `.score`. Execution and grading failures remain
failures, rather than successful wrong answers.

The bubblewrap service has a private network/PID/mount namespace, no `/proc` or
`/envarena`, a clean environment, a persistent 64 MiB `/work`, and 32 MiB `/tmp`. Each
call starts a fresh interpreter. Existing call/time/code/output limits apply. Children
use one CPU affinity, a 64-process RLIMIT, and 512 MiB address-space limits **per
process**. The Arena task's aggregate memory limit is separate; this is not identical to
the Docker backend's aggregate 512 MiB cgroup limit. The interpreter/base libraries are
read-only. The startup isolation probe is included in `environment.json`.

## External CPU network probe

`network_probe_server.py` is a standalone Python standard-library server. Copy this
single file to a CPU development host; it does not import AReaL or require a GPU:

```bash
python3 network_probe_server.py --bind 0.0.0.0 --port 8000 --output "$PROBE_DIR"
```

It writes a private `key` file, `server.json` and credential-free `requests.jsonl`.
Transfer the key file to the caller and use its value through an environment variable
for an `eval` run. You can instead pre-create `$PROBE_DIR/key` with an explicit
disposable fixture value shared with the caller; never use a real model or Arena token
for this scripted endpoint. Keep the server running while submitting one task with the
published external Harness, `--model scripted-network-probe` and the host's reachable
origin. `/healthz` reports the server instance ID. A browser-authenticated
port-forwarding URL does not by itself establish that Arena can reach the service.

The server asks Hermes to run authored `print(6 * 7)` code, then returns the tool's
output in a final answer. It supports JSON/SSE responses, records completion IDs for
matching against `call_trace.json`, and refuses requests containing `areal_episode`.
Usage is synthetic. This tests external connectivity and tool execution; it does not
solve the selected DAPO question, so even score zero can be a successful protocol probe.
An `external` transport label in the Harness answer is not proof of a real model; record
these runs as scripted probes and keep them out of training and quality results.

The `0.1.3` runtime can represent a Hermes provider connection failure as an incomplete
answer with reward zero. Platform `DONE`, Reward `OK`, or the client's completed task
count therefore cannot certify connectivity. Inspect `answer.completed`, successful
request completion IDs, and the expected tool roundtrip. Structured provider failure
classification is still required before accepting this runner for RL training.

## CPU smoke and verification

A smoke recipe has only `prompt_sha256` and authored Python `code`, with no gold. The
scripted model requests that computation and derives its final answer from the real tool
output. On Arena, run the `smoke` entrypoint with bubblewrap. For portable local
development only, `smoke.py --local-trusted-fixture` permits the exact authored code
through an explicit transport double; this option is refused inside Arena.

Check Task DONE, TaskResult OK, the reward's raw provenance, answer completion, both SDK
requests, the tool output and the request/tool/request trace edges. Smoke samples mark
`model_mode=mock`, synthetic usage and `training_eligible=false`. They demonstrate
integration and must not be used as model-quality or RL evidence.

The real RL caller and native Trainer still own sessions, reward submission, trajectory
sealing/export and consumption. Verify actual model token capture and these boundaries
against the running AReaL inference service before training.
