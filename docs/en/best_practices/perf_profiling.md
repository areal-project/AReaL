# Performance Profiling

AReaL provides a lightweight profiling infrastructure through `perf_tracer` to help you
identify performance bottlenecks in distributed training workflows. The tracer emits
Chrome Trace-compatible events that can be visualized in Perfetto or chrome://tracing,
making it easy to correlate computation, communication, and I/O across multiple ranks.

**Key capabilities**:

- Flexible tracing APIs: decorators (`@trace_perf`, `@session_context`,
  `@trace_session`), context managers (`trace_scope`/`atrace_scope`,
  `trace_session_phase`/`atrace_session_phase`), and markers (`instant`)
- **Per-session lifecycle tracking** (task registration → session creation → generation
  → reward → finalization) with derived metrics (total time, generation time, tool call
  time, reward calculation time)
- **Task-session hierarchy** for tracking dataset-level tasks and their sample-level
  sessions

## Quick start

### 1. Enable tracing in your config

Add a {ref}`PerfTracerConfig <section-perf-tracer>` to your training script's YAML
config or CLI overrides:

```yaml
perf_tracer:
  enabled: true
  experiment_name: ${experiment_name}  # Reuse top-level metadata
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}        # Shared filesystem path
  save_interval: 1                     # Write traces every step
  session_tracer:
    enabled: true                      # Track per-session lifecycle
    flush_threshold: 100               # Buffer 100 sessions before flushing
```

Configure the tracer in your YAML configuration file.

### 2. Initialize the tracer

Call `perf_tracer.configure()` once per rank at startup:

```python
from areal.utils import perf_tracer

if config.perf_tracer is not None:
    perf_tracer.configure(config.perf_tracer, rank=rank)
```

The global tracer is now active for this process.

### 3. Run your training and collect traces

Execute your training script as usual. The tracer automatically writes events to
`fileroot/logs/.../perf_tracer/traces-r{rank}.jsonl`. For multi-rank jobs, each rank
produces its own file.

```bash
python examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml scheduler.type=local
```

### 4. View traces in Perfetto

Convert JSONL to JSON and open in [Perfetto](https://ui.perfetto.dev/) or
chrome://tracing:

```bash
python -m areal.tools.perf_trace_converter logs/**/perf_tracer/traces-*.jsonl merged.json
```

## Profiling patterns and APIs

### Pattern 1: Trace entire functions with `@trace_perf`

**Use case**: Understand total time spent in key methods (train_batch, forward,
ppo_update, etc.).

**API**: `@trace_perf(name, category=...)`

- Decorator that wraps sync/async functions
- Automatically records start/end timestamps
- Handles exceptions gracefully

**Example** (from `areal/engine/fsdp_engine.py`):

```python
from areal.utils.perf_tracer import trace_perf

@trace_perf("fsdp_engine.train_batch")
def train_batch(self, input_: dict[str, Any], loss_fn, loss_weight_fn):
    # Training logic here
    ...
```

This creates a "complete event" (duration span) named `fsdp_engine.train_batch` in the
Chrome Trace output.

### Pattern 2: Trace code blocks with `trace_scope` / `atrace_scope`

**Use case**: Profile specific code sections without extracting them into methods.

**API**: Context managers for sync/async code

- `with trace_scope(name, category, args)`: Sync context
- `async with atrace_scope(name, category, args)`: Async context

**Example**:

```python
from areal.utils import perf_tracer
from areal.utils.perf_tracer import Category

with perf_tracer.trace_scope(
    "train.rollout",
    args={"global_step": global_step, "epoch_step": step},
):
    batch = actor.prepare_batch(dataloader, n_samples)
    # Rollout generation happens here
```

The `args` dict attaches metadata (step numbers, batch size, etc.) to the event, visible
in the trace viewer.

### Pattern 3: Track session lifecycles with `@trace_session`

**Use case**: Measure per-session timing for async rollout workflows (e.g., how long
does a single prompt take from submission to reward calculation?).

**API**: `@trace_session(phase)`

- Decorator for sync/async methods that participate in session processing
- Automatically reads `session_id` populated by `@session_context()`
- Records `mark_{phase}_start` and `mark_{phase}_end` events

**Example** (from `areal/workflow/rlvr.py`):

```python
from areal import workflow_context
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_perf,
    trace_session,
)

class RLVRWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine: InferenceEngine, data: dict[str, Any]):
        # WorkflowExecutor automatically sets task_id before calling this method
        # Each sample will register its own session_id

        # Generate responses and collect rewards for n_samples
        resp, reward = self._collect_samples(engine, req, prompt_str, data)

        # Build result tensors
        return self._build_result_tensors(resp, reward)

    @session_context()
    async def _collect_samples(
        self,
        engine: InferenceEngine,
        req: ModelRequest,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[ModelResponse, float, str]:
        """Generate one sample and compute its reward with session tracing."""
        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)

        reward = await self._compute_rewards(
            resp,
            prompt_str,
            task_data,
        )

        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        return resp, reward

    @trace_session("reward")
    async def _compute_rewards(self, resp, prompt_str, task_data):
        """Compute rewards with automatic phase tracing."""
        completion_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt_str, completion_str,
            resp.input_tokens, resp.output_tokens,
            **task_data
        )
        return reward
```

#### `session_context` decorator

`session_context()` registers a new session for each invocation once a `task_id` already
exists in the current context. Use it on the workflow method that handles a single
sample so that downstream helpers (e.g., `atrace_session_phase`, `@trace_session`) can
read the `session_id` from contextvars.

```python
from areal.utils.perf_tracer import atrace_session_phase, session_context

class MiniWorkflow:
    @session_context()
    async def collect(self, engine, request):
        async with atrace_session_phase("generate"):
            return await engine.agenerate(request)
```

**How it works**:

1. `WorkflowExecutor` calls `perf_tracer.set_task_id()` before invoking `arun_episode`
1. `arun_episode` spawns multiple `_collect_samples` calls (one per sample)
1. Each `_collect_samples` applies the `@perf_tracer.session_context()` decorator to
   register sessions automatically and place `task_id` / `session_id` into context
   variables
1. Context variables store the active task/session IDs transparently
1. Child async functions inherit this context automatically
1. `@trace_session("reward")` reads the session ID and logs phase start/end events
1. Session traces appear in `session_tracer/sessions-r{rank}.jsonl` with computed
   metrics like `reward_s`, `generate_s`

### Pattern 4: Manual phase scopes with `atrace_session_phase` and `trace_session_phase`

**Use case**: Trace phases that aren't cleanly extractable into methods (e.g., inline
generation loops or post-processing) while reusing the session context created in
Pattern 3.

**API**:

```python
async with atrace_session_phase(
    phase,
    start_payload: dict[str, Any] | None = None,
    end_payload: dict[str, Any] | None = None,
):
    ...

with trace_session_phase(
    phase,
    start_payload: dict[str, Any] | None = None,
    end_payload: dict[str, Any] | None = None,
):
    ...
```

- Context managers for session-phase tracing (async and sync variants)
- Automatically emit `mark_{phase}_start` / `mark_{phase}_end` events and attach
  optional payloads for richer trace metadata

**Example** (continuing the `session_context` workflow from Pattern 3):

```python
@session_context()
async def _collect_samples(..., n_attempts: int = 1):
    async with perf_tracer.atrace_session_phase(
        "generate",
        start_payload={"attempts": n_attempts},
    ):
        response = await engine.agenerate(req)

    reward, completion_str = await self._compute_rewards(response, prompt_str, data)

    with perf_tracer.trace_session_phase(
        "postprocess",
        end_payload={"accepted": response.accepted},
    ):
        filtered = self._postprocess(response)

    return filtered, reward, completion_str
```

The async scope emits timing for `engine.agenerate`, while the sync scope covers any
CPU-side post-processing. Both share the `session_id` implicitly provided by the
`@session_context()` decorator, so their events land in the same session trace record.

### Pattern 5: Add instant markers with `instant()`

**Use case**: Mark specific points in time (e.g., "batch prepared", "queue state
snapshot").

**API**: `perf_tracer.instant(name, category, args)`

- Creates a point-in-time marker (not a duration)
- Useful for events that don't have a meaningful duration

**Example** (from `areal/infra/workflow_executor.py`):

```python
perf_tracer.instant(
    "workflow_executor.prepare_batch",
    category="scheduler",
    args={"data": len(data)}
)

perf_tracer.instant(
    "workflow_executor.wait",
    category="scheduler",
    args={
        "queue_size": runner.get_output_queue_size(),
        "pending_results": len(pending_results),
    }
)
```

### Pattern 6: Manual session lifecycle events with `trace_session_event`

**Use case**: Track session lifecycle at orchestration level (submission, execution,
consumption).

**API**:
`perf_tracer.trace_session_event(method, session_id=None, task_id=None, **payload)`

- Manually record lifecycle events for session tracking
- Used by `WorkflowExecutor` to track full session lifecycle
- Events: `mark_finalized`, and phase events via `@trace_session` decorator
- Parameters: Use `session_id=` to target a specific session, or `task_id=` to target
  all sessions in a task

**Example** (from `areal/infra/workflow_executor.py`):

```python
from areal.utils.perf_tracer import trace_session_event

# Run workflow
traj = await workflow.arun_episode(engine, data)

# Mark execution end with status
if should_accept:
    trace_session_event(
        "mark_finalized",
        session_id=session_id,
        status="accepted",
    )
else:
    trace_session_event(
        "mark_finalized",
        session_id=session_id,
        status="rejected",
        reason="stale_weight",
    )
```

## Session lifecycle tracking

Enable `perf_tracer.session_tracer.enabled=true` to track per-session metrics beyond
just performance spans. This is useful for diagnosing queueing issues and staleness.

### Understanding Task-Session Hierarchy

AReaL's session tracer uses a two-level hierarchy to track rollout execution:

- **Task** (dataset-level): Represents one data point from your dataset. Registered once
  per `arun_episode` call.
- **Session** (sample-level): Represents one generated sample. When `n_samples > 1`, one
  task spawns multiple sessions.

**Example**: If your config sets `n_samples=4`, each dataset item creates:

- 1 task (registered by `WorkflowExecutor`)
- 4 sessions (registered in `_collect_samples`, one per generation)

This hierarchy enables:

- Tracking per-sample metrics (generation time, reward)
- Aggregating statistics per dataset item (acceptance rate across samples)
- Debugging which specific samples within a task failed or were rejected

### What gets tracked

Each session record includes:

- **Task/Session hierarchy**: `task_id` (dataset-level), `session_id` (sample-level)
- **Lifecycle timestamps**: `submit_ts`, `finalized_ts`
- **Status**: `pending`, `accepted`, `rejected`, `failed`, `dropped`
- **Phases**: Multiple executions of `generate`, `reward`, `toolcall` with start/end
  times
- **Derived metrics**: `total_s`, `generate_s`, `reward_s`, `toolcall_s`
- **Context**: `reason` (optional, for rejected/failed sessions)

### Output format

Session traces are written to `session_tracer/sessions-r{rank}.jsonl`. Each line is a
JSON object:

```json
{
    "task_id": 23,
    "session_id": 93,
    "rank": 0,
    "status": "accepted",
    "submit_ts": 7939251.674969524,
    "finalized_ts": 7939254.632833603,
    "total_s": 2.957864078693092,
    "generate_s": 2.65427936706692,
    "reward_s": 0.133724981918931,
    "toolcall_s": 0.156789012345678,
    "phases": {
        "generate": [
            {
                "start_ts": 7939251.674977085,
                "end_ts": 7939254.329256452
            }
        ],
        "reward": [
            {
                "start_ts": 7939254.32926108,
                "end_ts": 7939254.462986062
            }
        ],
        "toolcall": [
            {
                "start_ts": 7939254.463123456,
                "end_ts": 7939254.619912468
            }
        ]
    }
}
```

### Adding custom phases

Workflows that introduce new stages (e.g., verifier passes, safety filters) can emit
dedicated phase spans with a small extension to `areal/utils/perf_tracer.py`:

1. **Declare start/end events** in `SessionTraceEvent`:

   ```python
   class SessionTraceEvent(str, Enum):
        VALIDATION_START = "validation_start"
        VALIDATION_END = "validation_end"
   ```

1. **Register the phase** by appending a `PhaseSpec` in
   `SessionRecord.default_phase_configs()` (set `allow_multiple` or `ready_on_complete`
   as needed).

1. **Expose derived metrics** by adding a helper like `_compute_validation_time()` and a
   matching `FieldSpec` in `SessionRecord.default_field_specs()` if you want
   `validation_s` in the JSONL output.

1. **Map the trace methods** by updating `_SESSION_TRACE_METHOD_TO_EVENT` with
   `"mark_validation_start"` and `"mark_validation_end"` pointing to the new enum
   entries.

After these changes, `@trace_session("validation")`,
`trace_session_phase("validation")`, and `atrace_session_phase("validation")` work out
of the box—the context managers will emit the new events, and the session tracer will
serialize timing for the additional phase alongside the built-in metrics.

#### Plotting custom phases

Use the helper in `areal/tools/plot_session_trace.py` to visualize additional session
phases once they are recorded.

1. **Color the new phase** – extend `SEGMENT_STYLES` with a label and color so the
   lifecycle timeline renders the span distinctly:

   ```python
   # areal/tools/plot_session_trace.py
   SEGMENT_STYLES["validation"] = {
        "label": "Validation",
        "color": "#14b8a6",
   }
   ```

   The timeline renderer automatically picks up every phase present in the trace payload
   (it now augments the default generate/reward/toolcall order with any new keys).

1. **Expose distribution metrics (optional)** – if `SessionRecord` emits a derived
   `validation_s` field, add it to `DURATION_COLUMNS` and `HISTOGRAM_METRICS` so the
   summary charts display per-phase histograms alongside the defaults.

1. **Render the report** – point the plotting script at the session JSONL files and
   enable lifecycle figures:

   ```bash
   python -m areal.tools.plot_session_trace \
     logs/**/session_tracer/sessions-r*.jsonl \
     --consumer-batch-size 512 \
     --enable-lifecycle
   ```

   This produces HTML summaries under the same directory, including
   `sessions-lifecycle-r*.html` which highlights the new phase in the timeline.

The script derives the plot order from `DEFAULT_PHASE_ORDER` plus any phase names it
finds in the trace payload, so you only need to extend the style/metric dictionaries
when introducing new phases.

## PyTorch Profiler integration

`perf_tracer` can launch the PyTorch profiler for targeted steps so you capture kernel-
and op-level data without paying the cost every iteration.

### Configure profiler trigger steps

- Set `PerfTracerConfig.profile_steps` to a list of global steps (0-indexed) where the
  profiler should run. Example:

  ```yaml
  perf_tracer:
      enabled: true
      profile_steps: [49, 99]
  ```

- When any `trace_scope` or `atrace_scope` carries `args={"global_step": step}` and
  `step` matches one of the configured entries, `perf_tracer` will check if profiling
  should be triggered for this `global_step`.

- Passing `enable_profiler=True` to `trace_scope`/`atrace_scope`: the scope profiles
  **only** when *both* the flag is set and `global_step` matches one of the configured
  `profile_steps`. Without the schedule, manual requests are ignored; without the flag,
  scheduled steps stay dormant.

## Troubleshooting

**Q: Traces are empty or missing events**

A: Ensure `perf_tracer.save(force=True)` runs before exit. Check that
`perf_tracer.configure()` was called with the correct rank.

**Q: Session traces show all `status: "pending"`**

A: Lifecycle events (`mark_finalized`) aren't being recorded. Verify `WorkflowExecutor`
is calling `trace_session_event()` or your custom workflow implements the full
lifecycle.

**Q: Perfetto can't open my trace**

A: JSONL format requires conversion. Use the provided converter tool or manually wrap in
a JSON array:

```bash
python -m areal.tools.perf_trace_converter traces.jsonl trace.json
```

**Q: Some sessions show `null` for phase `end_ts`**

A: This occurs when `engine.agenerate()` throws an exception that propagates to
`arun_episode`. The orchestrator then calls
`trace_session_event("mark_finalized", task_id=task_id, ...)`, which finalizes **all
sessions** under that task—including ones whose phases were interrupted mid-execution,
leaving `end_ts: null`.

**Solution**: Remote inference engines must **not throw exceptions** from `agenerate()`.
Handle errors internally and return error responses instead.

## See also

- {ref}`CLI Reference: PerfTracerConfig <section-perf-tracer>`
- [Workflow customization guide](../customization/agent.md)
- [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/)
- [Perfetto UI](https://ui.perfetto.dev/)
