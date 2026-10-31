# Elastic revision adapter

弹性训练的CPU执行环境与候选模拟已移至
[areal/elastic/simulation](../../../elastic/simulation/README.md)，供重配置策略独立使用。本目录负责领域评测、代码修订接入和接受后的部署适配；`online.py`仍是原生修订流程的host桥接。

`RevisionLoop` proposes one Python component module, evaluates it, and either atomically
activates it or gives aggregate feedback to the agent for another bounded attempt. Each
store targets **simulator or optimizer**; the other component stays fixed.

The
[compact architecture diagram](../../../../examples/autopilot/elastic/README.md#policy-and-revision-loops)
shows the policy loop and the two separate revision loops.

`evaluator.py` supplies evaluation and acceptance criteria; `autopilot/revision/loop.py`
sends its feedback to the agent and applies the acceptance decision. The
[independent Expert model](../../../elastic/policy/reference/README.md) provides a
frozen offline reference alongside successive Basic simulator versions. Its
implementation remains in the source package; the R10 experiment driver is in the
[local development archive](../../../../../artifacts/elastic-policy-development-20261021/README.md).
A common experiment adapter remains separate work. It does not direct revisions or
supply training labels by default.

- **Simulator:** independent held-out measurements supply feasibility, wall time and GPU
  seconds. Calibration is visible to the agent; held-out labels stay in the evaluator.
  The score combines relative prediction error and feasibility error rate. Inaccurate
  incumbent feasibility is scored, so a candidate can repair it; execution failures
  remain invalid evaluations.
- **Optimizer:** fixed observations, historical windows, forecast horizon, resource/time
  limits and reference simulator. Reference simulation and query accounting run in the
  parent through RPC. The evaluator recomputes the returned plan's cost and progress;
  candidate scores, predictions and query counts are ignored.
- **Acceptance:** valid cases, no feasibility/guard violations, and a strictly improved
  score (default minimum relative improvement 0.5%). Optimizer mean reference GPU cost
  may not regress by default. Its score combines relative oracle cost and reference
  query overhead; wall time, update delay, transitions, samples and tokens are reported.
- **Persistence:** proposals, inputs, evaluation, acceptance receipt and source hash per
  attempt. `active.json` changes only after acceptance. A store rejects another
  benchmark or acceptance context; failed/duplicate candidates leave the active pointer
  unchanged.
- **Execution:** bounded subprocess wall time and output, process-group cleanup, no GPU
  visibility in evaluation workers. Candidate Python is trusted code; subprocess
  isolation is not a hostile-code sandbox.

## Native runtime bridge

`online.py` consumes an explicit committed-host protocol implemented by
`RevisionAdapter`. Native scope is `native-async-v1`, with horizon=1 and one fixed Actor
layout. Model service-time revision and optimizer acceptance remain separate. Serial
joint models below are retained as offline references. Ordinary policy assembly and
decisions live in `elastic/policy/adapter.py::PolicyAdapter`;
`native.py::RevisionAdapter` adds revision and activation to that same decision entry
point. AReaL's default `ElasticMonitor` supplies both ordinary policy history and
revision windows. Its `weights_version` host field names the applied version;
checkpoints may lag.

- `elastic/monitor/collector.py` owns observations, phase collection, token accounting
  and measurement recovery; `history.py` provides bounded `RuntimeHistory` storage. It
  records actual config, workload, causal planning history and measured service phases
  after publication and end-of-update hooks. `checkpointed` distinguishes periodic saves
  from skipped saves. Here the host's `committed` boundary means a completed, published
  update; the full checkpoint may lag. Fault recovery restores its bounded history and
  replays unsaved batches through native PPO.
- `RuntimeRevisionLoop` freezes a window after at least four updates, then runs CPU
  revisions in one background worker. Defaults: every four updates, at most three
  windows, two proposals per component per window. Started proposals consume their
  durable budget, including interrupted attempts.
- `runtime_problems` splits whole updates chronologically: the first 60% (at least two)
  supply agent calibration; remaining measured phases supply private timing labels.
  Model acceptance scores conditional phase seconds and feasibility, not full-window
  accuracy, reward or unseen-configuration accuracy.
- `LifecycleRevisionCycle` accepts or retains the model, freezes its executable source,
  and re-evaluates the current optimizer on that model before optimizer revision.
  Oracle, RPC queries and final scoring share one immutable finite prediction table.
  Reusing a model in a later window retains its original acceptance and adds
  current-window revalidation.
- `phase.py` fixes phase order, Rollout waits and GPU accounting. Only service-time
  components are revised; unvalidated extra GPU charges are rejected. Same-instance
  repetition and worker/parent/deployment prediction probes check reproducibility on the
  fixed inputs.
- `policy.py` exports accepted executable bundles with both component sources, profiles,
  settings and evaluation proofs. Even an unchanged optimizer requires revalidation
  after a model change. Preloading/probing runs outside the execution gate; activation
  atomically changes the policy pointer with no active update and fully published
  committed weights. Requests bind policy generation and weight version. An uncertain
  commit closes admission until native recovery follows the durable pointer.

See
[native integration](../../../../examples/autopilot/elastic/README.md#run-and-policy-integration).
CPU state-machine integration covers acceptance, rejection, interrupted activation,
restart and subsequent training. The historical, retired-executor
[Qwen3-0.6B H20 validation](../../../../../artifacts/repository-archive-20261021/files/examples/autopilot/elastic/validation/runtime-revision-20261019.json)
also covers both component acceptances, policy activation, real continued PPO training,
exact model/Adam checks and coordinator-object recovery. Its deterministic proposer and
synthetic trajectories do not establish LLM revision ability or quality/cost gains.
Checkpoints used host tmpfs; a harness bind-mount cleanup error was separately fixed and
replayed, with raw exit and cleanup records retained.

## Serial joint runtime revisions

`serial.py` adds conditional timing evaluation and a model→optimizer cycle for
`OnlineJointPolicy`. The serial evaluator, bundle verification and cycle remain
available for offline analysis. Their former live executor was removed; serial bundles
cannot activate on the native host.

- Factories implement `build_serial_model(profile)` →
  `predict_timing(SerialTimingInput)` → `SerialTimingEstimate`. Only seconds/unknown are
  revisable. Fixed code owns coverage, allocations, transition profiles and accounting.
- Split whole committed updates chronologically; calibration is agent-visible. Model
  validation uses actual full configurations/workload counts, calibration-only timing
  history and private labels. Episode source configurations are captured before
  generation. Unknown source attribution is skipped, never synthesized. Numerical
  feedback is suppressed for groups smaller than five; final holdout stays separate.
- Model acceptance requires identical coverage and lower error without phase/config
  regression. The frozen model then supplies the joint optimizer's cost/query/time
  evaluation. Calibration planning windows supply public deployment probes; no timing
  validation labels are exported. Deployment loads predictor code, not a finite table.
- `serial_policy.py` binds separate M/O acceptance, current-window revalidation,
  executable probes, fixed profile/settings, horizon, idle timeout and code identities.
  Even a retained optimizer is revalidated after model revision. Activation preserves
  full effective phases/episodes, history, both cursors, weights and config generation.

The
[joint H20 record](../../../../../artifacts/repository-archive-20261021/files/examples/autopilot/elastic/validation/joint-runtime-revision-20261019.json)
historically covered both component acceptances, live activation, subsequent automatic
joint config application and recovery through seven real PPO updates. Its proposals and
workloads are controlled fixtures; the full Linux CPU suite passed 481 tests with one
CUDA RNG skip.

This gate conditions on observed output counts; it does not establish future workload
forecast accuracy, unseen-config accuracy, reward preservation or cost savings.

## Historical model evaluation

`model_evaluator.py` adds `HistoricalModelEvaluator` for actual configurations,
workloads and measurements from one training history. It currently evaluates Python
predictor objects directly, or candidate source through `HistoricalSourceEvaluator` and
`RevisionLoop`. Neither Expert labels nor measurements at unseen configurations are
required for historical acceptance.

- `HistoricalCase` records the original run/step span, split, measurement source, actual
  inputs and measured metrics. Missing labels use `None`; zero is a real observation.
  Provide `execution_id` for the model, hardware, runtime and remaining execution
  options, and an audited timing boundary. Joint Actor profile IDs must resolve to the
  recorded full configuration.
- The evaluator freezes the cases, primary metrics, thresholds and reference code.
  Calibration labels are available through `.calibration`; prediction receives only
  `HistoricalInput`. Cross-split overlap and fitting on validation/final spans are
  rejected. `fitted_on` is caller-declared provenance, not an execution sandbox. Final
  holdout cases are never predicted or scored here.
- `BasicHistoryPredictor` adapts existing Actor, Rollout and joint models. Actor-only
  evaluation works without Rollout data. Actor totals include only logprob, advantage
  and parameter update; provide separately audited measurements for other boundaries. No
  fitting occurs in the adapter.
- Actor input modes are `logical` or `given_packing`. Rollout uses `given_external`;
  joint uses `given_external` or `given_packing_and_external`. Given packing is a
  conditional timing diagnostic. Given external durations are inputs, so these
  predictions do not validate an environment-wait model. `external_seconds` cannot be
  primary; primary Rollout total also requires primary, independently measured
  `inference_seconds` and its prediction for each total. Joint evaluation requires a
  verified committed, drained initial state and a measured window through shutdown.
  Unsupported traces remain uncovered even if a custom predictor returns numbers.
- Reports retain per-metric WAPE, nearest-rank P90 relative error, coverage, config and
  workload slices, and local detail rows. WAPE uses summed absolute error over summed
  measurements on covered cells. Zero labels contribute to WAPE; P90 excludes them and
  reports its sample count. An all-zero group has undefined WAPE and uses MAE in
  seconds/GPU-seconds for acceptance. Primary error is a macro average over metric
  groups; freeze the selection before evaluating candidates.
- Acceptance compares exactly the incumbent's covered cells, disallows coverage loss and
  checks per-metric/config/workload regression. Added coverage has its own error limit,
  including slices. Defaults are >0.5% relative primary improvement, zero regression
  tolerance and 0.1 maximum new-coverage error; these are configurable policy choices,
  not validated Tau2 thresholds. An empty incumbent requires an explicit
  `bootstrap_max_error` and complete labelled coverage. Comparisons accept only
  unchanged reports issued by the same evaluator instance; rebuild/re-evaluate both
  versions after changing the evaluation context.
- `.feedback()` excludes detail rows and absolute error statistics, and suppresses error
  values for groups with fewer than five covered cells. Local reports retain full
  diagnostics. Repeated validation feedback is development data, not an unbiased final
  test or a guarantee against label inference.

With company-local `cases`, frozen model bundles and calibration provenance:

```python
evaluation = HistoricalModelEvaluator(
    cases,
    primary_metrics=("recompute_logp_seconds", "train_step_seconds"),
)
before = evaluation.evaluate(basic_v0, "basic-v0-snapshot", fitted_on=fit_spans)
after = evaluation.evaluate(basic_v1, "basic-v1-snapshot", fitted_on=fit_spans)
decision = evaluation.compare(after, before)
feedback = after.feedback()
```

Each predictor implements `predict(HistoricalInput) -> HistoricalPrediction`; an
unavailable metric remains absent/`None`, with a reason. Candidate exceptions or
non-finite/negative outputs invalidate evaluation. Freeze Expert separately and run the
same cases for comparison. Historical accuracy alone does not establish accuracy on new
configurations, closed-loop training quality or cost savings.

## Revise historical model code

The `model` target reuses the bounded worker, proposal agents and durable revision
store. An explicit initial factory and JSON settings bind the starting model; there is
no implicit model/hardware/profile substitution. With the evaluator above:

```python
from pathlib import Path
from areal.autopilot.domains.elastic import (
    HistoricalSourceEvaluator, load_accepted_model,
)
from areal.autopilot.revision import RevisionLoop

source_evaluation = HistoricalSourceEvaluator(
    evaluation, settings=fixed_model_settings, fitted_on=prefit_spans,
)
result = RevisionLoop(
    agent, source_evaluation, Path("outputs/model-revision"),
    initial_source=Path("basic_v0.py").read_text(),
).run(max_rounds=3)
# Available once a candidate has been accepted; loads an offline predictor.
accepted = load_accepted_model(Path("outputs/model-revision"), source_evaluation)
```

Each candidate implements `build_model(calibration_cases, settings)` and returns a
`predict(HistoricalInput) -> HistoricalPrediction` object. Calibration cases are typed
`HistoricalCase` values. Settings carry caller-supplied model parameters and execution
identity; use `BasicHistoryPredictor` to compose the existing models.

The worker receives calibration labels and supported validation inputs only.
Applicability checks, validation labels, scoring and acceptance stay in the parent;
final holdout inputs/labels never enter the worker. Declared prior fit spans and all
supplied calibration spans are checked against protected history. Original run/step
identities must remain consistent across windows. Predictors are trusted Python;
subprocess bounds do not restrict filesystem access.

Factory, worker, evaluator and fixed Basic library code, settings, history, acceptance
rules, worker command and limits bind the evaluation context. Factory/type errors,
timeouts and nonfinite predictions are invalid. Predictors must remain frozen:
successful predictions are repeated within each worker and in a second independent
worker, with the deadline applying to each run. The accepted prediction fingerprint is
checked again on resume and offline loading. This checks reproducibility on the fixed
validation inputs; it does not serialize model parameters or prove behavior on unseen
inputs. Loading executes trusted factory code in the caller.

Aggregate revision feedback retains small-group suppression, including the incumbent
score and resumed attempts. Full detail rows are saved locally in
`history-evaluation.json` after the worker exits. Only a successful parent acceptance
receipt promotes `active.json`; duplicates and rejected attempts preserve the active
source. Loading requires the matching evaluation context.

Reuse the same initial source when resuming a store. Changed history/settings,
boundaries, criteria or protected code require a new evaluator/store; a prior accepted
source can seed that next window with declared fit provenance. Online history
collection, revision triggers and runtime hot replacement remain separate work. The
existing lifecycle CLI below keeps its original targets; historical model revision
currently uses this Python API. No model API call occurs unless the caller explicitly
supplies an LLM-backed agent.

## Revise joint optimizer code

`joint_optimizer` uses one frozen model version and explicit finite scenarios/configs:

```python
from areal.autopilot.domains.elastic import JointOptimizerEvaluator
from areal.autopilot.revision import RevisionLoop

search_evaluation = JointOptimizerEvaluator(
    frozen_basic_model, windows, configs, model_version=model_version,
    max_queries=query_budget, max_job_seconds=job_time_limit,
)
result = RevisionLoop(
    agent, search_evaluation, Path("outputs/joint-revision"),
    initial_source=Path("optimizer_v0.py").read_text(),
).run(max_rounds=3)
```

Implement `build_joint_optimizer(settings)` returning an object with
`optimize(JointScenario, JointModel) -> JointSearchResult`. Worker settings contain
typed `JointConfig` candidates; the default initial source uses `JointOptimizer`. Only
the returned config is trusted as a selection, and the parent checks it against the
frozen model. Claimed predictions, query counts and elapsed time are ignored.

The evaluator snapshots the model for all explicit configs, including each window's
current config. The snapshot digest and model version bind the context; workers get only
queried predictions. Every RPC query, including duplicates and failed attempts, is
checked in the parent. Caught violations still fail evaluation. Workload, initial state,
full config identities, GPU limits, job-time bound and `switch_margin` remain fixed.
Windows without an eligible reference config are rejected up front.

The quality score is mean normalized regret:
`1 + (selected_cost - minimum_eligible_cost) / max(minimum_eligible_cost, 1 GPU-s)`.
Acceptance requires no per-window cost regression by default, then either significant
quality improvement or preserved quality with fewer queries/faster solving. Fewer
queries cannot hide slower execution beyond the fixed timing tolerance; speed alone
cannot justify more queries. Criteria are frozen by the evaluator.

Each source runs at least three times with matching configs/query counts. The parent
measures the full worker wall time, including startup, source import, factory, RPC and
cleanup; medians and absolute/relative gates control timing acceptance. Per-case
factory/solve/RPC timings remain diagnostic and must meet `max_optimizer_seconds`;
`CandidateRunner.timeout_seconds` bounds each complete repetition. Reference snapshot
construction is separate evaluation setup, reported as `reference_evaluations`. Timing
uses frozen-table RPC, so live model-query performance requires separate measurement.

Detailed runs are saved to `joint-evaluation.json`; the revision store retains source,
acceptance receipt and active pointer. Resume checks the accepted config/cost/query
fingerprint before replacing the baseline. Changed model/scenarios/budgets/criteria
require a fresh evaluator/store; a model revision must be frozen and optimizer versions
re-evaluated under that new context. This is offline configuration search; runtime
application remains separate from the offline cycle below.

## Run one model-to-optimizer cycle

`PolicyRevisionCycle` runs one bounded offline model revision followed by optimizer
revision under the effective model. It reuses a joint evaluator's scenarios, configs,
constraints and budgets; the template's old prediction table is replaced.

```python
from pathlib import Path
from areal.autopilot.domains.elastic import PolicyRevisionCycle

cycle = PolicyRevisionCycle(
    source_evaluation, search_evaluation, Path("outputs/policy-cycle"),
    initial_model_source=Path("model_v0.py").read_text(),
    initial_optimizer_source=Path("optimizer_v0.py").read_text(),
)
result = cycle.run(
    model_agent, optimizer_agent, max_model_rounds=3, max_optimizer_rounds=3,
)
```

The model factory must return a historical predictor with an explicit
`.joint.evaluate(JointScenario, JointConfig) -> JointPrediction` export.
`BasicHistoryPredictor` already provides the `.joint` slot; supply its complete model
bundle. The cycle adds that requirement to model revision and checks the export on the
fixed scenarios/configs. It neither assembles missing components nor substitutes an
Expert model. Caller-supplied search windows must exclude final holdout data.

Each candidate reproduces historical predictions and joint exports in two independent
workers, with repeatability checks within each worker. Exporting cannot mutate the
historical fitted state. Accepted models and unchanged baselines use identical checks.
The handoff binds the effective source, history receipt and joint snapshot, then creates
a new optimizer context. The initial optimizer is evaluated again there; quality
comparison uses only that new model's scores. Missing/inconsistent exports or an invalid
optimizer baseline stop the cycle.

`model/` and `optimizer/` retain separate revision stores and immutable `stage.json`
receipts. `handoff.json` binds their model version; `result.json` marks completion.
Recovering a cycle verifies the same sources, historical predictions and exported joint
table. Started attempts consume the original total round budgets, including
interruptions. A durable acceptance can recover a missing active-pointer publication.
Completed stages make no new agent calls; completed-cycle reruns still recheck model and
search execution. Initial sources publish atomically and are never rewritten.

Reuse the same root only with identical initial sources, data, budgets and criteria. Use
a new root and explicit fit provenance for the next window/cycle. This API does not
collect training windows, trigger online revisions or apply the resulting policy to a
running trainer. All worker repetitions and recovery checks remain offline evaluation
overhead, separate from the optimizer's reported solve time.

## Run the existing lifecycle revision loop

Use the normal AReaL environment. To generate candidates with an actual LLM, configure
`OPENAI_API_KEY` and explicitly select a model:

```bash
python -m examples.autopilot.revising \
  --target simulator --profile profile.json --cases heldout.json \
  --output /path/to/revision-store --max-rounds 3 \
  --llm-model YOUR_MODEL_ID
```

The existing OpenAI SDK calls a Chat Completions endpoint. `OPENAI_BASE_URL` or
`--llm-base-url` selects a compatible service; `--llm-api-key-env` selects the
credential variable. `--llm-timeout`, `--llm-max-output-tokens` and optional
`--llm-reasoning-effort` bound/configure each call. Strict JSON Schema is the default;
use `--llm-response-format json_object` when the service supports only JSON mode. Both
modes validate the returned fields. There are no hidden SDK retries or format fallbacks.
`--llm-token-parameter max_tokens` selects the legacy output-limit field for backends
that require it (default: `max_completion_tokens`; OpenRouter accepts both).

The model receives the actual component API sources, calibration, incumbent code,
previous candidate code and aggregate feedback. Held-out labels stay in the evaluator.
Each successful proposal records model identity, token usage, duration and prompt hash
in `proposal.json`. The CLI uses the bounded external-command adapter; accepted code
still passes the same independent evaluator. API calls occur only when explicitly
selecting the LLM proposer.

Add `--demo --llm-model YOUR_MODEL_ID` to use synthetic inputs with a live model. Plain
`--demo` uses a deterministic code proposer without a model call:

```bash
python -m examples.autopilot.revising --target simulator --demo --output /tmp/revise-sim
python -m examples.autopilot.revising --target optimizer --demo --output /tmp/revise-opt
```

For [Step 3.7 Flash on OpenRouter](https://openrouter.ai/stepfun/step-3.7-flash),
configure `OPENROUTER_API_KEY`, then run:

```bash
python -m examples.autopilot.revising \
  --target simulator --demo --output /tmp/revise-step37 --max-rounds 2 \
  --llm-model stepfun/step-3.7-flash --llm-base-url https://openrouter.ai/api/v1 \
  --llm-api-key-env OPENROUTER_API_KEY --llm-token-parameter max_tokens \
  --llm-reasoning-effort low --llm-max-output-tokens 16384
```

For real calibration and held-out cases:

```bash
python -m examples.autopilot.revising \
  --target simulator --profile profile.json --cases heldout.json \
  --output /path/to/revision-store --max-rounds 3 \
  --agent-command python /path/to/agent_adapter.py
```

The adapter receives request JSON on stdin (target, incumbent source/hash, previous
candidate source, calibration, fixed settings, API and aggregate feedback). Return one
JSON object on stdout:

```json
{"source": "...Python module...", "rationale": "..."}
```

Factories:

```python
def build_simulator(workflow):  # WorkflowModel -> evaluate(obs, forecast, plan, history)
    ...


def build_optimizer(settings):  # dict -> optimize(obs, supplied_simulator, history)
    ...
```

A simulator returns `SimulationResult`. An optimizer returns `SearchResult` containing
its decision and a complete plan over the evaluator's fixed forecast. Existing
components and `BasicOptimizer` can be composed or wrapped. Evaluation does not alter
training/runtime code or launch training.

`profile.json` uses the elastic example's version-1 measurements, compatibility and
inference-residency fields. Simulator cases contain `observation`, `history`,
`forecast`, `plan`, and independent `observed` labels (`feasible`, plus `seconds` and
`total_gpu_seconds` for feasible cases). Optimizer cases contain only `observation` and
`history`; the evaluator constructs forecasts from history. The demo's `demo_inputs()`
provides executable examples of both schemas. `--settings` optionally loads the fixed
`BasicOptimizer` settings. Keep calibration/validation separation when preparing cases.

## Use an accepted component

Native execution accepts complete independently evaluated bundles through
`RevisionAdapter(bundle=path)` or `prepare_policy()` / `apply_policy()` with
`PolicyRequest`. The native host verifies scope, execution identity, both acceptance
proofs, component sources, fixed candidate set and idle timeout. Activation changes the
policy pointer only; history and training weights persist. Legacy serial/lifecycle
receipts cannot be reused as native acceptance. Configuration transactions reject layout
changes before any worker mutation.

`apply_accepted_revision()` remains an offline component loader; it does not bypass the
native bundle gate. The
[LLM demonstration](../../../../../artifacts/repository-archive-20261021/files/examples/autopilot/revising-llm-validation-20261018.json)
uses synthetic cases and does not establish workflow generalization, reward preservation
or production savings. Retain a separate final audit set.

Native Monitor failures pause revision and use the Basic timeout baseline. Training
recovery compatibility, measurement applicability and accepted-bundle source proofs are
independent: missing measurement state starts a fresh series without discarding valid
optimizer/data progress; changed protected sources still reject old bundles. A revision
root is bound to one history series and must be replaced when that series changes. See
[AReaL Elastic](../../../elastic/README.md) for collection and recovery rules.
