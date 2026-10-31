# Elastic training with AReaL v2.1.0

The execution entry is native `PPOTrainer`. With `actor_lifecycle.mode=on_demand`,
Rollout and CPU data services survive Actor worker release and restart. PPO computation,
checkpointing and publication stay in the native trainer. Current verification and GPU
blockers: [PROGRESS](../../../PROGRESS.md); research scope: [plan](../PLAN.md).

The former serial `ElasticLoop`, `QueuedTrainer`, `TrainGroup`, fixture engines and SWE
runner have been removed. Their durable state mechanisms are shared under
`areal/elastic/state` and `control`; raw history and models also belong to AReaL.
AutoPilot provides generic revision plus an Elastic adapter. Historical
[M1](../../../../artifacts/repository-archive-20261021/files/examples/autopilot/elastic/milestones/m1/README.md)
and
[M2](../../../../artifacts/repository-archive-20261021/files/examples/autopilot/elastic/milestones/m2/README.md)
records describe the retired executor, not GPU acceptance of the current path.

## Policy and revision loops

```text
Agent Service → resident Rollout → durable episode inbox → complete eligible batch
  → PPOTrainer / OnDemandActor → native PPO update → weight publication + acknowledgement
  → periodic checkpoint → evaluation / cleanup / resume → release decision / next batch
  → if releasing: latest full checkpoint + cursor → destroy Actor workers
                         ↑                         ↓
              PolicyAdapter ← ElasticMonitor → completed-update RuntimeHistory
                         ↑                              ↓
              RevisionAdapter ← frozen measured window
                         ↑                         ↓
                 accepted bundle ← model gate → frozen model → optimizer gate
```

AReaL's `elastic/policy/adapter.py::PolicyAdapter` assembles timeout or measured Basic
and supplies the single runtime decision entry. AutoPilot's `RevisionAdapter` extends it
with revision and accepted-bundle activation. AReaL's default `ElasticMonitor` supplies
observations, phase measurements and causal history to both. Neither creates workers or
performs PPO. The native executor retains batch, version, checkpoint, publication,
minimum-residence and release safety checks.

On-demand always selects an Elastic policy. The default `actor_lifecycle.policy=basic`
currently uses `IdleTimeoutPolicy`; the runtime has no separate idle-timeout rule. Full
model-based Basic remains a later integration. `actor_lifecycle.policy=reference` is
reserved and fails explicitly before services start until Reference scheduling exists.

CPU条件模拟位于
[areal/elastic/simulation](../../../areal/elastic/simulation/README.md)，独立于AutoPilot修订流程。

## Code map

| Responsibility                                                       | Current module                                            |
| -------------------------------------------------------------------- | --------------------------------------------------------- |
| Native PPO execution and per-update ordering                         | `areal/trainer/rl_trainer.py`                             |
| Actor lifecycle, publication, save and recovery coordination         | `areal/elastic/runtime.py`                                |
| Durable inbox, queue and checkpoint                                  | `areal/elastic/state/`                                    |
| Full native checkpoint I/O and per-rank LR/RNG state                 | `areal/v2/training_service/{checkpoint,runtime_state}.py` |
| Process ownership, native control and activation/config transactions | `areal/elastic/control/`                                  |
| Runtime observations and measured history                            | `areal/elastic/monitor/`                                  |
| Basic models, simulator and search                                   | `areal/elastic/policy/basic/`                             |
| Independent optional Expert reference                                | `areal/elastic/policy/reference/`                         |
| Generic proposal, evaluation orchestration, acceptance and store     | `areal/autopilot/revision/`                               |
| Elastic evaluators, worker, bundle verification and native bridge    | `areal/autopilot/domains/elastic/`                        |
| Real online application                                              | `examples/hermes/{train,hermes}.py`                       |

In `control/`, `policy_hooks.py` defines optional revision polling/recovery callbacks,
`policy_activation.py` manages policy activation, `config_application.py` manages
configuration transactions and the fixed-layout executor, and `boundary.py` provides
only the execution interface and snapshots. The runtime's boundary import does not load
configuration transactions. Each executor decodes its schema; the control layer does not
import Basic planning models.

`monitor/collector.py::ElasticMonitor` is enabled for every on-demand run. It owns live
observation assembly, batch workload extraction, phase collection, completed-token
accounting and measurement recovery. `history.py` supplies bounded persistence;
`trace.py` parses existing logs on demand. Policy decisions use an in-memory causal
window; revision freezes completed records from the same store. Update records are
published after `finish_update()` resumes Rollout. Recovery timings remain separate
lifecycle events, and observation sequence numbers preserve history order on restart.

## Run and policy integration

Use the normal Hermes training entry with a verified local model, explicit disjoint GPU
pools, sufficient checkpoint space and a new persistent `state_root`. Default lifecycle
mode remains `resident`; on-demand requires v2 online mode and supported native
backends. See [ActorLifecycleConfig](../../../areal/api/cli_args.py) for capability
validation.

The existing `examples/hermes/train.py` accepts `--elastic-policy` for AutoPilot
activation/revision integration, `--elastic-profile path.json` for measured Basic,
optional `--elastic-settings path.json` and `--elastic-bundle path.json` for accepted
deployment. These flags require `actor_lifecycle.mode=on_demand`; normal config
arguments continue to pass unchanged. Profile/settings alone select AReaL's ordinary
`PolicyAdapter`. `--elastic-policy` or `--elastic-bundle` explicitly selects AutoPilot's
`RevisionAdapter` and its activation transactions. With no flags, on-demand creates the
ordinary adapter, uses Basic timeout and collects default Monitor history.

A custom native entry can enable policy activation/revision without changing PPO:

```python
from areal import PPOTrainer
from areal.autopilot.domains.elastic.native import RevisionAdapter

adapter = RevisionAdapter()
with PPOTrainer(config, elastic_policy=adapter) as trainer:
    trainer.train()
```

A measured Basic run can use `areal.elastic.policy.adapter.PolicyAdapter` directly,
supplying `profile=` and `settings={"horizon": 1}` without importing AutoPilot. This
ordinary path uses the supplied measurements and the shared live Monitor history. The
generated `native-profile.json` identifies measurement applicability: training/model,
physical placement, tokenizer, sampling, CPU hooks and timing-relevant semantic source
hashes. Comments, formatting and offline trace parsing do not affect it. Collect and
audit measurements under that identity. Queue/checkpoint compatibility binds training
recovery separately; accepted bundles retain exact protected-source verification.
Ordinary checkpoints may carry optional `monitor-state.json`. Missing or incompatible
measurement state starts a fresh history series while restoring valid model state and
logical counts. An existing policy activation store still requires an activation-capable
adapter on resume; its accepted proof is not bypassed. A revision root bound to an older
history series must be replaced before revisions resume.

For bounded automatic revision, construct `RuntimeRevisionLoop` with the same profile,
`settings=runtime.settings`, `scope="native-async-v1"`, `horizon=1` and matching
`idle_seconds`, then attach it as `runtime.revision` before `trainer.train()`. Explicit
proposal agents and durable window/round budgets are required; see the
[Elastic revision contract](../../../areal/autopilot/domains/elastic/README.md).

Accepted bundles can be preloaded with `RevisionAdapter(bundle=path)`, or prepared and
applied with `PolicyRequest` at a completed, fully published boundary. Both components
are independently evaluated; loading checks scope, source/proof hashes, execution
identity, settings and idle timeout. An uncertain activation closes admission until
recovery follows the durable pointer. History survives policy replacement.

## Behavior and boundaries

- One fixed Actor layout and one fixed Rollout placement per run. Native configuration
  transactions accept the identical configuration only; topology changes fail before
  mutation. Serial joint models and accepted bundles remain offline references.
- A complete eligible batch wakes Actor; no predictive prewarm. Default release requires
  idle ≥60s and residence ≥120s, no active update and confirmed published weights. Basic
  owns the idle timeout; runtime enforces minimum residence and execution safety. Zero
  idle timeout requests release immediately once those constraints allow it. Production
  starting point: B=4, S=2; lifecycle smoke: S=0, idle=5s, residence=0.
- Weight publication precedes HF saving and periodic recovery checkpoints. HF saving
  follows `saver.freq_*`; full recovery saving follows `recover.mode` and
  `recover.freq_*`. Evaluation, cleanup and logging finish before serving resumes or
  release is considered. Planned idle release and clean shutdown save the latest full
  state regardless of periodic recovery being disabled. A snapshot from before the
  end-of-update CPU hooks is refreshed before release.
- Runtime and control snapshots expose `applied_version`, `published_version` and
  `checkpoint_version`. `publish_update()` advances live PPO/publication progress;
  checkpoint saving is separate. `OnlineInbox.status()` reports the durable cursor as
  `checkpoint_version`. The shared planning schema keeps its legacy `committed` and
  `committed_*` work counters; these describe completed/published work, not durability.
  SQLite and the shared planning schema remain unchanged. Runtime logical counts are
  saved as `data_progress` with the strict full-state checkpoint; legacy counts are
  rebuilt from the validated queue prefix. Optional `monitor-state.json` is written
  after that commit; `monitor-pending.json` supplements pending timings, and completed
  same-version snapshots supersede old sidecars. Histories are isolated under
  `monitor/<measurement_id>/<series_id>/`. Metric I/O/schema failures fall back to Basic
  timeout and pause revision until recovery reopens measurements. They do not roll back
  successful PPO updates or prevent full-state release saving. Critical checkpoint and
  queue failures remain strict.
- Applied, published and durable-checkpoint versions may differ. Fault recovery pauses
  serving, restores the last full checkpoint (or pinned base), reconciles serving
  weights and replays unsaved batches through native PPO. Unsealed episodes from the
  abandoned branch are explicitly rejected. Payloads survive publication until
  checkpoint recovery; unknown owners or communication outcomes remain fenced.
- Native phase measurements include actual CPU hooks and distinguish release boundaries
  from committed updates. Missing phase durations remain unknown. Fixed-layout, one-step
  prediction and revision do not establish asynchronous makespan, GPU-hour savings,
  cross-layout accuracy or training quality.
- CPU tests exercise real durable stores and protocol logic with fake workers/model
  endpoints. Actual FSDP/Megatron restore, Agent Service → GSM8K → Hermes training,
  asynchronous overlap and resource cleanup still require GPU acceptance.

## Archived development tools

Offline CLIs, the R10 experiment driver, conditional replay helpers and their dedicated
validation cases are in the
[local development archive](../../../../artifacts/elastic-policy-development-20261021/README.md).
The source package retains numerical models, simulators, optimizers, built-in profiles
and required test inputs. Existing-log parsing is under
`areal/elastic/monitor/trace.py`.

External-catalog tests still use `AREAL_TEST_EXPERT_CATALOG`; the explicit catalog must
have audited source/NumPy compatibility. Historical replay evidence remains distinct
from native GPU acceptance. Original observations and methods are preserved in the
[original archive](../../../../artifacts/repository-archive-20261021/README.md).
