# Process Rewards

Process rewards attach a training signal to each generated turn, in addition to
the trajectory's outcome reward. The v1 OpenAI proxy and v2 inference data proxy
score complete trajectories before exporting them to PPO/GRPO. Scorers are supplied
as Python classes; no external judge service is required by AReaL itself.

## Configuration

Add the following settings to an agent training configuration:

```yaml
rollout:
  _version: v1
  agent:
    chat_template_type: concat
    export_style: concat
    prm:
      enabled: true
      advantage_shaping:
        mode: process_weighted
      scorers:
        - path: examples.prm.scorers.LengthBudgetScorer
          weight: 1.0
          kwargs:
            max_output_tokens: 128
actor:
  token_rewards_as_adv: true
```

An empty scorer list or `enabled: false` disables proxy scoring. Scorers currently
require concat export, concat chat templates, token-backed interactions, and direct
process advantages. To use the v2 inference service, set `rollout._version: v2`
with the same scorer configuration. Individual export and string-only external
model responses are not supported. This integration scores completed training
trajectories, not the optional Agent Service's reconstructed chat history.

Active PRM scorers are rejected with v2 external-model mode (`rollout.api_url`),
because that path stores string-only responses without token-backed interactions.
This does not restrict ordinary SGLang/vLLM backends, including pre-existing servers
passed through `server_infos`. External mode remains available when PRM is disabled
or the scorer list is empty.

## Scorer Contract

Subclass `areal.reward.prm.BaseScorer`, set a unique class-level `name`, and implement
`async evaluate(interaction, ctx)`. Return an unweighted scalar or a tensor with
shape `[interaction.model_response.output_len]`. A scalar is broadcast across
every output token in the turn; a dense tensor retains its token positions. The
runner multiplies each scorer's result by its weight and adds the contributions.

`ctx["messages"]` contains the full conversation for the current branch. Inputs
are read-only. Scorers can be shared by concurrent sessions, so keep request state
inside the coroutine. External calls should use asynchronous clients with bounded
timeouts and retries. An exception or `None` rejects the trajectory; an intentional
zero score must be returned as `0.0`.

Subclass `BaseTrajectoryScorer` for joint scoring of the full conversation. Its
`evaluate_trajectory(interactions, ctx)` receives turns in parent-before-child
order and returns a mapping from interaction IDs to unweighted rewards. Missing
IDs receive zero; unknown IDs are rejected.

For structured monitoring, override `prepare_result` or `evaluate_result` to return
`PRMScorerResult(reward, observations)`. Each `PRMMetricObservation` declares its
scope, target ID, value type and aggregations. Boolean observations support `count`
and `rate`; numeric observations support `sum` and `mean`. Schemas must stay stable
for the same scorer and metric. The example length-budget scorer demonstrates
this interface without adding dependencies.

### Scorer Lifecycle

`BaseScorer.aclose()` is an optional asynchronous cleanup hook with a no-op
default. Override it to flush buffered audit records, join background writers,
and close clients owned by the scorer. Bound external I/O inside the hook;
the runner does not add a cleanup timeout or retry policy.

`PRMRunner.aclose()` closes scorers it created from configuration, including
disabled scorers, in reverse creation order. Scorer instances passed directly
to the runner are borrowed: their caller remains responsible for closing them.
Each owned instance is closed at most once. Concurrent close calls wait for the
same cleanup; ordinary failures are logged and collected in an `ExceptionGroup` after all
owned scorers have been attempted. Cancellation propagates, and subsequent close
calls do not retry failed or interrupted hooks.

Use `await PRMRunner.create(config)` when constructing a runner in async code.
If scorer resolution, construction, or configuration validation fails, it awaits
cleanup of all previously constructed owned scorers before re-raising the original
startup error. Ordinary cleanup failures are logged and attached as error notes;
external cancellation during cleanup still propagates. Borrowed instances are not
closed. A scorer whose own constructor raises before returning remains responsible
for cleaning up resources acquired inside that constructor.

Callers must drain scoring before closing the runner. New `run()` calls are
rejected once closing starts. The v2 data proxy creates its runner during service
startup with the async factory and awaits cleanup during lifespan shutdown, before closing its inference
bridge and HTTP client. Those resources are still cleaned up if scorer cleanup
fails. A PRM startup error also closes the bridge and HTTP client without letting
ordinary service cleanup failures replace that startup error. The synchronous
`PRMRunner(config)` interface remains available for compatibility, without async
startup rollback. The v1 proxy still uses it and does not yet invoke `aclose()`
automatically.

## Advantage Shaping

Scoring produces `token_rewards` aligned to generated tokens; prompt tokens are
zero. PPO shifts these rewards to next-token prediction positions. With direct
process advantages, shaping happens **after** GAE and advantage normalization and
does not alter critic returns:

- `additive`: add the weighted process reward to the outcome advantage.
- `gvpo`: a negative process reward marks a failed token. Its negative outcome
  advantage is multiplied by `1 + negative_scale`; an approximately zero advantage
  becomes `-zero_penalty`; a positive advantage becomes zero.
- `process_weighted`: process rewards must be in `[0, 1]`. Non-negative outcome
  advantages are multiplied by the process reward. For negative advantages, a
  positive process reward replaces the advantage; a zero process reward preserves
  the negative advantage. This mode rejects `mask_no_eos_with_zero: true`.

The low-level actor also supports folding uniform turn rewards into GAE with
`token_rewards_as_adv: false`. Configured proxy scorers deliberately reject this
mode until the conversion of dense rewards to whole-turn totals is specified.

## Branches, Filtering And Metrics

Each exported root-to-leaf branch is cloned before scoring. Shared ancestors thus
receive branch-local scores without modifying another branch or the session
cache. A scorer commits only after all its results pass validation. The proxy
publishes metrics only when every branch succeeds, otherwise it rejects the
session's trajectories.

Metrics include `prm_turn_reward/<scorer>`, `prm_trajectory_reward/<scorer>` and
`prm_metric/<scope>/<scorer>/<metric>/<aggregation>`. Structured observations are
unweighted; reward metrics include scorer weights. Worker means and rates are
weighted by observation counts, while counts and sums are added. Evaluation uses
the `eval-rollout` namespace.

### v2 Export And Metric Transport

The v2 controller forwards the existing `PRMConfig` to each inference data proxy.
Scoring runs after ready trajectories are collected and before v2's existing
group outcome-reward normalization. Independent sessions are scored concurrently,
with branches within a session scored in order. Process token rewards are not
group-normalized. No online per-step scoring or new advantage formula is introduced.

A missing or failed PRM-scored session rejects the entire requested group, even
when outcome-reward normalization is disabled. An explicit discard request skips
scoring. Sessions are cleaned up on success, scoring failure, and cancellation
when `remove_session` is true. This does not add partial-group acceptance.
The online workflow retains the persistent HITL session (`__hitl__`) and consumes
only the selected ready trajectory, so new interactions and subsequent ready
trajectories survive an in-flight export. Ordinary session-key exports still
remove their session.

The export response carries `prm_stats` containing the existing typed turn results
and per-branch scorer totals. The workflow records them through the same metric
recorders as v1, outside the HTTP retry boundary. Means and rates therefore retain
their observation counts; concurrent exports do not drain a shared server-side
statistics buffer. Each successfully scored session contributes observations even
if another member rejects the group. These are scoring metrics, not counts of
samples eventually used by the optimizer. Failed sessions publish no partial
branch observations. Requests without PRM retain the original response shape.

Export consumes a ready trajectory once. Retrying an already consumed trajectory
does not run its scorer again. As with existing v2 export, a lost response is not
replayed: this is not an exactly-once delivery guarantee for trajectories or metrics.

For grouped rollout filtering, `examples.swe.filter_function` provides
`filter_mixed_or_penalized_all_wrong`: it retains mixed-outcome groups and
all-wrong groups that still have negative process signals. The
`filter_mixed_or_penalized_all_wrong_mask_no_eos` variant respects the actor's
truncation mask. Apply these filters to complete sample groups.
