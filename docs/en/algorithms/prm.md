# Process Rewards

Process rewards attach a training signal to each generated turn, in addition to
the trajectory's outcome reward. The v1 OpenAI proxy scores complete trajectories
before exporting them to PPO/GRPO. Scorers are supplied as Python classes; no
external judge service is required by AReaL itself.

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
require v1 rollout, concat export, concat chat templates, and direct process
advantages. Individual export and the v2 agent service are not supported.

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

For grouped rollout filtering, `examples.swe.filter_function` provides
`filter_mixed_or_penalized_all_wrong`: it retains mixed-outcome groups and
all-wrong groups that still have negative process signals. The
`filter_mixed_or_penalized_all_wrong_mask_no_eos` variant respects the actor's
truncation mask. Apply these filters to complete sample groups.
