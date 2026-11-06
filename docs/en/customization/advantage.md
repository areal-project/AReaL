# Custom advantage computation

`actor.advantage` selects the whole advantage computation or individual stages. With the
default configuration, existing GAE, reward normalization, and PRM settings retain their
behavior. The configuration contains import paths and JSON-serializable keyword
arguments; each actor worker resolves its functions at initialization.

```yaml
actor:
  advantage:
    compute:
      implementation: default
      kwargs: {}
    outcome_reward_shaping: null
    estimator: null
    process_adv_shaping: null
```

The default pipeline applies length penalties, outcome shaping, reward scaling and
normalization, temporal estimation, advantage normalization, then process advantage
shaping. Loss aggregation and optimizer minibatch scheduling are unchanged.

| Stage                    | Built-in implementations               | Default                                    |
| ------------------------ | -------------------------------------- | ------------------------------------------ |
| `outcome_reward_shaping` | `identity`                             | Unmodified outcome reward                  |
| `estimator`              | `token_gae`, `turn_gae`                | Follows `actor.gae_timestep_unit`          |
| `process_adv_shaping`    | `additive`, `gvpo`, `process_weighted` | Existing PRM selection, otherwise additive |

Every stage also accepts an importable dotted Python function path and `kwargs`. `auto`
selects the stage default. Process `auto` does not accept kwargs. Explicit process
selection wins over call-local PRM selection; explicit GVPO kwargs override legacy GVPO
values, while unspecified values inherit them. Existing PRM settings remain supported.

```yaml
actor:
  token_rewards_as_adv: true
  advantage:
    estimator:
      implementation: turn_gae
      kwargs: {}
    process_adv_shaping:
      implementation: gvpo
      kwargs:
        negative_scale: 0.2
        zero_penalty: 0.4
```

Turn GAE requires rollout `turn_ids`. Its token KL penalties affect actor advantages but
not critic returns. Token GAE includes KL in the recurrence. A custom GAE lambda
receives lengths in the selected estimator's timestep unit, including when the estimator
overrides `gae_timestep_unit`. Existing `GAELambdaContext` functions remain supported;
`GAELambdaInputs` is an alias for the same type.

## Stage functions

Outcome functions accept `OutcomeRewardInputs` from `areal.trainer.ppo.advantage`. Its
`data` contains rollout tensors after the length penalty, `config` is the actor
configuration, and `meta` contains trajectory and logical rollout grouping. Return
`OutcomeResult(rewards=...)` with one floating-point reward per tensor row:

```python
from areal.trainer.ppo.advantage import OutcomeResult, OutcomeRewardInputs


def scale_outcome(inputs: OutcomeRewardInputs, *, factor: float) -> OutcomeResult:
    return OutcomeResult(rewards=inputs.data["rewards"] * factor)
```

Use `implementation: my_package.my_module.scale_outcome` and `kwargs: {factor: 0.5}`.
The module must be available on every actor worker. Functions must preserve tensor row
order, grouping, and device. Do not modify the input mapping or its tensors in place.

When reward normalization uses explicit logical rollout references, custom shaping must
also return `OutcomeResult.normalization_rewards`: one shaped reference per logical
rollout, before bias, scaling, and clipping. Read the raw references and row mapping
from `meta.rollout_groups`. Without explicit references, equal rows within a logical
rollout allow reference inference; differing rows require explicit shaped references.
`meta.traj_group_sizes` counts tensor rows per prompt, while `meta.logical_group_sizes`
counts logical rollouts per prompt. Shaped references require complete metadata produced
by `concat_batch`; they do not change membership or row counts. Distributed plugins
should use framework metadata rather than constructing partial metadata themselves.

Temporal functions accept `TemporalInputs` from `areal.trainer.ppo.advantage_algorithms`
and return `TemporalResult(advantages, returns, total_rewards)`. Inputs use next-token
prediction alignment. Outcome, KL, and folded process rewards are separate channels. The
explicit `bootstrap_values` come from each trajectory's actual final valid token.
Callable lambda functions can be resolved with `compute_gae_lambda` in
`areal.trainer.ppo.gae`, passing the estimator's timestep unit. `total_rewards` is the
token-aligned monitoring channel.

Process functions accept `ProcessAdvantageInputs` and return a tensor of advantages.
They receive normalized advantages, aligned direct token rewards (possibly `None`), and
the loss mask. They do not change critic returns. Built-in GVPO and process-weighted
shaping require `token_rewards_as_adv=True`; process-weighted shaping also requires
`mask_no_eos_with_zero=False`.

## Replacing the full computation

Set `compute.implementation` to a dotted function path accepting `AdvantageInputs` and
returning `AdvantageResult`. A full override cannot be combined with stage settings. The
default compute does not accept kwargs; configure its stages instead.

`AdvantageResult` supplies advantages, returns, loss mask, aligned log probabilities,
outcome rewards, total rewards, and KL rewards. Outputs must be finite floating-point
tensors on the expected device and with the expected shape; masks must be binary. The
identity outcome stage also preserves legacy integer task rewards. The optional `extras`
mapping may contain only `input_token_loss_mask`, `is_truncated`,
`mopd_behavior_logprobs`, `prox_logp`, and `ref_logp`. It cannot replace input IDs,
values, original rewards, or other rollout fields. A full override owns alignment and
must supply the derived metadata needed by its training mode, including immutable
behavior log probabilities when combining with MOPD.
