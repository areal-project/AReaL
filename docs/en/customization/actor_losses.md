# Composable actor losses

`actor.losses` defines named token objectives with independent support, reduction, and
coefficients. An empty mapping preserves the existing PPO/PRM/MOPD objective. A nonempty
mapping replaces that implicit objective: include a policy term when policy optimization
is desired.

```yaml
actor:
  loss_aggregation: token_mean
  losses:
    policy:
      implementation: policy
      reduction: prompt_mean
    exploration:
      implementation: entropy
      coefficient: 0.01
  megatron:
    entropy_requires_grad: true
```

The built-ins are `policy` (PPO, SAPO, or CISPO using the actor defaults), `entropy`
(negative entropy), and `mopd` (multi-teacher distillation). Entropy must be
differentiable; for Megatron enable `entropy_requires_grad` and use a compatible LM-head
configuration. Loss plugins are resolved once on each actor worker.

## Configuration and precedence

Each term accepts `implementation`, `enabled`, `coefficient`, `reduction`, `mask_key`,
and `kwargs`. Coefficients must be finite and nonnegative. Disabled or zero-coefficient
terms are not imported or evaluated; at least one term must be active. `kwargs` must be
JSON-serializable and cannot override runtime arguments.

An omitted `reduction` inherits `actor.loss_aggregation`. This is the single default
entry point; migrate source configurations using `default_loss_reduction` to
`loss_aggregation`. Explicit built-in kwargs override actor defaults, except settings
that control actor preprocessing (`prox_logp_method` and `use_decoupled_loss`), which
must agree with the actor.

`mask_key` defaults to `loss_mask`. It selects a binary token-shaped batch tensor,
intersected with `loss_mask` before optimizer splitting. The resulting support fixes
normalization; subsequent rejection filtering changes only the numerator. Zero reward
does not imply absent support. Custom masks must already use prediction (next-token
label) positions, matching the post-advantage `loss_mask`; they are not shifted
automatically. Raw rollout `turn_ids` are shifted separately.

## Reductions

| Reduction     | Objective                                                                               |
| ------------- | --------------------------------------------------------------------------------------- |
| `token_mean`  | Mean over supported tokens.                                                             |
| `seq_mean`    | Mean of supported-token means over active responses.                                    |
| `constant`    | Token sum divided by active responses times `actor.loss_aggregation_divisor`.           |
| `turn_mean`   | Mean of supported-token means over active turns.                                        |
| `prompt_mean` | Equal active-prompt weighting, with equal supported-token weighting within each prompt. |

Each loss uses its own support and denominator. A globally empty named term contributes
a differentiable zero. `turn_mean` requires raw token-aligned `turn_ids`; the actor
shifts them to prediction positions. Turn weights and counts are prepared before packing
or microbatch splitting. IDs may repeat across responses but must be nonnegative and
nondecreasing on supported tokens within a response, and smaller than its padded
sequence length.

`prompt_mean` preserves the existing response-level optimizer schedule. A prompt may
span optimizer steps. For `K` realized steps and `G` globally active prompt groups for a
term, each step uses the full-update coefficients scaled by `K/G`. At fixed parameters,
averaging step contributions recovers the full-update objective. This does not imply
that sequential parameter updates equal one large optimizer update. Fully masked groups
do not contribute to `G`.

The engine receives one summed numerator and one normalizer per name. The coefficient
multiplies the numerator exactly once. All ranks and microbatches retain the same name
schema, including empty terms and transport-only ranks.

## Custom token losses

Use an importable dotted callable with explicit keyword-only runtime arguments:

```python
import torch


def custom_loss(*, outputs, batch, mask, scale=1.0):
    return torch.where(mask, -scale * outputs["entropy"], 0.0)
```

`outputs` includes differentiable `logprobs`, `entropy`, and available vocabulary
statistics. `batch` includes prepared advantages and the current policy version. Both
mappings are read-only; do not mutate their tensors. Return a differentiable
floating-point tensor with the same shape and device as `mask`. Values on support must
be finite. Do not reduce the result or renormalize after numerator filtering. The
callback may receive padded or packed tensors and must handle both layouts.

## Compatibility and limits

- The unnamed objective and its diagnostics retain their existing behavior.
- Named terms report their own weighted loss means. Named policy terms also report
  clipping, importance-weight, KL, and rejection statistics; the complete legacy
  proximal-approximation diagnostic set is not reproduced for each named term.
- Named MOPD uses immutable `mopd_behavior_logprobs`, not recomputed PPO logprobs. Its
  coefficient and `importance_ratio_cap` kwarg determine the actual named objective.
  `mopd.loss` still controls whether teacher targets and RL advantages are prepared; it
  does not add a second implicit loss or override named coefficients. Enable teacher
  targets when using a named MOPD term. A policy term cannot be combined with
  pure-distillation preprocessing.
- Legacy single-teacher `teacher_logp` and forward-dependent M2PO selection are
  unsupported with explicit named losses; their unnamed paths remain available.
- Tree-packed training supports only `token_mean` terms.
- Separate process-reward loss routing, efficiency presets, and turn-reward ingestion
  are outside this interface. Existing PRM advantage shaping remains available through
  the advantage pipeline.
