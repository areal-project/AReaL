# Partial rollouts across policy versions

Enable continuation buffering and training token filtering together:

```yaml
rollout:
  enable_partial_rollout: true
actor:
  mask_stale_tokens: true
  max_token_version_gap: 1
```

Both switches default to false. The maximum token version gap defaults to 1 and
must be a non-negative integer. Rollout buffering is supported by the v1 remote
inference engine and the v2 inference bridge.

When generation is aborted, its request, generated tokens, behavior log
probabilities, token versions, and segment boundaries remain in the engine's
`partial_rollout_buffer`. The request resumes after a higher policy version is
published and generation is unpaused. It keeps the generated prefix as input and
uses only the remaining generation budget. Cancellation removes the buffer entry.
The workflow and its group remain alive; this mechanism does not reject an entire
trajectory because an earlier segment is old.

`ModelResponse.generation_segments` contains `(start, end, policy_version)` tuples,
with half-open offsets into output tokens. `output_versions` carries the equivalent
per-token attribution into the training trajectory's `versions` tensor. Existing
trajectory dumps also contain `version_rle`, recording consecutive version runs.

Before GAE, the actor compares its current engine version against each generated
token's version. A token is masked only when this difference is **strictly greater**
than `max_token_version_gap`. Prompt and attention masks are preserved. Stale
tokens are skipped by GAE and advantage normalization, their final advantages and
returns are zero, and PPO uses the resulting loss mask. A gap equal to the threshold
is permitted. If an optimizer minibatch has no trainable tokens on any rank,
both actor and critic skip that optimizer update; a locally empty rank still
participates when a peer has valid tokens. The current generation version receives
no special exemption from this rule if a completed trajectory waits in the queue
before training.

With masking enabled, locally empty ranks also participate in batch-level
advantage normalization. This prevents a rank with only stale actions from
skipping a collective that other ranks need for their valid actions.

The training statistics include:

- `stale_generated_tokens`: number of originally trainable generated tokens.
- `stale_masked_tokens`: number removed by the version-gap rule.
- `stale_masked_ratio`: removed tokens divided by originally trainable tokens.
- `stale_partially_masked_trajectories`: trajectories retaining fresh actions while
  their old actions are masked.
- `ppo_actor/update/stale_empty_minibatch_fraction`: optimizer minibatches skipped
  because their effective loss mask is globally empty.

These statistics are accumulated through the existing distributed stats tracker.
With masking disabled, version metadata is not required by this feature and no
new masking statistics are emitted. Existing rejection-sampling settings remain
independent and may further filter tokens.

## Validation

`tests/test_partial_rollout_versions.py` covers threshold boundaries, context
preservation, PPO gradients, all-stale rows, disabled behavior, and the actual
remote-generation abort/resume loop with a controlled backend. The v2 bridge test
suite covers its buffer/resume path too. These are behavior tests, not evidence of
GPU training. A two-process Gloo test checks advantage normalization with one
empty rank and one rank retaining valid actions.

`scripts/rsi_gsm8k_validate.sh OUTPUT_DIR` runs regression tests and fresh disabled
and enabled GSM8K training jobs in an existing Slurm allocation. It reads
`validation/environment.json`, or the file named by `RSI_VALIDATION_ENV`, for the
allocation, container, model, dataset and bind paths. Each job writes its own
configs and logs. Actual training completion, mixed-version trajectories and
nonzero masking statistics must be checked before declaring the feature validated.
