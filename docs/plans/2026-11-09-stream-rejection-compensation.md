# Stream rejection compensation

## Agreed behavior

Preserve the configured base stream mixture after dynamic filtering by inserting
same-stream replacement prompts at dispatch time. Training batches keep consuming
available accepted results; there are no stream quotas, result buckets, or waits for
particular sources. Existing concurrency, prefetch, metrics, and staleness gates remain
in force.

For each stream, maintain unpaid replacement debt and queued/running reservations. A
rejected base group adds one debt. Select the greatest positive debt minus reservations
and reserve before enqueueing. An accepted replacement settles its debt; a rejected
replacement releases its reservation without adding another debt. Cancellation releases
reservations without classifying unfinished base work as a rejection. A rejects three/B
rejects one leads to three A/one B replacements.

Limit consecutive replacement draws to the dataloader batch size, then allow one base
draw without discarding debt. This prevents a permanently failing source from starving
healthy sources. Persist this counter across batches and checkpoints.

## Data and integration

`StreamDataset` marks training rows by a source key. The existing dataset sampler
provides the base stream schedule. `StreamDataLoader` advances independent source
cursors for both base and replacement draws; a replacement does not consume the next
base schedule slot. Source exhaustion starts another source-local cycle, reshuffled when
configured. Epoch length and the existing mixture weighting policy are preserved.

`RolloutController.prepare_batch` reserves a draw when producing a task. Its remote
completion callback provides feedback before results are consumed, including rejected
groups, exceptions, and timeouts. The generic dispatcher and training batch assembly are
unchanged. Multi-Stream Arena training opts in on the v1 single-controller path;
validation and unsupported execution modes retain their ordinary dataloaders.

## Recovery and limits

The loader checkpoint stores base-loader state, buffered schedule entries, source
cursors/shuffle epochs, and total unpaid debt under a shared lock. Recovery restores no
reservations because queued/running tasks do not survive. Existing async recovery still
does not replay completed but unconsumed results. Flat-loader checkpoints cannot be
loaded as stream-loader checkpoints. This implementation targets single-controller
rollout and rejects unsupported distributed sampler configurations.

Compensation restores source supply over time; it guarantees neither exact individual
batch proportions nor progress from a permanently failing source. No acceptance-rate
prediction or speculative oversampling is introduced.

## Validation

CPU tests cover unequal filtering, weighted base schedules, repeated rejection,
in-flight reservation accounting, concurrent feedback, cancellation, checkpoint
recovery, failure paths, and fast-stream result consumption while another source is
still running. GPU/multi-node performance validation is separate from these tests.
