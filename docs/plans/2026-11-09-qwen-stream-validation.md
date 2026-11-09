# Qwen Arena stream validation

Validate the two stream metrics and replacement commits through the actual Qwen recipe,
with the reward predicate disabled. Share the supported-path dataset wrapper with the
standard SWE entrypoint; evaluation retains its original dataset.

For this experiment, opt into `QWEN_EQUAL_STREAM_SAMPLING=1`. Give each source the same
number of base dispatch slots, rounded to complete batches. Preserve every original row
and the native per-source cursors and rejection feedback. Results remain
completion-ordered; individual batches need not have equal source counts. Default
recipes retain their original source proportions.

Remove the deployment's older accepted-batch quota override. Keep batch size 64, 16
nodes, actor DP2, 12 samples, minimum 8 usable samples, and reward standard
normalization. Remove the two-step training cap while retaining 100 epochs.

Repair multi-stream cancellation without widening ownership: match only model names
recorded in the trial's own rollout log. Test foreign-model exclusion and missing-stream
refusal. Run the stream/controller/workflow tests in the actor image and full pre-commit
checks before committing. Real rollout, optimizer update, weight synchronization, and
subsequent rollout are separate runtime gates.
