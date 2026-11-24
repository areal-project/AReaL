# Ascend nightly CI

Install the reusable workflows, scripts, and benchmark config on `ascend` first. On
`main`, install only `.github/workflows/ascend-nightly-trigger.yml`. It calls
`ascend-nightly.yml@ascend` daily at 04:00 UTC (9 p.m. Vancouver the previous day) or on
manual dispatch. The `e2e` job in that workflow calls `ascend-nightly-e2e.yml`.

Both training jobs use the `linux-aarch64-a3-800i-16` runner and the same A3 nightly
container image as Ascend PR CI. Allow the workflow token to create and update the
`ascend-nightly-results` branch.

The runner needs storage for the model cache and one full HF export at a time. Jobs
resolve model weights from the ModelScope and Hugging Face caches and prepare the
Geometry3K dataset before offline training. For preloaded inputs, set
`ASCEND_NIGHTLY_GRPO_MODEL`, `ASCEND_NIGHTLY_SFT_MODEL`, or `ASCEND_NIGHTLY_GEOMETRY3K`
to paths visible inside the container. The dataset path must contain `geometry3k` for
AReaL's dataset selection.

GRPO uses the nightly Qwen3.5-35B-A3B config. SFT uses the nightly Qwen3.6-27B
Geometry3K config with the Ray scheduler. Each job trains for two epochs, validates each
epoch, and saves once at the end of epoch two. The job checks the step sequence, final
checkpoint metadata and shard index before discarding temporary training files.

The results branch stores the last attempted source SHA and the last successful
benchmark snapshots. An unchanged or documentation-only diff skips both training jobs.
Use **Re-run failed jobs** to retry a failed training job at the same revision. For each
successful suite, the publish job compares its metrics with its last successful snapshot
and writes the tables and charts to the Actions summary. The first successful run
establishes the baseline.
