# Qwen3.8-27B Arena RL

The default submission selects `train_fine_released64_bs16.yaml`, a bs16/gs12 training
preset. Vision encoder and projection are frozen, and SGLang's visual embedding cache
uses 100 MB. The cache setting is applied after the site profile so an inherited
`SGLANG_VLM_CACHE_SIZE_MB=0` cannot disable it. `train.yaml` retains the earlier preset
and can be selected explicitly with `QWEN_TRAIN_CONFIG=train.yaml` in the site profile.

The example profile uses experiment `qwen38-27b-arena-rl` and generates a dated trial
name. Choose a unique trial name for each new run. W&B, Slurm names, and log directories
use this same experiment/trial identity.

The recipe provides controller and actor entry points and Slurm submission preflights.

Actor/controller use **areal-dev-sglang-0821.sif**, with the historical dependency
overlay providing **Megatron-Core 0.18.2**, **megatron-bridge 0.5.1** and **AWEX
0.8.2**. Rollout uses **deepep-base-85b53914.sif**, containing **SGLang
0.5.18.dev10+g85b539146**; the proxy overlay supplies **litellm 1.83.7** and **fastuuid
0.14.0**. Keep the two images and their dependency paths separate. The recipe references
these installed packages; it does not vendor runtime wheels or private environment
files.

## Launch

1. Copy `site.env.example` to `site.env` next to it, or set `QWEN_ENV_FILE` to an
   external profile. Keep the local profile Git-ignored because it contains credentials.
   Fill the site paths, URLs, mounts, and credentials from the working environment.
   `QWEN_REPO` identifies the local source checkout; it does not have to be clean.

1. Set a new experiment/trial and shared run directory. `QWEN_RUN_DIR` stores the
   config, reward mapping, and reports. Set `QWEN_OUTPUT_ROOT` to a shared directory
   visible to every role; logs and checkpoints use
   `logs/<user>/<experiment_name>/<trial_name>/` and
   `checkpoints/<user>/<experiment_name>/<trial_name>/`. The submitting user is
   preserved in these paths even when submission uses `sudo`.

1. Run from the local checkout:

   ```bash
   # Optional: sync and verify the snapshot without submitting any jobs.
   bash examples/swe/rl/qwen38_27b/submit.sh --sync-only

   # Sync (or reuse the identical snapshot), then submit the controller.
   bash examples/swe/rl/qwen38_27b/submit.sh
   ```

The launcher uses `sudo` when necessary to copy code and the selected profile into
`$QWEN_RUN_DIR/source/code`. `QWEN_SYNC_DIR` can override the snapshot directory; it
must be on shared storage visible to all containers. Source and destination must not
overlap. Tracked files include local edits and deletions; non-ignored new files under
`areal/`, `examples/`, `tests/`, and `scripts/` are also included. Git metadata, virtual
environments, Python/tool caches, unrelated `.env` files, and root-level scratch files
are excluded. The selected profile is copied explicitly to the snapshot's recipe
directory. Relative symlinks must resolve within the snapshot; profiles sourcing
additional files must use shared paths for those files.

The example profile generates a date/time trial name once per submission and passes it
to every role. To reuse a trial (including one staged with `--sync-only`), pass its name
explicitly:
`QWEN_ARENA_TRIAL=<existing-trial> bash examples/swe/rl/qwen38_27b/submit.sh`. The
timestamp has minute precision, so use an explicit distinct name for additional runs
submitted within the same minute.

A SHA-256 manifest records the actual copied source and profile; Git HEAD is retained
only as provenance. Submission and controller startup verify the manifest without
requiring Git in the snapshot. The shared repository, profile, and run paths are pinned
in every role. An existing trial snapshot is reused only if identical; changed code or
profile requires a new trial/run directory, so active workers' code is never
overwritten.

Before submitting, replace `your-stream-id` under `econfig.arena_streams` in the default
YAML with an active Arena Stream ID. The default preset uses the selected Stream's
harness, LLM protocol, and reward reference. Verify that the Stream has suitable
defaults for your tasks. If you explicitly configure a `reward_mapping_file`, keep that
file when recovering and make it readable by all roles.

The controller verifies the source snapshot and durable storage before starting the
training entrypoint. Configuration and dependencies follow normal AReaL initialization.
Arena and Slurm cleanup remain scoped to this trial. Native SGLang logs rotate under the
trial's `native/` log directory. Existing NUMA settings in `train.yaml` are preserved,
including `SGLANG_NUMA_BIND_V2=0`.

## Opt-in v1 rollout timing

Copy the selected training YAML to a new recipe-local name, such as
`train_trace.yaml`, set `QWEN_TRAIN_CONFIG=train_trace.yaml` in `site.env`, and add this
block under `rollout.agent` in the copied YAML:

```yaml
    rollout_trace:
      enabled: true
      collect_interval_seconds: 60
      log_interval_seconds: 600
      long_tail_seconds: 300
      top_n: 20
```

The controller appends one bounded `Rollout long tails` snapshot to `main.log` each
`log_interval_seconds` while eligible sessions are active. It shows Arena task and
attempt IDs, elapsed time, current phase and phase age, completed plus ongoing
generation time, reward, pause, other, last progress, request count, and Arena retry
count. `collect_interval_seconds` controls how often the controller reads proxy
events; live log time resolution is therefore bounded by that interval. A phase
change between polls is still reconstructed from timestamped events.

Each completed training step emits `timeperf/rollout/generate`, `reward`,
`pause_wait`, and `other`; `toolcall` appears only if an explicit tool interval is
available. The child values add to the existing `timeperf/rollout` step wall time.
Concurrent sessions share each wall-clock slice in proportion to active session
counts. `generate` is the proxy's full model request interval, including request
setup and streaming; it is not SGLang kernel time. Arena reward includes waiting for
the reward sandbox, creating it, and evaluating, using the Task event timestamps.
Unobserved time, including tool execution and pause without explicit timestamps,
stays in `other`. The current Arena Task trace exposes no dedicated tool spans, so
this recipe does not produce `toolcall`; it also has no session-level pause signal.
Cross-node clocks must be synchronized for accurate attribution.

For a small local one-step proxy smoke run, use
`bash examples/agent_workflow/verify_rollout_trace_minimal.sh` on a configured CUDA
host. The expected result is a completed step with additive
`timeperf/rollout/*` values; a `Rollout long tails` line appears only if the
step stays active past the short smoke script's log interval.
For this Qwen/Arena chain, use
`bash examples/swe/rl/qwen38_27b/verify_rollout_trace_e2e.sh` after preparing the
opt-in YAML and site profile. The expected log contains the joined Arena
`task_id / attempt_id`; after a completed reward trace, the step's `reward` time
reflects its explicit reward interval. The end-to-end script submits the job and
waits up to five hours for the first step's timing fields; the submitted training
job continues after that check. These scripts have not been run locally.

## Recovery

Reuse the same profile, trial, run directory and source snapshot. For an explicit
recovery, set `QWEN_REQUIRE_RECOVER=1` and `QWEN_PREVIOUS_SUBMISSION` to the previous
attempt's immutable JSON receipt containing `controller_job_id`, `actor_job_id`,
`rollout_job_id` and `cleanup_job_id`. All four jobs must be terminal. The host checks
Slurm state; the actor container checks the matching fresh report and the published
checkpoint manifest. These checks do not replace actual distributed optimizer loading.

The independent experiment observer and logp/GDN diagnostic candidates are outside this
recipe. FA3 adapters, forced stable top-k, cache isolation and tensor replay patches are
not added by these files.
