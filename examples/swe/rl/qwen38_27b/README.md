# Qwen3.8-27B Arena RL

This recipe uses the dense 27B training settings from the 19-task Graybox experiment and
selects the 64-task Whitebox B Stream for the next run. The branch starts from upstream
`main` at `448de523`. Arena categorized retries, RPC/checkpoint fixes and
required-recovery checks are already upstream. The additional core fix carries
`de8d6796`: resolve rollout dump paths without importing metrics SDKs in inference
workers.

The new files provide the dense controller/actor entry points, the exact training
parameters and Slurm submission preflights. They do not modify the running trial.

| Setting       | Value                                                               |
| ------------- | ------------------------------------------------------------------- |
| Sampling      | temperature 1, top-p 1, top-k 100000                                |
| Batch / group | bs16 / gs12; consumer batch 16; rollout concurrency 1500; queue 256 |
| Training      | TP4 / PP4 / CP4 / EP1; CPU Adam offload                             |
| Inference     | 16 TP4 replicas; CUDA graph, memory saver and radix cache enabled   |
| Attention     | dense provider `auto`; native GDN defaults; no PLE override         |
| LM head       | native projection; chunked logits disabled                          |
| Reward        | group normalization uses standard deviation                         |
| R3            | disabled                                                            |
| Recovery      | auto; model/optimizer/LR/RNG state retained; save every two hours   |

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
   config, reward mapping, and reports. `QWEN_OUTPUT_ROOT` defaults to
   `/storage/openpsi/experiments`; logs and checkpoints use
   `logs/<user>/<experiment_name>/<trial_name>/` and
   `checkpoints/<user>/<experiment_name>/<trial_name>/`. The submitting user is
   preserved in these paths even though submission uses `sudo`, as in the previous sxj
   wrapper.

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

The YAML selects `whitebox-b-overfit64-20261013` with
`whitebox-unified-b-areal@0.1.1-no-budget`. Dataset setup generates
`$QWEN_RUN_DIR/reward-mapping.json` from each published Env version's original reward
reference and checks reward publication before creating rollout workers. Keep this
mapping when recovering; all roles must be able to read it.

The controller verifies the source snapshot and durable storage before starting the
training entrypoint. Configuration and dependencies follow normal AReaL initialization.
Arena and Slurm cleanup remain scoped to this trial. Native SGLang logs rotate under the
trial's `native/` log directory. Existing NUMA settings in `train.yaml` are preserved,
including `SGLANG_NUMA_BIND_V2=0`.

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
