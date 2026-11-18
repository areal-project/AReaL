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

## Ray

Use `ray_worker.sh controller` inside the controller image on an **existing** Ray
cluster. Pass `--config /shared/experiment/train.yaml` to reuse a Slurm experiment's
training configuration, including multiple Arena streams and per-task reward mappings.
Without `--config`, the default remains `train_fine_released64_bs16.yaml`. The entry
keeps the dense Qwen controller decorator and `qwen27_actor_worker` CPU optimizer
offload, and converts deployment settings to Ray.

Resource requirements come from `actor.backend` and `rollout.backend`, not a fixed node
count. Actor GPUs are DP × TP × PP × CP; colocated rollout must use the same total GPUs.
The actor is packed into nodes using `cluster.n_gpus_per_node` (framework default: 8),
including a partial last node. For multiple node bundles this must match the worker node
GPU capacity: larger nodes could pack multiple actor launchers on one node, which
grouped colocation cannot use. `cluster.n_nodes`, when supplied, must be large enough.
For example, actor `d1p2t4c2` with rollout `d4t4` uses two 8-GPU nodes; actor `d2p4t4c4`
with rollout `d32t4` uses sixteen. A legacy `allocation_mode`, if present, must agree
with the per-role backends. This entry supports Megatron actor separation with forked
SGLang colocation targeting actor. Each inference instance (TP × PP GPUs) must fit
within, and evenly partition, the actor GPUs on its node. Cross-node inference instances
are unsupported. Single-GPU inference instances need a nonempty rollout
`ray_runtime_env` (for example a role image) to use the role wrapper.

Prepare the controller and both role images with compatible Ray/Python versions and
role-specific dependencies. Set `QWEN_RAY_ACTOR_IMAGE` and `QWEN_RAY_ROLLOUT_IMAGE` to
OCI references in the profile. They become each role's `ray_runtime_env.image_uri`: Ray
starts the containers through Podman on the raylet node. The wrapper executes inside
that container; it does not launch Singularity or another raylet. See
[Ray container deployment](../../../ray/README.md) for Podman, GPU access, shared mounts
and Kubernetes prerequisites. This path does not require Slurm services.

`QWEN_RAY_PYTHON` selects the controller interpreter. Independently set
`QWEN_RAY_ACTOR_PYTHON` and `QWEN_RAY_ROLLOUT_PYTHON` to the native Python paths inside
their images (default `python3`). Do not force the controller's interpreter or
site-packages onto the worker images. For example, a training image using a virtualenv
should select that virtualenv's Python to avoid mixing CUDA/cuDNN installations. Proxy
forks use the rollout worker's interpreter.

For explicit Podman mounts/devices, put `ray_runtime_env.container` with `image` and
`run_options` in each role's `scheduling_spec` in a custom YAML and leave the profile's
image fields empty. Do not combine a profile image with `container`. These deployment
settings are preserved in the generated configuration. The images must contain an
importable AReaL installation before Ray deserializes the launcher; the shell wrapper's
later `PYTHONPATH` setup cannot provide that bootstrap requirement.

Optional `QWEN_RAY_ACTOR_PYTHONPATH` and `QWEN_RAY_ROLLOUT_PYTHONPATH` select
role-specific package directories before Python starts. Leave them empty when the image
contains the dependencies. A profile can use `QWEN_RUNTIME_ROLE` (`controller`, `actor`,
or `rollout`) to select native library paths. Do not change the scheduler-provided
GPU/rank variables. The profile is a shell script: it may source existing private
credential files and set native library paths by role. A Slurm `site.env` often only
contains paths; recreate its runtime's credential-loading order and dependency overlays
in the Ray profile. Do not source a Slurm `runtime.sh` that installs task prologs,
assumes SIF packages or infers roles from Slurm job names.

1. Make the checkout, model, output storage and private profile readable at the same
   absolute paths on every node. Use a fixed checkout for the run; this Ray entry does
   not create the Slurm submission's source snapshot. Nodes need enough GPUs for the
   derived placement and enough Ray CPU/memory resources for the actor requests (32 CPUs
   and 512 GiB per full node in the default preset). CPU optimizer and rollout processes
   also need sufficient physical host memory.

1. Copy `ray.env.example` to an external private profile. Source your existing
   credentials there. Set the model/output paths, explicit experiment/trial identity,
   shared run directory and `RAY_ADDRESS`. For the default single-stream template, set
   its stream/harness/reward overrides. For multiple streams, leave those overrides
   empty and define each stream in the YAML. `sampling_weight`, `reward_mapping_file`,
   optional harnesses and expected reward references are preserved. A mapping and an
   explicit reward reference are mutually exclusive. The native Arena loader validates
   task bindings and resolves missing harness/reward defaults from Arena. There is no
   need to supply Slurm images, partitions, reservations or mounts.

1. Prepare the configuration without starting Ray workers or training:

   ```bash
   export QWEN_ENV_FILE=/shared/private/qwen27-ray.env
   bash examples/swe/rl/qwen38_27b/ray_worker.sh controller --prepare-only
   ```

   This writes `$QWEN_RUN_DIR/train-ray.yaml` with credential references, not expanded
   credentials. Review it before running. To edit training settings, copy the bs16
   preset and pass `--config /shared/path/custom.yaml`; this option also works with
   `--prepare-only`. `QWEN_RECIPE_DIR` points to the input YAML's directory in both
   controller and workers, so `${oc.env:QWEN_RECIPE_DIR}/reward-mappings/...` keeps
   referring to the experiment's files. Mount that directory in both role images.
   Changing an existing generated config requires a new run directory, so a running
   trial's config cannot be overwritten accidentally.

1. Verify controller/worker RPC and training communication, worker access to Arena
   Online, and **Arena LLM gateway access back to each rollout proxy's advertised
   address and port**. Successful model registration does not verify that callback. Then
   start the driver in your Ray job platform:

   ```bash
   bash examples/swe/rl/qwen38_27b/ray_worker.sh controller
   ```

The driver connects to the selected cluster without falling back to local Ray and checks
the derived per-node GPU requirements against live nodes. This checks total physical
capacity; busy resources may still make Ray wait, and CPU/memory requests still need
separate capacity planning. `--prepare-only` does not check the target images, network
or cluster. Validate real multimodal inference, rewards, CPU optimizer allocation,
initial AWEX synchronization and the first optimizer update before treating a submitted
job as healthy.

The role wrapper ignores `SLURM_JOB_NAME`, preserves Ray's device/rank assignment, uses
unique process cache/log paths, and explicitly enables the historical external
multimodal processor. Visual embedding cache remains 100 MB. The profile's pixel limit
and image alignment settings must match for proxy and SGLang.

For a new experiment choose a fresh trial. If its log/checkpoint directory already
exists, startup fails unless `--resume` is supplied. Resume requires an actual loaded
checkpoint; it must not silently start at step zero. Stop the previous Ray job and its
workers before resuming. The Ray entry does not execute the Slurm job-state preflight or
install an `afterany` cleanup job. Configure your Ray job platform to terminate the
driver/workers and arrange trial-scoped Arena task cleanup after a hard kill; do not run
the Slurm cleanup script against Ray jobs.

Task-specific [SWE-MM and GSM8K drivers](../../qwen27b/README.md) provide Ray Jobs
submission examples, a standalone GSM8K preset, and repeatable `--set KEY=VALUE`
configuration overrides.
