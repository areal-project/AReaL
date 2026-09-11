# Arena single- and multi-Stream training

Arena supplies datasets, runs the external Harness, and returns terminal rewards. AReaL
serves the policy through its rollout proxy and trains on the recorded tokens. This
integration does not require AReaL-SWEAgent, PRM, or RewardSystem.

## Configuration

`arena_grpo.yaml` is a two-node FSDP/SGLang smoke profile: eight training GPUs, eight
inference GPUs, two samples per prompt, and one training step. The request budget is
32767 tokens against a 32768-token SGLang context, leaving its required one-token
margin. Set these environment variables to your deployment's values:

```bash
export AREAL_DIR=/path/to/AReaL
export AREAL_IMAGE=/path/to/compatible-image.sif
export AREAL_PYTHON=/path/to/python/in/container
export AREAL_FILEROOT=/shared/path/to/experiments
export MODEL_PATH=/shared/path/to/Qwen3-4B-Instruct
export ARENA_CREDENTIALS_FILE=/shared/private/arena.env
```

Keep only credentials and the Arena API base in the private credentials file. It must
export `ARENA_OPENAPI_BASE`, `ARENA_OPENAPI_TOKEN`, `ARENA_LLM_API_KEY`, and
`SWE_RL_ADMIN_API_KEY`. Load it in the controller environment as well. Workers load the
same file; credentials are not embedded in worker commands or committed YAML.

`ARENA_LLM_API_KEY` must authenticate to Arena's model gateway. It is not an upstream
model-provider key. If your deployment uses the same Arena bearer token for both APIs,
set `export ARENA_LLM_API_KEY="$ARENA_OPENAPI_TOKEN"` in the credentials file.
Control-plane preflight success alone does not validate model-gateway authentication.

The example assumes a compatible container with Slurm available to the controller,
shared storage, and network reachability from Arena to rollout proxies. Use the included
submission script after setting the site-specific mounts:

```bash
# Comma-separated Apptainer binds supplied by your cluster setup.
# Workers need the shared checkout, model, credentials and output paths.
export AREAL_WORKER_MOUNTS="$SITE_SHARED_MOUNTS"
# Controller additionally needs working Slurm commands and authentication.
export AREAL_CONTROLLER_MOUNTS="$SITE_SHARED_MOUNTS,$SITE_SLURM_MOUNTS"
export AREAL_CONTAINER_BIN=apptainer  # or singularity, available on compute nodes
export SBATCH_PARTITION="$SITE_PARTITION"
export SBATCH_TIMELIMIT=01:00:00      # worker limit
export ARENA_CONTROLLER_TIME=01:30:00
export ARENA_STREAM_ID=your-stream

bash examples/swe/submit_arena.sh --check-arena
bash examples/swe/submit_arena.sh
```

The script prints the submitted Slurm job ID. It allocates a **CPU-only controller** (4
CPUs, 16 GB); config validation and Arena preflight run inside the container before
AReaL submits any GPU workers. `--check-arena` stops after preflight. Check
`$AREAL_FILEROOT/submissions/$TRIAL_NAME/controller-<job-id>.log` and `preflight.json`;
set `TRIAL_NAME` explicitly to identify a run, otherwise a unique name is generated and
used as the controller job name. Use `squeue` to find the job and its name.

`CONFIG_PATH` defaults to `examples/swe/arena_grpo.yaml`. For multi-stream training:

```bash
export CONFIG_PATH=examples/swe/arena_multi_stream.yaml
# Fill in the Stream/Harness/reward entries before submitting.
bash examples/swe/submit_arena.sh
```

Use a custom YAML to change the GPU topology or training settings. Both preflight and
training resolve that same file, including its Hydra defaults. The supplied profile uses
two GPU nodes; the controller does not reserve those nodes itself. Optional Slurm site
settings use the usual `SBATCH_ACCOUNT`, `SBATCH_RESERVATION`, and related environment
variables. The worker scheduler retains its existing default shared-storage bind;
`AREAL_WORKER_MOUNTS` adds the mounts required by your site.

Keep the checkout and configuration unchanged while the job is queued or running; use a
dedicated checkout for each revision. Scripts do not copy or snapshot code. All paths
must be visible at the same locations inside the containers. The credentials file is
sourced only in the controller container and workers, never printed or embedded in
generated commands. In the controller, explicit launch paths, Stream ID and API base
take precedence over older credential files that also contain deployment settings.
Workers source the file again, so remove unrelated deployment settings from it.

For one Stream, set `ARENA_STREAM_ID`. For multiple Streams, edit the literal
`econfig.arena_streams` entries in `arena_multi_stream.yaml` and select that config. Pin
each Stream's actual Harness and reward key/version. Every group of `n_samples` uses one
Stream. The default epoch interleaves all source rows without replacement; a positive
`arena_mixture_epoch_size` selects a weighted subset, must not exceed the union, and
must be divisible by the training batch size. Rejection can change the mixture that
actually reaches training.

Before submission, extract the Stream list and validate it without launching tasks:

```bash
python -m examples.swe.arena_stream_config_projection \
  examples/swe/arena_multi_stream.yaml | base64 --decode > /tmp/arena-streams.yaml
python -m examples.swe.arena_preflight \
  --streams-file /tmp/arena-streams.yaml --base-url "$ARENA_OPENAPI_BASE"
```

The preflight checks Stream/Harness state, reward version, dataset availability, and
recent task health. A new Stream with no history can use `--min-terminal-tasks 0`; this
does not bypass the other checks.

## Routing and rewards

`session_gateway` reuses a registration for each rollout worker and binds every request
to an individual proxy session. The registered key only permits generation; it cannot
end sessions, assign rewards, or export trajectories. Registrations are probed while
tasks are active and cleaned up when the worker is destroyed. `gateway` retains the
older per-rollout registration mode; `direct` requires private network access from the
Harness to the proxy.

Only the result envelope's top-level `score` is used as reward. Per-Stream thresholds
and optional `reward_transform_fn` hooks can transform it. Heterogeneous `raw` payloads
are retained for audit, not parsed for training rewards. Model-attributed failures can
retain interactions with zero reward; system and ambiguous failures are rejected. Result
shards are written under `arena_result_dump_dir` when configured.

## Port provenance and scope

This port applies selected changes from `swe-dev` commits `da1da65c7`, `1d0fc7ba0`,
`714f733a0`, `586d7cc06`, `84f7bb341`, `ddb7d7f3c`, `2b994c10c`, and `9ecbde97c`. It
preserves main's processor cache, sample identity, and cancellation cleanup. Training
engine APIs, mean-only reward normalization, internal Astra launch scripts, AWEX/Qwen3.8
changes, and PRM/RewardSystem are outside this port.
