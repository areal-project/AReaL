# Arena single- and multi-Stream training

Arena supplies datasets, runs the external Harness, and returns terminal rewards. AReaL
serves the policy through its rollout proxy and trains on the recorded tokens. This
integration does not require AReaL-SWEAgent, PRM, or RewardSystem.

## Configuration

`arena_grpo.yaml` is a two-node FSDP/SGLang smoke profile: eight training GPUs, eight
inference GPUs, two samples per prompt, and one training step. Set these environment
variables to your deployment's values:

```bash
export AREAL_DIR=/path/to/AReaL
export AREAL_IMAGE=/path/to/compatible-image.sif
export AREAL_PYTHON=/path/to/python/in/container
export AREAL_FILEROOT=/shared/path/to/experiments
export MODEL_PATH=/shared/path/to/Qwen3-4B-Instruct
export ARENA_CREDENTIALS_FILE=/shared/private/arena.env
```

The private credentials file must export `ARENA_OPENAPI_BASE`, `ARENA_OPENAPI_TOKEN`,
`ARENA_LLM_API_KEY`, and `SWE_RL_ADMIN_API_KEY`. Load it in the controller environment
as well. Workers load the same file; credentials are not embedded in worker commands or
committed YAML. The example assumes a compatible container with Slurm available to the
controller, shared storage, and network reachability from Arena to rollout proxies. Use
your site's established Slurm controller submission wrapper to run:

```bash
python -m examples.swe.train_swe_rl --config examples/swe/arena_grpo.yaml
```

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
