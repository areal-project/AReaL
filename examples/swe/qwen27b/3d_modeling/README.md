# Qwen27 Blender 3D modeling on Ray

This baseline combines the BS64/GS12 subtask training recipe with a Blender workload:
four upstream datasets, each executed through a no-skill harness and a self-implement
harness. It uses the existing [SWE-MM Ray driver](../README.md) and colocated actor and
rollout GPUs. It requires an existing Ray cluster and native role images.

## Training contract

The training settings follow the BS64/GS12 base recipe, rather than the Blender run's
BS32 settings. In particular:

| Setting                                      | Value                                  |
| -------------------------------------------- | -------------------------------------- |
| Prompt groups per batch / samples per group  | 64 / 12                                |
| Train steps / epoch limit                    | 1000 / 1000                            |
| Temperature / top-p / top-k                  | 1.0 / 1.0 / 100000                     |
| Generated tokens / context                   | 65536 / 262144                         |
| Training reward normalization                | Center groups; do not divide by std    |
| Minimum usable group size                    | 8                                      |
| Actor / rollout backend                      | `megatron:d1p4t4c4e1` / `sglang:d16t4` |
| Grouped rollout concurrency / consumer batch | 16 / 64                                |
| SGLang max running requests per instance     | 1000                                   |
| Learning rate / warmup steps                 | 3e-6 / 2                               |
| Weight synchronization                       | AWEX, colocated                        |
| Recovery / proxy PRM                         | Disabled / disabled                    |

Optimizer, loss, filtering, evaluation generation settings, cache settings, timeouts,
and vision freezing are retained from the base recipe. Evaluation generation has
`reward_normalization=false`; its stored `reward_normalization_use_std=true` is also
retained. W&B is enabled and requires the external site's settings. This is a fresh
training baseline; use a distinct trial/run directory. To resume, explicitly configure
recovery from the start of that experiment; `--resume` alone does not enable it.

All eight stream entries have `sampling_weight=1`, as in the Blender recipe. This
retains native dataset mixture semantics; it does not impose equal counts in each
training batch. Leave `QWEN_EQUAL_STREAM_SAMPLING` unset to retain that schedule. Each
upstream dataset shares one frozen per-task reward mapping across its two harness
variants. Keep both harness versions and reward key/version mappings pinned. Shared
upstream IDs require distinct explicit harnesses and unique entry names; duplicate
entries with the same harness or an inherited harness default are rejected.

The frozen run's name included “JudgeStep20Start”, but its AReaL config disables proxy
PRM and its model entry contains no local step-20 judge switch. This example does not
invent such a switch. Any judge scheduling inside the externally hosted harness/reward
implementation must be verified with that service; the run name alone is not evidence.

## Site bindings and submission

The example includes a [portable Ray profile](ray.env.example),
[stream bindings template](bindings.env.example), and
[credentials template](credentials.env.example). Create the deployment directory on the
destination machine; no pre-existing `/shared/private` path is required:

```bash
recipe="$PWD/examples/swe/qwen27b/3d_modeling"
mkdir -p ../qwen27-blender-site/reward-mappings
cp "$recipe/ray.env.example" ../qwen27-blender-site/ray.env
cp "$recipe/bindings.env.example" ../qwen27-blender-site/bindings.env
cp "$recipe/credentials.env.example" ../qwen27-blender-site/credentials.env
chmod 600 ../qwen27-blender-site/*.env
```

Set the destination model path, role images/interpreters, and a unique trial in
`ray.env`. Supply four stream IDs and two full harness versions in `bindings.env`, and
copy the four frozen reward mappings to `reward-mappings/stream-{1,2,3,4}.json`. Fill
`credentials.env`, or make it source the destination site's existing private
environment. A prepared private deployment bundle can replace these copy steps; it must
include the same files and reward mappings. Do not publish that private bundle.

The entire deployment directory can be moved to another machine: mappings, credentials,
and default output paths resolve relative to `ray.env`/`bindings.env`. Only model paths,
images, and service access need destination-specific configuration. The upstream Arena
service must still provide the pinned streams, harnesses, and reward versions; copying
JSON mappings does not copy the remote datasets or services.

Within the destination Ray cluster, make the checkout, deployment directory, model, and
output paths visible at the same absolute paths on every node and in both role images
(shared mounts or identical mounted copies). A path on the submitting laptop is not
automatically visible to Ray workers. See the [Ray setup](../README.md).

```bash
export QWEN_ENV_FILE="$(cd ../qwen27-blender-site && pwd)/ray.env"
bash examples/swe/qwen27b/3d_modeling/ray.sh --prepare-only
bash examples/swe/qwen27b/3d_modeling/ray.sh
```

`--prepare-only` writes the Ray configuration without contacting Ray or Arena; it does
not validate remote harness availability or run training. For Ray Jobs, use the
dashboard address and a fixed shared checkout:

```bash
export QWEN_REPO="$PWD"
runtime_env=$(python3 -c 'import json, os; print(json.dumps({"env_vars": {"QWEN_ENV_FILE": os.environ["QWEN_ENV_FILE"]}}))')
ray job submit --address "$RAY_DASHBOARD_ADDRESS" --no-wait \
  --runtime-env-json "$runtime_env" \
  -- bash "$QWEN_REPO/examples/swe/qwen27b/3d_modeling/ray.sh"
```

The default parallel layout needs 64 shared GPUs (eight 8-GPU nodes). Node count is
inferred from the two backends and `cluster.n_gpus_per_node`; there is no node list or
fixed node count. For another scale, override both backends and the physical GPU count
per node, and verify model memory capacity. For example, the following prepares a 32-GPU
layout; it has not been validated for this model's full context length:

```bash
bash examples/swe/qwen27b/3d_modeling/ray.sh --prepare-only \
  --set actor.backend=megatron:d1p4t4c2e1 \
  --set rollout.backend=sglang:d8t4 \
  --set cluster.n_gpus_per_node=8
```

Use a new trial/run directory when changing settings. `--config` as the first argument
selects a different complete YAML; repeated `--set KEY=VALUE` overrides work as with the
shared driver. Training batch composition remains completion-ordered, with the current
branch's native stream rejection compensation.
