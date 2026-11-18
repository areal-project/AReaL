# Qwen27 Ray tasks

These drivers reuse the Qwen27 role wrappers, native Ray role images, CPU optimizer
offload, and colocated AWEX updates. They connect to an existing Ray cluster. They do
not start Slurm jobs or create a cluster. See the
[Ray runtime setup](../rl/qwen38_27b/README.md#ray) for image, Podman, shared mount,
GPU, and network requirements.

## Private site profile

Copy [ray.env.example](../rl/qwen38_27b/ray.env.example) outside the checkout. Set
`QWEN_MODEL`, `QWEN_OUTPUT_ROOT`, `QWEN_EXPERIMENT_NAME`, `QWEN_ARENA_TRIAL`,
`QWEN_RUN_DIR`, the role images/interpreters, and `RAY_ADDRESS` there. The historical
name `QWEN_ARENA_TRIAL` also identifies GSM8K runs. Each task needs a distinct trial and
run directory. Keep the profile, fixed checkout, model, data, and reward mappings at the
same absolute paths on every node and inside every role image.

SWE-MM profiles source the existing private Arena environment. Leave the four
single-stream overrides empty when using a multi-stream configuration. W&B credentials
are needed only when the configuration enables an authenticated W&B service.

For GSM8K, set `QWEN_GSM8K_PATH` to a complete local Hugging Face GSM8K snapshot,
including its `README.md` metadata defining the `main` configuration. It must be
loadable with `datasets.load_dataset(path=..., name="main", split="train")`. No Arena
credentials are required. The bundled preset disables W&B. The wrappers run offline; a
cached dataset ID also works if `HF_HOME` points to a populated shared cache. Do not
rely on the default per-process cache. `QWEN_GSM8K_ENABLE_THINKING=false` is the
default; set it to `true` in the profile to enable thinking.

## SWE-MM

Pass the original subtask YAML explicitly. Stream weights, reward mappings, sampling,
batch size, and parallelism remain as specified in that file. Relative environment
references using `QWEN_RECIPE_DIR` resolve to its directory in every worker.

```bash
export QWEN_ENV_FILE=/shared/private/qwen27-swemm-ray.env
bash examples/swe/qwen27b/ray_swemm.sh \
  --config /shared/experiments/subtask/train.yaml --prepare-only

# Run from the controller environment inside the Ray cluster.
bash examples/swe/qwen27b/ray_swemm.sh \
  --config /shared/experiments/subtask/train.yaml
```

The same entry supports text-only SWE streams; the stream's harness and task definitions
select the workload. No private stream IDs or reward mappings are bundled here.

## GSM8K

[gsm8k.yaml](gsm8k.yaml) uses batch size 8, 8 samples per question, 25 updates, 4096
generated tokens, and an 8192-token context. It uses the native single-turn
`RLVRWorkflow` and `gsm8k_reward_fn`; it does not call Arena or an external LLM gateway.
Reward normalization is per group in the actor. Vision modules stay frozen. This is a
separate short GSM8K preset, not an assertion of numerical equivalence to an Arena run.
Evaluation and HF export are disabled by default; optional recovery remains enabled.

```bash
export QWEN_ENV_FILE=/shared/private/qwen27-gsm8k-ray.env
bash examples/swe/qwen27b/ray_gsm8k.sh --prepare-only
bash examples/swe/qwen27b/ray_gsm8k.sh
```

The example layout is actor `d2p2t4c1e1` and rollout `d4t4` (16 shared GPUs), packed
into the configured `cluster.n_gpus_per_node` (8 by default). Node count is inferred.
CPU/memory requests are per actor rank (4 CPUs and 64 GiB in the preset); size these for
the cluster and CPU optimizer. A different layout or dataset can be supplied with
`--config /shared/custom.yaml`. YAML must be complete, without Hydra `defaults`.

Both drivers accept repeatable `--set KEY=VALUE` overrides before writing the generated
config. For example, a 32-GPU GSM8K layout on 8-GPU nodes:

```bash
bash examples/swe/qwen27b/ray_gsm8k.sh \
  --set actor.backend=megatron:d4p2t4c1e1 \
  --set rollout.backend=sglang:d8t4 \
  --set cluster.n_gpus_per_node=8 \
  --set total_train_steps=50
```

Use a new trial/run directory for a changed config. With a custom YAML containing
`allocation_mode`, update it to match the new backends, or explicitly remove the legacy
metadata using `--set allocation_mode=null`. Update a supplied `cluster.n_nodes` limit
as needed. Each inference instance must fit on one node; configured GPUs per node must
match the physical Ray workers when using multiple nodes. GPU capacity validation does
not guarantee that a new layout will fit the model's GPU/host memory.

## Submit through Ray Jobs

Run the Ray CLI from an environment that can reach the dashboard. Set
`RAY_DASHBOARD_ADDRESS` to its actual HTTP address; this is distinct from the driver's
`RAY_ADDRESS` (normally `auto` inside a Ray Job). Set `QWEN_REPO` to the fixed, shared
checkout path visible to the Ray head and workers. The controller environment must
already include the training dependencies; role image settings affect workers only.

```bash
ray job submit --address "$RAY_DASHBOARD_ADDRESS" --no-wait \
  --runtime-env-json '{"env_vars":{"QWEN_ENV_FILE":"/shared/private/qwen27-swemm-ray.env"}}' \
  -- bash "$QWEN_REPO/examples/swe/qwen27b/ray_swemm.sh" \
  --config /shared/experiments/subtask/train.yaml

ray job submit --address "$RAY_DASHBOARD_ADDRESS" --no-wait \
  --runtime-env-json '{"env_vars":{"QWEN_ENV_FILE":"/shared/private/qwen27-gsm8k-ray.env"}}' \
  -- bash "$QWEN_REPO/examples/swe/qwen27b/ray_gsm8k.sh"
```

Replace the illustrative shared paths with the actual profile and configuration paths.
Prepare each config first from the same controller environment, or submit a Ray Job with
`--prepare-only` before the training job. This writes `$QWEN_RUN_DIR/train-ray.yaml`
without allocating training workers. It does not validate images, Arena reachability,
model initialization, or PPO updates. The two training jobs share the pool; submit them
sequentially if there is capacity for only one.

Use `ray job status --address "$RAY_DASHBOARD_ADDRESS" <job-id>` and `ray job logs` with
the returned job ID. Validate generation, reward, an optimizer update, and generation
with updated weights. To resume, stop the old job and workers, retain the same config,
profile/trial and storage, then pass `--resume`; a missing checkpoint is an error.
