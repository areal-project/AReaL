# Quickstart

Welcome to the **AReaL** Quickstart Guide! This guide demonstrates how to run an AReaL
experiment training an LLM on the GSM8K dataset using the GRPO algorithm with
function-based rewards. Ensure you've completed
[the installation and environment setup](installation.md) before proceeding.

## Running the Experiment (on a single node)

To run the experiment, you will need:

- Training script:
  [examples/math/gsm8k_rl.py](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_rl.py)
- Config YAML:
  [examples/math/gsm8k_grpo.yaml](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_grpo.yaml)

Our training scripts will automatically download the dataset (openai/gsm8k) and model
(Qwen/Qwen2-1.5B-Instruct). To run the example with default configuration, execute from
the repository directory:

```
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml scheduler.type=local experiment_name=<your experiment name> trial_name=<your trial name>
```

> **Note**: For distributed experiments across multiple nodes, see
> [Distributed Experiments with Ray or Slurm](#distributed-experiments-with-ray-or-slurm).

## Modifying configuration

All available configuration options are listed in
[areal/api/cli_args.py](https://github.com/areal-project/AReaL/blob/main/areal/api/cli_args.py).
To customize the experiment (models, resources, algorithm options), you can:

1. Edit the YAML file directly at
   [examples/math/gsm8k_grpo.yaml](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_grpo.yaml).
1. Add command-line options:
   - For existing options in the YAML file, directly add the option:
     `actor.path=Qwen/Qwen3-1.7B`.
   - For other options in `cli_args.py`, but not in the YAML file, add with a prefix
     "+": `+sglang.attention_backend=triton`.

For example, here is the command to launch a customized configuration, based on our
GSM8K GRPO example:

```
python3 examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    rollout.backend=sglang:d2p1t1 actor.backend=fsdp:d2p1t1 \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node=4 \
    gconfig.max_new_tokens=2048 \
    train_dataset.batch_size=1024 \
    +sglang.attention_backend=triton
```

To enable [Hugging Face Kernels](https://github.com/huggingface/kernels) in training,
add the train engine overrides explicitly:

```bash
python3 examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    +actor.attn_impl=kernels-community/flash-attn \
    +actor.use_kernels=true
```

Apply the same overrides to `critic` or `teacher` if those engines should also use
kernels.

(distributed-experiments-with-ray-or-slurm)=

## Distributed Experiments with Ray or Slurm

For distributed experiments across multiple nodes, you can use Ray or Slurm schedulers.
After setting up your Ray or Slurm cluster, launch experiments by specifying the
appropriate scheduler type:

```
# Launch with Ray scheduler. 4 nodes (4 GPUs each), 3 nodes for generation, 1 node for training.
python3 examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=ray \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    rollout.backend=sglang:d12p1t1 actor.backend=fsdp:d4p1t1 \
    cluster.n_nodes=4 \
    cluster.n_gpus_per_node=4

# Launch with Slurm scheduler. 16 nodes (8 GPUs each), 12 nodes for generation, 4 nodes for training
python3 examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=slurm \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    rollout.backend=sglang:d96p1t1 actor.backend=fsdp:d32p1t1 \
    cluster.n_nodes=16 \
    cluster.n_gpus_per_node=8
```

Additional references:

- For more options for schedulers, check `SchedulerConfig` in
  [areal/api/cli_args.py](https://github.com/areal-project/AReaL/blob/main/areal/api/cli_args.py).
- Ray cluster setup guide (see installation.md for distributed setup) for a guide on how
  to set up a ray cluster.

> **Important Note**: Ensure the total GPUs across `rollout.backend` and `actor.backend`
> matches your cluster configuration
> (`#GPUs == cluster.n_nodes * cluster.n_gpus_per_node`)

<!--
> **Notes**: Before launching distributed experiments, please check if your per-engine `backend` fields match your cluster configuration. Make sure the total GPUs allocated by `rollout.backend` and `actor.backend` equals `cluster.n_nodes * cluster.n_gpus_per_node`.
> **Note**: Ray and Slurm launchers only work for distributed experiments with more than 1 node (`cluster.n_nodes > 1`). They allocate GPUs for training and generation at the granularity of **nodes**, which means the number of GPUs allocated for generation and training must be integer multiples of `cluster.n_gpus_per_node`.
-->

## Legacy: SPMD Mode with Dedicated Launchers

AReaL also supports SPMD (Single Program Multiple Data) mode via dedicated launchers.
This mode is maintained for backwards compatibility but the single-controller mode
(direct script execution with `scheduler.type`) is now the recommended approach for most
use cases.

In SPMD mode, the launcher manages process spawning via `torchrun` and sets
`AREAL_SPMD_MODE=1`. Each GPU worker runs the full training script independently, with
coordination handled through PyTorch distributed primitives.

```bash
# SPMD mode with local launcher (legacy)
python3 -m areal.infra.launcher.local examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml

# SPMD mode with Ray launcher (legacy)
python3 -m areal.infra.launcher.ray examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml

# SPMD mode with Slurm launcher (legacy)
python3 -m areal.infra.launcher.slurm examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml
```

## Distributed Experiments on Cloud or K8s with SkyPilot

If you want to directly run an experiment on cloud or your own Kubernetes
infrastructure, we recommend you to use SkyPilot. After installing and setting up
SkyPilot (see {ref}`Install SkyPilot <install-skypilot>`), you could launch a
distributed experiment based on our SkyPilot example (two 8xA100 GPU nodes) with one
command line:

```bash
# Launch on GCP
sky launch -c areal-test examples/skypilot/ray_cluster.sky.yaml --infra gcp
# Launch on AWS
sky launch -c areal-test examples/skypilot/ray_cluster.sky.yaml --infra aws
# Launch on your K8s Cluster
sky launch -c areal-test examples/skypilot/ray_cluster.sky.yaml --infra k8s
```

Check
[Running AReaL with SkyPilot](https://github.com/areal-project/AReaL/blob/main/examples/skypilot/README.md),
for more details about the examples. Check
[SkyPilot Documentation](https://docs.skypilot.co/en/latest/docs/index.html) for more
information about SkyPilot.

## Next Steps

Check [Getting Started with AReaL](gsm8k_grpo.md) for a complete code walkthrough on the
GRPO GSM8K Example.

Customization guides:

- [Custom dataset](../customization/dataset.md)
- [Custom agentic/RVLR rollout workflows](../customization/agent.md)
