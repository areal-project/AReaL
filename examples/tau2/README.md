# Customer Service Agent Training with Tau2 Benchmark

## Overview

This example demonstrates how to train customer service agents using the
[$\\tau^2$-Bench](https://github.com/sierra-research/tau2-bench) with AReaL's PPO/GRPO
training pipeline. The $\\tau^2$-Bench provides realistic customer service simulation
environments across multiple domains (retail, airline, telecom) where agents must help
with user's request by both using agent tools and guiding users using their tools.

## Code Architecture

- `train.py`: Training script that creates tau2 datasets and runs PPO training with the
  `Tau2AgentWorkflow`.
- `agent.py`: Implements `Tau2AgentWorkflow` which runs tau2 simulations. The
  implementation is completely independent from AReaL (except for logging, which you can
  replace with other logging tools). AReaL's proxy server will automatically connects to
  the workflow and runs it with self-hosted inference servers for RL training.
- `utils.py`: Common utilities including `Tau2EnvConfig`, `Tau2PPOConfig`, and
  `Tau2RunInfo` dataclasses. Also patches tau2's cost calculation to silently handle
  self-hosted models.

## Running the Example

### Prerequisites

Please make sure AReaL is setup and working following the
[installation guide](https://areal-project.github.io/AReaL/en/tutorial/installation.html).

1. Install the (forked) tau2-bench package:

```bash
pip install git+https://github.com/dhh1995/tau2-bench.git@dhh/async-and-custom-completion
```

Note that the training relies on the async version of the agent and user simulator in
the [tau2-bench package](https://github.com/dhh1995/tau-bench). These changes will be
merged into the original tau2-bench repository later.

1. Setup the `TAU2_DATA_DIR` environment variable:

```bash
export TAU2_DATA_DIR=/path/to/tau2-bench/data
```

For multi-node experiment with slurm, this can be set in the config file under
`actor.scheduling_spec[0].env_vars.TAU2_DATA_DIR`.

### Configuration Files

Four example configurations are provided:

| Config                         | Model           | Cluster           | Allocation                                             | Use Case                                |
| ------------------------------ | --------------- | ----------------- | ------------------------------------------------------ | --------------------------------------- |
| `config_1.7b_airline.yaml`     | Qwen3-1.7B      | 1 node, 8 GPUs    | `sglang:d6+archon:d2`                                  | Small-scale local training              |
| `config_8b_airline.yaml`       | Qwen3-8B        | 3 nodes, 24 GPUs  | `sglang:d16+archon:d8`                                 | Multi-node Slurm training               |
| `config_30b_moe_airline.yaml`  | Qwen3-30B-A3B   | 8 nodes, 64 GPUs  | `sglang:d8t4+megatron:(attn:d4p4t2\|ffn:d2p4e4)`       | Multi-node Slurm training for MOE model |
| `config_235b_moe_airline.yaml` | Qwen3-235B-A22B | 10 nodes, 80 GPUs | `sglang:d4t8+megatron:(attn:d1p12t4c1\|ffn:d1p12t1e4)` | Multi-node Slurm training for MOE model |

### Prepare User Simulator Server

You need to setup a user simulator server if using self-hosted LLMs. For example, using
[Qwen with SGLang](https://qwen.readthedocs.io/en/latest/deployment/sglang.html):

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-72B \
    --host 0.0.0.0 \
    --port 8000 \
    --tool-call-parser qwen25 \
    --chat-template ./qwen3_nonthinking.jinja \
    --dp-size 2 \
    --tp-size 4
```

Update the `econfig.user_llm_base_url` in your config to point to this server.

### Training Commands

NOTE: Following commands should be executed from root directory of this repository.

#### Single Node (1.7B Model)

On a single 8x GPU node with our offical image
(ghcr.io/areal-project/areal-runtime:latest), run:

```bash
python3 examples/tau2/train.py \
    --config examples/tau2/config_1.7b_airline.yaml \
    experiment_name=$experiment_name \
    trial_name=$trial_name \
    econfig.user_llm_base_url=http://localhost:8000/v1/ # your user LLM address
```

#### Multi-Node Slurm

On a SLURM cluster with at least 3 8x GPU nodes, directly run from a intermediate server
with AReaL and SLURM cli installed:

```bash
python3 examples/tau2/train.py \
    --config examples/tau2/config_8b_airline.yaml \
    experiment_name=$experiment_name \
    trial_name=$trial_name \
    cluster.fileroot=/path/to/shared/storage \
    cluster.name_resolve.nfs_record_root=/path/to/shared/storage/name_resolve \
    econfig.user_llm_base_url=http://localhost:8000/v1/ # your user LLM address
```

### Tau2 Related Configuration Options

| Option                           | Default   | Description                                                                     |
| -------------------------------- | --------- | ------------------------------------------------------------------------------- |
| `econfig.domain`                 | `telecom` | Tau2 domain: `airline`, `retail`, or `telecom`                                  |
| `econfig.max_steps`              | `100`     | Maximum number of steps per trajectory                                          |
| `econfig.add_thinking_tool`      | `false`   | Whether to use thinking as a tool for the agent                                 |
| `econfig.solo_mode`              | `false`   | If true, agent handles both agent and user roles (no user simulator needed)     |
| `econfig.user_llm_base_url`      | `null`    | Base URL of the user simulator LLM server                                       |
| `econfig.user_llm`               | `null`    | Model name for user simulator (e.g., `openai/self-hosted-Qwen2.5-72B`)          |
| `econfig.user_llm_args`          | `null`    | Arguments for user LLM (e.g., `{temperature: 0.0, max_completion_tokens: 512}`) |
| `econfig.turn_discount`          | `1.0`     | Discount factor for turn-based learning                                         |
| `econfig.invalid_format_penalty` | `0.1`     | Penalty for invalid format in completions                                       |

## Results

The following figure shows the training reward curves for the two configurations,
trained using the Archon engine:

<p align="left">
  <img src="reward.png" width="400">
</p>

- **Green line**: Qwen3-1.7B model (`config_1.7b_airline.yaml`)
- **Purple line**: Qwen3-8B model (`config_8b_airline.yaml`)

We also provide example configs for MoE models: `config_30b_moe_airline.yaml`
(Qwen3-30B-A3B) and `config_235b_moe_airline.yaml` (Qwen3-235B-A22B). The following
figure shows the training reward curve of Qwen3-30B-A3B model trained using the Megatron
engine:

<p align="left">
  <img src="reward_moe.png" width="400">
</p>

For reward curves of experiments on a larger scale, please refer to the
[AReaL Tau2 paper](https://arxiv.org/abs/2601.22607).

## Notes

1. **Trajectory logging**: Trajectories are dumped as `json` and `txt` files in the
   `generated/` directory under `cluster.fileroot`. You can analyze these for debugging
   and evaluation.

1. **Tree training**: The configs enable `enable_tree_training=true` by default, which
   optimizes training by sharing prefix computations across rollouts with the same
   prompt. This option can largely accelerate training but will possibly increase GPU
   memory usage if `actor.mb_spec.max_tokens_per_mb` is large. And this setting may
   cause instability during the training of the MoE model.

## Customization

We have released the training data and a trained model from this pipeline. You can use
the
[open-source Tau2 dataset](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data)
to reproduce results from the [AReaL Tau2 paper](https://arxiv.org/abs/2601.22607), or
directly download the resulting model
[AReaL-SEA-235B-A22B](https://huggingface.co/inclusionAI/AReaL-SEA-235B-A22B) trained
with this data and pipeline.
