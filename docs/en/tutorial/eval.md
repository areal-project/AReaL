# Evaluation

AReaL supports distributed inference using the same controller infrastructure as
training. This allows you to leverage existing workflows and schedulers to scale
evaluation across multiple GPUs and nodes.

**Note:** AReaL provides distributed inference for your trained model, not a complete
evaluation pipeline with dataset retrieval and metrics computation. You can use
third-party evaluation frameworks with AReaL checkpoints directly --- no conversion
required since AReaL saves HuggingFace-compatible checkpoints.

## Quick Start

Run evaluation on GSM8K:

```bash
python3 examples/math/gsm8k_eval.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    actor.path=/path/to/checkpoint
```

For distributed evaluation:

```bash
# With Ray (3 nodes, 12 GPUs)
python3 examples/math/gsm8k_eval.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=ray \
    rollout.backend=sglang:d12p1t1 \
    cluster.n_nodes=3

# With Slurm (12 nodes, 96 GPUs)
python3 examples/math/gsm8k_eval.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=slurm \
    rollout.backend=sglang:d96p1t1 \
    cluster.n_nodes=12
```

## Evaluation Metrics

Select an appropriate dataset and metrics for your task, then integrate the evaluation
logic as a workflow. See the [Agentic RL guide](./agentic_rl.md) for details.

Example with an agentic math evaluator (the evaluation code is independent with AReaL):

```python
from agents import Agent, OpenAIProvider, RunConfig, SQLiteSession, function_tool
from agents import Runner as OpenAIRunner
from math_verify import parse, verify
from openai import AsyncOpenAI


@function_tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@function_tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def math_reward_fn(completions: str, answer: str) -> float:
    return float(verify(parse(completions), parse(answer)))


class MathAgent:
    async def run(self, data, **extra_kwargs):
        http_client = extra_kwargs.get("http_client")
        base_url = extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0)

        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            model="default",
            tracing_disabled=True,
        )
        agent = Agent(
            name="RLVR Math with Calculator",
            instructions="Answer math questions using the calculator tools.",
            tools=[add, multiply],
        )
        result = await OpenAIRunner.run(
            agent,
            input=data["messages"][-1]["content"],
            session=SQLiteSession("math"),
            run_config=run_config,
        )
        return math_reward_fn(result.final_output, data["answer"])
```

## Architecture

Evaluation uses a single-controller architecture without training workers:

```
Controller Process
    │
    └─> Inference Engine Controller (SGLang/vLLM)
        ├─> Scheduler creates inference workers
        ├─> Submits evaluation tasks with workflow
        └─> Collects results and computes metrics
```

The controller orchestrates evaluation from a CPU process while inference workers run on
GPUs.

## Implementation

See
[`examples/math/gsm8k_eval.py`](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_eval.py)
for a complete example. The key pattern:

```python
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import GRPOConfig, SGLangConfig, load_expr_config, vLLMConfig
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.infra import LocalScheduler, RayScheduler, SlurmScheduler

# Load config and parse rollout backend
config, _ = load_expr_config(args, GRPOConfig)
rollout_alloc = ModelAllocation.from_str(config.rollout.backend)

# Initialize scheduler based on config
if config.scheduler.type == "local":
    scheduler = LocalScheduler(exp_config=config)
elif config.scheduler.type == "ray":
    scheduler = RayScheduler(exp_config=config)
elif config.scheduler.type == "slurm":
    scheduler = SlurmScheduler(exp_config=config)

# Select inference engine and build server args
if rollout_alloc.backend == "sglang":
    engine_cls = RemoteSGLangEngine
    server_args = SGLangConfig.build_args(
        sglang_config=config.sglang,
        tp_size=rollout_alloc.tp_size,
        base_gpu_id=0,
    )
elif rollout_alloc.backend == "vllm":
    engine_cls = RemotevLLMEngine
    server_args = vLLMConfig.build_args(
        vllm_config=config.vllm,
        tp_size=rollout_alloc.tp_size,
        pp_size=rollout_alloc.pp_size,
    )

# Create controller and initialize
eval_rollout = engine_cls.as_controller(config.rollout, scheduler)
eval_rollout.initialize(
    role="eval-rollout",
    rollout_alloc=rollout_alloc,
    server_args=server_args,
)

# Define workflow and its configuration
workflow = "areal.workflow.rlvr.RLVRWorkflow"
workflow_kwargs = dict(
    reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
    gconfig=config.gconfig,
    tokenizer=config.tokenizer_path,
    enable_thinking=False,
)

# Submit evaluation tasks
cnt = 0
for data in valid_dataloader:
    for item in data:
        eval_rollout.submit(
            item,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            group_size=config.gconfig.n_samples,
        )
        cnt += 1

# Wait for completion and collect results
eval_rollout.wait(cnt, timeout=None)
eval_stats = eval_rollout.export_stats()
```

This follows the same controller pattern as training but without training components.

## Configuration

Evaluation reuses the same config structure as training. You can use an existing
training config directly with the evaluation script.

```yaml
experiment_name: gsm8k-eval
trial_name: eval0
seed: 1

rollout:
  backend: "sglang:d4p1t1"  # Inference-only allocation
  max_concurrent_rollouts: 256

scheduler:
  type: local  # or 'ray', 'slurm'
  # max_head_offpolicyness is set to 1e12 internally for eval

gconfig:
  n_samples: 8
  temperature: 1.0
  max_new_tokens: 1024

actor:
  path: Qwen/Qwen2.5-1.5B-Instruct
  dtype: bfloat16
  scheduling_spec:
    - task_type: worker
      port_count: 2
      gpu: 1
      cmd: python3 -m areal.infra.rpc.rpc_server

valid_dataset:
  name: gsm8k
  split: test
  batch_size: 32
```

## Logging Results

Use `tabulate_stats` to format evaluation metrics:

```python
from areal.utils.printing import tabulate_stats

eval_stats = eval_rollout.export_stats()
logger.info(f"Evaluation Results: {tabulate_stats(eval_stats)}")
```

## Custom Workflows

Reuse training workflows or create custom ones. See the
[Agentic RL tutorial](../tutorial/agentic_rl.md) and
[Customization: Rollout Workflows](../customization/agent.md) for complete guides.

## Next Steps

- {ref}`Distributed Experiments <distributed-experiments-with-ray-or-slurm>`
- [Customization: Workflows](../customization/agent.md)
- [Agentic RL Tutorial](../tutorial/agentic_rl.md)
- [Large MoE Training](../tutorial/megatron.md)
