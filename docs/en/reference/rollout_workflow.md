# RolloutWorkflow Reference

This document describes the `RolloutWorkflow` abstraction, the core interface for
implementing rollout generation in AReaL's reinforcement learning pipeline.

**Notes**:

1. This page targets developers seeking a deep understanding of the codebase. For
   agentic RL training, use the high-level API described in the
   [Agentic RL Guide](../tutorial/agentic_rl.md).

1. **Legacy pattern**: Directly subclassing `RolloutWorkflow` is considered legacy and
   should not be used proactively. For new agentic RL workflows, use the
   [agent workflow pattern](./agent_workflow.md) with `async def run()` instead.

## Overview

A `RolloutWorkflow` defines how to generate training trajectories from input data. It
encapsulates the logic for:

- Tokenizing prompts and preparing model inputs
- Calling the inference engine to generate completions
- Computing rewards for generated outputs
- Packaging results into tensor dictionaries for training

## Interface

```python
from areal.api.workflow_api import RolloutWorkflow

class RolloutWorkflow(ABC):
    @abstractmethod
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None | dict[str, InteractionWithTokenLogpReward]:
        """Run a single episode of the workflow."""
        ...
```

### Parameters

| Parameter | Type              | Description                                     |
| --------- | ----------------- | ----------------------------------------------- |
| `engine`  | `InferenceEngine` | Inference engine for generating model responses |
| `data`    | `dict[str, Any]`  | A single sample from the dataloader             |

### Return Types

The `arun_episode` method supports three return types:

| Return Type                                 | Description                                                                                        |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `dict[str, torch.Tensor]`                   | Standard tensor format for training                                                                |
| `dict[str, InteractionWithTokenLogpReward]` | Token-level interactions (auto-converted to tensors); produced by the high-level `ArealOpenAI` API |
| `None`                                      | Rejected trajectory, excluded from training                                                        |

## Tensor Dictionary Format

When returning a tensor dictionary, the following fields are expected:

| Field            | Shape                   | Type    | Required | Description                         |
| ---------------- | ----------------------- | ------- | -------- | ----------------------------------- |
| `input_ids`      | `[batch_size, seq_len]` | int32   | Yes      | Token IDs (prompt + completion)     |
| `attention_mask` | `[batch_size, seq_len]` | bool    | Yes      | Valid token mask                    |
| `loss_mask`      | `[batch_size, seq_len]` | int32   | No       | Completion token mask (1 = train)   |
| `logprobs`       | `[batch_size, seq_len]` | float32 | No       | Log probabilities per token         |
| `rewards`        | `[batch_size]`          | float32 | No       | Per-sequence rewards                |
| `versions`       | `[batch_size, seq_len]` | int32   | No       | Weight version when token generated |

Example return value:

```python
return {
    "input_ids": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32),
    "attention_mask": torch.ones(1, 5, dtype=torch.bool),
    "loss_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
    "logprobs": torch.tensor([[0.0, 0.0, -0.5, -0.3, -0.2]], dtype=torch.float32),
    "rewards": torch.tensor([1.0], dtype=torch.float32),
    "versions": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
}
```

## Workflow Context

Inside `arun_episode`, access the execution context via the `workflow_context` module.
Each workflow instance has its own isolated context:

```python
from areal.infra import workflow_context

async def arun_episode(self, engine, data):
    # Get current execution context
    ctx = workflow_context.get()

    # Check if running in evaluation mode
    if ctx.is_eval:
        # Use different parameters for evaluation
        ...

    # Get task ID for logging
    task_id = ctx.task_id

    # Get stats scope based on mode ("rollout" or "eval-rollout")
    scope = workflow_context.stat_scope()
```

## Trajectory Dumping

When `InferenceEngineConfig.dump_to_file=True`, trajectories are automatically saved to
disk for debugging and analysis.

### Configuration

```yaml
rollout:
  dump_to_file: true
  fileroot: "/path/to/logs"
  tokenizer_path: "model/tokenizer" # Required for text decoding
```

### Output Location

Trajectories are saved to:

```
{fileroot}/{experiment_name}/{trial_name}/[rollout|eval-rollout]/{version}/{task_id}.jsonl
```

Example:

```
/tmp/areal/my_exp/trial1/rollout/5/42.jsonl
```

### Output Format

Each line in the JSONL file contains:

```json
{
  "task_id": 42,
  "sample_idx": 0,
  "seqlen": 256,
  "prompt_len": 128,
  "head_version": 5,
  "tail_version": 5,
  "reward": 1.0,
  "prompt": "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
  "completion": "The answer is 4.<|im_end|>"
}
```

## Grouped Rollout

Grouped rollout runs the same workflow multiple times per input prompt, producing
diverse completions for training. This is useful for algorithms like GRPO that benefit
from multiple samples per prompt.

### Configuration

Set `group_size` when submitting rollouts:

```python
engine.submit(
    data=sample,
    workflow=MyWorkflow,
    workflow_kwargs={...},
    group_size=4,  # Run workflow 4 times per input
)
```

Or via CLI:

```yaml
rollout:
  group_size: 4
```

### How It Works

When `group_size > 1`, the workflow is wrapped in `GroupedRolloutWorkflow`:

1. The wrapper runs `arun_episode` concurrently `group_size` times using
   `asyncio.gather`
1. Results are merged based on their type:
   - **Tensor dictionaries**: Concatenated along the batch dimension
   - **InteractionWithTokenLogpReward dicts**: Merged into a single dictionary
1. If some runs return `None` (rejected), only valid results are kept
1. If all runs return `None`, the entire grouped result is `None`

### Output Shape

With `group_size=4` and a workflow returning `[1, seq_len]` tensors, the grouped output
has shape `[4, seq_len]` (4 samples concatenated).

### Implementation

From `areal/infra/remote_inf_engine.py`:

```python
class GroupedRolloutWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine, data):
        # Run N times concurrently
        results = await asyncio.gather(
            *[self.workflow.arun_episode(engine, data)
              for _ in range(self.group_size)]
        )

        # Filter None results
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return None

        # Merge based on result type
        if all_interaction_dicts(valid_results):
            return merge_dicts(valid_results)
        else:
            return concat_padded_tensors(valid_results)
```

## Implementing Custom Workflows

To create a custom workflow:

1. **Subclass `RolloutWorkflow`**:

```python
from areal.api.workflow_api import RolloutWorkflow

class MyWorkflow(RolloutWorkflow):
    def __init__(self, tokenizer, gconfig, **kwargs):
        self.tokenizer = tokenizer
        self.gconfig = gconfig

    async def arun_episode(self, engine, data):
        # 1. Prepare input
        input_ids = self.tokenizer.encode(data["prompt"])

        # 2. Generate completion
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig,
            tokenizer=self.tokenizer,
        )
        resp = await engine.agenerate(req)

        # 3. Compute reward
        reward = self.compute_reward(resp, data)

        # 4. Return tensor dict (or None to reject)
        if reward < 0:
            return None

        return self.build_tensor_dict(resp, reward)
```

2. **Register with trainer**:

```python
trainer.train(
    workflow=MyWorkflow,
    workflow_kwargs={
        "tokenizer": tokenizer,
        "gconfig": config.gconfig,
    },
)
```

## Workflow Resolution

Workflows can be specified in multiple ways:

| Format         | Example                          | Description                |
| -------------- | -------------------------------- | -------------------------- |
| Instance       | `MyWorkflow(...)`                | Pre-instantiated workflow  |
| Class          | `MyWorkflow`                     | Class (requires kwargs)    |
| String path    | `"my_module.MyWorkflow"`         | Dynamic import             |
| Agent workflow | Any class with `async def run()` | Wrapped with proxy support |

The training system automatically resolves these to `RolloutWorkflow` instances.

## See Also

- [Agentic RL Tutorial](../tutorial/agentic_rl.md) - Training with agent frameworks
- [Adding Custom Workflows](../customization/agent.md) - Step-by-step guide
