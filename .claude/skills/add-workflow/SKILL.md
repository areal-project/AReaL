---
name: add-workflow
description: Guide for adding a new RolloutWorkflow to AReaL. Use when user wants to create a new workflow.
---

# Add Workflow

Add a new RolloutWorkflow implementation to AReaL.

## When to Use

This skill is triggered when:

- User asks "how do I add a workflow?"
- User wants to create a new RolloutWorkflow
- User mentions implementing a custom rollout

## Prerequisites

Before starting, ensure you understand:

- The workflow's purpose and requirements
- Input/output data format
- Reward function to use

## Step-by-Step Guide

### Step 1: Create Workflow File

Create `areal/workflow/<name>.py`:

```python
import uuid
from typing import Any, Callable

import torch

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging

logger = logging.getLogger("MyWorkflow")


class MyWorkflow(RolloutWorkflow):
    """Description of your workflow."""

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer,
        reward_fn: Callable,
    ):
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Run a single episode. MUST be async and non-blocking."""

        # 1. Prepare input_ids from data
        input_ids = self.tokenizer.apply_chat_template(
            data["messages"],
            tokenize=True,
            add_generation_prompt=True,
        )

        # 2. Build ModelRequest
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=list(input_ids),
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )

        # 3. Generate completion (async)
        resp: ModelResponse = await engine.agenerate(req)

        # 4. Compute reward (async)
        prompt_str = self.tokenizer.decode(input_ids)
        completion_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt_str,
            completion_str,
            resp.input_tokens,
            resp.output_tokens,
            **data,
        )

        # 5. Return results in expected format
        return {
            "input_ids": torch.tensor(resp.input_tokens),
            "output_ids": torch.tensor(resp.output_tokens),
            "reward": torch.tensor(reward),
        }
```

### Step 2: Register in __init__.py

Add to `areal/workflow/__init__.py`:

```python
from areal.workflow.<name> import MyWorkflow

__all__ = [
    # ... existing exports
    "MyWorkflow",
]
```

### Step 3: Update Entry Script

Update your training script to use the new workflow:

```python
trainer.train(
    workflow="areal.workflow.<name>.MyWorkflow",
    # ... other args
)
```

### Step 4: Add Tests

Create `tests/test_<name>_workflow.py`:

```python
import pytest
from areal.workflow.<name> import MyWorkflow

@pytest.mark.asyncio
async def test_workflow_basic():
    # Test basic functionality
    pass
```

## Reference Implementations

| Workflow           | File                            | Description                |
| ------------------ | ------------------------------- | -------------------------- |
| MultiTurnWorkflow  | `areal/workflow/multi_turn.py`  | Multi-turn conversation    |
| RLVRWorkflow       | `areal/workflow/rlvr.py`        | RL with verifiable rewards |
| VisionRLVRWorkflow | `areal/workflow/vision_rlvr.py` | Vision + RLVR              |

## Key Requirements

1. **Async**: `arun_episode` must be `async def` and non-blocking
1. **No sync I/O**: Use `aiofiles` for file operations
1. **Wrap rewards**: Use `AsyncRewardWrapper` for reward functions
1. **Tensor format**: Output tensors should be `[batch, seq_len, ...]`
1. **Use helpers**: `concat_padded_tensors` for combining outputs

## Common Mistakes

- ❌ Using `open()` instead of `aiofiles.open()`
- ❌ Forgetting to `await` async calls
- ❌ Not wrapping reward function with `AsyncRewardWrapper`
- ❌ Wrong tensor shape conventions

______________________________________________________________________

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .claude/skills/add-workflow/SKILL.md
Invocation: /add-workflow <name>

## Purpose

Step-by-step guide for adding new RolloutWorkflow implementations.

## How to Update

### When Workflow API Changes
1. Update the code template in Step 1
2. Update the required imports
3. Update the method signature if changed

### When New Patterns Emerge
1. Add to "Reference Implementations" table
2. Update "Key Requirements" if new requirements added

================================================================================
-->
