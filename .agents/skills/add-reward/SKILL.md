---
name: add-reward
description: Guide for adding a new reward function to AReaL. Use when user wants to create a reward function.
---

# Add Reward

Add a new reward function to AReaL.

## When to Use

This skill is triggered when:

- User asks "how do I add a reward function?"
- User wants to implement custom rewards
- User mentions reward computation

## Step-by-Step Guide

### Step 1: Create Reward File

Create `areal/reward/<name>.py`:

```python
from typing import Any

from areal.utils import logging

logger = logging.getLogger("MyReward")


def <name>_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids,
    completion_ids,
    answer: str | None = None,
    **kwargs: Any,
) -> float:
    """Compute reward for a single completion.

    Args:
        prompt: Prompt string
        completions: Completion string (model output)
        prompt_ids: Tokenized prompt IDs
        completion_ids: Tokenized completion IDs
        answer: Ground truth answer from dataset (optional)
        **kwargs: Additional data from dataset

    Returns:
        Reward value (float), typically 0.0 or 1.0
    """
    try:
        # Extract answer from completion
        extracted = _extract_answer(completions)

        # Compare with ground truth
        if answer is not None and extracted == str(answer):
            return 1.0
        return 0.0
    except Exception:
        logger.warning("Exception in reward computation", exc_info=True)
        return 0.0


def _extract_answer(completion: str) -> str:
    """Extract the answer from a completion string.

    Implement your extraction logic here.
    """
    # Example: Extract content from \boxed{}
    import re

    match = re.search(r"\\boxed\{([^}]+)\}", completion)
    if match:
        return match.group(1).strip()
    return completion.strip()
```

### Step 2: Register in __init__.py

Update `areal/reward/__init__.py`:

```python
# Add to VALID_REWARD_FN
VALID_REWARD_FN = [
    # ... existing reward functions
    "<name>",
]

# Add to get_reward_fn function
def get_reward_fn(name: str, **kwargs):
    # ... existing code
    elif name == "<name>":
        from areal.reward.<name> import <name>_reward_fn
        return <name>_reward_fn
```

### Step 3: Handle Blocking Operations

If your reward function uses blocking operations (e.g., API calls, model inference), the
workflow will wrap it with `AsyncRewardWrapper`:

```python
# In your workflow
from areal.reward import AsyncRewardWrapper

self.reward_fn = AsyncRewardWrapper(reward_fn)

# Then call it asynchronously
rewards = await self.reward_fn(prompt, completions, **data)
```

### Step 4: Add Tests

Create `tests/test_<name>_reward.py`:

```python
import pytest
from areal.reward.<name> import <name>_reward_fn

def test_reward_correct_answer():
    reward = <name>_reward_fn(
        prompt="What is 2+2?",
        completions="The answer is \\boxed{4}",
        prompt_ids=None,
        completion_ids=None,
        answer="4",
    )
    assert reward == 1.0

def test_reward_wrong_answer():
    reward = <name>_reward_fn(
        prompt="What is 2+2?",
        completions="The answer is \\boxed{5}",
        prompt_ids=None,
        completion_ids=None,
        answer="4",
    )
    assert reward == 0.0
```

## Reference Implementations

| Reward     | File                              | Description                  |
| ---------- | --------------------------------- | ---------------------------- |
| GSM8K      | `areal/reward/gsm8k.py`           | Math answer verification     |
| Geometry3K | `areal/reward/geometry3k.py`      | Geometry answer verification |
| CLEVR      | `areal/reward/clevr_count_70k.py` | Counting verification        |
| MathVerify | `areal/reward/math_verify.py`     | General math verification    |

## Function Signature

All reward functions must follow this signature:

```python
def reward_fn(
    prompt: str,               # Input prompt string
    completions: str,          # Model completion string
    prompt_ids,                # Tokenized prompt
    completion_ids,            # Tokenized completion
    **kwargs: Any,             # Additional data from dataset (e.g., answer)
) -> float:                    # Reward value (typically 0.0 or 1.0)
```

**Note**: The reward function is called once per sample. Batching is handled by
`AsyncRewardWrapper` in the workflow.

## Key Requirements

1. **Deterministic**: Same inputs should produce same outputs
1. **Return float**: Output is a single float value per sample
1. **No blocking in async context**: Use `AsyncRewardWrapper` if needed
1. **Logging**: Use `areal.utils.logging`, not `print`
1. **Handle exceptions**: Return 0.0 on error, don't raise

## Common Mistakes

- Returning a tensor instead of a float
- Expecting batched inputs (reward is called per sample)
- Non-deterministic behavior
- Blocking operations without `AsyncRewardWrapper`
- Raising exceptions instead of returning 0.0
