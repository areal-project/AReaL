# Reward

This guide explains how to customize reward functions for reinforcement learning in
AReaL. AReaL supports **rule-based reward functions** and **generative (LLM-as-judge)
reward models**.

## Reward Function Signature

All reward functions used with `RLVRWorkflow` follow this signature:

```python
def my_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    **kwargs,
) -> float:
    """
    Args:
        prompt: The input prompt string.
        completions: The model-generated response string.
        prompt_ids: Token IDs of the prompt.
        completion_ids: Token IDs of the completion.
        **kwargs: Additional dataset fields (e.g., answer, solution).

    Returns:
        A scalar reward value (float).
    """
```

The `**kwargs` receives all other fields from your dataset. For example, if your dataset
contains an `"answer"` column, you can access it as `kwargs["answer"]` or add `answer`
as a named parameter.

## Rule-Based Reward Functions

The simplest reward functions compare the model's output to a ground truth answer using
rules.

### Example: Math Answer Verification

```python
# my_project/rewards.py
from areal.reward import get_math_verify_worker

def my_math_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    """Verify math answer using AReaL's built-in math verifier."""
    try:
        worker = get_math_verify_worker()
        return worker.verify(str(completions), str(answer))
    except Exception:
        return 0.0
```

The built-in `MathVerifyWorker` handles LaTeX and expression extraction with
configurable precision. See `areal/reward/__init__.py` for details.

### Example: Format + Accuracy Composite Reward

```python
import re

def composite_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    """Reward that combines format compliance and answer accuracy."""
    format_score = 0.0
    accuracy_score = 0.0

    # Check if response follows the expected format (e.g., uses \boxed{})
    if re.search(r"\\boxed\{.+\}", completions):
        format_score = 0.2

    # Check answer accuracy
    from areal.reward import get_math_verify_worker
    try:
        accuracy_score = get_math_verify_worker().verify(
            str(completions), str(answer)
        ) * 0.8
    except Exception:
        pass

    return format_score + accuracy_score
```

## Built-in Reward Functions

AReaL ships with the following reward functions:

| Function | Module Path | Dataset |
|----------|------------|---------|
| `gsm8k_reward_fn` | `areal.reward.gsm8k.gsm8k_reward_fn` | GSM8K math |
| `geometry3k_reward_fn` | `areal.reward.geometry3k.geometry3k_reward_fn` | Geometry3K |
| `clevr_count_70k_reward_fn` | `areal.reward.clevr_count_70k.clevr_count_70k_reward_fn` | CLEVR Count |

## Using Reward Functions in Training

Pass your reward function to the workflow by **module path string**. This allows AReaL
to serialize and distribute the function across workers:

```python
from areal import PPOTrainer

workflow_kwargs = dict(
    reward_fn="my_project.rewards.my_math_reward_fn",  # importable string path
    gconfig=config.gconfig,
    tokenizer=config.tokenizer_path,
)

with PPOTrainer(config, train_dataset=train_dataset) as trainer:
    trainer.train(
        workflow="areal.workflow.rlvr.RLVRWorkflow",
        workflow_kwargs=workflow_kwargs,
    )
```

You can also pass the function object directly (works when not using distributed
serialization):

```python
from my_project.rewards import my_math_reward_fn

workflow_kwargs = dict(
    reward_fn=my_math_reward_fn,  # direct function reference
    gconfig=config.gconfig,
    tokenizer=tokenizer,
)
```

## Generative Reward Model (LLM-as-Judge)

For tasks where rule-based verification is insufficient (e.g., open-ended generation,
creative writing, instruction following), you can use a **generative reward model** that
prompts another LLM to score the response.

### Example: LLM-as-Judge Reward Function

```python
import re

def llm_judge_reward_fn(
    prompt, completions, prompt_ids, completion_ids, **kwargs
) -> float:
    """Use an external LLM API to judge response quality."""
    import openai

    judge_prompt = f"""Rate the following response on a scale of 0 to 10.
Only output the numeric score.

Question: {prompt}
Response: {completions}

Score:"""

    client = openai.OpenAI(
        base_url="http://localhost:8000/v1",  # local vLLM/SGLang server
        api_key="unused",
    )
    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
        max_tokens=16,
    )
    score_text = response.choices[0].message.content.strip()

    # Extract numeric score
    match = re.search(r"(\d+(?:\.\d+)?)", score_text)
    if match:
        score = float(match.group(1))
        return min(score / 10.0, 1.0)  # Normalize to [0, 1]
    return 0.0
```

### Handling Slow Reward Functions

Generative reward models can be slow. AReaL's `AsyncRewardWrapper` automatically wraps
your reward function for async execution with timeout handling:

```python
from areal.api import AsyncRewardWrapper

# The wrapper is applied automatically when you pass reward_fn to RLVRWorkflow.
# To customize timeout and concurrency:
async_reward = AsyncRewardWrapper(
    reward_fn=llm_judge_reward_fn,
    timeout_seconds=30,   # Increase for slow models (default: 15s)
    max_workers=4,        # Number of parallel reward computations
    max_retries=3,        # Auto-recovery from crashes
)
```

The `RLVRWorkflow` automatically wraps your reward function with `AsyncRewardWrapper`,
so you typically don't need to create one manually. The default timeout is 15 seconds —
if your reward function is slower, consider optimizing it or batching requests.

## Registering a New Built-in Reward Function

To add a reward function to AReaL's built-in collection (so it can be auto-selected by
dataset name):

1. Create your reward module at `areal/reward/my_dataset.py`
2. Register it in `areal/reward/__init__.py`:

```python
# In areal/reward/__init__.py
VALID_REWARD_FN = ["clevr_count_70k", "geometry3k", "my_dataset"]

def get_custom_reward_fn(path: str, **kwargs):
    # ... existing entries ...
    elif "my_dataset" in path:
        from .my_dataset import my_dataset_reward_fn
        return my_dataset_reward_fn
```

See the [Add Reward skill guide](https://github.com/inclusionAI/AReaL/blob/main/.claude/skills/add-reward/SKILL.md)
for the full step-by-step process.
