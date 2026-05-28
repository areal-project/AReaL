---
name: add-dataset
description: Guide for adding a new dataset loader to AReaL. Use when user wants to add a new dataset.
---

# Add Dataset

Add a new dataset loader to AReaL.

## When to Use

This skill is triggered when:

- User asks "how do I add a dataset?"
- User wants to integrate a new dataset
- User mentions creating a dataset loader

## Step-by-Step Guide

### Step 1: Create Dataset File

Create `areal/dataset/<name>.py`:

```python
from datasets import Dataset, load_dataset


def get_<name>_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
) -> Dataset:
    """Load dataset for SFT training.

    Args:
        path: Path to dataset (HuggingFace hub or local path)
        split: Dataset split (train/validation/test)
        tokenizer: Tokenizer for processing
        max_length: Maximum sequence length (optional)

    Returns:
        HuggingFace Dataset with processed samples
    """
    dataset = load_dataset(path=path, split=split)

    def process(sample):
        # Tokenize the full sequence (prompt + response)
        seq_token = tokenizer.encode(
            sample["question"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])
        # Loss mask: 0 for prompt, 1 for response
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["question", "answer"])

    if max_length is not None:
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_<name>_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
) -> Dataset:
    """Load dataset for RL training.

    Args:
        path: Path to dataset
        split: Dataset split
        tokenizer: Tokenizer for length filtering
        max_length: Maximum sequence length

    Returns:
        HuggingFace Dataset with prompts and answers for reward computation
    """
    dataset = load_dataset(path=path, split=split)

    def process(sample):
        messages = [
            {
                "role": "user",
                "content": sample["question"],
            }
        ]
        return {"messages": messages, "answer": sample["answer"]}

    dataset = dataset.map(process).remove_columns(["question"])

    if max_length is not None:

        def filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
```

### Step 2: Register in __init__.py

Update `areal/dataset/__init__.py`:

```python
# Add to VALID_DATASETS
VALID_DATASETS = [
    # ... existing datasets
    "<name>",
]

# Add to _get_custom_dataset function
def _get_custom_dataset(name: str, ...):
    # ... existing code
    elif name == "<name>":
        from areal.dataset.<name> import get_<name>_sft_dataset, get_<name>_rl_dataset
        if dataset_type == "sft":
            return get_<name>_sft_dataset(path, split, max_length, tokenizer)
        else:
            return get_<name>_rl_dataset(path, split, max_length, tokenizer)
```

### Step 3: Add Config (Optional)

If the dataset needs special configuration, add to `areal/api/cli_args.py`:

```python
@dataclass
class TrainDatasetConfig:
    # ... existing fields
    <name>_specific_field: Optional[str] = None
```

### Step 4: Add Tests

Create `tests/test_<name>_dataset.py`:

```python
import pytest
from areal.dataset.<name> import get_<name>_sft_dataset, get_<name>_rl_dataset

def test_sft_dataset_loads(tokenizer):
    dataset = get_<name>_sft_dataset("path/to/data", split="train", tokenizer=tokenizer)
    assert len(dataset) > 0
    assert "input_ids" in dataset.column_names
    assert "loss_mask" in dataset.column_names

def test_rl_dataset_loads(tokenizer):
    dataset = get_<name>_rl_dataset("path/to/data", split="train", tokenizer=tokenizer)
    assert len(dataset) > 0
    assert "messages" in dataset.column_names
    assert "answer" in dataset.column_names
```

## Reference Implementations

| Dataset    | File                               | Description              |
| ---------- | ---------------------------------- | ------------------------ |
| GSM8K      | `areal/dataset/gsm8k.py`           | Math word problems       |
| Geometry3K | `areal/dataset/geometry3k.py`      | Geometry problems        |
| CLEVR      | `areal/dataset/clevr_count_70k.py` | Visual counting          |
| HH-RLHF    | `areal/dataset/hhrlhf.py`          | Helpfulness/Harmlessness |
| TORL       | `areal/dataset/torl_data.py`       | Tool-use RL              |

## Required Fields

### SFT Dataset

```python
{
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
    ]
}
```

### RL Dataset

```python
{
    "messages": [
        {"role": "user", "content": "..."},
    ],
    "answer": "ground_truth_for_reward",
    # Optional metadata for reward function
}
```

## Common Mistakes

- Returning `List[Dict]` instead of HuggingFace `Dataset`
- Using Python loops instead of `dataset.map()`/`filter()`
- Missing `"messages"` field for RL datasets
- Wrong message format (should be list of dicts with `role` and `content`)
- Not registering in `__init__.py`

______________________________________________________________________

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .opencode/skills/add-dataset/SKILL.md
Invocation: /add-dataset <name>

## Purpose

Step-by-step guide for adding new dataset loaders.

## How to Update

### When Dataset API Changes
1. Update the code templates
2. Update required fields section
3. Update registration example

### When New Dataset Types Added
1. Add to "Reference Implementations" table
2. Add any new required fields

================================================================================
-->
