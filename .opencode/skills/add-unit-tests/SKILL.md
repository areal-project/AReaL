---
name: add-unit-tests
description: Guide for adding unit tests to AReaL. Use when user wants to add tests for new functionality or increase test coverage.
---

# Add Unit Tests

Add unit tests to AReaL following the project's testing conventions.

## When to Use

This skill is triggered when:

- User asks "how do I add tests?"
- User wants to increase test coverage
- User needs to write tests for new functionality
- User wants to understand AReaL testing patterns

## Step-by-Step Guide

### Step 1: Understand Test Types

AReaL has two main test categories:

| Test Type             | Purpose                            | Location Pattern                   | How It Runs                                |
| --------------------- | ---------------------------------- | ---------------------------------- | ------------------------------------------ |
| **Unit Tests**        | Test individual functions/modules  | `tests/test_<module>_<feature>.py` | Directly via pytest                        |
| **Distributed Tests** | Test distributed/parallel behavior | `tests/torchrun/run_*.py`          | Via torchrun (called by pytest subprocess) |

**Note**: All tests are invoked via pytest. Distributed tests use `torchrun` but are
still called from pytest test files.

### Step 2: Create Test File Structure

Create test file with naming convention: `test_<module>_<feature>.py`

```python
import pytest
import torch

# Import the module to test
from areal.dataset.gsm8k import get_gsm8k_sft_dataset
from tests.utils import get_dataset_path  # Optional test utilities
# For mocking tokenizer: from unittest.mock import MagicMock
```

### Step 3: Write Test Functions

Follow Arrange-Act-Assert pattern:

```python
def test_function_under_condition_returns_expected():
    """Test that function returns expected value under condition."""
    # Arrange
    input_data = 5
    expected_output = 10

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected_output
```

### Step 4: Add Pytest Markers and CI Strategy

Use appropriate pytest markers:

| Marker                                  | When to Use                                                  |
| --------------------------------------- | ------------------------------------------------------------ |
| `@pytest.mark.slow`                     | Test takes > 10 seconds (excluded from CI by default)        |
| `@pytest.mark.ci`                       | Slow test that must run in CI (use with `@pytest.mark.slow`) |
| `@pytest.mark.asyncio`                  | Async test functions                                         |
| `@pytest.mark.skipif(cond, reason=...)` | Conditional skip                                             |
| `@pytest.mark.parametrize(...)`         | Parameterized tests                                          |

**CI Test Strategy**:

- `@pytest.mark.slow`: Excluded from CI by default (CI runs `pytest -m "not slow"`)
- `@pytest.mark.slow` + `@pytest.mark.ci`: Slow but must run in CI
- No marker: Runs in CI (fast unit tests)

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_feature():
    tensor = torch.tensor([1, 2, 3], device="cuda")
    # ... assertions

@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_with_parameters(batch_size):
    # Parameterized test

@pytest.mark.slow
def test_slow_function():
    # Excluded from CI by default

@pytest.mark.slow
@pytest.mark.ci
def test_slow_but_required_in_ci():
    # Slow but must run in CI
```

### Step 5: Mock Distributed Environment

For unit tests that need distributed mocks:

```python
import torch.distributed as dist

def test_distributed_function(monkeypatch):
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    result = distributed_function()
    assert result == expected
```

### Step 6: Handle GPU Dependencies

Always skip gracefully when GPU unavailable:

```python
CUDA_AVAILABLE = torch.cuda.is_available()

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_gpu_function():
    tensor = torch.tensor([1, 2, 3], device="cuda")
    # ... assertions
```

## Key Requirements (Based on testing.md)

### Mocking Distributed

- Use `torch.distributed.fake_pg` for unit tests
- Mock `dist.get_rank()` and `dist.get_world_size()` explicitly
- Don't mock internals of FSDP/DTensor

### GPU Test Constraints

- **Always skip gracefully** when GPU unavailable
- Clean up GPU memory: `torch.cuda.empty_cache()` in fixtures
- Use smallest possible model/batch for unit tests

### Assertions

- Use `torch.testing.assert_close()` for tensor comparison
- Specify `rtol`/`atol` explicitly for numerical tests
- Avoid bare `assert tensor.equal()` - no useful error message

## Reference Implementations

| Test File                        | Description                            | Key Patterns                                      |
| -------------------------------- | -------------------------------------- | ------------------------------------------------- |
| `tests/test_utils.py`            | Utility function tests                 | Fixtures, parametrized tests                      |
| `tests/test_examples.py`         | Integration tests with dataset loading | Dataset path resolution, success pattern matching |
| `tests/test_fsdp_engine_nccl.py` | Distributed tests                      | Torchrun integration                              |

## Common Mistakes

- **Missing test file registration**: Ensure file follows `test_*.py` naming
- **GPU dependency without skip**: Always use `@pytest.mark.skipif` for GPU tests
- **Incorrect tensor comparisons**: Use `torch.testing.assert_close()` not
  `assert tensor.equal()`
- **Memory leaks in GPU tests**: Clean up with `torch.cuda.empty_cache()`
- **Mocking too much**: Don't mock FSDP/DTensor internals
- **Unclear test names**: Follow `test_<what>_<condition>_<expected>` pattern
- **No docstrings**: Add descriptive docstrings to test functions

## Integration with Other Skills

This skill complements other AReaL development skills:

- **After `/add-dataset`**: Add tests for new dataset loaders
- **After `/add-workflow`**: Add tests for new workflows
- **After `/add-reward`**: Add tests for new reward functions
- **With expert agents**: Reference this skill when planning test implementation

## Running Tests

```bash
# First check GPU availability (many tests require GPU)
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

# Run specific test file
uv run pytest tests/test_<name>.py

# Skip slow tests (CI default)
uv run pytest -m "not slow"

# Run with verbose output
uv run pytest -v

# Run distributed tests (requires torchrun and multi-GPU)
# Note: Usually invoked via pytest test files
torchrun --nproc_per_node=2 tests/torchrun/run_<test>.py
```

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .opencode/skills/add-unit-tests/SKILL.md
Invocation: /add-unit-tests

## Purpose

Step-by-step guide for adding unit tests to AReaL.

## How to Update

### When Testing Conventions Change
1. Update "Key Requirements" section based on `testing.md`
2. Update test examples to match new patterns
3. Update reference implementations

### When Test Types Need Update
1. Update "Understand Test Types" table (currently two main types)
2. Add new examples if needed
3. Update common mistakes

### Integration with Other Skills
Ensure references to other skills (`/add-dataset`, `/add-workflow`, `/add-reward`) remain accurate.

================================================================================
-->
