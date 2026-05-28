---
paths:
  - '**/tests/**'
  - '*_test.py'
  - test_*.py
---

# Testing Rules

## Pytest Markers

| Marker                                  | When to Use          |
| --------------------------------------- | -------------------- |
| `@pytest.mark.slow`                     | Takes > 10 seconds   |
| `@pytest.mark.asyncio`                  | Async test functions |
| `@pytest.mark.skipif(cond, reason=...)` | Conditional skip     |
| `@pytest.mark.parametrize(...)`         | Parameterized tests  |

## Test Structure

```python
def test_<what>_<condition>_<expected>():
    """Test that <what> does <expected> when <condition>."""
    # Arrange
    ...
    # Act
    ...
    # Assert
    ...
```

## Mocking Distributed

- Use `torch.distributed.fake_pg` for unit tests
- Mock `dist.get_rank()` and `dist.get_world_size()` explicitly
- Don't mock internals of FSDP/DTensor - use integration tests

## GPU Test Constraints

- **Always skip gracefully** when GPU unavailable:
  ```python
  CUDA_AVAILABLE = torch.cuda.is_available()

  @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
  def test_gpu_feature():
      ...
  ```
- Clean up GPU memory: `torch.cuda.empty_cache()` in fixtures
- Use smallest possible model/batch for unit tests

## Fixtures

- Prefer `tmp_path` over manual temp directories
- Use `monkeypatch` for environment variables
- Scope expensive fixtures appropriately (`session` > `module` > `function`)

## Assertions

- Use `torch.testing.assert_close()` for tensor comparison
- Specify `rtol`/`atol` explicitly for numerical tests
- Avoid bare `assert tensor.equal()` - no useful error message
