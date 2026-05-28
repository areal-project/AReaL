---
paths:
  - areal/api/**
---

# API & Config Rules

## Dataclass Conventions

```python
@dataclass
class XxxConfig:
    """One-line description.

    Attributes:
        field_name: Description with default explained.
    """
    # Required fields first (no default)
    required_field: str

    # Optional fields with defaults
    optional_field: int = 32

    # Internal fields last (underscore prefix)
    _internal: str = field(default="", repr=False)
```

## Field Ordering

1. Required fields (no default)
1. Common optional fields
1. Advanced/rare optional fields
1. Internal fields (`_prefix`)

## Validation

- Use `__post_init__` for validation
- Raise `ValueError` with clear message:
  ```python
  def __post_init__(self):
      if self.batch_size <= 0:
          raise ValueError(f"batch_size must be positive, got {self.batch_size}")
  ```

## Backward Compatibility

- **Adding fields**: Add with default value (safe)
- **Removing fields**: Deprecate first, remove in next major version
- **Renaming fields**: Add new field, keep old with deprecation warning
- **Changing types**: Avoid; use Union if necessary

## CLI Integration

- Fields exposed to CLI must have clear `help` in metadata
- Use `Literal` for enum-like choices
- Avoid complex nested types in CLI-exposed configs

## Documentation

- All public configs must have docstring
- Document constraints (e.g., "must be power of 2")
- Include example values for non-obvious fields
