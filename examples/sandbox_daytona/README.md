# Daytona sandbox example

This example shows how to use `DaytonaRunner` from plain Python code, such as a reward
function, evaluation script, or data-prep utility.

## Install

```bash
uv sync --extra sandbox
export DAYTONA_API_KEY=your-key
```

## Run

```bash
uv run python examples/sandbox_daytona/reward_example.py
```

## Notes

- `DaytonaRunner` is the synchronous entry point for non-async user code.
- Each runner instance creates one sandbox and keeps it warm across multiple `run()`
  calls.
