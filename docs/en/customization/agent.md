# Custom Agent Workflows

This guide shows how to create custom agents for RL training. AReaL supports any agent
framework (OpenAI Agents SDK, LangChain, CAMEL-AI, etc.) with minimal integration.

**Notes**:

1. Agent workflows are supported on `local` and `slurm` schedulers only. The `ray`
   scheduler is incompatible with the HTTP proxy architecture.

1. For internal architecture details, see the
   [Agent Workflow Reference](../reference/agent_workflow.md).

## Quick Start

An agent workflow is any class with an `async def run(data, **extra_kwargs)` method that
returns a reward. AReaL automatically wraps it for RL training.

```python
class MyAgent:
    async def run(self, data, **extra_kwargs):
        # Get injected client and URL
        http_client = extra_kwargs.get("http_client")
        base_url = extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")

        # Use standard OpenAI SDK
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )

        response = await client.chat.completions.create(
            model="default",
            messages=data["messages"],
        )

        # Return reward (float or dict[str, float])
        return compute_reward(response, data["answer"])
```

Pass the agent to the trainer:

```python
trainer.train(workflow="my_module.MyAgent")
```

## Method Signature

The `run` method must follow this signature:

```python
async def run(self, data: dict, **extra_kwargs) -> float | dict[str, float]
```

| Parameter      | Description                                           |
| -------------- | ----------------------------------------------------- |
| `data`         | A sample from your dataset (dict with your data keys) |
| `extra_kwargs` | AReaL-injected arguments (see below)                  |
| **Return**     | `float`: reward for last completion                   |
|                | `dict[str, float]`: maps completion IDs to rewards    |

### Injected Arguments

AReaL injects these arguments via `extra_kwargs`:

| Key           | Type                | Description                                  |
| ------------- | ------------------- | -------------------------------------------- |
| `base_url`    | `str`               | URL to AReaL's proxy server                  |
| `api_key`     | `str`               | Session-wise API key to AReaL's proxy server |
| `http_client` | `httpx.AsyncClient` | Shared HTTP client (reduces overhead)        |

## Execution Modes

AReaL supports two execution modes, configured via `rollout.agent.mode`:

### Inline Mode (Default)

The agent runs in the same process as the rollout worker. Recommended for most use
cases.

```yaml
rollout:
  agent:
    mode: inline
```

**Requirements:**

- The `run` method must be `async`
- Use `extra_kwargs["base_url"]` for LLM calls
- Optionally use `extra_kwargs["http_client"]` to reduce overhead

**Advantages:**

- No serialization overhead
- Direct access to shared HTTP client
- Lower latency

### Subprocess Mode

The agent runs in a separate process pool. Use this when your agent code is not
async-compatible or uses libraries that conflict with the main process.

```yaml
rollout:
  agent:
    mode: subproc
    subproc_max_workers: 4  # Process pool size
```

**Requirements:**

- The agent class must be picklable (serializable)
- Read `OPENAI_BASE_URL` from environment instead of `extra_kwargs`

**Example:**

```python
import os
from openai import OpenAI  # Sync client is OK

class MySyncAgent:
    async def run(self, data, **extra_kwargs):
        # In subproc mode, base_url and api_key come from environment
        client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_key="DUMMY",  # Not used by AReaL
        )

        response = client.chat.completions.create(
            model="default",
            messages=data["messages"],
        )

        return compute_reward(response, data["answer"])
```

**Note:** The method signature remains `async def run(...)` even in subprocess mode, but
AReaL wraps the call with `asyncio.run()` internally. You can use synchronous code
inside the method.

**Trade-offs:**

- Pickling overhead for agent and data
- No access to shared HTTP client
- Higher latency per call
- Useful for non-async libraries or process isolation

## Reward Assignment

### Simple Reward

Return a single float to assign reward to the last LLM completion:

```python
async def run(self, data, **extra_kwargs):
    # ... agent logic ...
    return 1.0 if is_correct else 0.0
```

### Per-Completion Rewards

For multi-turn conversations, return a dict mapping completion IDs to rewards:

```python
async def run(self, data, **extra_kwargs):
    # ... multi-turn agent logic ...
    return {
        "completion-id-1": 0.5,
        "completion-id-2": 1.0,
    }
```

Access completion IDs from the response:

```python
response = await client.chat.completions.create(...)
completion_id = response.id  # Use this ID for reward mapping
```

## Configuration

Agent workflow settings are in `rollout.agent`:

```yaml
rollout:
  agent:
    mode: inline              # "inline" or "subproc"
    turn_discount: 0.9        # Reward discount for earlier turns
    export_style: individual  # "individual" or "concat"
    subproc_max_workers: 4    # Process pool size (subproc mode only)
```

| Field                 | Default      | Description                               |
| --------------------- | ------------ | ----------------------------------------- |
| `mode`                | `inline`     | Execution mode                            |
| `turn_discount`       | `1.0`        | Geometric discount for multi-turn rewards |
| `export_style`        | `individual` | How to export interactions for training   |
| `subproc_max_workers` | `4`          | Max worker processes for subprocess mode  |

## See Also

- [Agentic RL Tutorial](../tutorial/agentic_rl.md) - End-to-end training examples
- [Async Workflow Best Practices](../best_practices/workflow.md) - Writing efficient
  inline async agent workflows
- [Agent Workflow Reference](../reference/agent_workflow.md) - Internal architecture
