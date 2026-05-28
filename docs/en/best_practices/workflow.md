# Writing Agent Workflows

This guide covers best practices for implementing efficient and robust `RolloutWorkflow`
classes and agent workflows in AReaL.

For the difference between `RolloutWorkflow` and agent workflows, see the
[Agentic RL Guide](../tutorial/agentic_rl.md).

## Best Practices

### Use Async/Await Throughout

All workflow methods should be async and use `await` for I/O-bound operations. This
enables concurrent execution across multiple rollouts.

```python
# Or the `run` method in agent workflows
async def arun_episode(self, engine, data):
    # Correct: await the engine call
    resp = await engine.agenerate(req)

    # Correct: await other LLM calls
    async with AsyncOpenAI() as client:
        resp = await client.chat.completions.create(...)

    # Incorrect: blocking calls stall other rollouts
    # resp = engine.generate(req)  # Don't do this
    # resp = OpenAI().chat.completions.create(...)  # Don't do this

    # Await HTTP requests with reused client
    session = await workflow_context.get_aiohttp_session()
    async with session.get(url) as response:
        result = await response.json()

    # Await file operations (use aiofiles)
    async with aiofiles.open(path, "r") as f:
        content = await f.read()
```

### Wrap Expensive Reward Functions

Use `AsyncRewardWrapper` for reward functions involving CPU-intensive computation,
external API calls, or any blocking operation. `AsyncRewardWrapper` dispatches reward
computation to a dedicated process pool.

```python
from areal.api.reward_api import AsyncRewardWrapper

class MyWorkflow(RolloutWorkflow):
    def __init__(self, reward_fn, ...):
        # Wrap the reward function once during initialization
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)

    async def arun_episode(self, engine, data):
        resp = await engine.agenerate(req)
        # Await the wrapped reward function
        reward = await self.async_reward_fn(
            prompt_str,
            completion_str,
            **data,
        )
```

### Avoid Heavy Initialization

Place expensive setup logic in `__init__`, not in `arun_episode`. The `arun_episode`
method runs for every rollout, so repeated initialization wastes resources.

### Reuse HTTP Clients via Workflow Context

Reuse HTTP clients across requests instead of creating new ones. AReaL provides shared
clients through `workflow_context` with automatic lifecycle management.

When using OpenAI, Anthropic, or other SDK clients, pass the shared HTTP client:

```python
from openai import AsyncOpenAI
from areal.infra import workflow_context

class MyAgentWorkflow:
    async def run(self, data, **extra_kwargs):
        # Get pre-configured client from extra_kwargs
        http_client = extra_kwargs.get("http_client")
        base_url = extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")

        # Pass to SDK constructor
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )

        response = await client.chat.completions.create(...)
```

## See Also

- [Debugging Guide](../best_practices/debugging.md) - Debugging customized workflows
- [RolloutWorkflow Reference](../reference/rollout_workflow.md) - API documentation
- [Agentic RL Guide](../tutorial/agentic_rl.md) - Training with agent frameworks
