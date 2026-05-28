# 编写 Agent Workflow

本指南涵盖在 AReaL 中实现高效且健壮的 `RolloutWorkflow` 类和 Agent Workflow 的最佳实践。

有关 `RolloutWorkflow` 与 Agent Workflow 之间的区别，请参阅
[Agentic RL 指南](../tutorial/agentic_rl.md)。

## 最佳实践

### 全程使用 Async/Await

所有 Workflow 方法都应该是 async 的，并对 I/O 密集型操作使用 `await`。这使得跨多个 rollouts 的并发执行成为可能。

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

### 包装开销大的奖励函数

对于涉及 CPU 密集型计算、外部 API 调用或任何阻塞操作的奖励函数，请使用 `AsyncRewardWrapper`。`AsyncRewardWrapper`
将奖励计算分派到专用的进程池。

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

### 避免繁重的初始化

将开销大的设置逻辑放在 `__init__` 中，而不是 `arun_episode` 中。`arun_episode` 方法为每次 rollout
运行，因此重复初始化会浪费资源。

### 通过 Workflow 上下文重用 HTTP 客户端

跨请求重用 HTTP 客户端，而不是创建新客户端。AReaL 通过 `workflow_context` 提供具有自动生命周期管理的共享客户端。

使用 OpenAI、Anthropic 或其他 SDK 客户端时，传递共享的 HTTP 客户端：

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

## 另请参阅

- [调试指南](../best_practices/debugging.md) - 调试自定义 Workflow
- [RolloutWorkflow 参考文档](../reference/rollout_workflow.md) - API 文档
- [Agentic RL 指南](../tutorial/agentic_rl.md) - 使用 Agent 框架进行训练
