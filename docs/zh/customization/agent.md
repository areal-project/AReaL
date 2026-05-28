# 自定义 Agent Workflow

本指南介绍如何为 RL 训练创建自定义 Agent。AReaL 支持任何 Agent 框架（OpenAI Agents SDK、LangChain、CAMEL-AI
等），只需少量集成工作。

**注意**：

1. Agent Workflow 仅在 `local` 和 `slurm` 调度器上受支持。`ray` 调度器与 HTTP 代理架构不兼容。

1. 有关内部架构详情，请参阅 [Agent Workflow 参考文档](../reference/agent_workflow.md)。

## 快速开始

Agent Workflow 是任何具有 `async def run(data, **extra_kwargs)` 方法并返回奖励的类。AReaL 自动为其包装以便进行 RL
训练。

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

将 Agent 传递给训练器：

```python
trainer.train(workflow="my_module.MyAgent")
```

## 方法签名

`run` 方法必须遵循此签名：

```python
async def run(self, data: dict, **extra_kwargs) -> float | dict[str, float]
```

| 参数           | 描述                                     |
| -------------- | ---------------------------------------- |
| `data`         | 来自数据集的样本（包含数据键的字典）     |
| `extra_kwargs` | AReaL 注入的参数（见下文）               |
| **返回**       | `float`：最后一个补全的奖励              |
|                | `dict[str, float]`：将补全 ID 映射到奖励 |

### 注入的参数

AReaL 通过 `extra_kwargs` 注入这些参数：

| 键            | 类型                | 描述                              |
| ------------- | ------------------- | --------------------------------- |
| `base_url`    | `str`               | AReaL 代理服务器的 URL            |
| `api_key`     | `str`               | AReaL 代理服务器的会话级 API 密钥 |
| `http_client` | `httpx.AsyncClient` | 共享 HTTP 客户端（减少开销）      |

## 执行模式

AReaL 支持两种执行模式，通过 `rollout.agent.mode` 配置：

### Inline 模式（默认）

Agent 与 rollout 工作进程在同一进程中运行。推荐用于大多数场景。

```yaml
rollout:
  agent:
    mode: inline
```

**要求**：

- `run` 方法必须是 `async`
- 使用 `extra_kwargs["base_url"]` 进行 LLM 调用
- 可选择使用 `extra_kwargs["http_client"]` 来减少开销

**优势**：

- 无序列化开销
- 直接访问共享 HTTP 客户端
- 更低的延迟

### Subprocess 模式

Agent 在单独进程池中运行。当您的 Agent 代码不兼容 async 或使用与主进程冲突的库时，使用此模式。

```yaml
rollout:
  agent:
    mode: subproc
    subproc_max_workers: 4  # Process pool size
```

**要求**：

- Agent 类必须是可 pickle（可序列化）的
- 从环境变量而不是 `extra_kwargs` 读取 `OPENAI_BASE_URL`

**示例**：

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

**注意**：即使在子进程模式下，方法签名仍然是 `async def run(...)`，但 AReaL 会在内部用 `asyncio.run()`
包装调用。您可以在方法内部使用同步代码。

**权衡**：

- Agent 和数据的 pickle 开销
- 无法访问共享 HTTP 客户端
- 每次调用延迟更高
- 适用于非 async 库或进程隔离

## 奖励分配

### 简单奖励

返回单个浮点数来为最后一个 LLM 补全分配奖励：

```python
async def run(self, data, **extra_kwargs):
    # ... agent logic ...
    return 1.0 if is_correct else 0.0
```

### 每个补全的奖励

对于多轮对话，返回将补全 ID 映射到奖励的字典：

```python
async def run(self, data, **extra_kwargs):
    # ... multi-turn agent logic ...
    return {
        "completion-id-1": 0.5,
        "completion-id-2": 1.0,
    }
```

从响应中获取补全 ID：

```python
response = await client.chat.completions.create(...)
completion_id = response.id  # Use this ID for reward mapping
```

## 配置

Agent Workflow 设置在 `rollout.agent` 中：

```yaml
rollout:
  agent:
    mode: inline              # "inline" or "subproc"
    turn_discount: 0.9        # Reward discount for earlier turns
    export_style: individual  # "individual" or "concat"
    subproc_max_workers: 4    # Process pool size (subproc mode only)
```

| 字段                  | 默认         | 描述                       |
| --------------------- | ------------ | -------------------------- |
| `mode`                | `inline`     | 执行模式                   |
| `turn_discount`       | `1.0`        | 多轮奖励的几何折扣         |
| `export_style`        | `individual` | 如何导出交互以进行训练     |
| `subproc_max_workers` | `4`          | 子进程模式的最大工作进程数 |

## 另请参阅

- [Agentic RL 教程](../tutorial/agentic_rl.md) - 端到端训练示例
- [异步 Workflow 最佳实践](../best_practices/workflow.md) - 编写高效的 inline async Agent Workflow
- [Agent Workflow 参考文档](../reference/agent_workflow.md) - 内部架构
