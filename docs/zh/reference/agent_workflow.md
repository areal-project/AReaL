# 代理工作流

本文档描述 AReaL 的代理工作流系统，该系统支持使用代理框架训练语言模型，同时捕获用于强化学习的 token 级别数据。

**注意：**

1. 本页面向希望深入理解代码库的开发者。实践指南请参见 [Agentic RL Guide](../tutorial/agentic_rl.md)。

1. 首先阅读 [`RolloutWorkflow` 参考](../reference/rollout_workflow.md)，因为代理工作流建立在
   `RolloutWorkflow` 之上。

1. **调度器兼容性**：代理工作流仅在 `local` 和 `slurm` 调度器上受支持。`ray` 调度器与 HTTP 代理架构不兼容。

## 概述

代理工作流允许使用流行的代理框架（OpenAI Agents SDK、CAMEL-AI、LangChain 等）训练模型，而无需修改其核心逻辑。AReaL 自动捕获 RL
训练所需的 token 级别信息，同时保留代理的原始行为。

主要优势：

- **灵活性**：支持任何使用 OpenAI/Anthropic 消息协议的框架
- **统一开发**：基准测试、评估和 RL 训练使用相同代码
- **算法正确性**：Token 级别跟踪避免训练-推理不匹配

挑战在于代理框架通过不暴露 token ID 和对数概率的高级 API 与 LLM 交互。AReaL 通过以下方式解决此问题：

1. **拦截 LLM 调用**通过代理服务器或直接客户端
1. **跟踪 token 级别数据**在 `InteractionCache` 中
1. **构建对话树**用于多轮奖励传播
1. **导出训练就绪的张量**并正确归因奖励

## 与 RolloutWorkflow 的关系

代理工作流不是单独的抽象——它们通过 `OpenAIProxyWorkflow` 自动包装为 `RolloutWorkflow`：

```
用户的代理代码（async def run())
           ↓
   OpenAIProxyWorkflow（包装器）
           ↓
   RolloutWorkflow.arun_episode()
           ↓
   dict[str, InteractionWithTokenLogpReward]
           ↓
   用于训练的张量字典
```

任何具有 `async def run(data, **extra_kwargs)` 方法的类都被识别为代理工作流，在传递给训练器时自动包装。

## 两种集成范式

AReaL 提供两种集成代理框架的方法：

| 方面             | 代理方式                             | 直接方式                                    |
| ---------------- | ------------------------------------ | ------------------------------------------- |
| **代码修改**     | 无（仅更改 `base_url` 和 `api_key`） | 必须接受 `ArealOpenAI` 客户端               |
| **通信**         | 通过代理服务器的 HTTP                | 直接引擎调用                                |
| **框架支持**     | 任何 OpenAI 兼容框架                 | 接受自定义客户端的框架                      |
| **性能**         | HTTP 开销（最小）                    | 无 HTTP 开销                                |
| **引擎状态访问** | 有限                                 | 完全访问                           用于\*\* |
| \*\*推荐         | 现有代理、第三方框架                 | 遗留代码。**不要主动使用。**                |

具体示例见 [Agentic RL Guide](../tutorial/agentic_rl.md)。

### 代理方式

代理方式使代理代码独立于 AReaL。您的代理使用标准的 OpenAI/Anthropic 消息协议，指向 AReaL 代理服务器的定制 `base_url`。

AReaL 的训练器在 RL 训练期间自动提供 `base_url` 和 `http_client`。

```python
class MyAgent:
    async def run(self, data, **extra_kwargs):
        # AReaL 注入这些 kwargs
        http_client = extra_kwargs.get("http_client")
        base_url = extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")

        # 标准 OpenAI SDK 使用方式
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

        # 返回奖励（float）或奖励字典
        return compute_reward(response, data["answer"])
```

### 直接方式

> **遗留模式**：使用 `ArealOpenAI` 和 `RolloutWorkflow`
> 的直接方式被视为遗留方式，不应用于新项目。请优先使用上述代理方式，使代理代码独立于 AReaL 内部实现。

直接方式使用 `ArealOpenAI`，它扩展了 `AsyncOpenAI` 并直接绑定到推理引擎。此方式需要工作流继承 `RolloutWorkflow` 并使用
`arun_episode` 中的引擎。

```python
from areal.experimental.openai import ArealOpenAI

class MyWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine, data):
        # 创建绑定到引擎的客户端
        client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)

        # 像标准 OpenAI 客户端一样使用
        response = await client.chat.completions.create(
            model="default",
            messages=data["messages"],
        )

        # 设置奖励并导出
        reward = compute_reward(response, data["answer"])
        client.set_last_reward(reward)
        client.apply_reward_discount(turn_discount=0.9)

        return client.export_interactions(style="individual")
```

## 执行模式

代理方式支持两种执行模式，通过 `rollout.agent.mode` 配置：

### 内联模式（默认）

代理在 Rollout 工作器的同一进程中运行。AReaL 直接调用代理的 `run` 方法作为异步协程，通过 `extra_kwargs` 传递
`base_url`、`api_key` 和 `http_client`。

```yaml
rollout:
  agent:
    mode: inline
```

**特性：**

- 无序列化开销
- 直接访问共享 HTTP 客户端
- 延迟更低
- 需要异步代码

### 子进程模式

代理在独立的进程池（`ProcessPoolExecutor`）中运行。AReaL 序列化代理和数据，在子进程中执行，然后反序列化结果。

```yaml
rollout:
  agent:
    mode: subproc
    subproc_max_workers: 4  # 进程池大小
```

**特性：**

- 代理必须可序列化（picklable）
- `OPENAI_BASE_URL` 和 `OPENAI_API_KEY` 设置为环境变量
- 代理从环境变量而非 `extra_kwargs` 读取 `base_url` 和 `api_key`
- 允许在 `run()` 中使用同步代码（AReaL 用 `asyncio.run()` 包装）
- 代理和数据的序列化开销
- 用于非异步库或进程隔离

**子进程示例：**

```python
import os
from openai import OpenAI  # 同步客户端也可以

class MySyncAgent:
    async def run(self, data, **extra_kwargs):
        # 在 subproc 模式下，base_url 和 api_key 来自环境
        client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            api_key="DUMMY",
        )

        response = client.chat.completions.create(
            model="default",
            messages=data["messages"],
        )

        return compute_reward(response, data["answer"])
```

## 架构

### 代理服务器

检测到代理工作流时，AReaL 会启动运行 FastAPI 服务器的代理工作器，实现 OpenAI 兼容端点。

```
┌─────────────────────────────────────────────────────────────────┐
│                         PPOTrainer                              │
│         （检测代理工作流，初始化代理）                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RolloutController                            │
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐                          │
│  │   Rollout    │     │    Proxy     │  FastAPI 服务器           │
│  │   Worker     │◄────│    Worker    │  /v1/chat/completions    │
│  │              │     │              │  /v1/responses           │
│  │ SGLang/vLLM  │     │              │  /v1/messages            │
│  └──────────────┘     └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

**关键文件：** `areal/experimental/openai/proxy/proxy_rollout_server.py`

### 四进程架构（代理）

代理模式在代理和推理引擎之间引入代理服务器：

```
│ 控制器进程 │  │ Rollout Worker (RPC) │  │ Proxy Worker │  │ GPU 进程 │
│                    │  │                      │  │              │  │             │
│ RolloutController  │  │  Flask HTTP 服务器    │  │ FastAPI HTTP │  │ SGLang/vLLM │
│        │           │  │        │             │  │    服务器    │  │      │      │
│        ▼           │  │   /call endpoint     │  │ OpenAI API   │  │ Inference   │
│ BatchTaskDispatcher│  │        │             │  │ 兼容         │  │   Engine    │
│   （后台线程）      │  │        ▼             │  │      │       │  │      │      │
│        │           │  │   Engine Thread      │  │      │       │  │      │      │
│        │           │  │        │             │  │      │       │  │      │      │
│        │    HTTP   │  │        ▼             │  │      │       │  │      │      │
│ submit ├────POST───┼─>│   RemoteInfEngine    │  │      │       │  │      │      │
│ task 1 │           │  │        │             │  │      │       │  │      │      │
│        │           │  │        ▼             │  │      │       │  │      │      │
│ submit │           │  │ OpenAIProxyWorkflow  │  │      │       │  │      │      │
│ task 2 │           │  │        │             │  │      │       │  │      │      │
│        │           │  │  OpenAIProxyClient ──┼──┼──────┤       │  │      │      │
│ submit │           │  │        │             │  │      │       │  │      │      │
│ task 3 │           │  │   agent.run()        │  │      │       │  │      │      │
│        │           │  │        │             │  │      │       │  │      │      │
│        │           │  │        ▼             │  │      │       │  │      │      │
│        │           │  │   OpenAI API 调用 ───┼──┼─>  /chat/ ───┼──┼─> generate  │
│        │           │  │        │             │  │ completions  │  │    tokens   │
│        │           │  │        │             │  │      │       │  │      │      │
│        │           │  │ ChatCompletion <────┼──┼──────<───────┼──┼──────┘      │
│        │           │  │        │             │  │   (cached)   │  │             │
│        │           │  │        │             │  │      │       │  │             │
│        │           │  │        ▼             │  │      │       │  │             │
│        │           │  │     reward           │  │      │       │  │             │
│        │           │  │        │             │  │      │       │  │             │
│   set_reward() ─────┼──┼─>  /rl/      │  │             │
│        │           │  │        │             │  │ set_reward   │  │             │
│        │           │  │        ▼             │  │      │       │  │             │
│        │           │  │     ...              │  │      │       │  │             │
│        │           │  │        │             │  │      │       │  │             │
│        │           │  │        ▼             │  │      │       │  │             │
│        │           │  │    trajectory        │  │      │       │  │             │
│        │           │  │        │             │  │      │       │  │             │
│    collect<────────┼──┼────────┘             │  │      │       │  │             │
│                    │  │                      │  │              │  │             │
└────────────────────┴──┴──────────────────────┴──┴──────────────┴──┴─────────────┘
```

`OpenAIProxyWorkflow` 包含一个 `OpenAIProxyClient`，管理代理服务器的会话生命周期。关键交互包括：

- **chat/completions**：将代理的 OpenAI API 调用路由到推理引擎，缓存 token 级别数据
- **set_reward**：为回复分配 RL 训练奖励

### 数据流详情

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                               Rollout Worker + Proxy Worker                                │
│                                                                                            │
│  ┌─────────────────────┐      ┌──────────────────────────────────────────────────────────┐ │
│  │ OpenAIProxyWorkflow │      │               ProxyRolloutServer (FastAPI)               │ │
│  │                     │      │                                                          │ │
│  │ 1. grant_capacity()─┼─────>│                                                          │ │
│  │                     │      │                                                          │ │
│  │ 2. start_session() ─┼─────>│ → SessionData 创建                                        │ │
│  │    ← session_id, session_api_key │ │
│  │                     │      │                                                          │ │
│  │ 3. agent.run()      │      │   ┌──────────────────────────────────────────────────┐   │ │
│  │    │                │      │   │                   ArealOpenAI                    │   │ │
│  │    └─> OpenAI 调用 ─┼─────>│   │                                                  │   │ │
│  │                     │      │   │  /chat/completions                               │   │ │
│  │                     │      │   │    → 分词，engine.agenerate() ───────────────┼───┼─┼──┐
│  │                     │      │   │    → 缓存在 InteractionCache    <──────────────┼───┼─┼──┤
│  │    ChatCompletion  <┼──────┤   │    → 返回 ChatCompletion                       │   │ │  │
│  │                     │      │   │                                                  │   │ │  │
│  │                     │      │   └──────────────────────────────────────────────────┘   │ │  │
│  │                     │      │                                                          │ │  │
│  │ 4. set_reward()    ─┼─────>│ → 奖励存储在 InteractionCache                           │ │  │
│  │                     │      │                                                          │ │  │
│  │ 5. end_session()   ─┼─────>│ → 会话标记完成                                           │ │  │
│  │                     │      │                                                          │ │  │
│  │ 6. export_          │      │                                                          │ │  │
│  │    trajectories()  ─┼─────>│ → 应用折扣，to_tensor_dict()                              │ │  │
│  │    → tensors       <┼──────┤                                                          │ │  │
│  └─────────────────────┘      └──────────────────────────────────────────────────────────┘ │  │
│                                                                                            │  │
└────────────────────────────────────────────────────────────────────────────────────────────┘  │
                                                                                                │
                                             ┌──────────────────────────────────────────────────┘
                                             │
                                             ▼
                           ┌─────────────────────────────────────────────────────────┐
                           │                  GPU Process (SGLang/vLLM)              │
                           │                                                         │
                           │   Continuous batching, KV cache, tensor parallelism     │
                           └─────────────────────────────────────────────────────────┘
```

### 代理端点

| 端点                        | 认证                                | 用途                        |
| --------------------------- | ----------------------------------- | --------------------------- |
| `POST /grant_capacity`      | Admin 密钥                          | 预留槽位（过期控制）        |
| `POST /rl/start_session`    | Admin 密钥                          | 创建唯一会话 ID             |
| `POST /v1/chat/completions` | Session 密钥                        | OpenAI chat completions API |
| `POST /v1/responses`        | Session 密钥                        | OpenAI responses API        |
| `POST /v1/messages`         | Session 密钥                        | Anthropic Messages API      |
| `POST /rl/set_reward`       | Session 密钥                        | 为交互分配奖励              |
| `POST /rl/end_session`      | Session 密钥                        | 标记会话完成                |
| `POST /export_trajectories` | Admin 密钥 + body 中的 `session_id` | 导出带奖励折扣的轨迹        |

## 会话生命周期

每个代理执行遵循以下生命周期：

```
1. 预留容量
   POST /grant_capacity → 过期控制

2. 启动会话
   POST /rl/start_session → 返回 session_id 和唯一 API 密钥

3. 代理执行（多次 LLM 调用）
   POST /v1/chat/completions（授权头中带 session API 密钥）
     → 代理服务器对消息分词
     → 引擎生成带对数概率的 token
     → 响应存储在 InteractionCache
     → ChatCompletion 返回给代理

4. 分配奖励
   POST /rl/set_reward（带 session API 密钥）
     Body: {"reward": 1.0}                           → 最后回复
     Body: {"interaction_id": "...", "reward": 0.5}  → 特定回复

5. 结束会话
   POST /rl/end_session（带 session API 密钥）

6. 导出轨迹
   POST /export_trajectories（带 admin API 密钥，body: {session_id: ..., discount: 0.9})
     → 应用奖励反向传播
     → 返回 InteractionWithTokenLogpReward 对象
     → 清理会话和 API 密钥映射
```

## Token 级别跟踪

### InteractionCache

`InteractionCache`（扩展 `OrderedDict`）存储以 completion ID 为键的
`InteractionWithTokenLogpReward` 对象。

**关键文件：** `areal/experimental/openai/cache.py`

**父子解析**：添加新交互时，缓存通过检查现有交互的消息是否为新消息的前缀来找到其父交互：

```python
# 父：[system, user]
# 子：[system, user, assistant, user]
# → 子的父设置为父
```

### InteractionWithTokenLogpReward

此数据类存储带 token 级别信息的回复数据：

```python
@dataclass
class InteractionWithTokenLogpReward:
    model_response: ModelResponse | None  # 引擎的 token ID、对数概率
    reward: float | None
    parent: InteractionWithTokenLogpReward | None
    messages: list[dict]                  # 输入消息
    output_message_list: list[dict] | None
    completion: ChatCompletion | None     # OpenAI 响应对象
```

**关键文件：** `areal/experimental/openai/types.py`

`to_tensor_dict()` 方法转换为训练格式：

```python
{
    "input_ids": torch.tensor([...], dtype=torch.int32),
    "loss_mask": torch.tensor([0]*input_len + [1]*output_len, dtype=torch.int32),
    "logprobs": torch.tensor([0]*input_len + output_logprobs, dtype=torch.float32),
    "versions": torch.tensor([...], dtype=torch.int32),
    "attention_mask": torch.ones(..., dtype=torch.bool),
    "rewards": torch.tensor([reward], dtype=torch.float32),
}
```

## 奖励系统

### 分配

奖励可以通过两种方式分配：

1. **从 `run()` 方法返回**：

   - `float`：应用于最后回复
   - `dict[str, float]`：将 completion ID 映射到奖励

1. **显式 API 调用**（直接方式）：

   ```python
   client.set_last_reward(1.0)
   client.set_reward(completion_id, 0.5)
   ```

### 反向传播

对于多轮对话，奖励通过几何折扣沿对话树向后传播：

```
# 对话树：
A → B → C（叶子，reward=1.0）

# 折扣=0.9：
C.reward = 1.0
B.reward = 0 + 1.0 × 0.9 = 0.9
A.reward = 0 + 0.9 × 0.9 = 0.81
```

按逆拓扑顺序处理（叶子优先），确保子奖励在传播给父奖励之前先确定。

### 配置

```python
# 直接方式
client.apply_reward_discount(turn_discount=0.9)
interactions = client.export_interactions(style="individual")

# 代理方式（通过导出端点）
POST /export_trajectories
Body: {"discount": 0.9, "style": "individual"}
```

## 工作流解析

将工作流传递给训练器时，AReaL 按以下方式解析：

**关键文件：** `areal/infra/remote_inf_engine.py`（`_resolve_workflow` 方法）

```python
def _resolve_workflow(workflow, workflow_kwargs, group_size, proxy_addr):
    # 1. RolloutWorkflow 实例 → 直接使用
    # 2. RolloutWorkflow 类 → 用 kwargs 实例化
    # 3. 字符串路径 → 导入并递归解析
    # 4. 有 run() 方法 → 用 OpenAIProxyWorkflow 包装

    if not isinstance(resolved, RolloutWorkflow):
        resolved = OpenAIProxyWorkflow(
            agent=resolved,
            proxy_addr=proxy_addr,
            ...
        )

    # 如需要应用分组
    if group_size > 1:
        resolved = GroupedRolloutWorkflow(resolved, group_size)

    return resolved
```

## OpenAIProxyWorkflow

`OpenAIProxyWorkflow` 将用户代理包装为 `RolloutWorkflow`：

**关键文件：** `areal/experimental/openai/proxy/workflow.py`

```python
class OpenAIProxyWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine, data):
        # 1. 授予容量
        await self._grant_capacity(http_session)

        # 2. 创建代理客户端（管理会话）
        proxy_client = OpenAIProxyClient(...)

        async with proxy_client:
            # 3. 使用会话 API 密钥运行代理
            rewards = await self._run_agent(proxy_client.session_api_key, data)

            # 4. 分配奖励
            if isinstance(rewards, float):
                await proxy_client.set_last_reward(rewards)
            elif isinstance(rewards, dict):
                for id, reward in rewards.items():
                    await proxy_client.set_reward(id, reward)

        # 5. 导出交互
        return await proxy_client.export_interactions(
            discount=self.discount,
            style=self.export_style,
        )
```

`_run_agent` 方法处理两种执行模式：

- **内联**：直接将 `agent.run()` 作为协程调用
- **子进程**：提交到 `ProcessPoolExecutor`，设置 `OPENAI_BASE_URL` 环境变量，用 `asyncio.run()` 包装

## ArealOpenAI 客户端

`ArealOpenAI` 类扩展 `AsyncOpenAI` 用于直接引擎集成：

**关键文件：** `areal/experimental/openai/client.py`

### 关键方法

| 方法                                   | 描述                 |
| -------------------------------------- | -------------------- |
| `chat.completions.create(...)`         | OpenAI 兼容聊天 API  |
| `responses.create(...)`                | OpenAI responses API |
| `set_reward(id, reward)`               | 为特定交互设置奖励   |
| `set_last_reward(reward)`              | 为最后交互设置奖励   |
| `apply_reward_discount(turn_discount)` | 应用反向奖励折扣     |
| `export_interactions(style)`           | 导出用于训练         |

### 导出样式

| 样式         | 描述                                                                |
| ------------ | ------------------------------------------------------------------- |
| `individual` | 将所有交互作为单独条目返回。轨迹可能共享前缀。                      |
| `concat`     | 构建对话树，仅返回叶子节点。仅对具有匹配 token 序列的线性对话有效。 |

## 公共 API

```python
from areal.experimental.openai import (
    ArealOpenAI,                     # 直接方式客户端
    InteractionWithTokenLogpReward,  # Token 级别数据结构
    OpenAIProxyClient,               # 代理会话的 HTTP 客户端
    OpenAIProxyWorkflow,             # 工作流包装器
)
```

## 使用代理轨迹训练

完整的代理 episode 可能包含多次 LLM 交互（轮次）。对于训练，这些被视为独立的输入-输出-奖励元组：

```
轮次 1：[system, user]                         → output_1 → reward_1（折扣后）
轮次 2：[system, user, asst, user]             → output_2 → reward_2（折扣后）
轮次 3：[system, user, asst, user, asst, user] → output_3 → reward_3（最终）
```

每个元组包含用于策略梯度计算的完整 token 级别数据：输入 token ID、输出 token ID 和对数概率。折扣奖励确保 RL 目标正确地将最终结果归因于早期行动。

### Token 一致性保证

由于 AReaL 存储推理期间使用的实际 token（而非重新分词的文本），Rollout 和训练之间不存在分词不匹配的风险。发送到推理引擎的 token 正是用于梯度计算的
token。

### 使用树注意力高效训练

多轮轨迹通常共享长 token 前缀，这可能由于冗余计算而减慢训练速度。AReaL 通过前缀共享树注意力解决这一问题，该方法仅计算一次共享前缀上的注意力。

## 另见

- [RolloutWorkflow 参考](./rollout_workflow.md) - 核心工作流抽象
- [Agentic RL Guide](../tutorial/agentic_rl.md) - 实践训练指南
- [Workflow Best Practices](../best_practices/workflow.md) - 实现技巧
