# 智能体强化学习

本指南演示如何使用 AReaL 配合流行的智能体框架（如
[CAMEL-AI](https://github.com/camel-ai/camel)、[OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
等）来训练智能体模型，使您能够利用它们的智能体编排能力，同时使用 AReaL 的分布式强化学习训练系统。

## 概述

AReaL 中智能体 RL
的核心设计理念是**统一的训练和部署**。换句话说，我们期望用户使用相同的代码进行训练和评估，无需任何更改。然而，这通常很困难，因为智能体框架存在以下问题：

1. **缺乏 token 级别访问**：智能体框架通过高级 API（如 OpenAI 的聊天补全 API）与语言模型交互，这些 API 不公开 RL 训练所需的 token
   ID 和对数概率。

1. **没有奖励机制**：智能体框架专为推理设计，没有内置的奖励函数。RL 训练需要奖励信号来指导策略优化。

1. **并行化有限**：标准智能体使用涉及顺序执行，难以高效收集 RL 训练所需的多样化轨迹。

AReaL 通过提供以下功能来解决这些限制：

1. **带 token 级别跟踪的代理模型客户端**：AReaL 设置了一个 HTTP 代理服务器，将所有 LLM 调用路由到 AReaL 的推理引擎（SGLang 或
   vLLM）。每次交互都会自动跟踪完整的 token 级别信息。

1. **奖励分配和传播**：AReaL 提供灵活的奖励系统，允许您将奖励分配给特定的交互或整个轨迹。系统自动构建对话树并支持奖励反向传播。

1. **并行轨迹收集**：AReaL 的工作流系统支持多个智能体实例的并发执行，允许您为每个查询收集多样化的轨迹。

我们在下面演示了几个具体示例。更多示例可以在
[`workflow/` 目录](https://github.com/areal-project/AReaL/tree/main/areal/workflow)中找到。

> **调度器兼容性**：使用代理方法的智能体工作流仅在 `local` 和 `slurm` 调度器上受支持。`ray` 调度器不支持，因为 Ray 的基于 actor
> 的编程模型与需要 worker 之间持久连接的 HTTP 代理服务器本质上不兼容。

## 示例

### 使用 OpenAI 智能体训练

#### 第 1 步：构建可运行的智能体

实现一个标准的智能体循环并进行工具调用。这段代码与 AReaL 无关——这个智能体可以使用适当的 API 密钥和基础 URL 与 OpenAI 官方模型一起运行。

将以下内容放在 AReaL 可以导入的 `my_agent.py` 中：

```python
from agents import (
    Agent,
    OpenAIProvider,
    RunConfig,
    SQLiteSession,
    function_tool,
)
from agents import Runner as OpenAIRunner
from math_verify import parse, verify
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

@function_tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@function_tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def math_reward_fn(completions: str, answer: str) -> float:
    ans = parse(completions)
    gold = parse(answer)
    return float(verify(ans, gold))

class MathAgent:
    async def run(self, data, **extra_kwargs):
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None) or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key", None) or os.getenv("OPENAI_API_KEY")
        # 重要提示：替换 `base_url` 和 `api_key`
        client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0)
        content = data["messages"][-1]["content"]
        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            model="default",  # 不需要传递
            tracing_disabled=True,
        )
        agent = Agent(
            name="RLVR Math with Calculator",
            instructions="Answer the user's math questions using the available calculator tools. Don't give the answer directly, you must use tools to do the mathematical calculation.",
            tools=[add, multiply],
        )
        session = SQLiteSession("math")
        result = await OpenAIRunner.run(
            agent, input=content, session=session, run_config=run_config
        )
        # 返回奖励
        return math_reward_fn(completions=result.final_output, answer=data["answer"])
```

#### 第 2 步：集成智能体

将智能体路径传递给训练器：

```python
import sys

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config, GRPOConfig
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer

def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset
    ) as trainer:
        trainer.train(
            workflow="my_agent.MathAgent"
        )

if __name__ == "__main__":
    main(sys.argv[1:])
```

完整的 OpenAI Agents 训练示例位于
[**`examples/agent_workflow/`**](https://github.com/areal-project/AReaL/blob/main/examples/agent_workflow/)。要在单节点上运行示例：

```bash
python3 examples/agent_workflow/train.py \
    --config examples/agent_workflow/config.yaml \
    scheduler.type=local workflow=my_agent.MathAgent
```

### 使用 CAMEL-AI 训练

> **旧模式注意**：使用 `ArealOpenAI` 与 `RolloutWorkflow` 的直接方法被视为旧模式。对于新项目，建议使用代理方法（如上方的 OpenAI
> 智能体示例），这样可以让您的智能体代码独立于 AReaL。

CAMEL-AI 是一个开源的模块化框架，用于构建智能多智能体系统。它提供灵活的智能体架构，可以处理复杂的对话流程、工具调用和多智能体交互。

#### 第 1 步：编写 CAMEL 智能体

一个典型的 CAMEL 智能体编写起来很简单：

```python
from camel.agents import ChatAgent

# 创建一个基本的 CAMEL 智能体
agent = ChatAgent(
    system_message="You are a helpful math assistant.",
    model="gpt-4o-mini",
)

# 运行智能体
response = await agent.astep("Solve: 2 + 2 = ?")
print(response.msg.content)
```

#### 第 2 步：转换为可训练的 RL 智能体

要使这个智能体可以与 AReaL 一起训练，请将模型替换为 AReaL 的 OpenAI 兼容模型：

```python
from camel.agents import ChatAgent
from areal.experimental.camel.openai_model import AReaLOpenAICompatibleModel
from areal.experimental.openai import ArealOpenAI

# 创建 AReaL 的 OpenAI 兼容客户端
client = ArealOpenAI(engine=engine, tokenizer=tokenizer)

# 将模型替换为 AReaL 的 OpenAI 兼容模型
agent = ChatAgent(
    system_message="You are a helpful math assistant.",
    model=AReaLOpenAICompatibleModel(
        openai_client=client,
        tokenizer=tokenizer,
        model_type="areal"
    )
)

# 现在客户端（ArealOpenAI）会记录 token 级别的信息
response = await agent.astep("Solve: 2 + 2 = ?")
```

#### 第 3 步：添加奖励评估

智能体响应后，检查答案是否正确并设置奖励：

```python
def math_reward_fn(result, answer):
    """简单的奖励函数：正确返回 1.0，否则返回 0.0。"""
    return 1.0 if result.strip() == answer.strip() else 0.0

# 运行智能体
response = await agent.astep("Solve: 2 + 2 = ?")

# 评估并设置奖励
reward = math_reward_fn(response.msg.content, "4")
client.set_last_reward(reward)
```

#### 第 4 步：将智能体包装为可重用类

要将智能体集成到 AReaL 的训练管道中，请将其包装在一个管理智能体生命周期和奖励评估的类中：

```python
from areal.api.reward_api import AsyncRewardWrapper
from transformers import PreTrainedTokenizerFast

class CamelMathAgent:
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer
        # 为异步执行包装奖励函数
        self.async_reward_fn = AsyncRewardWrapper(math_reward_fn)

    async def run_agent(self, data, client: ArealOpenAI):
        """在数据集样本上运行智能体。"""
        # 使用 AReaL OpenAI 兼容模型创建智能体
        agent = ChatAgent(
            system_message="You are a helpful math assistant.",
            model=AReaLOpenAICompatibleModel(...),
        )

        # 运行智能体
        response = await agent.astep(data["messages"][-1]["content"])
        content = response.msg.content

        # 评估奖励并在客户端上设置
        reward = await self.async_reward_fn(result=content, answer=data["answer"])
        client.set_last_reward(reward)

        return reward
```

#### 第 5 步：创建 rollout 工作流

将智能体集成到 AReaL 的 `RolloutWorkflow` 中：

```python
from areal.api.workflow_api import RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters

class CamelRLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.agent = CamelMathAgent(tokenizer=self.tokenizer)

    async def arun_episode(self, engine, data):
        """运行一个训练 episode。"""
        # 每个轨迹创建一个客户端
        client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)

        # 运行智能体
        reward = await self.agent.run_agent(data=data, client=client)

        # 对多轮对话应用奖励折扣
        client.apply_reward_discount(turn_discount=0.9)
        # 导出带有 token 级别数据的交互
        return client.export_interactions(style="individual")
```

**关键点：**

- **并行 episode 执行**：AReaL 的训练循环在多个样本上并行调用 `arun_episode`
- **奖励折扣**：对于多轮对话，奖励会通过对话树向后折扣
- **交互导出**：所有带有 token 级别数据和奖励的交互都会导出用于 RL 训练

#### 第 6 步：运行训练

在 AReaL 的训练循环中使用该工作流：

```python
workflow = CamelRLVRWorkflow(
    gconfig=config.gconfig,
    tokenizer=tokenizer,
)

# AReaL 将为每个批次调用 workflow.arun_episode()
```

查看[完整的训练脚本](https://github.com/areal-project/AReaL/blob/main/examples/camel/train.py)以获取完整的工作实现。

### 更多示例

除了上述两个示例外，AReaL 还支持与各种其他智能体框架和 SDK 的集成：

- **Claude Agent SDK**：使用 Anthropic 的 Claude Agent SDK 和 MCP
  工具训练智能体。请参阅[Claude 示例](https://github.com/areal-project/AReaL/blob/main/areal/workflow/anthropic/claude_math_agent.py)，了解带有计算器工具的数学智能体。

- **LangChain**：将 LangChain 智能体与 AReaL 的训练基础设施集成。请参阅
  [LangChain 示例](https://github.com/areal-project/AReaL/blob/main/areal/workflow/langchain/math_agent.py)了解详情。

## 底层原理

有关 AReaL 智能体训练基础设施的详细解释，包括代理服务器架构、会话生命周期、token
级别跟踪和奖励反向传播，请参阅[智能体工作流参考](../reference/agent_workflow.md)。

参考文档涵盖的关键主题：

- **两种集成范式**：代理方法与直接方法
- **架构**：代理服务器、端点和推理引擎层
- **会话生命周期**：容量预留、会话管理和导出
- **Token 级别跟踪**：`InteractionCache` 和 `InteractionWithTokenLogpReward`
- **奖励系统**：分配方法和反向传播算法
- **工作流解析**：AReaL 如何检测和包装智能体工作流
