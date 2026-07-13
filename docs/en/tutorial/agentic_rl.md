# Agentic Reinforcement Learning

This guide demonstrates how to use AReaL to train agentic models with popular agent
frameworks, e.g., [CAMEL-AI](https://github.com/camel-ai/camel),
[OpenAI Agents SDK](https://github.com/openai/openai-agents-python), etc, enabling you
to leverage their agent orchestration capabilities while using AReaL's distributed
reinforcement learning training system.

## Overview

The core design philosophy of agentic RL in AReaL is **unified training and
deployment**. In other words, we expect the user to use the same code for both training
and evaluation without any change. However, this is usually difficult because agent
frameworks:

1. **Lack token-level access**: Agent frameworks interact with language models through
   high-level APIs (e.g., OpenAI's chat completion API), which do not expose token IDs
   and log probabilities needed for RL training.

1. **Have no reward mechanism**: Agent frameworks are designed for inference and do not
   have built-in reward functions. RL training requires reward signals to guide policy
   optimization.

1. **Have limited parallelization**: Standard agent usage involves sequential execution,
   making it difficult to efficiently collect diverse trajectories needed for RL
   training.

AReaL addresses these limitations by providing:

1. **Proxy model client with token-level tracking**: AReaL sets up an HTTP proxy server
   that routes all LLM calls to AReaL's inference engine (SGLang or vLLM). Every
   interaction is automatically tracked with complete token-level information.

1. **Reward assignment and propagation**: AReaL provides a flexible reward system that
   allows you to assign rewards to specific interactions or entire trajectories. The
   system automatically builds conversation trees and supports reward backpropagation.

1. **Parallel trajectory collection**: AReaL's workflow system enables concurrent
   execution of multiple agent instances, allowing you to collect diverse trajectories
   for each query.

We demonstrate several concrete examples below. More examples can be found in the
[`workflow/` directory](https://github.com/areal-project/AReaL/tree/main/areal/workflow).

> **Scheduler Compatibility**: Agent workflows with the proxy approach are supported on
> `local` and `slurm` schedulers only. The `ray` scheduler is not supported because
> Ray's actor-based programming model is inherently incompatible with HTTP proxy servers
> that require persistent connections between workers.

## Examples

### Training with OpenAI Agent

#### Step 1: Build a Runnable Agent

Implement a standard agent loop with tool calling. This code snippet has nothing to do
with AReaL—this agent can run with OpenAI's official models with a proper API key and
base URL.

Place the following content in `my_agent.py` where AReaL can import:

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
        # IMPORTANT: replace the `base_url` and `api_key`
        client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0)
        content = data["messages"][-1]["content"]
        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            model="default",  # no need to pass
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
        # return reward
        return math_reward_fn(completions=result.final_output, answer=data["answer"])
```

#### Step 2: Integrate the Agent

Pass the agent path to the trainer:

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

The full working OpenAI Agents training example is located in
[**`examples/agent_workflow/`**](https://github.com/areal-project/AReaL/blob/main/examples/agent_workflow/).
To run the example on a single node:

```bash
python3 examples/agent_workflow/train.py \
    --config examples/agent_workflow/config.yaml \
    scheduler.type=local workflow=my_agent.MathAgent
```

### Training with CAMEL-AI

> **Legacy Pattern**: The direct approach using `ArealOpenAI` with `RolloutWorkflow` is
> considered legacy. For new projects, prefer the proxy approach (like the OpenAI Agent
> example above) which keeps your agent code independent from AReaL.

CAMEL-AI is an open-source, modular framework for building intelligent multi-agent
systems. It provides a flexible agent architecture that can handle complex dialogue
flows, tool calling, and multi-agent interactions.

#### Step 1: Write a CAMEL Agent

A typical CAMEL agent is straightforward to write:

```python
from camel.agents import ChatAgent

# Create a basic CAMEL agent
agent = ChatAgent(
    system_message="You are a helpful math assistant.",
    model="gpt-4o-mini",
)

# Run the agent
response = await agent.astep("Solve: 2 + 2 = ?")
print(response.msg.content)
```

#### Step 2: Convert to an RL-Trainable Agent

To make this agent trainable with AReaL, replace the model with AReaL's
OpenAI-compatible model:

```python
from camel.agents import ChatAgent
from areal.experimental.camel.openai_model import AReaLOpenAICompatibleModel
from areal.experimental.openai import ArealOpenAI

# Create AReaL's OpenAI-compatible client
client = ArealOpenAI(engine=engine, tokenizer=tokenizer)

# Replace the model with AReaL's OpenAI-compatible model
agent = ChatAgent(
    system_message="You are a helpful math assistant.",
    model=AReaLOpenAICompatibleModel(
        openai_client=client,
        tokenizer=tokenizer,
        model_type="areal"
    )
)

# Now the client (ArealOpenAI) records token-level information
response = await agent.astep("Solve: 2 + 2 = ?")
```

#### Step 3: Add Reward Evaluation

After the agent responds, check if the answer is correct and set the reward:

```python
def math_reward_fn(result, answer):
    """Simple reward function: 1.0 if correct, 0.0 otherwise."""
    return 1.0 if result.strip() == answer.strip() else 0.0

# Run the agent
response = await agent.astep("Solve: 2 + 2 = ?")

# Evaluate and set reward
reward = math_reward_fn(response.msg.content, "4")
client.set_last_reward(reward)
```

#### Step 4: Wrap the Agent in a Reusable Class

To integrate the agent into AReaL's training pipeline, wrap it in a class that manages
the agent lifecycle and reward evaluation:

```python
from areal.api.reward_api import AsyncRewardWrapper
from transformers import PreTrainedTokenizerFast

class CamelMathAgent:
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer
        # Wrap reward function for async execution
        self.async_reward_fn = AsyncRewardWrapper(math_reward_fn)

    async def run_agent(self, data, client: ArealOpenAI):
        """Run agent on a dataset sample."""
        # Create agent with AReaL OpenAI-compatible model
        agent = ChatAgent(
            system_message="You are a helpful math assistant.",
            model=AReaLOpenAICompatibleModel(...),
        )

        # Run agent
        response = await agent.astep(data["messages"][-1]["content"])
        content = response.msg.content

        # Evaluate reward and set on client
        reward = await self.async_reward_fn(result=content, answer=data["answer"])
        client.set_last_reward(reward)

        return reward
```

#### Step 5: Create the Rollout Workflow

Integrate the agent into AReaL's `RolloutWorkflow`:

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
        """Run one training episode."""
        # Create one client per trajectory
        client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)

        # Run agent
        reward = await self.agent.run_agent(data=data, client=client)

        # Apply reward discounting for multi-turn conversations
        client.apply_reward_discount(turn_discount=0.9)
        # Export interactions with token-level data
        return client.export_interactions(style="individual")
```

**Key points:**

- **Parallel episode execution**: AReaL's training loop calls `arun_episode` in parallel
  across multiple samples.
- **Reward discounting**: For multi-turn conversations, rewards are discounted backward
  through the conversation tree.
- **Interactions export**: All interactions with token-level data and rewards are
  exported for RL training.

#### Step 6: Run the Training

Use the workflow in AReaL's training loop:

```python
workflow = CamelRLVRWorkflow(
    gconfig=config.gconfig,
    tokenizer=tokenizer,
)

# AReaL will call workflow.arun_episode() for each batch
```

See the
[complete train script](https://github.com/areal-project/AReaL/blob/main/examples/camel/train.py)
for a full working implementation.

### More Examples

Beyond the two examples above, AReaL supports integration with various other agent
frameworks and SDKs:

- **Claude Agent SDK**: Train agents using Anthropic's Claude Agent SDK with MCP tools.
  See the
  [Claude example](https://github.com/areal-project/AReaL/blob/main/areal/workflow/anthropic/claude_math_agent.py)
  for a math agent with calculator tools.

- **LangChain**: Integrate LangChain agents with AReaL's training infrastructure. See
  the
  [LangChain example](https://github.com/areal-project/AReaL/blob/main/areal/workflow/langchain/math_agent.py)
  for details.

## Online Mode

For scenarios where the agent code runs _outside_ of AReaL (e.g., external agent
runtimes, human evaluators, or third-party services), use **online proxy mode**. AReaL
exposes an OpenAI-compatible HTTP API that any application can connect to, and all
interactions are automatically collected as RL training data.

See the [Online RL Training Guide](online_proxy.md) for a complete walkthrough, and the
[OpenClaw example](https://github.com/areal-project/AReaL/tree/main/examples/openclaw)
for a working end-to-end demo.

## Under the Hood

For a detailed explanation of how AReaL's agentic training infrastructure works,
including the proxy server architecture, session lifecycle, token-level tracking, and
reward backpropagation, see the
[Agent Workflow Reference](../reference/agent_workflow.md).

Key topics covered in the reference:

- **Two integration paradigms**: Proxy approach vs. direct approach
- **Architecture**: Proxy server, endpoints, and inference engine layers
- **Session lifecycle**: Capacity reservation, session management, and export
- **Token-level tracking**: `InteractionCache` and `InteractionWithTokenLogpReward`
- **Reward system**: Assignment methods and backpropagation algorithm
- **Workflow resolution**: How AReaL detects and wraps agent workflows
