# SPDX-License-Identifier: Apache-2.0

import os

from agents import (
    Agent,
    ModelSettings,
    OpenAIProvider,
    RunConfig,
    SQLiteSession,
    function_tool,
)
from agents import Runner as OpenAIRunner
from math_verify import parse, verify
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from areal.api import AsyncRewardWrapper


def math_reward_fn(completions: str, answer: str) -> float:
    ans = parse(completions)
    gold = parse(answer)
    return float(verify(ans, gold))


class MathAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()
        self.kwargs.pop("max_tokens", None)
        self.kwargs.pop("max_turns", None)
        self._reward_fn = AsyncRewardWrapper(math_reward_fn)

    async def run(self, data: dict, **extra_kwargs):
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None) or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key", None) or os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0
        )
        comp: ChatCompletion = await client.chat.completions.create(
            messages=data["messages"], model="default", **self.kwargs
        )

        return await self._reward_fn(
            completions=comp.choices[0].message.content, answer=data["answer"]
        )


class MultiTurnMathAgent:
    def __init__(self, max_turns: int = 8, **kwargs):
        self.max_turns = max_turns
        self.kwargs = kwargs.copy()
        self.kwargs.pop("max_tokens", None)
        self._reward_fn = AsyncRewardWrapper(math_reward_fn)

    async def run(self, data: dict, **extra_kwargs):
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None) or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key", None) or os.getenv("OPENAI_API_KEY")
        messages = data["messages"].copy()
        rewards = {}
        client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0
        )
        for _ in range(self.max_turns):
            response: ChatCompletion = await client.chat.completions.create(
                messages=messages,
                model="default",
                **self.kwargs,
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            reward = await self._reward_fn(
                completions=message.content, answer=data["answer"]
            )
            rewards[response.id] = reward
            if reward == 1:
                break
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": "Your answer is either wrong or not parsable to the reward function. You may misunderstand the original question. "
                        "Please carefully read the original question, check the previous errors, and try to answer it again.",
                    }
                )
        return rewards


@function_tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@function_tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b


@function_tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@function_tool
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


@function_tool
def power(a: float, b: float) -> float:
    """Raise a to the power of b."""
    return a**b


@function_tool
def sqrt(a: float) -> float:
    """Calculate the square root of a number."""
    if a < 0:
        raise ValueError("Cannot compute square root of a negative number.")
    return a**0.5


class MathToolAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()
        self.kwargs.pop("max_tokens", None)
        self.kwargs.pop("max_turns", None)
        self._reward_fn = AsyncRewardWrapper(math_reward_fn)

    async def run(self, data: dict, **extra_kwargs):
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None) or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key", None) or os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0
        )
        content = data["messages"][-1]["content"]
        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            model="default",
            tracing_disabled=True,
            model_settings=ModelSettings(**self.kwargs),
        )
        agent = Agent(
            name="RLVR Math with Calculator",
            instructions="Answer the user's math questions using the available calculator tools. Don't give the answer directly, you must use tools to do the mathematical calculation.",
            tools=[
                add,
                subtract,
                multiply,
                divide,
                power,
                sqrt,
            ],
        )
        session = SQLiteSession("math")
        result = await OpenAIRunner.run(
            agent, input=content, session=session, run_config=run_config
        )

        return await self._reward_fn(
            completions=result.final_output, answer=data["answer"]
        )
