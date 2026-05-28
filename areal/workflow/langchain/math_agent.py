# SPDX-License-Identifier: Apache-2.0

"""LangChain-based math agent implementation for AReaL training.

This module provides math agents using LangChain that integrate with AReaL's
proxy-based training infrastructure via the AgentWorkflow pattern.
"""

import math
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from math_verify import parse, verify

from areal.api import AsyncRewardWrapper


def math_reward_fn(completions: str, answer: str) -> float:
    """Calculate reward based on math answer correctness."""
    ans = parse(completions)
    gold = parse(answer)
    return float(verify(ans, gold))


# Tool definitions using LangChain's @tool decorator
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide the first number by the second."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


@tool
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return base**exponent


@tool
def sqrt(n: float) -> float:
    """Calculate the square root of a number."""
    if n < 0:
        raise ValueError("Cannot compute square root of a negative number.")
    return math.sqrt(n)


CALCULATOR_TOOLS = [add, subtract, multiply, divide, power, sqrt]


class MathAgent:
    """Simple single-turn math agent using LangChain ChatOpenAI.

    This agent uses the AReaL proxy to generate responses and calculates
    rewards based on math answer correctness.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()

    async def run(self, data: dict, **extra_kwargs) -> float:
        """Run the agent on a single problem.

        Args:
            data: Input data containing "messages" and "answer"
            **extra_kwargs: Contains base_url, api_key, and http_client from proxy

        Returns:
            Reward value for this trajectory
        """
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None) or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key", None) or os.getenv("OPENAI_API_KEY")

        # Build LangChain ChatOpenAI with proxy settings
        llm = ChatOpenAI(
            model="default",
            base_url=base_url,
            api_key=api_key,
            temperature=self.kwargs.get("temperature", 1.0),
            top_p=self.kwargs.get("top_p", 1.0),
            max_tokens=self.kwargs.get("max_completion_tokens", 1024),
            http_async_client=http_client,
        )

        # Use ainvoke for async execution
        response = await llm.ainvoke(data["messages"])
        completion_text = response.content or ""

        # Calculate reward
        reward_fn = AsyncRewardWrapper(math_reward_fn)
        return await reward_fn(completions=completion_text, answer=data["answer"])


class MathToolAgent:
    """Math agent with calculator tools using LangChain's create_agent.

    This agent uses LangChain's official agent API with calculator tools
    and integrates with AReaL's proxy for training.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()

    async def run(self, data: dict, **extra_kwargs) -> float:
        """Run the agent with tool-calling capabilities.

        Args:
            data: Input data containing "messages" and "answer"
            **extra_kwargs: Contains base_url, api_key, and http_client from proxy

        Returns:
            Reward value for this trajectory
        """
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None) or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key", None) or os.getenv("OPENAI_API_KEY")

        # Build LangChain ChatOpenAI with proxy settings
        llm = ChatOpenAI(
            model="default",
            base_url=base_url,
            api_key=api_key,
            temperature=self.kwargs.get("temperature", 1.0),
            top_p=self.kwargs.get("top_p", 1.0),
            max_tokens=self.kwargs.get("max_completion_tokens", 1024),
            http_async_client=http_client,
        )

        # Create agent using LangChain's create_agent function
        system_prompt = (
            "You are a math assistant with calculator tools. "
            "You MUST use the provided calculator tools (add, subtract, multiply, divide, power, sqrt) "
            "to perform ALL mathematical calculations. Do not calculate in your head. "
            "Show your step-by-step reasoning and use tools for each calculation. "
            "After completing calculations, provide your final answer."
        )

        agent = create_agent(
            model=llm,
            tools=CALCULATOR_TOOLS,
            system_prompt=system_prompt,
        )

        # Extract user content from messages
        user_content = data["messages"][-1]["content"]

        # Run the agent
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_content}]}
        )

        # Extract the final response
        final_message = result["messages"][-1]
        final_answer = (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )

        # Calculate reward
        reward_fn = AsyncRewardWrapper(math_reward_fn)
        return await reward_fn(completions=final_answer, answer=data["answer"])
