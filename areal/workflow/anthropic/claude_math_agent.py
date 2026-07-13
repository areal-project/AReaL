# SPDX-License-Identifier: Apache-2.0

import math
import os
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
    create_sdk_mcp_server,
    tool,
)
from math_verify import parse, verify

from areal.api import AsyncRewardWrapper


def math_reward_fn(completions: str, answer: str) -> float:
    ans = parse(completions)
    gold = parse(answer)
    return float(verify(ans, gold))


@tool("add", "Add two numbers", {"a": float, "b": float})
async def add(args: dict[str, Any]) -> dict[str, Any]:
    result = args["a"] + args["b"]
    return {"content": [{"type": "text", "text": str(result)}]}


@tool("subtract", "Subtract two numbers", {"a": float, "b": float})
async def subtract(args: dict[str, Any]) -> dict[str, Any]:
    result = args["a"] - args["b"]
    return {"content": [{"type": "text", "text": str(result)}]}


@tool("multiply", "Multiply two numbers", {"a": float, "b": float})
async def multiply(args: dict[str, Any]) -> dict[str, Any]:
    result = args["a"] * args["b"]
    return {"content": [{"type": "text", "text": str(result)}]}


@tool("divide", "Divide two numbers (b must not be zero)", {"a": float, "b": float})
async def divide(args: dict[str, Any]) -> dict[str, Any]:
    if args["b"] == 0:
        return {
            "content": [{"type": "text", "text": "Error: Division by zero"}],
            "is_error": True,
        }
    result = args["a"] / args["b"]
    return {"content": [{"type": "text", "text": str(result)}]}


@tool("power", "Raise base to the exponent power", {"base": float, "exponent": float})
async def power(args: dict[str, Any]) -> dict[str, Any]:
    result = args["base"] ** args["exponent"]
    return {"content": [{"type": "text", "text": str(result)}]}


@tool("sqrt", "Calculate square root of a non-negative number", {"n": float})
async def sqrt(args: dict[str, Any]) -> dict[str, Any]:
    if args["n"] < 0:
        return {
            "content": [{"type": "text", "text": "Error: Cannot sqrt negative number"}],
            "is_error": True,
        }
    result = math.sqrt(args["n"])
    return {"content": [{"type": "text", "text": str(result)}]}


# Create MCP server with calculator tools.
calculator_server = create_sdk_mcp_server(
    name="calc",
    version="1.0.0",
    tools=[add, subtract, multiply, divide, power, sqrt],
)

# List of allowed MCP tools: mcp__<server_name>__<tool_name>
CALCULATOR_MCP_TOOLS = [
    "mcp__calc__add",
    "mcp__calc__subtract",
    "mcp__calc__multiply",
    "mcp__calc__divide",
    "mcp__calc__power",
    "mcp__calc__sqrt",
]


class MathToolAgent:
    """Math agent with calculator tools using claude_agent_sdk"""

    SYSTEM_PROMPT = (
        "Answer the user's math questions using the available calculator tools. "
        "Don't give the answer directly, you must use tools to do the mathematical calculation."
    )

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def run(self, data: dict, **extra_kwargs) -> float:
        """Run the agent on a math problem.

        Args:
            data: Input data containing "messages" and "answer"
            **extra_kwargs: Contains base_url, api_key, and http_client from proxy (inline mode)

        Returns:
            Reward value for this trajectory
        """

        content = data["messages"][-1]["content"]

        base_url = extra_kwargs.get("base_url", None) or os.getenv("ANTHROPIC_BASE_URL")
        api_key = extra_kwargs.get("api_key", None) or os.environ.get(
            "ANTHROPIC_API_KEY"
        )

        # Set SDK timeout in current process BEFORE creating ClaudeSDKClient
        # The SDK reads CLAUDE_CODE_STREAM_CLOSE_TIMEOUT from os.environ during __init__,
        # not from ClaudeAgentOptions.env (which only passes to subprocess)
        os.environ["CLAUDE_CODE_STREAM_CLOSE_TIMEOUT"] = "300000"  # 300s timeout

        env = {}
        if base_url:
            env.update(
                {
                    "ANTHROPIC_BASE_URL": base_url,
                    "ANTHROPIC_API_KEY": api_key,
                    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(
                        extra_kwargs.get("max_completion_tokens", 1024)
                    ),
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1",
                    "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1",
                }
            )

        # Configure agent options with MCP server for calculator tools
        options = ClaudeAgentOptions(
            system_prompt=self.SYSTEM_PROMPT,
            tools=CALCULATOR_MCP_TOOLS,
            mcp_servers={"calc": calculator_server},
            allowed_tools=CALCULATOR_MCP_TOOLS,
            max_turns=self.kwargs.get("max_turns", 10),
            env=env,
        )

        # Use ClaudeSDKClient for bidirectional, interactive conversations
        final_output = ""
        async with ClaudeSDKClient(options=options) as client:
            await client.query(content)

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            final_output += block.text

        reward_fn = AsyncRewardWrapper(math_reward_fn)
        reward = await reward_fn(completions=final_output, answer=data["answer"])
        return reward
