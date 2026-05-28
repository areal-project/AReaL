"""Utilities for Tau2 benchmark training with AReaL."""

import sys
from dataclasses import dataclass, field

import tau2.utils.llm_utils
import yaml
from litellm import completion_cost
from litellm.main import ModelResponse
from loguru import logger
from pydantic import BaseModel
from tau2.data_model.message import Message
from tau2.data_model.simulation import RewardInfo
from tau2.data_model.tasks import Task

from areal.api.cli_args import PPOConfig


@dataclass
class Tau2EnvConfig:
    """Environment configuration for Tau2 benchmark."""

    domain: str = field(
        default="telecom",
        metadata={
            "help": "The tau2 domain name, e.g., 'retail', 'airline', 'telecom'."
        },
    )
    max_steps: int = field(
        default=100, metadata={"help": "Maximum number of steps per episode."}
    )
    add_thinking_tool: bool = field(
        default=False, metadata={"help": "Whether to add a thinking tool."}
    )
    solo_mode: bool = field(
        default=False, metadata={"help": "Whether to use solo mode."}
    )
    user_llm_base_url: str | None = field(
        default=None,
        metadata={"help": "The base URL of the user LLM."},
    )
    user_llm: str | None = field(
        default=None,
        metadata={"help": "The user LLM to use, default to the gpt-4.1 model."},
    )
    user_llm_args: dict | None = field(
        default=None, metadata={"help": "The arguments for the user LLM."}
    )
    turn_discount: float = field(
        default=1.0, metadata={"help": "Discount factor for turn-based learning."}
    )
    invalid_format_penalty: float = field(
        default=0.1, metadata={"help": "Penalty for invalid format in completions."}
    )


@dataclass
class Tau2PPOConfig(PPOConfig):
    """PPO configuration with Tau2-specific settings."""

    econfig: Tau2EnvConfig = field(default_factory=Tau2EnvConfig)


# Configure loguru logger for tau2-bench package
# This runs at import time, so workers will also have this configuration
logger.remove()
# Log to stderr by default, will be captured by the worker's log system
logger.add(sys.stderr, level="INFO", format="{time} {level} {message}")


def _get_response_cost_silent(response: ModelResponse) -> float:
    """Get cost from response, silently returning 0.0 for unmapped models.

    This is a patched version of tau2.utils.llm_utils.get_response_cost that
    suppresses the error log when LiteLLM doesn't have pricing info for a model
    (e.g., self-hosted models like 'openai/self-hosted-Qwen2.5-72B').

    The original function logs an error via logger.error(e) which is noisy.
    This version silently returns 0.0 for unmapped models.
    """
    # Parse fine-tuned model names (reuse tau2's helper)
    response.model = tau2.utils.llm_utils._parse_ft_model_name(response.model)
    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        # Silently return 0.0 for unmapped models (e.g., self-hosted models)
        return 0.0
    return cost


# Patch tau2.utils.llm_utils.get_response_cost with our silent version
tau2.utils.llm_utils.get_response_cost = _get_response_cost_silent


class Tau2RunInfo(BaseModel):
    """Information about a Tau2 simulation run."""

    reward: float
    agent_time: list[float]
    user_time: list[float]
    messages: list[Message]
    task: Task
    reward_info: RewardInfo | None = None
    error: str | None = None

    def __str__(self):
        s = f"[REWARD]: {self.reward}\n\n"
        s += "[TASK]\n"
        s += yaml.dump(self.task.model_dump()) + "\n"
        if self.reward_info:
            s += "[REWARD_INFO]\n"
            s += yaml.dump(self.reward_info.model_dump()) + "\n"
        s += f"[TURNS COUNT]: {len(self.messages)}\n"
        s += "[MESSAGES]\n"
        for message in self.messages:
            turn_idx = message.turn_idx
            role = message.role
            content = message.content or ""
            usage = getattr(message, "usage", {})
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                content += "\n[TOOL_CALLS]\n"
                content += yaml.dump(
                    [tool_call.model_dump() for tool_call in tool_calls]
                )
            s += f"[{turn_idx}][{role}]: {content}\n"
            if usage:
                s += f"[{turn_idx}][{role}][USAGE]: {yaml.dump(usage)}\n"
        if len(self.agent_time):
            s += f"[AGENT_TIME]: total {sum(self.agent_time)}, avg {sum(self.agent_time) / len(self.agent_time)}\n"
        if len(self.user_time):
            s += f"[USER_TIME]: total {sum(self.user_time)}, avg {sum(self.user_time) / len(self.user_time)}\n"
        if self.error:
            s += f"[ERROR]: {self.error}\n"
        return s
