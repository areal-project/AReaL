from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


@dataclass
class AgentRLConfig(GRPOConfig):
    reward_fn_path: str = field(
        default="areal.reward.gsm8k.gsm8k_reward_fn",
        metadata={
            "help": "The path to the reward function. Should follow the API in `areal/api/reward_api.py`."
        },
    )
    agent_builder_path: str = field(
        default="areal.workflow.openai_agent.math_agent.build_math_agent",
        metadata={
            "help": "The path to the OpenAI agent builder. The function should return an `Agent` object with OpenAI SDK."
        },
    )
    agent_builder_kwargs: dict = field(
        default_factory=dict,
        metadata={
            "help": "The initialization arguments for the agent builder function."
        },
    )
