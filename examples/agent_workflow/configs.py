from dataclasses import dataclass, field

from areal.api.cli_args import PPOConfig


@dataclass
class AgentConfig(PPOConfig):
    """Configuration for Agent training experiments.

    Extends PPOConfig with agent-specific settings for agent-based
    workflows using OpenAI-compatible APIs (OpenAI, Anthropic, LangChain, etc.).
    """

    workflow: str = field(
        default="areal.workflow.langchain.math_agent.MathAgent",
        metadata={"help": "Path to the workflow class for training."},
    )
    eval_workflow: str = field(
        default="areal.workflow.langchain.math_agent.MathAgent",
        metadata={"help": "Path to the workflow class for evaluation."},
    )
    max_turns: int = field(
        default=10,
        metadata={"help": "Maximum turns for multi-turn agents."},
    )
