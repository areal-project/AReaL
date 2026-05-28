from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


@dataclass
class MultiTurnGRPOConfig(GRPOConfig):
    agent_run_args: dict = field(
        default_factory=dict,
        metadata={"help": "Arguments for running the agent."},
    )
    export_style: str = field(
        default="concat",
        metadata={
            "help": "Export style for the completions. By default export_style=concat."
        },
    )
