from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


@dataclass
class ALFWorldConfig(GRPOConfig):
    agent_run_args: dict = field(
        default_factory=lambda: {"max_steps": 30},
        metadata={"help": "Arguments for running the ALFWorld agent."},
    )
    export_style: str = field(
        default="concat",
        metadata={"help": "Export style for the completions."},
    )
    turn_discount: float = field(
        default=0.9,
        metadata={"help": "Reward discount across turns."},
    )
