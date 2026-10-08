"""Configuration for the τ³-Bench Qwen3.5 GRPO example."""

from dataclasses import dataclass, field

from examples.multi_turn_math.config import MultiTurnGRPOConfig
from examples.tau3.agent import Tau3EnvConfig


@dataclass
class Tau3GRPOConfig(MultiTurnGRPOConfig):
    """GRPO configuration with text τ³-Bench simulator options."""

    econfig: Tau3EnvConfig = field(default_factory=Tau3EnvConfig)
