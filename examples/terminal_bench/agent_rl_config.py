from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


@dataclass
class TaskTimeouts:
    _reset_env: float = 1800.0
    _reset_agent: float = 120.0
    agent_astep: float = 300.0
    _evaluate_completion_sync: float = 1200.0
    _cleanup: float | None = None


@dataclass
class AgentRLConfig(GRPOConfig):
    n_trajs: int = field(default=1)
    max_tokens_per_trajectory: int = field(default=32768)
    max_iteration: int = field(default=3)
    max_workers: int = field(default=25)
    non_think_mode: bool = field(default=True)
    async_training: bool = field(default=False)
    task_timeouts: TaskTimeouts = field(default_factory=TaskTimeouts)
    filter_uniform_reward: bool = field(default=False)
    encourage_completion_reward: bool = field(default=False)
