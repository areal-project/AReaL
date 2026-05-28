# SPDX-License-Identifier: Apache-2.0

"""AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"""

from .version import __version__  # noqa

from .infra import (
    RolloutController,
    StalenessManager,
    TrainController,
    WorkflowExecutor,
    current_platform,
    workflow_context,
)


def __getattr__(name: str):
    if name in ("DPOTrainer", "PPOTrainer", "RWTrainer", "SFTTrainer"):
        from .trainer import DPOTrainer, PPOTrainer, RWTrainer, SFTTrainer

        _map = {
            "DPOTrainer": DPOTrainer,
            "PPOTrainer": PPOTrainer,
            "RWTrainer": RWTrainer,
            "SFTTrainer": SFTTrainer,
        }
        globals().update(_map)
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DPOTrainer",
    "PPOTrainer",
    "RolloutController",
    "RWTrainer",
    "SFTTrainer",
    "StalenessManager",
    "TrainController",
    "WorkflowExecutor",
    "current_platform",
    "workflow_context",
]
