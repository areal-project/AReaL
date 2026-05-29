# SPDX-License-Identifier: Apache-2.0

"""AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"""

from .version import __version__  # noqa


_INFRA_NAMES = frozenset(
    {
        "RolloutController",
        "StalenessManager",
        "TrainController",
        "WorkflowExecutor",
        "current_platform",
        "workflow_context",
    }
)

_TRAINER_NAMES = frozenset(
    {
        "DPOTrainer",
        "PPOTrainer",
        "RWTrainer",
        "SFTTrainer",
    }
)


def __getattr__(name: str):
    if name in _INFRA_NAMES:
        from . import infra as _infra

        value = getattr(_infra, name)
        globals()[name] = value
        return value
    if name in _TRAINER_NAMES:
        from . import trainer as _trainer

        value = getattr(_trainer, name)
        globals()[name] = value
        return value
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
