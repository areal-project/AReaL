# SPDX-License-Identifier: Apache-2.0

"""AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"""

import importlib

from .version import __version__  # noqa

_lazy_exports = {
    "RolloutController": ("areal.infra", "RolloutController"),
    "StalenessManager": ("areal.infra", "StalenessManager"),
    "TrainController": ("areal.infra", "TrainController"),
    "WorkflowExecutor": ("areal.infra", "WorkflowExecutor"),
    "current_platform": ("areal.infra", "current_platform"),
    "workflow_context": ("areal.infra", "workflow_context"),
}


def __getattr__(name: str):
    if name == "infra":
        module = importlib.import_module("areal.infra")
        globals()[name] = module
        return module
    if name in _lazy_exports:
        module_name, attribute_name = _lazy_exports[name]
        module = importlib.import_module(module_name)
        attribute = getattr(module, attribute_name)
        globals()[name] = attribute
        return attribute
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


def __dir__():
    return sorted(set(globals()) | set(__all__) | {"infra"})


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
