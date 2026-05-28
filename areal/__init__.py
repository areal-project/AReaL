# SPDX-License-Identifier: Apache-2.0

"""AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"""

from .version import __version__  # noqa


# Heavy submodules (infra, trainer) are loaded lazily via PEP 562
# ``__getattr__`` so that ``import areal`` stays light. This is the load-
# bearing precondition for the ``areal`` console-script's lightness
# invariant (see ``tests/experimental/test_cli_lightness.py``): importing
# ``areal.experimental.cli.main`` transitively runs ``areal/__init__.py``,
# and if any eager top-level import here pulled in torch / ray / megatron
# / fastapi, the CLI could no longer be installed on a login node without
# the training stack.
#
# Backwards-compat: ``areal.RolloutController`` etc. still work because
# attribute access triggers ``__getattr__``; only bare ``import areal``
# changes behavior (now light).
_INFRA_NAMES = frozenset({
    "RolloutController",
    "StalenessManager",
    "TrainController",
    "WorkflowExecutor",
    "current_platform",
    "workflow_context",
})

_TRAINER_NAMES = frozenset({
    "DPOTrainer",
    "PPOTrainer",
    "RWTrainer",
    "SFTTrainer",
})


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
