# SPDX-License-Identifier: Apache-2.0

"""AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"""

# ruff: noqa: E402  # deliberate: allocator env must be set before areal imports

# The CUDA allocator config must be set per role BEFORE any areal submodule
# import: `python -m areal.infra.rpc.rpc_server` executes this __init__ first,
# and the `from .infra` chain below initializes CUDA and locks the allocator
# config, so setting the env later (even at rpc_server module top) has no
# effect. Training roles enable expandable_segments to curb reserved-memory
# fragmentation under colocation; inference roles keep it off because sglang
# fails engine init with expandable segments. Pure-stdlib argv parsing only,
# so nothing here can initialize CUDA.
import os as _os
import sys as _sys


def _early_set_alloc_conf() -> None:
    role = ""
    for _i, _a in enumerate(_sys.argv):
        if _a == "--role" and _i + 1 < len(_sys.argv):
            role = _sys.argv[_i + 1]
        elif _a.startswith("--role="):
            role = _a.split("=", 1)[1]
    is_inference = ("rollout" in role.lower()) or ("sglang" in role.lower())
    # AREAL_ACTOR_ALLOC_CONF overrides the training-side allocator config;
    # an empty string leaves the environment untouched.
    conf = _os.environ.get("AREAL_ACTOR_ALLOC_CONF", "expandable_segments:True")
    if role and not is_inference and conf.strip():
        _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf.strip()


_early_set_alloc_conf()

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
