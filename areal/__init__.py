# SPDX-License-Identifier: Apache-2.0

"""AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"""

# The per-role CUDA allocator config must be set BEFORE importing any areal
# submodule: the `from .infra` chain below initializes CUDA and locks in the
# allocator config, and it runs before module bodies such as rpc_server.py —
# so setting the env var at the top of rpc_server.py is already too late.
# The first lines of this file are the only early-enough location.
# Only when an AWEX colocate setup explicitly sets AWEX_ACTOR_ALLOC_CONF do
# the training roles (actor/ref) opt in early; inference roles (rollout /
# SGLang) keep it off because expandable segments break SGLang engine init.
# Parse argv with pure stdlib only — importing anything CUDA-adjacent here
# would defeat the purpose.
import os as _os
import sys as _sys


def _merge_alloc_conf(_existing: str, _extra: str) -> str:
    _existing = _existing.strip()
    _extra_parts = [_p.strip() for _p in _extra.split(",") if _p.strip()]
    if not _existing:
        return ",".join(_extra_parts)
    _existing_keys = {
        _p.split(":", 1)[0].split("=", 1)[0].strip()
        for _p in _existing.split(",")
        if _p.strip()
    }
    _merged = [_existing]
    for _part in _extra_parts:
        _key = _part.split(":", 1)[0].split("=", 1)[0].strip()
        if _key not in _existing_keys:
            _merged.append(_part)
    return ",".join(_merged)


def _early_set_alloc_conf() -> None:
    role = ""
    for _i, _a in enumerate(_sys.argv):
        if _a == "--role" and _i + 1 < len(_sys.argv):
            role = _sys.argv[_i + 1]
        elif _a.startswith("--role="):
            role = _a.split("=", 1)[1]
    is_inference = ("rollout" in role.lower()) or ("sglang" in role.lower())
    # AWEX colocate config opts in with AWEX_ACTOR_ALLOC_CONF. Empty/unset keeps
    # default allocator behavior for non-colocate training runs.
    conf = _os.environ.get("AWEX_ACTOR_ALLOC_CONF", "")
    if role and not is_inference and conf.strip():
        _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _merge_alloc_conf(
            _os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""), conf
        )


_early_set_alloc_conf()

from .version import __version__  # noqa

from .infra import (  # noqa: E402
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
