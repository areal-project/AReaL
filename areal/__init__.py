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


def _early_hide_controller_devices() -> None:
    """Hide accelerators from the single-controller process by default.

    The single-controller trainer only orchestrates remote workers and should hold
    no device context. But merely importing areal pulls in transformers, which
    eagerly imports the torchao quantizer; torchao probes the GPU at import
    (has_triton -> torch.cuda.current_device) and creates a ~hundreds-of-MiB CUDA
    context. The only reliable way to prevent that probe is to make the accelerator
    invisible to this process *before* those imports run.

    Controller-like processes hide by default. Set
    AREAL_HIDE_CONTROLLER_DEVICES=0 to opt out. Every process runs this at
    ``import areal``, so device-owning workers are protected by positive worker
    signals rather than by trying to identify every controller entry point.

    As belt-and-suspenders (the local scheduler spawns workers with the controller's
    full env inherited), we still never hide from a process that carries a
    device-owning-worker signal: --role (local/slurm RPC workers), AREAL_ROLE_WORKER
    (Ray engine actors, which carry no --role in argv), or the run-level AREAL_SPMD_MODE
    (torchrun / launcher SPMD ranks). The original visibility is stashed so the
    scheduler can still enumerate the worker device pool (important on shared nodes
    where only a subset of GPUs is assigned). Uses only stdlib, to avoid triggering the
    very CUDA init we are trying to prevent.
    """

    def _is_device_owning_worker() -> bool:
        if _os.environ.get("AREAL_SPMD_MODE", "").lower() in ("1", "true"):
            return True
        if _os.environ.get("AREAL_ROLE_WORKER", "").lower() in ("1", "true"):
            return True
        return any(_a == "--role" or _a.startswith("--role=") for _a in _sys.argv)

    # Device-owning workers must never be hidden, even if they inherited the opt-in
    # flag from the controller's environment. --role covers local/slurm RPC workers;
    # AREAL_ROLE_WORKER covers Ray engine actors (no --role in their argv); the
    # run-level AREAL_SPMD_MODE covers torchrun / launcher SPMD ranks.
    if _is_device_owning_worker():
        return
    hide_flag = _os.environ.get("AREAL_HIDE_CONTROLLER_DEVICES", "").lower()
    if hide_flag in ("0", "false"):
        return
    if hide_flag not in ("", "1", "true"):
        return

    # Idempotent: don't re-process if a re-import happens.
    if _os.environ.get("AREAL_CONTROLLER_HIDDEN_DEVICE_ENV"):
        return

    def _has_dev(_prefix: str) -> bool:
        try:
            return any(
                _f.startswith(_prefix) and _f[len(_prefix) :].isdigit()
                for _f in _os.listdir("/dev")
            )
        except OSError:
            return False

    # Map physical accelerator -> its visibility env var, without importing torch.
    if _has_dev("nvidia"):
        _env_var = "CUDA_VISIBLE_DEVICES"
    elif _has_dev("davinci"):
        _env_var = "ASCEND_RT_VISIBLE_DEVICES"
    else:
        return

    _orig = _os.environ.get(_env_var)
    # Stash for the scheduler: record both the value and whether it was set at all.
    _os.environ["AREAL_CONTROLLER_HIDDEN_DEVICE_ENV"] = _env_var
    _os.environ["AREAL_CONTROLLER_ORIG_DEVICES_SET"] = "0" if _orig is None else "1"
    _os.environ["AREAL_CONTROLLER_ORIG_DEVICES"] = "" if _orig is None else _orig
    _os.environ[_env_var] = ""


_early_hide_controller_devices()

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
