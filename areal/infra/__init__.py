# SPDX-License-Identifier: Apache-2.0

"""Core components for AREAL."""

from importlib import import_module

__all__ = [
    "RemoteInfBackendProtocol",
    "RemoteInfEngine",
    "StalenessManager",
    "WorkflowExecutor",
    "check_trajectory_format",
    "RolloutController",
    "TrainController",
    "workflow_context",
    "Platform",
    "current_platform",
    "is_npu_available",
    "LocalScheduler",
    "RayScheduler",
    "SlurmScheduler",
    "LocalLauncher",
    "RayLauncher",
    "SlurmLauncher",
    "SGLangServerWrapper",
    "vLLMServerWrapper",
]

# 配置模块在加载时会访问 platforms，控制器则会读取这些配置。
_LAZY_IMPORTS = {
    "RemoteInfBackendProtocol": "remote_inf_engine",
    "RemoteInfEngine": "remote_inf_engine",
    "StalenessManager": "staleness_manager",
    "WorkflowExecutor": "workflow_executor",
    "check_trajectory_format": "workflow_executor",
    "RolloutController": "controller",
    "TrainController": "controller",
    "Platform": "platforms",
    "current_platform": "platforms",
    "is_npu_available": "platforms",
    "LocalScheduler": "scheduler",
    "RayScheduler": "scheduler",
    "SlurmScheduler": "scheduler",
    "LocalLauncher": "launcher",
    "RayLauncher": "launcher",
    "SlurmLauncher": "launcher",
    "SGLangServerWrapper": "launcher",
    "vLLMServerWrapper": "launcher",
}
_SUBMODULES = {*_LAZY_IMPORTS.values(), "workflow_context"}


def __getattr__(name: str):
    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    if name in _LAZY_IMPORTS:
        module = import_module(f".{_LAZY_IMPORTS[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
