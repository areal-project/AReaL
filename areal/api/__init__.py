# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "RolloutWorkflow",
    "AsyncRewardWrapper",
    "TrainEngine",
    "InferenceEngine",
    "Scheduler",
    "Worker",
    "Job",
    "AllocationType",
    "ModelAllocation",
    "ParallelStrategy",
    "FSDPParallelStrategy",
    "MegatronParallelStrategy",
    "ModelRequest",
    "ModelResponse",
    "WeightUpdateMeta",
    "SaveLoadMeta",
    "StepInfo",
    "FinetuneSpec",
    "ParamSpec",
    "RolloutStat",
    "LocalInfServerInfo",
    "WorkflowLike",
    "AgentWorkflow",
]

_LAZY_IMPORTS = {
    "TrainEngine": "areal.api.engine_api",
    "InferenceEngine": "areal.api.engine_api",
    "Scheduler": "areal.api.scheduler_api",
    "Worker": "areal.api.scheduler_api",
    "Job": "areal.api.scheduler_api",
    "AllocationType": "areal.api.alloc_mode",
    "ModelAllocation": "areal.api.alloc_mode",
    "ParallelStrategy": "areal.api.alloc_mode",
    "FSDPParallelStrategy": "areal.api.alloc_mode",
    "MegatronParallelStrategy": "areal.api.alloc_mode",
    "ModelRequest": "areal.api.io_struct",
    "ModelResponse": "areal.api.io_struct",
    "WeightUpdateMeta": "areal.api.io_struct",
    "SaveLoadMeta": "areal.api.io_struct",
    "StepInfo": "areal.api.io_struct",
    "FinetuneSpec": "areal.api.io_struct",
    "ParamSpec": "areal.api.io_struct",
    "RolloutStat": "areal.api.io_struct",
    "LocalInfServerInfo": "areal.api.io_struct",
    "WorkflowLike": "areal.api.workflow_api",
    "AgentWorkflow": "areal.api.workflow_api",
    "AsyncRewardWrapper": "areal.api.reward_api",
    "RolloutWorkflow": "areal.api.workflow_api",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
