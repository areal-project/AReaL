"""
Scaffolding Framework Example for AReaL.

This package provides the Scaffolding framework for composing inference-time
compute methods with AReaL's RL training pipeline. Core scaffolding primitives
are vendored from TensorRT-LLM under ``examples.scaffolding.core``.

Key Components:
- ScaffoldingWorkflow: RolloutWorkflow implementation that wraps ScaffoldingLlm
- RLVRRewardTask: Task for computing verifiable rewards
- RLVRRewardController: Controller for computing verifiable rewards
- PipelineTrajectoryMaker: Controller for composing generation and reward pipelines
- ChatTracer: TaskCollection for tracing multi-turn chat conversations
- TraceTrajectoryMaker: Controller that traces ChatTask objects during rollout
- TraceGenerationTask: Task for tracing multi-turn generation
- ChatRewardTask: Task for computing rewards on traced interactions
- CreateWorkerFromEngine: Creates a scaffolding Worker from AReaL's InferenceEngine
- SGLangWorker: Worker implementation for SGLang engines
"""

from examples.scaffolding.controllers import (
    ChatTracer,
    PipelineTrajectoryMaker,
    RLVRRewardController,
    TraceTrajectoryMaker,
)
from examples.scaffolding.task import (
    ChatRewardTask,
    RLVRRewardTask,
    TraceGenerationTask,
)
from examples.scaffolding.worker import CreateWorkerFromEngine, SGLangWorker
from examples.scaffolding.workflow import ScaffoldingWorkflow

__all__ = [
    "ScaffoldingWorkflow",
    "RLVRRewardTask",
    "RLVRRewardController",
    "PipelineTrajectoryMaker",
    "ChatTracer",
    "TraceTrajectoryMaker",
    "TraceGenerationTask",
    "ChatRewardTask",
    "CreateWorkerFromEngine",
    "SGLangWorker",
]
