"""Scaffolding framework primitives vendored from TensorRT-LLM.

This module re-exports the core scaffolding classes from the vendored copy
at ``core``, so the rest of the example can import them from a single location.
"""

from .core.controller import (
    BestOfNController,
    Controller,
    MajorityVoteController,
    NativeChatController,
    NativeGenerationController,
    NativeRewardController,
    ParallelProcess,
)
from .core.result import ScaffoldingOutput
from .core.scaffolding_llm import ScaffoldingLlm
from .core.task import (
    AssistantMessage,
    ChatTask,
    GenerationTask,
    OpenAIToolDescription,
    RoleMessage,
    StreamGenerationTask,
    SystemMessage,
    Task,
    TaskStatus,
    UserMessage,
)
from .core.task_collection import (
    TaskCollection,
    with_task_collection,
)
from .core.worker import OpenaiWorker, Worker

__all__ = [
    "AssistantMessage",
    "BestOfNController",
    "ChatTask",
    "Controller",
    "GenerationTask",
    "MajorityVoteController",
    "NativeChatController",
    "NativeGenerationController",
    "NativeRewardController",
    "OpenAIToolDescription",
    "OpenaiWorker",
    "ParallelProcess",
    "RoleMessage",
    "ScaffoldingLlm",
    "ScaffoldingOutput",
    "StreamGenerationTask",
    "SystemMessage",
    "Task",
    "TaskCollection",
    "TaskStatus",
    "UserMessage",
    "Worker",
    "with_task_collection",
]
