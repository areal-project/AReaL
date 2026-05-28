# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Vendored from TensorRT-LLM scaffolding framework.
# Core scaffolding primitives adapted for standalone use in AReaL.

from .controller import (
    BestOfNController,
    Controller,
    MajorityVoteController,
    NativeChatController,
    NativeGenerationController,
    NativeRewardController,
    ParallelProcess,
)
from .math_utils import (
    extract_answer_from_boxed,
    extract_answer_with_regex,
    get_digit_majority_vote_result,
)
from .result import ScaffoldingOutput
from .scaffolding_llm import ScaffoldingLlm
from .task import (
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
from .task_collection import TaskCollection, with_task_collection
from .worker import OpenaiWorker, Worker

__all__ = [
    "ScaffoldingLlm",
    "ParallelProcess",
    "Controller",
    "NativeChatController",
    "NativeGenerationController",
    "NativeRewardController",
    "MajorityVoteController",
    "BestOfNController",
    "Task",
    "GenerationTask",
    "StreamGenerationTask",
    "ChatTask",
    "OpenAIToolDescription",
    "RoleMessage",
    "UserMessage",
    "SystemMessage",
    "AssistantMessage",
    "Worker",
    "OpenaiWorker",
    "TaskStatus",
    "extract_answer_from_boxed",
    "extract_answer_with_regex",
    "get_digit_majority_vote_result",
    "TaskCollection",
    "with_task_collection",
    "ScaffoldingOutput",
]
