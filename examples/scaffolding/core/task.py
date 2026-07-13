# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Vendored from tensorrt_llm.scaffolding.task

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .result import ScaffoldingOutput


@dataclass
class Task:
    # Scaffolding delivers the task to the Worker by worker_tag.
    worker_tag: str = field(default=None)

    # For streaming output.
    streaming_output_flag: bool = field(default=False)
    streaming_output_list: list[Any] = field(default_factory=list)

    # Reserve for custom input params.
    custom_input_params: dict | None = None

    # Reserve for custom output params.
    custom_output_params: dict | None = None

    @staticmethod
    def create_from_prompt(prompt: str) -> "Task":
        pass

    def create_scaffolding_output(self) -> ScaffoldingOutput:
        pass

    def create_scaffolding_output_stream(self) -> list[ScaffoldingOutput]:
        pass


class TaskStatus(Enum):
    SUCCESS = "success"
    WORKER_NOT_SUPPORTED = "worker_not_supported"
    WORKER_EXECEPTION = "worker_exception"


@dataclass
class GenerationTask(Task):
    # input field
    input_tokens: list[int] | None = None
    input_str: str | None = None
    skip_tokenizer: bool = False
    skip_detokenizer: bool = False

    # sampling params for openai
    best_of: int | None = None
    echo: bool | None = False
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None
    num_logprobs: int | None = None
    max_tokens: int | None = None
    n: int = 1
    presence_penalty: float | None = 0.0
    seed: int | None = None
    stop: str | list[str] | None = field(default_factory=list)
    suffix: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    user: str | None = None
    ignore_eos: bool = False

    # sampling params
    top_k: int | None = None
    return_context_logits: bool | None = False

    # suggest to use Controller.WorkerTag
    worker_tag: str | None = None

    # result field
    output_str: str | None = None
    output_tokens: list[int] | None = None
    finish_reason: str | None = None
    context_logits: Any = None
    logprobs: Any = None
    customized_result_fields: dict[str, Any] = field(default_factory=dict)

    perf_metrics: dict[str, float] | None = None

    @staticmethod
    def create_from_prompt(prompt: str) -> "GenerationTask":
        task = GenerationTask()
        task.input_str = prompt
        task.skip_tokenizer = False
        task.skip_detokenizer = False
        return task

    def create_scaffolding_output(self) -> ScaffoldingOutput:
        return ScaffoldingOutput(self.output_str, self.output_tokens)


@dataclass
class StreamGenerationTask(GenerationTask):
    # input field
    cancel_flag: bool | None = field(default=False)
    streaming_step: int | None = field(default=1)

    # result field
    request_handle: Any = field(default=None)
    end_flag: bool = field(default=False)

    @staticmethod
    def create_from_generation_task(
        task: GenerationTask, streaming_step
    ) -> "StreamGenerationTask":
        stream_task = StreamGenerationTask()
        for k, v in task.__dict__.items():
            stream_task.__dict__[k] = v
        stream_task.streaming_step = streaming_step
        return stream_task


@dataclass
class RewardTask(Task):
    # input field
    input_tokens: list[int] | None = field(default=None)
    input_str: str | None = field(default=None)


@dataclass
class RoleMessage:
    role: str | None = field(default=None)
    content: str | None = field(default=None)
    prefix: str | None = field(default=None)

    def __str__(self) -> str:
        return json.dumps(
            {
                "role": self.role,
                "content": self.content,
            }
        )

    def __repr__(self) -> str:
        return f"{self.role}: {self.content}\n"

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(role=data["role"], content=data["content"])


@dataclass
class UserMessage(RoleMessage):
    def __init__(self, content: str, prefix: str | None = None):
        super().__init__(role="user", content=content, prefix=prefix)


@dataclass
class AssistantMessage(RoleMessage):
    reasoning: str | None = field(default=None)
    reasoning_content: str | None = field(default=None)
    tool_calls: list[Any] | None = field(default=None)

    def __init__(
        self,
        content: str,
        reasoning: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: list[Any] | None = None,
    ):
        super().__init__(role="assistant", content=content)
        self.reasoning = reasoning
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls

    def __str__(self) -> str:
        return json.dumps(
            {
                "role": "assistant",
                "content": self.content,
                "reasoning": self.reasoning,
                "reasoning_content": self.reasoning_content,
                "tool_calls": [str(tool) for tool in self.tool_calls]
                if self.tool_calls is not None
                else None,
            }
        )


@dataclass
class SystemMessage(RoleMessage):
    def __init__(self, content: str, prefix: str | None = None):
        super().__init__(role="system", content=content, prefix=prefix)


class ToolDescription:
    def __init__(self, name: str, description: str, parameters: dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

    def to_dict(self) -> dict[str, Any]:
        pass


class OpenAIToolDescription(ToolDescription):
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                },
            },
        }


@dataclass
class ChatTask(StreamGenerationTask):
    messages: list[RoleMessage] = field(default_factory=list)
    tools: Any = field(default=None)

    # for token counting
    enable_token_counting: bool = field(default=False)
    prompt_tokens_num: int = field(default=0)
    completion_tokens_num: int = field(default=0)
    reasoning_tokens_num: int = field(default=0)

    # for sub request marker
    sub_request_markers: list[tuple[str, int]] = field(default_factory=list)
    unique_id: int | None = field(default=None)

    def messages_to_dict_content(self, start_index: int = 0) -> list[Mapping[str, str]]:
        ret = []
        for message in self.messages[start_index:]:
            if message.content is not None:
                ret.append(message.to_dict())
        return ret

    def add_message(self, message: RoleMessage):
        self.messages.append(message)

    def add_messages(self, messages: list[RoleMessage]):
        self.messages.extend(messages)

    @staticmethod
    def create_from_prompt(
        user_prompt: str | None,
        system_prompts: list[SystemMessage] | None = None,
        tools: Any | None = None,
    ) -> "ChatTask":
        task = ChatTask()
        if system_prompts is not None:
            task.messages.extend(system_prompts)
        if user_prompt is not None:
            task.add_message(UserMessage(user_prompt))
        task.tools = tools
        return task

    @staticmethod
    def create_from_messages(
        messages: list[RoleMessage], tools: Any | None = None
    ) -> "ChatTask":
        task = ChatTask()
        task.messages = messages
        task.tools = tools
        return task
