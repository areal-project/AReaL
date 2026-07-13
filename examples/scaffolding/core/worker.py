# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Vendored from tensorrt_llm.scaffolding.worker
# TRTLLMWorker and MCPWorker omitted (require tensorrt_llm runtime).

import os
from abc import ABC
from collections.abc import Callable

import openai

from .task import AssistantMessage, ChatTask, GenerationTask, Task, TaskStatus


class Worker(ABC):
    def register_task_handler(
        self, task_cls: type[Task], handler: Callable[[object, Task], TaskStatus]
    ):
        worker_cls = type(self)
        worker_cls.task_handlers[task_cls] = handler

    async def run_task(self, task: Task) -> TaskStatus:
        worker_cls = type(self)
        if type(task) not in worker_cls.task_handlers:
            return TaskStatus.WORKER_NOT_SUPPORTED
        return await worker_cls.task_handlers[type(task)](self, task)

    task_handlers = {}

    def shutdown(self):
        pass

    async def async_shutdown(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self):
        self.shutdown()


# helper function
def add_param_if_not_none(params, key, candidate_values):
    for value in candidate_values:
        if value is not None:
            params[key] = value
            return


# helper function
def add_attr_if_not_none(obj, attr, candidate_values):
    for value in candidate_values:
        if value is not None:
            setattr(obj, attr, value)
            return


def is_deterministic_mode():
    """Check if SCAFFOLDING_DETERMINISTIC environment variable is set."""
    return int(os.environ.get("SCAFFOLDING_DETERMINISTIC", 0)) == 1


class OpenaiWorker(Worker):
    def __init__(
        self,
        async_client: openai.AsyncOpenAI,
        model: str,
        kv_cache_hint_enabled: bool = False,
    ):
        self.model = model
        self.async_client = async_client
        self.kv_cache_hint_enabled = kv_cache_hint_enabled

    def convert_task_params(self, task: GenerationTask | ChatTask):
        params = {
            "model": self.model,
            "extra_body": {},
        }

        if not isinstance(task, ChatTask):
            params["prompt"] = task.input_str
            add_param_if_not_none(params, "echo", [task.echo])

        add_param_if_not_none(params, "best_of", [task.best_of])
        add_param_if_not_none(params, "frequency_penalty", [task.frequency_penalty])
        add_param_if_not_none(params, "logit_bias", [task.logit_bias])
        add_param_if_not_none(params, "logprobs", [task.num_logprobs])
        add_param_if_not_none(params, "max_tokens", [task.max_tokens])
        add_param_if_not_none(params, "n", [task.n])
        add_param_if_not_none(params, "presence_penalty", [task.presence_penalty])
        add_param_if_not_none(params, "seed", [task.seed])
        add_param_if_not_none(params, "stop", [task.stop])
        add_param_if_not_none(params, "suffix", [task.suffix])
        add_param_if_not_none(params, "temperature", [task.temperature])
        add_param_if_not_none(params, "top_p", [task.top_p])
        add_param_if_not_none(params, "user", [task.user])

        # Override parameters for deterministic inference
        if is_deterministic_mode():
            params["temperature"] = 0.0
            params["top_p"] = 1.0
            params["n"] = 1
            if "seed" not in params or params["seed"] is None:
                params["seed"] = 42

        if hasattr(task, "sub_request_markers") and len(task.sub_request_markers) > 0:
            params["extra_body"]["agent_hierarchy"] = [task.sub_request_markers[-1]]

        return params

    def fill_generation_task_with_response(
        self, task: GenerationTask, response: openai.Completion
    ):
        task.output_str = response.choices[0].text
        task.output_tokens = response.choices[0].token_ids
        task.finish_reason = response.choices[0].finish_reason
        task.logprobs = response.choices[0].logprobs
        task.perf_metrics = response.perf_metrics

    async def generation_handler(self, task: GenerationTask) -> TaskStatus:
        params = self.convert_task_params(task)

        try:
            response = await self.async_client.completions.create(**params)
            self.fill_generation_task_with_response(task, response)

            return TaskStatus.SUCCESS

        except Exception as e:
            print("Openai client get exception: " + str(e))
            return TaskStatus.WORKER_EXECEPTION

    async def chat_handler(self, task: ChatTask) -> TaskStatus:
        params = self.convert_task_params(task)
        params["messages"] = task.messages_to_dict_content()
        params["model"] = self.model
        if task.tools is not None:
            params["tools"] = [tool.to_dict() for tool in task.tools]

        try:
            response = await self.async_client.chat.completions.create(**params)
            task.finish_reason = response.choices[0].finish_reason
            task.perf_metrics = response.perf_metrics
            content = response.choices[0].message.content
            reasoning = response.choices[0].message.reasoning
            reasoning_content = response.choices[0].message.reasoning_content
            tool_calls = response.choices[0].message.tool_calls
            task.messages.append(
                AssistantMessage(content, reasoning, reasoning_content, tool_calls)
            )
            if task.enable_token_counting:
                task.prompt_tokens_num = response.usage.prompt_tokens
                task.completion_tokens_num = response.usage.completion_tokens
                if (
                    hasattr(response.usage, "completion_tokens_details")
                    and response.usage.completion_tokens_details is not None
                ):
                    task.reasoning_tokens_num = (
                        response.usage.completion_tokens_details.reasoning_tokens
                    )

            return TaskStatus.SUCCESS

        except Exception as e:
            print("Openai chat client get exception: " + str(e))
            return TaskStatus.WORKER_EXECEPTION

    task_handlers = {
        GenerationTask: generation_handler,
        ChatTask: chat_handler,
    }
