"""
Worker implementations for Scaffolding Framework.

This module provides Worker implementations that wrap AReaL inference engines
for use with the scaffolding framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import openai

from areal.utils import logging

from ._compat import (
    AssistantMessage,
    ChatTask,
    GenerationTask,
    OpenaiWorker,
    TaskStatus,
)

if TYPE_CHECKING:
    from areal.engine.sglang_remote import RemoteSGLangEngine

worker_logger = logging.getLogger("SGLangWorker")


class SGLangWorker(OpenaiWorker):
    """Worker that wraps an SGLang engine for scaffolding.

    This worker connects to an SGLang server via its OpenAI-compatible API
    and handles generation and chat tasks.

    Parameters
    ----------
    async_client : openai.AsyncOpenAI
        The OpenAI async client configured to connect to SGLang server.
    model : str
        The model name to use for requests.
    engine : RemoteSGLangEngine
        The underlying SGLang engine (kept for reference and potential future use).
    """

    def __init__(
        self,
        async_client: openai.AsyncOpenAI,
        model: str,
        engine: RemoteSGLangEngine,
    ):
        super().__init__(async_client, model, kv_cache_hint_enabled=False)
        self.engine = engine

    async def chat_handler(self, task: ChatTask) -> TaskStatus:
        """Handle chat completion requests.

        This method extends the base OpenaiWorker's chat handler to also
        store the ChatCompletion object in the task for tracing purposes.

        Parameters
        ----------
        task : ChatTask
            The chat task to process.

        Returns
        -------
        TaskStatus
            The status of the task execution.
        """
        params = self.convert_task_params(task)
        params["messages"] = task.messages_to_dict_content()
        params["model"] = self.model
        if task.tools is not None:
            params["tools"] = [tool.to_dict() for tool in task.tools]

        try:
            worker_logger.info(
                "Sending chat request to %s (messages=%d) ...",
                self.async_client.base_url,
                len(params.get("messages", [])),
            )
            response = await self.async_client.chat.completions.create(**params)
            worker_logger.info("Chat response received.")

            # Store the completion in the task for tracing
            task.completion = response

            task.finish_reason = response.choices[0].finish_reason
            if hasattr(response, "perf_metrics"):
                task.perf_metrics = response.perf_metrics

            content = response.choices[0].message.content
            reasoning = getattr(response.choices[0].message, "reasoning", None)
            reasoning_content = getattr(
                response.choices[0].message, "reasoning_content", None
            )
            tool_calls = response.choices[0].message.tool_calls

            task.messages.append(
                AssistantMessage(content, reasoning, reasoning_content, tool_calls)
            )

            if task.enable_token_counting and response.usage:
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
            worker_logger.error("SGLang chat client exception: %s", e)
            return TaskStatus.WORKER_EXECEPTION

    async def generation_handler(self, task: GenerationTask) -> TaskStatus:
        """Handle text generation requests.

        Parameters
        ----------
        task : GenerationTask
            The generation task to process.

        Returns
        -------
        TaskStatus
            The status of the task execution.
        """
        params = self.convert_task_params(task)
        params["model"] = self.model
        if task.input_str is not None:
            params["prompt"] = task.input_str

        try:
            worker_logger.info(
                "Sending generation request to %s ...",
                self.async_client.base_url,
            )
            response = await self.async_client.completions.create(**params)
            worker_logger.info("Generation response received.")

            task.output_str = response.choices[0].text
            if hasattr(response.choices[0], "token_ids"):
                task.output_tokens = response.choices[0].token_ids
            task.finish_reason = response.choices[0].finish_reason
            if hasattr(response.choices[0], "logprobs"):
                task.logprobs = response.choices[0].logprobs
            if hasattr(response, "perf_metrics"):
                task.perf_metrics = response.perf_metrics

            return TaskStatus.SUCCESS

        except Exception as e:
            worker_logger.error("SGLang completion client exception: %s", e)
            return TaskStatus.WORKER_EXECEPTION

    # Register task handlers
    task_handlers = {
        GenerationTask: generation_handler,
        ChatTask: chat_handler,
    }


def CreateWorkerFromEngine(
    engine: RemoteSGLangEngine,
    model: str = "default",
) -> SGLangWorker:
    """Create a scaffolding Worker from an AReaL SGLang engine.

    This function creates a Worker that wraps the given SGLang engine,
    allowing it to be used with the scaffolding framework.
    The worker uses the SGLang server's OpenAI-compatible API.

    Parameters
    ----------
    engine : RemoteSGLangEngine
        The AReaL SGLang inference engine (must be initialized).
    model : str, optional
        The model name to use for API requests. Defaults to "default".

    Returns
    -------
    SGLangWorker
        A Worker instance that can be used with ScaffoldingLlm.

    Example
    -------
    ```python
    from areal.engine.sglang_remote import RemoteSGLangEngine
    from examples.scaffolding import CreateWorkerFromEngine

    # Initialize the engine
    engine = RemoteSGLangEngine(config)
    engine.initialize()

    # Create a worker
    worker = CreateWorkerFromEngine(engine)

    # Use with ScaffoldingLlm
    from examples.scaffolding._compat import ScaffoldingLlm, NativeGenerationController
    llm = ScaffoldingLlm(
        controller,
        {NativeGenerationController.WorkerTag.GENERATION: worker},
    )
    ```

    Raises
    ------
    RuntimeError
        If the engine is not initialized.
    """
    if not engine.initialized:
        raise RuntimeError(
            "Engine must be initialized before creating a worker. "
            "Call engine.initialize() first."
        )

    # Get the server address from the engine
    # The internal engine stores server info
    internal_engine = engine._engine
    server_addrs = internal_engine._server_addrs

    if not server_addrs:
        raise RuntimeError("No server addresses found in engine.")

    # Use the first server address for the OpenAI client
    # SGLang servers support OpenAI-compatible API at /v1/
    base_url = server_addrs[0]
    if not base_url.startswith("http"):
        base_url = f"http://{base_url}"
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    # Create an async OpenAI client pointing to the SGLang server
    async_client = openai.AsyncOpenAI(
        base_url=base_url,
        api_key="EMPTY",  # SGLang doesn't require API key by default
    )

    return SGLangWorker(
        async_client=async_client,
        model=model,
        engine=engine,
    )
