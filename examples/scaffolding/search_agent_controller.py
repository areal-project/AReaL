"""
SearchAgentController — multi-turn tool-calling loop as a scaffolding Controller.

This is the scaffolding-framework equivalent of ``MultiTurnReactAgent.run_agent``
from the tongyi_deepresearch example.  Instead of calling an ``ArealOpenAI``
client directly, it yields ``ChatTask`` objects through a
``NativeGenerationController`` so that the ``ScaffoldingLlm`` dispatches them
to the SGLang worker.

Tool execution (search / visit) happens **locally** inside ``process()`` —
only LLM generation goes through a Worker.
"""

from __future__ import annotations

import asyncio
from typing import Any

import json5

from areal.utils import logging

from ._compat import (
    ChatTask,
    Controller,
    NativeGenerationController,
    RoleMessage,
    Task,
    UserMessage,
)
from .real_tools import real_search, real_visit

logger = logging.getLogger("SearchAgentController")

OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"


class SearchAgentController(Controller):
    """Multi-turn search-agent controller for the scaffolding framework.

    Each call to :meth:`process` runs a loop of up to *max_turns* turns:

    1. Yield the ``ChatTask`` to the generation controller (LLM call).
    2. Parse the assistant reply for ``<tool_call>...</tool_call>``.
    3. If a tool call is found, execute it locally and append the tool
       response as a user message.
    4. If an ``<answer>`` tag is found, stop.
    5. If the token budget is exhausted, append a "please answer" nudge,
       do one final generation, and stop.

    Parameters
    ----------
    generation_controller : Controller
        Typically a ``NativeGenerationController`` that yields ``ChatTask``
        to the worker.
    tokenizer
        HuggingFace tokenizer for token counting.
    max_turns : int
        Maximum number of LLM calls per episode.
    max_total_tokens : int
        Soft token budget for the conversation.
    messages : list[dict] | None
        Initial chat messages (set per-episode before ``generate``).
    input_tokens : list[int] | None
        Tokenised input IDs (set per-episode before ``generate``).
    """

    # Re-use the generation worker tag from NativeGenerationController
    WorkerTag = NativeGenerationController.WorkerTag

    def __init__(
        self,
        generation_controller: Controller,
        tokenizer: Any,
        max_turns: int = 20,
        max_total_tokens: int = 32768,
        messages: list[dict] | None = None,
        input_tokens: list[int] | None = None,
    ):
        super().__init__()
        self.generation_controller = generation_controller
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.max_total_tokens = max_total_tokens
        self.max_total_tokens_before_finishing = int(max_total_tokens * 0.8)
        # Safety margin to account for _count_tokens approximation errors
        # (the simple template-based count can underestimate by ~200 tokens)
        # and SGLang's >= check (exactly max_context_length is rejected too).
        self._token_safety_margin = 256
        self.messages = messages if messages is not None else []
        self.input_tokens = input_tokens if input_tokens is not None else []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_tokens(self, messages: list[RoleMessage]) -> int:
        """Approximate token count for a list of ``RoleMessage`` objects."""
        parts = []
        for msg in messages:
            d = msg.to_dict() if hasattr(msg, "to_dict") else msg
            parts.append(f"<|im_start|>{d['role']}\n{d['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return len(self.tokenizer.encode("".join(parts)))

    async def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Dispatch a tool call to the appropriate tool."""
        if tool_name == "search":
            queries = tool_args.get("query", [])
            if isinstance(queries, str):
                queries = [queries]
            return await real_search(queries)
        if tool_name == "visit":
            urls = tool_args.get("url", [])
            if isinstance(urls, str):
                urls = [urls]
            goal = tool_args.get("goal", "")
            return await real_visit(urls, goal)
        return f"Error: Tool {tool_name} not found"

    # ------------------------------------------------------------------
    # Controller interface
    # ------------------------------------------------------------------

    def process(self, tasks: list[Task], **kwargs) -> Any:  # noqa: C901
        """Run the multi-turn search-agent loop.

        Parameters
        ----------
        tasks : list[Task]
            Ignored — the ``ChatTask`` is built from ``self.messages``.
        **kwargs
            Forwarded to the generation controller.

        Yields
        ------
        list[Task]
            Task lists for worker execution (one per LLM call).
        """
        # Build the ChatTask from stored messages
        role_messages = [RoleMessage.from_dict(m) for m in self.messages]
        chat_task = ChatTask.create_from_messages(role_messages)
        chat_task.stop = ["\n<tool_response>", "<tool_response>"]
        if self.input_tokens:
            chat_task.input_tokens = self.input_tokens

        for turn in range(self.max_turns):
            # --- Token budget check (before generation) -----------------------
            token_count = self._count_tokens(chat_task.messages)
            # Reserve room for max_new_tokens so the request won't exceed
            # the SGLang context window.
            max_new = self.generation_controller.sampling_params.get("max_tokens", 2048)
            if token_count + max_new > self.max_total_tokens:
                logger.info(
                    "Token budget approaching limit (%d + %d > %d); "
                    "requesting final answer.",
                    token_count,
                    max_new,
                    self.max_total_tokens,
                )
                chat_task.add_message(
                    UserMessage(
                        "You have now reached the maximum context length you can handle. "
                        "You should stop making tool calls and, based on all the information above, "
                        "think again and provide what you consider the most likely answer "
                        "in the following format:"
                        "<think>your final thinking</think>\n"
                        "<answer>your answer</answer>"
                    )
                )
                # Cap max_tokens directly on the task to stay within budget.
                # NativeGenerationController.process() only sets values when
                # task.max_tokens is None, so we must set it on the task itself.
                remaining = max(
                    256,
                    self.max_total_tokens
                    - self._count_tokens(chat_task.messages)
                    - self._token_safety_margin,
                )
                chat_task.max_tokens = remaining
                yield from self.generation_controller.process([chat_task], **kwargs)
                break

            # --- LLM generation ------------------------------------------------
            yield from self.generation_controller.process([chat_task], **kwargs)

            # Extract the assistant reply
            last_msg = chat_task.messages[-1]
            content = last_msg.content or ""

            # --- Tool call handling -------------------------------------------
            if "<tool_call>" in content and "</tool_call>" in content:
                tool_call_str = content.split("<tool_call>")[1].split("</tool_call>")[0]
                try:
                    tool_call = json5.loads(tool_call_str)
                    tool_name = tool_call["name"]
                    tool_args = tool_call.get("arguments", {})
                    # Execute tool (async → sync bridge)
                    loop = asyncio.new_event_loop()
                    try:
                        result = loop.run_until_complete(
                            self._execute_tool(tool_name, tool_args)
                        )
                    finally:
                        loop.close()
                except Exception as e:
                    result = (
                        f"Error: {e} Tool call must be valid JSON with "
                        f'"name" and "arguments" fields.'
                    )
                tool_response = f"{OBS_START}\n{result}{OBS_END}"
                chat_task.add_message(UserMessage(tool_response))

            # --- Check for final answer ---------------------------------------
            if "<answer>" in content and "</answer>" in content:
                break

        # --- Turn limit reached without answer --------------------------------
        if turn == self.max_turns - 1:
            last_content = chat_task.messages[-1].content or ""
            if "<answer>" not in last_content:
                chat_task.add_message(
                    UserMessage(
                        "Sorry, the number of LLM calls exceeds the limit. "
                        "You should stop making tool calls and, based on all "
                        "the information above, think again and provide what "
                        "you consider the most likely answer in the following format:"
                        "<think>your final thinking</think>\n"
                        "<answer>your answer</answer>"
                    )
                )
                # Cap max_tokens directly on the task to stay within budget.
                remaining = max(
                    256,
                    self.max_total_tokens
                    - self._count_tokens(chat_task.messages)
                    - self._token_safety_margin,
                )
                chat_task.max_tokens = remaining
                yield from self.generation_controller.process([chat_task], **kwargs)
