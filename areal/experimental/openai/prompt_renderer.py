# SPDX-License-Identifier: Apache-2.0

"""Incremental prompt rendering for multi-turn agent rollout.

In multi-turn tool-using rollouts (e.g., coding/search agents), re-running
Jinja2 chat templates and tokenization over the full message history on every
turn produces O(N^2) cumulative message processing over N turns.

This module provides incremental prompt rendering:
1. Turn 1 renders the full prompt and caches the base token prefix (without the
   final generation prompt).
2. Turns 2..N render only the newly appended delta messages against a bounded
   synthetic context, appending the resulting token slice to the parent's base
   prefix.
3. Automatically probes tokenizer template capability on first use to ensure
   100% token-for-token mathematical identity with canonical full-history
   rendering, safely falling back to full-history rendering for dynamic or
   unsupported templates.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from openai.types.chat import ChatCompletionToolParam

from areal.utils import logging
from areal.utils.hf_utils import apply_chat_template

if TYPE_CHECKING:
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

logger = logging.getLogger("PromptRenderer")


def _find_kth(lst: list[int], val: int, k: int) -> int:
    """Find the index of the k-th (1-indexed) occurrence of val in lst."""
    count = 0
    for idx, item in enumerate(lst):
        if item == val:
            count += 1
            if count == k:
                return idx
    return -1


_THINK_START = "<think>"
_THINK_END = "</think>"


def tools_signature(tools: Iterable[ChatCompletionToolParam] | None) -> str:
    """Return a stable signature for the tool set rendered into a prompt prefix."""
    if not tools:
        return ""
    try:
        return json.dumps(list(tools), sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(list(tools))


def contains_reasoning(message: dict[str, Any]) -> bool:
    """Check whether a message carries an inline reasoning block."""
    content = message.get("content")
    return isinstance(content, str) and _THINK_END in content


def has_superseded_reasoning(messages: Iterable[dict[str, Any]] | None) -> bool:
    """Check whether reasoning appears before the final user turn of a history.

    Templates such as Qwen3's keep the reasoning of the current turn but drop it
    from every turn preceding the last user message. Such a history cannot be
    served from an append-only token cache, because appending the new user turn
    is supposed to remove tokens that are already in the cached prefix.
    """
    if not messages:
        return False
    message_list = list(messages)
    last_user_idx = -1
    for idx, message in enumerate(message_list):
        if message.get("role") == "user":
            last_user_idx = idx
    if last_user_idx <= 0:
        return False
    return any(contains_reasoning(m) for m in message_list[:last_user_idx])


class IncrementalPromptRenderer:
    """Renders multi-turn agent prompts incrementally with token parity guarantees."""

    _capability_cache: dict[tuple[Any, ...], tuple[bool, bool]] = {}
    _dummy_d0_cache: dict[tuple[Any, ...], int] = {}
    _lock = threading.Lock()

    @classmethod
    def _get_cache_key(
        cls,
        tokenizer: PreTrainedTokenizerFast,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> tuple[Any, ...]:
        kw_items = (
            tuple(sorted((k, str(v)) for k, v in chat_template_kwargs.items()))
            if chat_template_kwargs
            else ()
        )
        return (
            id(tokenizer),
            getattr(tokenizer, "name_or_path", None),
            kw_items,
        )

    @classmethod
    def is_supported(
        cls,
        tokenizer: PreTrainedTokenizerFast,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> bool:
        """Check whether the tokenizer chat template supports incremental delta rendering."""
        return cls._get_capability(tokenizer, tools, chat_template_kwargs)[0]

    @classmethod
    def is_history_safe(
        cls,
        tokenizer: PreTrainedTokenizerFast,
        messages: Iterable[dict[str, Any]] | None,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> bool:
        """Check whether ``messages`` can be served from an append-only token cache."""
        if not has_superseded_reasoning(messages):
            return True
        return cls._get_capability(tokenizer, tools, chat_template_kwargs)[1]

    @classmethod
    def _get_capability(
        cls,
        tokenizer: PreTrainedTokenizerFast,
        tools: Iterable[ChatCompletionToolParam] | None,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> tuple[bool, bool]:
        """Return (delta rendering supported, reasoning history safe) for a tokenizer."""
        if not hasattr(tokenizer, "chat_template") or not tokenizer.chat_template:
            return False, False

        key = cls._get_cache_key(tokenizer, chat_template_kwargs)
        with cls._lock:
            if key in cls._capability_cache:
                return cls._capability_cache[key]

        # Probe capability with synthetic sequences
        capability = cls._probe_capability(tokenizer, tools, chat_template_kwargs)
        with cls._lock:
            cls._capability_cache[key] = capability

        supported, reasoning_safe = capability
        if supported:
            logger.debug(
                "Incremental prompt rendering verified and enabled for tokenizer: %s "
                "(reasoning history safe: %s)",
                getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
                reasoning_safe,
            )
        else:
            logger.debug(
                "Incremental prompt rendering not supported for tokenizer: %s; using full fallback.",
                getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
            )
        return capability

    @classmethod
    def _probe_capability(
        cls,
        tokenizer: PreTrainedTokenizerFast,
        tools: Iterable[ChatCompletionToolParam] | None,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> tuple[bool, bool]:
        """Run probes to verify token-for-token equality between incremental and full rendering.

        The first probe covers an append-only tool-calling delta. The second probe
        covers a cached prefix that already contains a reasoning block, which some
        templates rewrite once a later turn is appended.
        """
        kwargs = chat_template_kwargs or {}
        try:
            m1 = [{"role": "user", "content": "probe user query"}]
            delta = [
                {
                    "role": "assistant",
                    "content": "probe response",
                    "tool_calls": [
                        {
                            "id": "call_probe_1",
                            "type": "function",
                            "function": {
                                "name": "probe_tool",
                                "arguments": '{"param": "val"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_probe_1",
                    "name": "probe_tool",
                    "content": "probe result",
                },
            ]
            full_2 = apply_chat_template(
                tokenizer,
                m1 + delta,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                **kwargs,
            )
            base_1 = apply_chat_template(
                tokenizer,
                m1,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
                **kwargs,
            )
            dummy = [{"role": "user", "content": "x"}]
            d0 = apply_chat_template(
                tokenizer,
                dummy,
                add_generation_prompt=False,
                tokenize=True,
                **kwargs,
            )
            d_gen = apply_chat_template(
                tokenizer,
                dummy + delta,
                add_generation_prompt=True,
                tokenize=True,
                **kwargs,
            )
            if not isinstance(full_2, list) or not isinstance(base_1, list):
                return False, False
            incr_2 = base_1 + d_gen[len(d0) :]
            supported = full_2 == incr_2
        except Exception as e:
            logger.debug("PromptRenderer probe failed with error: %s", e)
            return False, False

        if not supported:
            return False, False
        return True, cls._probe_reasoning_history(
            tokenizer, tools, chat_template_kwargs
        )

    @classmethod
    def _probe_reasoning_history(
        cls,
        tokenizer: PreTrainedTokenizerFast,
        tools: Iterable[ChatCompletionToolParam] | None,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> bool:
        """Verify that a cached prefix containing reasoning survives appending a turn."""
        kwargs = chat_template_kwargs or {}
        try:
            history = [
                {"role": "user", "content": "probe user query"},
                {
                    "role": "assistant",
                    "content": f"{_THINK_START}probe reasoning{_THINK_END}probe answer",
                },
            ]
            delta = [{"role": "user", "content": "probe follow-up"}]
            base = apply_chat_template(
                tokenizer,
                history,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
                **kwargs,
            )
            full = apply_chat_template(
                tokenizer,
                history + delta,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                **kwargs,
            )
            dummy = [{"role": "user", "content": "x"}]
            d0 = apply_chat_template(
                tokenizer,
                dummy,
                add_generation_prompt=False,
                tokenize=True,
                **kwargs,
            )
            d_gen = apply_chat_template(
                tokenizer,
                dummy + delta,
                add_generation_prompt=True,
                tokenize=True,
                **kwargs,
            )
            if not isinstance(full, list) or not isinstance(base, list):
                return False
            return full == base + d_gen[len(d0) :]
        except Exception as e:
            logger.debug("PromptRenderer reasoning probe failed with error: %s", e)
            return False

    @classmethod
    def _get_dummy_d0_len(
        cls,
        tokenizer: PreTrainedTokenizerFast,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> int:
        key = cls._get_cache_key(tokenizer, chat_template_kwargs)
        with cls._lock:
            if key in cls._dummy_d0_cache:
                return cls._dummy_d0_cache[key]

        dummy = [{"role": "user", "content": "x"}]
        d0 = apply_chat_template(
            tokenizer,
            dummy,
            add_generation_prompt=False,
            tokenize=True,
            **(chat_template_kwargs or {}),
        )
        d0_len = len(d0)
        with cls._lock:
            cls._dummy_d0_cache[key] = d0_len
        return d0_len

    @classmethod
    def render_initial(
        cls,
        tokenizer: PreTrainedTokenizerFast,
        messages: list[dict[str, Any]],
        tools: Iterable[ChatCompletionToolParam] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list[int], list[int]]:
        """Render the initial turn's prompt tokens and base prefix tokens."""
        kwargs = chat_template_kwargs or {}
        prompt_token_ids = apply_chat_template(
            tokenizer,
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            **kwargs,
        )
        prompt_base_token_ids = apply_chat_template(
            tokenizer,
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            **kwargs,
        )
        return prompt_token_ids, prompt_base_token_ids

    @classmethod
    def render_incremental(
        cls,
        tokenizer: PreTrainedTokenizerFast,
        parent_base_token_ids: list[int],
        delta_messages: list[dict[str, Any]],
        tools: Iterable[ChatCompletionToolParam] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        parent_tools_signature: str | None = "",
        parent_messages: list[dict[str, Any]] | None = None,
    ) -> tuple[list[int], list[int]] | None:
        """Render prompt tokens for delta messages appended to parent base tokens.

        Only the delta is rendered, so the tool definitions baked into the parent
        prefix are reused as-is. ``parent_tools_signature`` records the tools that
        prefix was built with; when the current turn declares a different tool set
        the prefix is stale and rendering is refused.

        ``parent_messages`` are the messages the prefix was built from. Together
        with ``delta_messages`` they are checked against
        :meth:`is_history_safe`, so a prefix whose reasoning blocks a later turn
        would strip is never produced or consumed.

        Returns (prompt_token_ids, new_base_token_ids) or None on failure.
        """
        if not delta_messages:
            return None

        if (parent_tools_signature or "") != tools_signature(tools):
            logger.debug(
                "Tool set changed between turns; skipping incremental prompt rendering."
            )
            return None

        if not cls.is_history_safe(
            tokenizer,
            list(parent_messages or []) + delta_messages,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        ):
            logger.debug(
                "Chat template rewrites reasoning history; skipping incremental "
                "prompt rendering."
            )
            return None

        kwargs = chat_template_kwargs or {}
        try:
            dummy = [{"role": "user", "content": "x"}]
            d0_len = cls._get_dummy_d0_len(tokenizer, kwargs)
            d_gen = apply_chat_template(
                tokenizer,
                dummy + delta_messages,
                add_generation_prompt=True,
                tokenize=True,
                **kwargs,
            )
            d_no_gen = apply_chat_template(
                tokenizer,
                dummy + delta_messages,
                add_generation_prompt=False,
                tokenize=True,
                **kwargs,
            )
            if not isinstance(d_gen, list) or not isinstance(d_no_gen, list):
                return None
            prompt_token_ids = parent_base_token_ids + d_gen[d0_len:]
            new_base_token_ids = parent_base_token_ids + d_no_gen[d0_len:]
            return prompt_token_ids, new_base_token_ids
        except Exception as e:
            logger.debug("render_incremental failed: %s; falling back", e)
            return None

    @classmethod
    def render_concat_child_tokens(
        cls,
        tokenizer: PreTrainedTokenizerFast,
        parent_output_messages: list[dict[str, Any]],
        message_list: list[dict[str, Any]],
        tools: Iterable[ChatCompletionToolParam] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[int] | None:
        """Render child tokens for concat mode from a bounded synthetic context."""
        kwargs = chat_template_kwargs or {}
        try:
            dummy = [{"role": "user", "content": "x"}]
            d_delta = dummy + parent_output_messages + message_list
            d_gen = apply_chat_template(
                tokenizer,
                d_delta,
                add_generation_prompt=True,
                tokenize=True,
                **kwargs,
            )
            if not isinstance(d_gen, list):
                return None
            eos_token_id = tokenizer.eos_token_id
            dummy_parent_eos_count = len(dummy) + len(parent_output_messages)
            child_truncate_idx = _find_kth(d_gen, eos_token_id, dummy_parent_eos_count)
            if child_truncate_idx == -1 or child_truncate_idx + 1 >= len(d_gen):
                return None
            return d_gen[child_truncate_idx + 1 :]
        except Exception as e:
            logger.debug("render_concat_child_tokens failed: %s; falling back", e)
            return None
