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


class IncrementalPromptRenderer:
    """Renders multi-turn agent prompts incrementally with token parity guarantees."""

    _capability_cache: dict[tuple[Any, ...], bool] = {}
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
        if not hasattr(tokenizer, "chat_template") or not tokenizer.chat_template:
            return False

        key = cls._get_cache_key(tokenizer, chat_template_kwargs)
        with cls._lock:
            if key in cls._capability_cache:
                return cls._capability_cache[key]

        # Probe capability with a synthetic 2-turn sequence
        supported = cls._probe_capability(tokenizer, tools, chat_template_kwargs)
        with cls._lock:
            cls._capability_cache[key] = supported

        if supported:
            logger.debug(
                "Incremental prompt rendering verified and enabled for tokenizer: %s",
                getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
            )
        else:
            logger.debug(
                "Incremental prompt rendering not supported for tokenizer: %s; using full fallback.",
                getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
            )
        return supported

    @classmethod
    def _probe_capability(
        cls,
        tokenizer: PreTrainedTokenizerFast,
        tools: Iterable[ChatCompletionToolParam] | None,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> bool:
        """Run a probe to verify token-for-token equality between incremental and full rendering."""
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
                return False
            incr_2 = base_1 + d_gen[len(d0) :]
            return full_2 == incr_2
        except Exception as e:
            logger.debug("PromptRenderer probe failed with error: %s", e)
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
    ) -> tuple[list[int], list[int]] | None:
        """Render prompt tokens for delta messages appended to parent base tokens.

        Returns (prompt_token_ids, new_base_token_ids) or None on failure.
        """
        if not delta_messages:
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
