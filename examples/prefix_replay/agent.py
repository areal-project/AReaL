# SPDX-License-Identifier: Apache-2.0

"""Agent-style one-step rollout for replayed-prefix OPD."""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI

from areal.dataset.prefix_replay import (
    PREFIX_REPLAY_MAX_TOTAL_TOKENS_FIELD,
    PREFIX_REPLAY_METADATA_KEY,
)
from areal.utils import stats_tracker


class PrefixReplayLengthTruncated(RuntimeError):
    """Raised when the generated student action stops because of a length cap."""


class PrefixReplayEmptyAction(RuntimeError):
    """Raised when the proxy returns no completion tokens for a replay prefix."""


class PrefixReplayAgent:
    """Generate one student action from a structured offline teacher prefix.

    The OpenAI proxy records the token/logprob interaction. Returning
    ``{response.id: 0.0}`` attaches a zero scalar reward; MOPD supplies the
    optimization signal later by scoring the same action with the routed teacher.
    """

    def __init__(self, **generation_kwargs: Any) -> None:
        self.generation_kwargs = dict(generation_kwargs)
        self.generation_kwargs.setdefault("max_completion_tokens", 131071)

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> dict[str, float]:
        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("PrefixReplayAgent requires non-empty messages")

        base_url = extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY", "dummy")
        if not base_url:
            raise ValueError("base_url is required for PrefixReplayAgent")

        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=extra_kwargs.get("http_client"),
            max_retries=0,
        )
        request: dict[str, Any] = {
            "messages": messages,
            "model": "default",
            **self.generation_kwargs,
        }
        prefix_replay_metadata = data.get(PREFIX_REPLAY_METADATA_KEY)
        if isinstance(prefix_replay_metadata, dict):
            max_total_tokens = prefix_replay_metadata.get(
                PREFIX_REPLAY_MAX_TOTAL_TOKENS_FIELD
            )
            if max_total_tokens is not None:
                if (
                    not isinstance(max_total_tokens, int)
                    or isinstance(max_total_tokens, bool)
                    or max_total_tokens <= 1
                ):
                    raise ValueError(
                        "Prefix replay max_total_tokens must be greater than 1"
                    )
                extra_body = request.get("extra_body")
                if extra_body is not None and not isinstance(extra_body, dict):
                    raise TypeError("PrefixReplayAgent extra_body must be a mapping")
                configured_max_total_tokens = (extra_body or {}).get("max_total_tokens")
                if configured_max_total_tokens is not None:
                    if (
                        not isinstance(configured_max_total_tokens, int)
                        or isinstance(configured_max_total_tokens, bool)
                        or configured_max_total_tokens <= 1
                    ):
                        raise ValueError(
                            "PrefixReplayAgent extra_body max_total_tokens must be "
                            "greater than 1"
                        )
                    max_total_tokens = min(
                        max_total_tokens, configured_max_total_tokens
                    )
                request["extra_body"] = {
                    **(extra_body or {}),
                    "max_total_tokens": max_total_tokens,
                }
        tools = data.get("tools")
        if tools is not None:
            request["tools"] = tools

        response = await client.chat.completions.create(**request)
        choice = response.choices[0]
        finish_reason = choice.finish_reason
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)

        metrics: dict[str, float] = {
            "prefix_replay_length_truncated": float(finish_reason == "length"),
        }
        if isinstance(prompt_tokens, int):
            metrics["prefix_replay_prompt_tokens"] = float(prompt_tokens)
        if isinstance(completion_tokens, int):
            metrics["prefix_replay_action_tokens"] = float(completion_tokens)
        stats_tracker.get().scalar(**metrics)

        if finish_reason == "length":
            raise PrefixReplayLengthTruncated(
                f"Prefix replay action for response {response.id} hit length limit"
            )
        if completion_tokens == 0:
            raise PrefixReplayEmptyAction(
                f"Prefix replay action for response {response.id} has no tokens"
            )

        return {response.id: 0.0}


__all__ = [
    "PrefixReplayAgent",
    "PrefixReplayEmptyAction",
    "PrefixReplayLengthTruncated",
]
