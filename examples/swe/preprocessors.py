"""Message preprocessors for SWE-bench agent rollouts.

Preprocessors transform messages after Anthropic-to-OpenAI translation and
before they reach AReaL's OpenAI client. They remove volatile Claude Code
metadata that otherwise breaks prefix-based parent matching in concat export.
"""

import json
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class MessagePreprocessor(Protocol):
    """Protocol for message preprocessors."""

    def __call__(self, messages: list[dict]) -> list[dict]: ...


class StripAnthropicBillingHeader:
    """Remove per-request Anthropic billing header lines from system prompts."""

    _PATTERN = re.compile(r"^x-anthropic-billing-header:[^\n]*\n?", re.MULTILINE)

    def __call__(self, messages: list[dict]) -> list[dict]:
        if not messages:
            return messages
        first = messages[0]
        if first.get("role") == "system" and isinstance(first.get("content"), str):
            first["content"] = self._PATTERN.sub("", first["content"])
        return messages


class NormalizeSystemReminder:
    """Remove volatile ``currentDate`` lines from system reminders."""

    _PATTERN = re.compile(
        r"# currentDate\nToday's date is \d{4}-\d{2}-\d{2}\.\n?",
    )

    def __call__(self, messages: list[dict]) -> list[dict]:
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str) and "currentDate" in content:
                msg["content"] = self._PATTERN.sub("", content)
        return messages


class StripAllSystemReminders:
    """Remove all ``<system-reminder>...</system-reminder>`` blocks."""

    _PATTERN = re.compile(r"<system-reminder>[\s\S]*?</system-reminder>")

    def __call__(self, messages: list[dict]) -> list[dict]:
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str) and "<system-reminder>" in content:
                msg["content"] = self._PATTERN.sub("", content)
        return messages


class StripAnthropicCacheFields:
    """Strip Anthropic-specific fields that are not preserved in stored output."""

    def __call__(self, messages: list[dict]) -> list[dict]:
        for msg in messages:
            msg.pop("cache_control", None)
            msg.pop("thinking_blocks", None)
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                        tc.pop("cache_control", None)
                        tc.pop("provider_specific_fields", None)
                        fn = tc.get("function")
                        if isinstance(fn, dict):
                            fn.pop("cache_control", None)
        return messages


class NormalizeToolCallArguments:
    """Normalize tool_call arguments for deterministic comparison."""

    _TOOL_ARG_DEFAULTS: dict[tuple[str, str], object] = {
        ("Edit", "replace_all"): False,
    }

    @classmethod
    def _normalize_tool_arguments(
        cls,
        tool_name: str | None,
        args_dict: dict,
    ) -> dict:
        if tool_name is not None:
            for (tn, field), default in cls._TOOL_ARG_DEFAULTS.items():
                if tn == tool_name and args_dict.get(field) == default:
                    args_dict.pop(field, None)
        return args_dict

    def __call__(self, messages: list[dict]) -> list[dict]:
        for msg in messages:
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                        fn = tc.get("function")
                        if isinstance(fn, dict):
                            tool_name = fn.get("name")
                            args = fn.get("arguments")
                            if isinstance(args, str):
                                try:
                                    parsed = json.loads(args)
                                    parsed = self._normalize_tool_arguments(
                                        tool_name,
                                        parsed,
                                    )
                                    fn["arguments"] = json.dumps(
                                        parsed,
                                        sort_keys=True,
                                        ensure_ascii=False,
                                    )
                                except (json.JSONDecodeError, TypeError):
                                    pass
                if msg.get("role") == "assistant" and "content" not in msg:
                    msg["content"] = ""
        return messages
