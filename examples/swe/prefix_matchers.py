"""Custom prefix matchers for InteractionCache parent-child matching.

The default cache matcher requires exact message equality. SWE-bench agents,
especially Claude Code style agents routed through Anthropic-compatible APIs,
can rewrite tool arguments or tool output formatting between turns while still
representing the same conversation. The matcher here keeps concat export stable
for those known rewrites.
"""

from __future__ import annotations

from typing import Any

_logger: Any | None = None


def _get_logger() -> Any:
    """Lazily import AReaL logging so this standalone matcher stays lightweight."""
    global _logger
    if _logger is None:
        from areal.utils import logging

        _logger = logging.getLogger("SWEPrefixMatcher")
    return _logger


def _tool_call_ids(tool_calls: list[dict]) -> list[str]:
    """Extract ordered tool_call IDs from a tool_calls list."""
    return [tc.get("id", "") for tc in tool_calls if isinstance(tc, dict)]


def _content_for_comparison(message: dict):
    """Normalize lossy blank content on assistant tool-call messages only."""
    content = message.get("content", "")
    tool_calls = message.get("tool_calls")
    if (
        message.get("role") == "assistant"
        and isinstance(tool_calls, list)
        and tool_calls
        and (content is None or isinstance(content, str) and not content.strip())
    ):
        return ""
    return content


def _messages_match(a: dict, b: dict) -> bool:
    """Check whether two message dicts are semantically equivalent."""
    if a.get("role") != b.get("role"):
        return False

    role = a.get("role")

    if role == "tool":
        return a.get("tool_call_id") == b.get("tool_call_id")

    if _content_for_comparison(a) != _content_for_comparison(b):
        return False

    if a.get("thinking") != b.get("thinking"):
        return False

    a_tc = a.get("tool_calls")
    b_tc = b.get("tool_calls")
    if a_tc is not None or b_tc is not None:
        if a_tc is None or b_tc is None:
            return False
        if not isinstance(a_tc, list) or not isinstance(b_tc, list):
            return a_tc == b_tc
        if len(a_tc) != len(b_tc):
            return False
        if _tool_call_ids(a_tc) != _tool_call_ids(b_tc):
            return False

    return True


def swe_prefix_matcher(a: list[dict], b: list[dict]) -> bool:
    """Return True if ``a`` is a semantic prefix of ``b``."""
    if len(a) > len(b):
        return False
    for am, bm in zip(a, b):
        if am == bm:
            continue
        if not _messages_match(am, bm):
            return False
    return True


def _strict_tool_call_ids(tool_calls: Any) -> list[str] | None:
    """Return ordered tool-call IDs, or ``None`` for malformed calls."""
    if not isinstance(tool_calls, list) or not tool_calls:
        return None

    ids: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            return None
        tool_call_id = tool_call.get("id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return None
        ids.append(tool_call_id)
    return ids


def _warn_prompt_content_mismatch(
    *, kind: str, index: int, prefix_content: Any, full_content: Any, matched: bool
) -> None:
    _get_logger().warning(
        "%s content mismatch at message index %d; treating messages as %s. "
        "prefix_content=%r full_content=%r",
        kind,
        index,
        "matching" if matched else "not matching",
        prefix_content,
        full_content,
    )


def _tool_id_messages_match(
    prefix_message: dict,
    full_message: dict,
    *,
    index: int,
    first_prefix_user_index: int | None,
    first_full_user_index: int | None,
) -> bool:
    """Match one pair using stable tool IDs and selected prompt contents."""
    prefix_role = prefix_message.get("role")
    full_role = full_message.get("role")
    if prefix_role != full_role:
        return False

    prefix_content = prefix_message.get("content")
    full_content = full_message.get("content")

    if prefix_role == "system":
        if prefix_content != full_content:
            _warn_prompt_content_mismatch(
                kind="System prompt",
                index=index,
                prefix_content=prefix_content,
                full_content=full_content,
                matched=True,
            )
        return True

    if prefix_role == "user":
        is_first_prefix_user = index == first_prefix_user_index
        is_first_full_user = index == first_full_user_index
        if is_first_prefix_user != is_first_full_user:
            return False
        if is_first_prefix_user:
            if prefix_content != full_content:
                _warn_prompt_content_mismatch(
                    kind="Initial user prompt",
                    index=index,
                    prefix_content=prefix_content,
                    full_content=full_content,
                    matched=True,
                )
            return True
        if prefix_content != full_content:
            _warn_prompt_content_mismatch(
                kind="Follow-up user prompt",
                index=index,
                prefix_content=prefix_content,
                full_content=full_content,
                matched=False,
            )
            return False
        return True

    if prefix_role == "tool":
        prefix_tool_call_id = prefix_message.get("tool_call_id")
        full_tool_call_id = full_message.get("tool_call_id")
        return (
            isinstance(prefix_tool_call_id, str)
            and bool(prefix_tool_call_id)
            and prefix_tool_call_id == full_tool_call_id
        )

    if prefix_role == "assistant":
        prefix_tool_calls = prefix_message.get("tool_calls")
        full_tool_calls = full_message.get("tool_calls")
        prefix_has_tool_calls = bool(prefix_tool_calls)
        full_has_tool_calls = bool(full_tool_calls)
        if prefix_has_tool_calls or full_has_tool_calls:
            if not prefix_has_tool_calls or not full_has_tool_calls:
                return False
            prefix_tool_call_ids = _strict_tool_call_ids(prefix_tool_calls)
            full_tool_call_ids = _strict_tool_call_ids(full_tool_calls)
            if (
                prefix_tool_call_ids is None
                or full_tool_call_ids is None
                or prefix_tool_call_ids != full_tool_call_ids
            ):
                return False
            if prefix_message != full_message:
                _get_logger().warning(
                    "Tool-call content mismatch at message index %d with matching "
                    "ordered tool-call IDs %r; treating messages as matching. "
                    "prefix_message=%r full_message=%r",
                    index,
                    prefix_tool_call_ids,
                    prefix_message,
                    full_message,
                )
            return True

        if prefix_message != full_message:
            _get_logger().warning(
                "Plain assistant message mismatch at message index %d; treating "
                "messages as not matching. prefix_message=%r full_message=%r",
                index,
                prefix_message,
                full_message,
            )
            return False
        return True

    return prefix_message == full_message


def swe_tool_id_prefix_matcher(a: list[dict], b: list[dict]) -> bool:
    """Match a SWE conversation prefix by tool IDs and selected prompt text.

    Leading system messages and the first user prompt are assumed stable: a
    content difference is warned about but accepted. Tool-call messages match
    by their ordered ``tool_calls[*].id`` values, and tool responses match by
    ``tool_call_id``; their payload differences are ignored. Later user prompts
    must have identical content. Plain assistant messages, including reasoning,
    must be entirely equal so retries with different generated text cannot be
    conflated.
    """
    if len(a) > len(b):
        return False

    first_prefix_user_index = next(
        (index for index, message in enumerate(a) if message.get("role") == "user"),
        None,
    )
    first_full_user_index = next(
        (index for index, message in enumerate(b) if message.get("role") == "user"),
        None,
    )

    for index, (prefix_message, full_message) in enumerate(zip(a, b)):
        if prefix_message == full_message:
            continue
        if not _tool_id_messages_match(
            prefix_message,
            full_message,
            index=index,
            first_prefix_user_index=first_prefix_user_index,
            first_full_user_index=first_full_user_index,
        ):
            return False
    return True


def message_id_prefix_matcher(a: list[dict], b: list[dict]) -> bool:
    """Return True if ``a`` is a prefix of ``b`` based only on message IDs."""
    if len(a) > len(b):
        return False
    return all(
        "id" in prefix_message
        and "id" in full_message
        and prefix_message["id"] == full_message["id"]
        for prefix_message, full_message in zip(a, b)
    )
