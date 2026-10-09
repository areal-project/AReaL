# SPDX-License-Identifier: Apache-2.0

"""Normalize structured tool events once, before publishing a successful run."""

import uuid
from typing import Any


def normalize_tool_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign missing call IDs and validate associations without reordering events.

    Explicit IDs are opaque. A result without an ID may match only one pending
    call of the same name. Message IDs group calls from one assistant message;
    a group cannot reopen after a result or a call from another message.

    This validates reported associations, not transcript completeness: agents
    that only report calls for observability need not emit results. All state
    is local to this invocation, and input events are left untouched on failure.
    """
    normalized = [dict(event) for event in events]
    reserved_ids: set[str] = set()
    for event in normalized:
        if event.get("type") not in ("tool_call", "tool_result"):
            continue
        fields = (
            ("call_id", "message_id") if event["type"] == "tool_call" else ("call_id",)
        )
        for field in fields:
            value = event.get(field)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field} must be a non-empty string or None")
        if event.get("call_id") is not None:
            reserved_ids.add(event["call_id"])

    # Do not rely on run_id: callers may omit or reuse it across turns.
    prefix = f"call_{uuid.uuid4().hex}"
    counter = 0
    seen_calls: set[str] = set()
    pending: dict[str, str] = {}
    seen_messages: set[str] = set()
    current_message: str | None = None

    for event in normalized:
        event_type = event.get("type")
        if event_type not in ("tool_call", "tool_result"):
            continue
        name = event.get("name", "")
        call_id = event.get("call_id")

        if event_type == "tool_call":
            if call_id is None:
                call_id = f"{prefix}_{counter}"
                while call_id in reserved_ids:
                    counter += 1
                    call_id = f"{prefix}_{counter}"
                counter += 1
            if call_id in seen_calls:
                raise ValueError(f"Duplicate tool call_id {call_id!r}")

            message_id = event.get("message_id")
            if message_id is not None and message_id != current_message:
                if message_id in seen_messages:
                    raise ValueError(
                        f"message_id {message_id!r} cannot reopen after another "
                        "message or a tool result"
                    )
                seen_messages.add(message_id)
            current_message = message_id
            seen_calls.add(call_id)
            pending[call_id] = name
        else:
            current_message = None
            if call_id is None:
                candidates = [
                    cid for cid, tool_name in pending.items() if tool_name == name
                ]
                if not candidates:
                    raise ValueError(f"No pending tool call for result {name!r}")
                if len(candidates) != 1:
                    raise ValueError(
                        f"Ambiguous tool result for {name!r}; provide call_id"
                    )
                call_id = candidates[0]
            if call_id not in pending:
                reason = "Duplicate result for" if call_id in seen_calls else "Unknown"
                raise ValueError(f"{reason} tool call_id {call_id!r}")
            if pending[call_id] != name:
                raise ValueError(
                    f"Tool result name {name!r} does not match call_id {call_id!r} "
                    f"({pending[call_id]!r})"
                )
            del pending[call_id]

        event["call_id"] = call_id
    return normalized
