# SPDX-License-Identifier: Apache-2.0

"""Gateway protocol frame parsing for the Agent Service.

Implements the OpenClaw-like WebSocket protocol with three frame types:
``req`` (client→gateway), ``res`` (gateway→client), and ``event``
(gateway→client streaming). All frames are JSON text with a ``type``
discriminator.

Reference: https://docs.openclaw.ai/gateway/protocol
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FrameType(str, Enum):
    """WebSocket frame type discriminator."""

    REQ = "req"
    RES = "res"
    EVENT = "event"


class RequestMethod(str, Enum):
    """Supported request methods in ``req`` frames."""

    AGENT = "agent"


class QueueMode(str, Enum):
    """Queue mode for inbound messages when a session is already running.

    Controls how the gateway handles new ``req`` frames that arrive while
    a prior run for the same session is still active.

    Reference: https://docs.openclaw.ai/concepts/queue
    """

    COLLECT = "collect"
    """Coalesce queued messages into a single followup turn (default)."""

    FOLLOWUP = "followup"
    """Enqueue for the next agent turn after the current run ends."""


class RunStatus(str, Enum):
    """Status values for ``res`` frame payloads."""

    ACCEPTED = "accepted"
    COMPLETE = "complete"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Frame dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RequestFrame:
    """Client-to-gateway request frame.

    Attributes
    ----------
    id : str
        Unique request identifier, client-generated.
    method : RequestMethod
        The method to invoke (currently only ``agent``).
    params : dict[str, Any]
        Parameters including ``message``, ``sessionKey``, and
        ``idempotencyKey``.
    """

    id: str
    method: RequestMethod
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def message(self) -> str:
        return self.params.get("message", "")

    @property
    def session_key(self) -> str:
        return self.params.get("sessionKey", "")

    @property
    def idempotency_key(self) -> str:
        return self.params.get("idempotencyKey", "")

    @property
    def queue_mode(self) -> QueueMode:
        raw = self.params.get("queueMode", QueueMode.COLLECT)
        try:
            return QueueMode(raw)
        except ValueError:
            return QueueMode.COLLECT


@dataclass
class ResponseFrame:
    """Gateway-to-client response frame.

    Attributes
    ----------
    id : str
        Matching request ID from the originating ``req`` frame.
    ok : bool
        Whether the request succeeded.
    payload : dict[str, Any]
        Response payload containing ``runId`` and ``status``.
    """

    id: str
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return self.payload.get("runId", "")

    @property
    def status(self) -> str:
        return self.payload.get("status", "")


@dataclass
class EventFrame:
    """Gateway-to-client streaming event frame.

    Attributes
    ----------
    event : str
        Event category (e.g. ``"agent"``).
    payload : dict[str, Any]
        Event payload containing ``runId`` and optional ``delta`` /
        ``toolCall`` fields.
    """

    event: str
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return self.payload.get("runId", "")

    @property
    def delta(self) -> str | None:
        return self.payload.get("delta")

    @property
    def tool_call(self) -> dict[str, Any] | None:
        return self.payload.get("toolCall")


# Type alias for any frame.
Frame = RequestFrame | ResponseFrame | EventFrame


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def parse_frame(raw: str | bytes) -> Frame:
    """Parse a raw WebSocket text frame into a typed dataclass.

    Parameters
    ----------
    raw : str | bytes
        JSON-encoded WebSocket text frame.

    Returns
    -------
    Frame
        One of :class:`RequestFrame`, :class:`ResponseFrame`, or
        :class:`EventFrame`.

    Raises
    ------
    ValueError
        If the frame is missing required fields or has an unknown type.
    """
    data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw

    frame_type = data.get("type")
    if frame_type is None:
        raise ValueError("Frame missing 'type' field")

    if frame_type == FrameType.REQ:
        return RequestFrame(
            id=data["id"],
            method=RequestMethod(data.get("method", "agent")),
            params=data.get("params", {}),
        )
    if frame_type == FrameType.RES:
        return ResponseFrame(
            id=data["id"],
            ok=data.get("ok", True),
            payload=data.get("payload", {}),
        )
    if frame_type == FrameType.EVENT:
        return EventFrame(
            event=data.get("event", "agent"),
            payload=data.get("payload", {}),
        )
    raise ValueError(f"Unknown frame type: {frame_type!r}")


def serialize_frame(frame: Frame) -> str:
    """Serialize a frame dataclass to a JSON string.

    Parameters
    ----------
    frame : Frame
        A :class:`RequestFrame`, :class:`ResponseFrame`, or
        :class:`EventFrame`.

    Returns
    -------
    str
        JSON-encoded string ready to send over WebSocket.
    """
    if isinstance(frame, RequestFrame):
        return json.dumps(
            {
                "type": FrameType.REQ,
                "id": frame.id,
                "method": frame.method.value,
                "params": frame.params,
            }
        )
    if isinstance(frame, ResponseFrame):
        return json.dumps(
            {
                "type": FrameType.RES,
                "id": frame.id,
                "ok": frame.ok,
                "payload": frame.payload,
            }
        )
    if isinstance(frame, EventFrame):
        return json.dumps(
            {
                "type": FrameType.EVENT,
                "event": frame.event,
                "payload": frame.payload,
            }
        )
    raise TypeError(f"Cannot serialize {type(frame).__name__}")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_accepted_response(request_id: str, run_id: str) -> ResponseFrame:
    """Create an ``accepted`` response frame for a request."""
    return ResponseFrame(
        id=request_id,
        ok=True,
        payload={"runId": run_id, "status": RunStatus.ACCEPTED},
    )


def make_complete_response(
    request_id: str, run_id: str, summary: str = ""
) -> ResponseFrame:
    """Create a ``complete`` response frame for a finished run."""
    payload: dict[str, Any] = {"runId": run_id, "status": RunStatus.COMPLETE}
    if summary:
        payload["summary"] = summary
    return ResponseFrame(id=request_id, ok=True, payload=payload)


def make_failed_response(
    request_id: str, run_id: str, error: str = ""
) -> ResponseFrame:
    """Create a ``failed`` response frame for an errored run."""
    payload: dict[str, Any] = {"runId": run_id, "status": RunStatus.FAILED}
    if error:
        payload["error"] = error
    return ResponseFrame(id=request_id, ok=False, payload=payload)


def make_delta_event(run_id: str, delta: str) -> EventFrame:
    """Create a text delta streaming event."""
    return EventFrame(
        event="agent",
        payload={"runId": run_id, "delta": delta},
    )


def make_tool_call_event(run_id: str, tool_name: str, tool_args: str) -> EventFrame:
    """Create a tool call streaming event."""
    return EventFrame(
        event="agent",
        payload={
            "runId": run_id,
            "toolCall": {"name": tool_name, "args": tool_args},
        },
    )


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return f"run-{uuid.uuid4().hex[:12]}"
