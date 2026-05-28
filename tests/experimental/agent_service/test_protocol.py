"""Unit tests for the Agent Service protocol module."""

from __future__ import annotations

import json

import pytest

from areal.experimental.agent_service.protocol import (
    EventFrame,
    FrameType,
    RequestFrame,
    RequestMethod,
    ResponseFrame,
    RunStatus,
    generate_run_id,
    make_accepted_response,
    make_complete_response,
    make_delta_event,
    make_failed_response,
    make_tool_call_event,
    parse_frame,
    serialize_frame,
)


class TestFrameTypes:
    def test_frame_type_values(self):
        assert FrameType.REQ == "req"
        assert FrameType.RES == "res"
        assert FrameType.EVENT == "event"

    def test_request_method_values(self):
        assert RequestMethod.AGENT == "agent"

    def test_run_status_values(self):
        assert RunStatus.ACCEPTED == "accepted"
        assert RunStatus.COMPLETE == "complete"
        assert RunStatus.FAILED == "failed"


class TestRequestFrame:
    def test_basic_construction(self):
        frame = RequestFrame(
            id="req-1",
            method=RequestMethod.AGENT,
            params={"message": "hello", "sessionKey": "s1", "idempotencyKey": "idk-1"},
        )
        assert frame.id == "req-1"
        assert frame.method == RequestMethod.AGENT
        assert frame.message == "hello"
        assert frame.session_key == "s1"
        assert frame.idempotency_key == "idk-1"

    def test_empty_params_defaults(self):
        frame = RequestFrame(id="r", method=RequestMethod.AGENT)
        assert frame.message == ""
        assert frame.session_key == ""
        assert frame.idempotency_key == ""


class TestResponseFrame:
    def test_basic_construction(self):
        frame = ResponseFrame(
            id="req-1",
            ok=True,
            payload={"runId": "run-abc", "status": "accepted"},
        )
        assert frame.id == "req-1"
        assert frame.ok is True
        assert frame.run_id == "run-abc"
        assert frame.status == "accepted"

    def test_empty_payload_defaults(self):
        frame = ResponseFrame(id="r", ok=False)
        assert frame.run_id == ""
        assert frame.status == ""


class TestEventFrame:
    def test_delta_event(self):
        frame = EventFrame(
            event="agent",
            payload={"runId": "run-1", "delta": "Hello"},
        )
        assert frame.event == "agent"
        assert frame.run_id == "run-1"
        assert frame.delta == "Hello"
        assert frame.tool_call is None

    def test_tool_call_event(self):
        frame = EventFrame(
            event="agent",
            payload={
                "runId": "run-1",
                "toolCall": {"name": "read_file", "args": "/tmp/x"},
            },
        )
        assert frame.delta is None
        assert frame.tool_call == {"name": "read_file", "args": "/tmp/x"}

    def test_empty_payload(self):
        frame = EventFrame(event="agent")
        assert frame.run_id == ""
        assert frame.delta is None
        assert frame.tool_call is None


class TestParseFrame:
    def test_parse_req_frame(self):
        raw = json.dumps(
            {
                "type": "req",
                "id": "req-1",
                "method": "agent",
                "params": {"message": "hi", "sessionKey": "s1"},
            }
        )
        frame = parse_frame(raw)
        assert isinstance(frame, RequestFrame)
        assert frame.id == "req-1"
        assert frame.method == RequestMethod.AGENT
        assert frame.message == "hi"

    def test_parse_res_frame(self):
        raw = json.dumps(
            {
                "type": "res",
                "id": "req-1",
                "ok": True,
                "payload": {"runId": "run-1", "status": "complete"},
            }
        )
        frame = parse_frame(raw)
        assert isinstance(frame, ResponseFrame)
        assert frame.ok is True
        assert frame.run_id == "run-1"

    def test_parse_event_frame(self):
        raw = json.dumps(
            {
                "type": "event",
                "event": "agent",
                "payload": {"runId": "run-1", "delta": "chunk"},
            }
        )
        frame = parse_frame(raw)
        assert isinstance(frame, EventFrame)
        assert frame.delta == "chunk"

    def test_parse_bytes_input(self):
        raw = json.dumps({"type": "req", "id": "r1", "method": "agent"}).encode()
        frame = parse_frame(raw)
        assert isinstance(frame, RequestFrame)

    def test_parse_missing_type_raises(self):
        with pytest.raises(ValueError, match="missing 'type'"):
            parse_frame(json.dumps({"id": "r1"}))

    def test_parse_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown frame type"):
            parse_frame(json.dumps({"type": "unknown"}))


class TestSerializeFrame:
    def test_serialize_request(self):
        frame = RequestFrame(
            id="r1",
            method=RequestMethod.AGENT,
            params={"message": "hello"},
        )
        data = json.loads(serialize_frame(frame))
        assert data["type"] == "req"
        assert data["id"] == "r1"
        assert data["method"] == "agent"
        assert data["params"]["message"] == "hello"

    def test_serialize_response(self):
        frame = ResponseFrame(
            id="r1", ok=True, payload={"runId": "run-1", "status": "complete"}
        )
        data = json.loads(serialize_frame(frame))
        assert data["type"] == "res"
        assert data["ok"] is True

    def test_serialize_event(self):
        frame = EventFrame(event="agent", payload={"runId": "r", "delta": "x"})
        data = json.loads(serialize_frame(frame))
        assert data["type"] == "event"
        assert data["payload"]["delta"] == "x"

    def test_roundtrip_request(self):
        frame = RequestFrame(
            id="r1", method=RequestMethod.AGENT, params={"sessionKey": "s1"}
        )
        parsed = parse_frame(serialize_frame(frame))
        assert isinstance(parsed, RequestFrame)
        assert parsed.id == frame.id
        assert parsed.session_key == "s1"

    def test_roundtrip_response(self):
        frame = ResponseFrame(
            id="r1", ok=False, payload={"runId": "run-1", "status": "failed"}
        )
        parsed = parse_frame(serialize_frame(frame))
        assert isinstance(parsed, ResponseFrame)
        assert parsed.ok is False

    def test_roundtrip_event(self):
        frame = EventFrame(event="agent", payload={"runId": "r", "delta": "hi"})
        parsed = parse_frame(serialize_frame(frame))
        assert isinstance(parsed, EventFrame)
        assert parsed.delta == "hi"


class TestFactoryHelpers:
    def test_make_accepted_response(self):
        frame = make_accepted_response("req-1", "run-1")
        assert frame.id == "req-1"
        assert frame.ok is True
        assert frame.run_id == "run-1"
        assert frame.status == RunStatus.ACCEPTED

    def test_make_complete_response_no_summary(self):
        frame = make_complete_response("req-1", "run-1")
        assert frame.status == RunStatus.COMPLETE
        assert "summary" not in frame.payload

    def test_make_complete_response_with_summary(self):
        frame = make_complete_response("req-1", "run-1", summary="done")
        assert frame.payload["summary"] == "done"

    def test_make_failed_response(self):
        frame = make_failed_response("req-1", "run-1", error="timeout")
        assert frame.ok is False
        assert frame.status == RunStatus.FAILED
        assert frame.payload["error"] == "timeout"

    def test_make_delta_event(self):
        frame = make_delta_event("run-1", "Hello")
        assert frame.event == "agent"
        assert frame.delta == "Hello"

    def test_make_tool_call_event(self):
        frame = make_tool_call_event("run-1", "read_file", "/tmp/x")
        assert frame.tool_call == {"name": "read_file", "args": "/tmp/x"}


class TestGenerateRunId:
    def test_format(self):
        run_id = generate_run_id()
        assert run_id.startswith("run-")
        assert len(run_id) == 16  # "run-" + 12 hex chars

    def test_uniqueness(self):
        ids = {generate_run_id() for _ in range(100)}
        assert len(ids) == 100
