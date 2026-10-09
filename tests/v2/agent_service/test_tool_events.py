"""Tests for structured tool event normalization, independent of HTTP transport."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from areal.v2.agent_service.worker.events import normalize_tool_events


def _call(name="search", **ids):
    return {"type": "tool_call", "name": name, "args": "{}", **ids}


def _result(name="search", **ids):
    return {"type": "tool_result", "name": name, "result": "ok", **ids}


def test_parallel_calls_preserve_explicit_ids_and_event_order():
    """Same-name results may arrive backwards without rewriting opaque IDs."""
    events = [
        _call(call_id=" a ", message_id="m1"),
        {"type": "delta", "text": "progress"},
        _call(call_id="b", message_id="m1"),
        _result(call_id="b"),
        _result(call_id=" a "),
    ]
    assert normalize_tool_events(events) == events


def test_legacy_serial_calls_match_without_mutating_input():
    """Old two-argument reporting keeps working even for repeated tool names."""
    events = [_call(), _result(), _call(), _result()]
    before = deepcopy(events)
    normalized = normalize_tool_events(events)
    assert events == before
    assert normalized[0]["call_id"] == normalized[1]["call_id"]
    assert normalized[2]["call_id"] == normalized[3]["call_id"]
    assert normalized[0]["call_id"] != normalized[2]["call_id"]
    assert all("message_id" not in event for event in normalized)


def test_results_without_ids_match_only_the_unique_pending_name():
    """Fallback uses the name, not the most recently emitted call."""
    normalized = normalize_tool_events(
        [
            _call("search", call_id="a"),
            _call("lookup"),
            _result("search"),
            _result("lookup"),
        ]
    )
    assert normalized[2]["call_id"] == "a"
    assert normalized[3]["call_id"] == normalized[1]["call_id"]


@pytest.mark.parametrize(
    ("events", "error"),
    [
        pytest.param([_call(), _call(), _result()], "Ambiguous", id="ambiguous-name"),
        pytest.param([_result()], "No pending", id="orphan-result"),
        pytest.param(
            [_call(call_id="a"), _result(call_id="b")], "Unknown", id="unknown-id"
        ),
        pytest.param(
            [_result(call_id="a"), _call(call_id="a")],
            "Unknown",
            id="result-before-call",
        ),
        pytest.param(
            [_call(call_id="a"), _call(call_id="a")], "Duplicate", id="duplicate-call"
        ),
        pytest.param(
            [_call(call_id="a"), _result(call_id="a"), _result(call_id="a")],
            "Duplicate",
            id="duplicate-result",
        ),
        pytest.param(
            [_call(call_id="a"), _result("different", call_id="a")],
            "name",
            id="name-mismatch",
        ),
        pytest.param(
            [_call(call_id="a"), _result(call_id="a"), _call(call_id="a")],
            "Duplicate",
            id="reuse-completed-id",
        ),
    ],
)
def test_invalid_associations_raise_without_changing_events(events, error):
    """Invalid or insufficient identity must never silently select a call."""
    before = deepcopy(events)
    with pytest.raises(ValueError, match=error):
        normalize_tool_events(events)
    assert events == before


@pytest.mark.parametrize("invalid_id", ["", 42, False, []])
@pytest.mark.parametrize(
    ("event_type", "field"),
    [("tool_call", "call_id"), ("tool_call", "message_id"), ("tool_result", "call_id")],
)
def test_invalid_identifiers_are_rejected(event_type, field, invalid_id):
    """Only None means absent; malformed IDs must not be guessed or coerced."""
    with pytest.raises(ValueError, match=field):
        normalize_tool_events([{"type": event_type, field: invalid_id}])


@pytest.mark.parametrize(
    "intervening",
    [
        [_result(call_id="a")],
        [_call(call_id="b", message_id="m2")],
        [_call(call_id="b")],
    ],
)
def test_message_group_cannot_reopen_across_a_boundary(intervening):
    """Grouping cannot move a late call across results or another message."""
    with pytest.raises(ValueError, match="message_id"):
        normalize_tool_events(
            [
                _call(call_id="a", message_id="m1"),
                *intervening,
                _call(call_id="c", message_id="m1"),
            ]
        )


def test_separate_messages_and_call_only_reporting_remain_supported():
    """Do not require a complete tool transcript from observability-only agents."""
    events = [
        _call(call_id="a", message_id="m1"),
        _result(call_id="a"),
        _call(call_id="b", message_id="m2"),
    ]
    assert normalize_tool_events(events) == events


def test_generated_ids_avoid_explicit_ids_even_later_in_the_run():
    """ID generation reserves explicit IDs before allocating fallback IDs."""
    with patch("areal.v2.agent_service.worker.events.uuid.uuid4") as uuid4:
        uuid4.return_value.hex = "seed"
        events = normalize_tool_events(
            [
                _call(),
                _result(),
                _call(call_id="call_seed_0"),
            ]
        )
    assert events[0]["call_id"] != "call_seed_0"
    assert events[1]["call_id"] == events[0]["call_id"]
    assert events[2]["call_id"] == "call_seed_0"
