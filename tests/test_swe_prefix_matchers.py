from unittest.mock import MagicMock

import pytest

from examples.swe import prefix_matchers
from examples.swe.prefix_matchers import (
    message_id_prefix_matcher,
    swe_tool_id_prefix_matcher,
)


def _mock_warning(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_logger = MagicMock()
    monkeypatch.setattr(prefix_matchers, "_get_logger", lambda: mock_logger)
    return mock_logger.warning


@pytest.mark.parametrize(
    ("candidate_prefix", "full_messages", "warning_kind", "old_text", "new_text"),
    [
        (
            [
                {"role": "system", "content": "old system"},
                {"role": "user", "content": "same initial user"},
            ],
            [
                {"role": "system", "content": "new system"},
                {"role": "user", "content": "same initial user"},
            ],
            "System prompt",
            "old system",
            "new system",
        ),
        (
            [
                {"role": "system", "content": "same system"},
                {"role": "user", "content": "old initial user"},
            ],
            [
                {"role": "system", "content": "same system"},
                {"role": "user", "content": "new initial user"},
            ],
            "Initial user prompt",
            "old initial user",
            "new initial user",
        ),
    ],
)
def test_swe_tool_id_prefix_matcher_initial_prompt_mismatch_warns_and_matches(
    monkeypatch,
    candidate_prefix,
    full_messages,
    warning_kind,
    old_text,
    new_text,
):
    """Initial system and user text drift is visible but accepted."""
    warning = _mock_warning(monkeypatch)

    assert swe_tool_id_prefix_matcher(candidate_prefix, full_messages)
    assert warning.call_count == 1
    warning_args = warning.call_args.args
    assert warning_args[1] == warning_kind
    assert old_text in warning_args
    assert new_text in warning_args


def test_swe_tool_id_prefix_matcher_tool_call_content_mismatch_warns_and_matches(
    monkeypatch,
):
    """Tool arguments may drift when the ordered tool-call IDs are stable."""
    warning = _mock_warning(monkeypatch)
    candidate_prefix = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"a"}'},
                }
            ],
        }
    ]
    full_messages = [
        {
            "role": "assistant",
            "content": "rewritten assistant text",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"b"}'},
                }
            ],
        }
    ]

    assert swe_tool_id_prefix_matcher(candidate_prefix, full_messages)
    warning.assert_called_once()
    assert "Tool-call content mismatch" in warning.call_args.args[0]
    assert candidate_prefix[0] in warning.call_args.args
    assert full_messages[0] in warning.call_args.args


def test_swe_tool_id_prefix_matcher_tool_response_content_mismatch_matches_silently(
    monkeypatch,
):
    """Tool responses use only their non-empty tool_call_id for matching."""
    warning = _mock_warning(monkeypatch)
    candidate_prefix = [
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"}
    ]
    full_messages = [
        {"role": "tool", "tool_call_id": "call-1", "content": "new result"}
    ]

    assert swe_tool_id_prefix_matcher(candidate_prefix, full_messages)
    warning.assert_not_called()


def test_swe_tool_id_prefix_matcher_follow_up_user_mismatch_warns_and_rejects(
    monkeypatch,
):
    """Later user text drift is visible and prevents a prefix match."""
    warning = _mock_warning(monkeypatch)
    candidate_prefix = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "initial"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "old follow-up"},
    ]
    full_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "initial"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "new follow-up"},
    ]

    assert not swe_tool_id_prefix_matcher(candidate_prefix, full_messages)
    warning.assert_called_once()
    warning_args = warning.call_args.args
    assert warning_args[1] == "Follow-up user prompt"
    assert "old follow-up" in warning_args
    assert "new follow-up" in warning_args


def test_swe_tool_id_prefix_matcher_different_tool_call_id_rejects_without_warning(
    monkeypatch,
):
    """Different tool-call IDs identify different assistant messages."""
    warning = _mock_warning(monkeypatch)
    candidate_prefix = [{"role": "assistant", "tool_calls": [{"id": "call-1"}]}]
    full_messages = [{"role": "assistant", "tool_calls": [{"id": "call-2"}]}]

    assert not swe_tool_id_prefix_matcher(candidate_prefix, full_messages)
    warning.assert_not_called()


@pytest.mark.parametrize(
    ("candidate_message", "full_message"),
    [
        (
            {
                "role": "assistant",
                "content": "first answer",
                "reasoning_content": "same reasoning",
            },
            {
                "role": "assistant",
                "content": "retry answer",
                "reasoning_content": "same reasoning",
            },
        ),
        (
            {
                "role": "assistant",
                "content": "same answer",
                "reasoning_content": "first reasoning",
            },
            {
                "role": "assistant",
                "content": "same answer",
                "reasoning_content": "retry reasoning",
            },
        ),
    ],
)
def test_swe_tool_id_prefix_matcher_plain_assistant_mismatch_warns_and_rejects(
    monkeypatch, candidate_message, full_message
):
    """Plain assistant retries compare all text, including reasoning."""
    warning = _mock_warning(monkeypatch)

    assert not swe_tool_id_prefix_matcher([candidate_message], [full_message])
    warning.assert_called_once()
    warning_args = warning.call_args.args
    assert "Plain assistant message mismatch" in warning_args[0]
    assert warning_args[1] == 0
    assert candidate_message in warning_args
    assert full_message in warning_args


def test_swe_tool_id_prefix_matcher_token_limit_resume_rejects():
    """Arena's malformed token-limit continuation remains a separate branch."""
    candidate_prefix = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "initial"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "truncated output",
        },
    ]
    full_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "initial"},
        {"role": "user", "content": "Output token limit hit. Resume directly."},
    ]

    assert not swe_tool_id_prefix_matcher(candidate_prefix, full_messages)


def test_swe_tool_id_prefix_matcher_other_assistant_to_user_transition_rejects():
    """Unrecognized role changes remain real branches."""
    candidate_prefix = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "initial"},
        {"role": "assistant", "content": "answer"},
    ]
    full_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "initial"},
        {"role": "user", "content": "Please try a different approach."},
    ]

    assert not swe_tool_id_prefix_matcher(candidate_prefix, full_messages)


def test_message_id_prefix_matcher_same_ids_ignores_message_content():
    """Messages with the same IDs match regardless of their other fields."""
    candidate_prefix = [
        {"id": "msg-1", "role": "user", "content": "original"},
        {"id": "msg-2", "role": "assistant", "content": "answer"},
    ]
    full_messages = [
        {"id": "msg-1", "role": "assistant", "content": "rewritten"},
        {"id": "msg-2", "role": "tool", "content": "different"},
        {"id": "msg-3", "role": "user", "content": "next"},
    ]

    assert message_id_prefix_matcher(candidate_prefix, full_messages)


def test_message_id_prefix_matcher_different_id_returns_false():
    """Messages with different IDs do not match."""
    candidate_prefix = [{"id": "msg-1", "content": "same"}]
    full_messages = [{"id": "msg-2", "content": "same"}]

    assert not message_id_prefix_matcher(candidate_prefix, full_messages)


def test_message_id_prefix_matcher_missing_id_returns_false():
    """Messages without explicit IDs do not match each other implicitly."""
    candidate_prefix = [{"role": "user", "content": "same"}]
    full_messages = [{"role": "user", "content": "same"}]

    assert not message_id_prefix_matcher(candidate_prefix, full_messages)


def test_message_id_prefix_matcher_longer_candidate_returns_false():
    """A candidate longer than the full message list cannot be a prefix."""
    candidate_prefix = [{"id": "msg-1"}, {"id": "msg-2"}]
    full_messages = [{"id": "msg-1"}]

    assert not message_id_prefix_matcher(candidate_prefix, full_messages)
