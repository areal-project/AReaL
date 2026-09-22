import pytest

pytest.importorskip("litellm")
pytest.importorskip("openai")
pytest.importorskip("tau2")

from examples.tau2.agent import Tau2Runner


def test_clean_messages_preserves_user_tool_turns():
    messages = [
        {"role": "system", "content": "you are the user"},
        {"role": "user", "content": "Have you tried turning it off and on?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "check_status_bar", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "id": "call_1", "content": "Status bar shows: No Service"},
    ]

    cleaned = Tau2Runner._clean_messages(messages)

    assert cleaned == messages
    assert all(message is not original for message, original in zip(cleaned, messages))
