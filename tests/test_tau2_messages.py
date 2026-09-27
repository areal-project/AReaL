from copy import deepcopy

import pytest


def test_clean_messages_preserves_tool_history():
    pytest.importorskip("tau2")
    from examples.tau2.agent import Tau2Runner

    messages = [
        {"role": "system", "content": "you are the user"},
        {"role": "user", "content": "Have you tried turning it off and on?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "check_status_bar", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "Status bar shows: No Service",
        },
    ]
    original_messages = deepcopy(messages)

    cleaned = Tau2Runner._clean_messages(messages)

    assert cleaned == messages
    assert messages == original_messages
