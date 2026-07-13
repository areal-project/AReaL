# SPDX-License-Identifier: Apache-2.0

from areal.experimental.openai.client import _parse_tool_call_arguments


def test_string_arguments_are_parsed_to_dict():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "Bash", "arguments": '{"command": "ls"}'}}
            ],
        }
    ]

    out = _parse_tool_call_arguments(messages)

    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}


def test_non_string_arguments_are_left_unchanged():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "Bash", "arguments": {"command": "ls"}}}
            ],
        }
    ]

    out = _parse_tool_call_arguments(messages)

    assert out[0] is messages[0]
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}


def test_invalid_json_arguments_are_left_unchanged():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "Bash", "arguments": "not json"}}],
        }
    ]

    out = _parse_tool_call_arguments(messages)

    assert out[0]["tool_calls"][0]["function"]["arguments"] == "not json"


def test_messages_without_tool_calls_pass_through():
    messages = [{"role": "user", "content": "hi"}]

    out = _parse_tool_call_arguments(messages)

    assert out[0] is messages[0]
