# SPDX-License-Identifier: Apache-2.0

import pytest

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
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}'


def test_mapping_arguments_are_left_unchanged():
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


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ("not json", {"arguments": "not json"}),
        ('["a", "b"]', {"arguments": ["a", "b"]}),
        ("null", {"arguments": None}),
        (["a", "b"], {"arguments": ["a", "b"]}),
        (None, {"arguments": None}),
        (7, {"arguments": 7}),
    ],
)
def test_non_mapping_arguments_are_wrapped(arguments, expected):
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "Bash", "arguments": arguments}}],
        }
    ]

    out = _parse_tool_call_arguments(messages)

    assert out[0]["tool_calls"][0]["function"]["arguments"] == expected
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == arguments


def test_messages_without_tool_calls_pass_through():
    messages = [{"role": "user", "content": "hi"}]

    out = _parse_tool_call_arguments(messages)

    assert out[0] is messages[0]
