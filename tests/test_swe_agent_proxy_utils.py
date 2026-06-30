"""Tests for SWE-bench agent proxy helpers."""

from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from examples.swe.prefix_matchers import swe_prefix_matcher
from examples.swe.preprocessors import (
    NormalizeToolCallArguments,
    StripAllSystemReminders,
    StripAnthropicBillingHeader,
    StripAnthropicCacheFields,
)


def _interaction(
    interaction_id: str,
    messages: list[dict],
    output_message_list: list[dict],
) -> InteractionWithTokenLogpReward:
    interaction = InteractionWithTokenLogpReward(
        chat_template_type="concat",
        messages=messages,
        output_message_list=output_message_list,
    )
    interaction.interaction_id = interaction_id
    return interaction


def test_swe_prefix_matcher_ignores_tool_output_content():
    parent = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "old cwd"},
    ]
    child = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "new cwd"},
        {"role": "user", "content": "next"},
    ]

    assert swe_prefix_matcher(parent, child)


def test_interaction_cache_accepts_custom_prefix_matcher():
    cache = InteractionCache(session_id="test", prefix_matcher=swe_prefix_matcher)
    parent_messages = [{"role": "user", "content": "start"}]
    parent_output = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "old cwd"},
    ]
    child_messages = parent_messages + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "new cwd"},
        {"role": "user", "content": "next"},
    ]

    parent = _interaction("parent", parent_messages, parent_output)
    child = _interaction(
        "child",
        child_messages,
        [{"role": "assistant", "content": "done"}],
    )

    cache["parent"] = parent
    cache["child"] = child

    assert child.parent is parent


def test_swe_message_preprocessors_remove_volatile_anthropic_fields():
    messages = [
        {
            "role": "system",
            "content": "x-anthropic-billing-header: cch=abc\nkeep",
        },
        {
            "role": "user",
            "content": "<system-reminder>volatile</system-reminder>task",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "cache_control": {},
                    "provider_specific_fields": {"signature": "x"},
                    "function": {
                        "name": "Edit",
                        "arguments": '{"replace_all": false, "file_path": "a.py"}',
                        "cache_control": {},
                    },
                }
            ],
        },
    ]

    for preprocessor in (
        StripAnthropicBillingHeader(),
        StripAllSystemReminders(),
        StripAnthropicCacheFields(),
        NormalizeToolCallArguments(),
    ):
        messages = preprocessor(messages)

    assert messages[0]["content"] == "keep"
    assert messages[1]["content"] == "task"
    assert "cache_control" not in messages[1]
    tool_call = messages[2]["tool_calls"][0]
    assert "cache_control" not in tool_call
    assert "provider_specific_fields" not in tool_call
    assert "cache_control" not in tool_call["function"]
    assert tool_call["function"]["arguments"] == '{"file_path": "a.py"}'
    assert messages[2]["content"] == ""
