"""Tests for _normalize_messages_for_chat_template in ArealOpenAI client.

Covers two normalizations that align ArealOpenAI with SGLang's native
/v1/chat/completions preprocessing:
1. Content flattening: list content → string for templates that expect string
2. tool_calls arguments: JSON string → dict for Jinja2 tojson sort_keys alignment
"""

import json
from typing import Any


def _normalize_messages_for_chat_template(messages: list[dict[str, Any]]) -> None:
    """Copied from areal.experimental.openai.client to avoid heavy import chain."""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if content is not None and not isinstance(content, str):
            if not isinstance(content, list):
                content = list(content)
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    part = (
                        dict(part)
                        if hasattr(part, "items")
                        else {"type": "text", "text": str(part)}
                    )
                if "text" in part and "type" not in part:
                    part["type"] = "text"
                parts.append(part)
            if len(parts) == 1 and parts[0].get("type") == "text":
                msg["content"] = parts[0]["text"]
            else:
                msg["content"] = parts
        if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
            for tool_call in msg["tool_calls"]:
                func = tool_call.get("function", tool_call)
                if isinstance(func.get("arguments"), str):
                    try:
                        func["arguments"] = json.loads(func["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        pass


class TestContentFlattening:
    def test_single_text_part_flattened_to_string(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        _normalize_messages_for_chat_template(msgs)
        assert msgs[0]["content"] == "hello"

    def test_string_content_unchanged(self):
        msgs = [{"role": "user", "content": "hello"}]
        _normalize_messages_for_chat_template(msgs)
        assert msgs[0]["content"] == "hello"

    def test_none_content_unchanged(self):
        msgs = [{"role": "assistant", "content": None}]
        _normalize_messages_for_chat_template(msgs)
        assert msgs[0]["content"] is None

    def test_multi_part_content_kept_as_list(self):
        parts = [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ]
        msgs = [{"role": "user", "content": parts}]
        _normalize_messages_for_chat_template(msgs)
        assert isinstance(msgs[0]["content"], list)
        assert len(msgs[0]["content"]) == 2

    def test_part_missing_type_field_gets_text_type(self):
        msgs = [{"role": "user", "content": [{"text": "hello"}]}]
        _normalize_messages_for_chat_template(msgs)
        assert msgs[0]["content"] == "hello"

    def test_iterator_content_materialized(self):
        def content_iter():
            yield {"type": "text", "text": "from iterator"}

        msgs = [{"role": "user", "content": content_iter()}]
        _normalize_messages_for_chat_template(msgs)
        assert msgs[0]["content"] == "from iterator"

    def test_non_dict_part_converted(self):
        msgs = [{"role": "user", "content": [42]}]
        _normalize_messages_for_chat_template(msgs)
        assert msgs[0]["content"] == "42"


class TestToolCallsArgumentsParsing:
    def test_string_arguments_parsed_to_dict(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "flights", "limit": 5}',
                        }
                    }
                ],
            }
        ]
        _normalize_messages_for_chat_template(msgs)
        args = msgs[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict)
        assert args == {"query": "flights", "limit": 5}

    def test_dict_arguments_unchanged(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": {"query": "flights"},
                        }
                    }
                ],
            }
        ]
        _normalize_messages_for_chat_template(msgs)
        assert msgs[0]["tool_calls"][0]["function"]["arguments"] == {"query": "flights"}

    def test_invalid_json_arguments_kept_as_string(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "fn", "arguments": "not valid json"}}
                ],
            }
        ]
        _normalize_messages_for_chat_template(msgs)
        assert msgs[0]["tool_calls"][0]["function"]["arguments"] == "not valid json"

    def test_user_message_tool_calls_ignored(self):
        msgs = [
            {
                "role": "user",
                "tool_calls": [{"function": {"name": "fn", "arguments": '{"a": 1}'}}],
            }
        ]
        _normalize_messages_for_chat_template(msgs)
        assert msgs[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'

    def test_jinja2_tojson_key_ordering_aligned(self):
        """After parsing, Jinja2 tojson(sort_keys=True) produces alphabetically
        sorted keys, matching SGLang's native /v1/chat/completions path."""
        import jinja2

        args_str = '{"origin": "LAX", "destination": "SFO", "date": "2025-01-15"}'
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "search", "arguments": args_str}}],
            }
        ]
        _normalize_messages_for_chat_template(msgs)
        args_dict = msgs[0]["tool_calls"][0]["function"]["arguments"]

        env = jinja2.Environment()
        template = env.from_string("{{ val | tojson }}")
        rendered = template.render(val=args_dict)

        assert '"date"' in rendered
        assert rendered.index('"date"') < rendered.index('"destination"')
        assert rendered.index('"destination"') < rendered.index('"origin"')


class TestMixedMessages:
    def test_full_conversation_normalized(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": [{"type": "text", "text": "Book a flight"}]},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_flights",
                            "arguments": '{"origin": "LAX", "dest": "SFO"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": '{"flights": []}'},
        ]
        _normalize_messages_for_chat_template(msgs)

        assert msgs[0]["content"] == "You are helpful."
        assert msgs[1]["content"] == "Book a flight"
        assert isinstance(msgs[2]["tool_calls"][0]["function"]["arguments"], dict)
        assert msgs[3]["content"] == '{"flights": []}'

    def test_empty_messages_no_error(self):
        _normalize_messages_for_chat_template([])

    def test_non_dict_message_skipped(self):
        msgs = ["not a dict", {"role": "user", "content": "hello"}]
        _normalize_messages_for_chat_template(msgs)
        assert msgs[1]["content"] == "hello"
