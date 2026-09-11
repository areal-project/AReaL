from areal.experimental.openai.anthropic import translate_anthropic_request


def test_translate_anthropic_request_preserves_tool_round_trip():
    translated = translate_anthropic_request(
        {
            "model": "claude-compatible",
            "max_tokens": 64,
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            ],
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Read",
                            "input": {"path": "/tmp/example"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "example contents",
                        }
                    ],
                },
            ],
        }
    )

    tool_call = translated["messages"][0]["tool_calls"][0]
    assert tool_call["id"] == "tool-1"
    assert tool_call["function"]["name"] == "Read"
    assert translated["messages"][1]["tool_call_id"] == "tool-1"
