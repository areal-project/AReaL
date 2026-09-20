from areal.experimental.openai.anthropic import translate_anthropic_request


def test_anthropic_stream_queues_are_isolated(monkeypatch):
    from areal.experimental.openai import anthropic

    wrappers = []
    real_wrapper = anthropic.AnthropicStreamWrapper

    def capture_wrapper(**kwargs):
        wrapper = real_wrapper(**kwargs)
        wrappers.append(wrapper)
        return wrapper

    monkeypatch.setattr(anthropic, "AnthropicStreamWrapper", capture_wrapper)

    async def stream():
        if False:
            yield None

    first = anthropic.translate_anthropic_stream(stream(), "test-model")
    second = anthropic.translate_anthropic_stream(stream(), "test-model")
    wrappers[0].chunk_queue.append("pending-first-stream-event")
    assert list(wrappers[1].chunk_queue) == []
    assert first is not second


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
