"""CPU regressions for all-or-nothing Qwen3-Coder parsing and option wiring."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from areal.api import ModelResponse
from areal.api.cli_args import AgentConfig
from areal.experimental.openai import client as client_module
from areal.experimental.openai import tool_call_parser as parser
from areal.experimental.openai.client import ArealOpenAI

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "Ping", "parameters": {"type": "object"}},
    },
]
CALL = (
    "<tool_call>\n<function=Bash>\n"
    "<parameter=command>\npwd\n</parameter>\n"
    "<parameter=timeout>\n120\n</parameter>\n"
    "</function>\n</tool_call>"
)
EMPTY_CALL = "<tool_call>\n<function=Ping>\n</function>\n</tool_call>"


@pytest.fixture
def no_backend(monkeypatch):
    """Neither successful nor failed strict parses may invoke a backend."""
    backend = Mock(side_effect=AssertionError("Unexpected parser fallback"))
    monkeypatch.setattr(parser, "_process_tool_calls_sglang", backend)
    monkeypatch.setattr(parser, "_process_tool_calls_vllm", backend)
    return backend


def _parse(text, *, use_responses=False, finish_reason="stop", **kwargs):
    tools = (
        [{"type": "function", **tool["function"]} for tool in TOOLS]
        if use_responses
        else TOOLS
    )
    return parser.process_tool_calls(
        text=text,
        tools=tools,
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        finish_reason=finish_reason,
        use_responses=use_responses,
        tool_call_parser_allow_recovery=False,
        **kwargs,
    )


@pytest.mark.parametrize("use_responses", [False, True])
@pytest.mark.parametrize("finish_reason", ["stop", "length", "abort", "tool_calls"])
@pytest.mark.parametrize(
    "text",
    [
        pytest.param(CALL.replace("<function=Bash>", "<parameter=Bash>"), id="typo"),
        pytest.param(
            CALL.replace("</function>", "</function></function>"), id="extra-close"
        ),
        pytest.param(
            CALL.replace("</function>", "</function></function\n"), id="broken-close"
        ),
        pytest.param(
            CALL.replace("<function=Bash>", "garbage<function=Bash>"), id="body-garbage"
        ),
        pytest.param(
            CALL.replace("</function>", "garbage</function>"), id="param-garbage"
        ),
        pytest.param(
            CALL.replace("</parameter>", "</parameter></parameter>", 1),
            id="extra-param-close",
        ),
        pytest.param(
            CALL.replace("<parameter=command>", "<parameter=timeout>"),
            id="duplicate-param",
        ),
        pytest.param(
            CALL.replace("<parameter=command>", "<parameter=command<junk>"),
            id="broken-param-open",
        ),
        pytest.param(
            CALL.replace("<function=Bash>", "<function=Unknown>"), id="unknown-tool"
        ),
        pytest.param(CALL + "\n<tool_call><function=Bash>", id="partial-tail"),
        pytest.param(CALL + "\n<tool_call", id="partial-open"),
        pytest.param(CALL + "\ngarbage", id="suffix"),
        pytest.param(CALL + "\ngarbage\n" + CALL, id="between-calls"),
        pytest.param("<tool_call><function=Bash>" + CALL, id="partial-before-valid"),
        pytest.param("</function>" + CALL, id="stray-prefix"),
        pytest.param(CALL.removesuffix("</tool_call>"), id="missing-tool-close"),
        pytest.param(CALL.replace("</parameter>", "", 1), id="missing-param-close"),
        pytest.param(
            CALL.replace("pwd", "<parameter=nested>pwd</parameter>"), id="nested-param"
        ),
        pytest.param(
            CALL.replace("pwd", "literal </parameter> in code"), id="ambiguous-literal"
        ),
        pytest.param(
            CALL.replace("</function>", "</function><function=Ping></function>"),
            id="two-functions",
        ),
        pytest.param("<tool_call></function></tool_call>", id="observed-empty-call"),
        pytest.param(
            "<tool_call><function=Bash><parameter=command>pwd", id="bare-truncated-call"
        ),
    ],
)
def test_no_recovery_malformed_output_raises_model_error(
    no_backend, text, use_responses, finish_reason
):
    """Reject malformed protocol without executing or repairing any prefix."""
    with pytest.raises(parser.InvalidToolCallError, match="invalid_tool_json"):
        _parse(text, use_responses=use_responses, finish_reason=finish_reason)
    no_backend.assert_not_called()


@pytest.mark.parametrize("use_responses", [False, True])
def test_no_recovery_valid_parallel_and_empty_calls_parse(no_backend, use_responses):
    """Whitespace, natural prefix, empty arguments, and primitive casts work."""
    prefix = "<think>Discuss <tool_call> syntax.</think>\nInspecting.\n"
    calls, remaining, reason = _parse(
        prefix + CALL + "\n \t" + EMPTY_CALL + "<|im_end|>\n",
        use_responses=use_responses,
    )
    functions = calls if use_responses else [call.function for call in calls]
    assert [fn.name for fn in functions] == ["Bash", "Ping"]
    assert [json.loads(fn.arguments) for fn in functions] == [
        {"command": "pwd", "timeout": 120},
        {},
    ]
    assert remaining == prefix + "\n \t"
    assert reason == "tool_calls"
    no_backend.assert_not_called()


@pytest.mark.parametrize("use_responses", [False, True])
@pytest.mark.parametrize(
    "marker", ["<tool_call>", "<function=Bash>", "<parameter=command>", "</parameter>"]
)
def test_no_recovery_prefilled_reasoning_markers_allow_valid_call(
    no_backend, use_responses, marker
):
    """Protocol examples in prefilled thinking cannot invalidate a real call."""
    prefix = f"Discuss {marker} syntax.</think>\nInspecting.\n"

    calls, remaining, reason = _parse(
        prefix + CALL, use_responses=use_responses, reasoning_prefilled=True
    )

    function = calls[0] if use_responses else calls[0].function
    assert function.name == "Bash"
    assert json.loads(function.arguments) == {"command": "pwd", "timeout": 120}
    assert remaining == prefix
    assert reason == "tool_calls"
    no_backend.assert_not_called()


def test_no_recovery_prefilled_reasoning_markers_allow_final_text(no_backend):
    """A protocol example in thinking is not a malformed final answer."""
    text = "Discuss <tool_call> syntax.</think>Task complete."

    assert _parse(text, reasoning_prefilled=True) == (None, text, "stop")
    no_backend.assert_not_called()


@pytest.mark.parametrize(
    "call",
    [
        CALL.replace("<function=Bash>", "<parameter=Bash>"),
        CALL.removesuffix("</tool_call>"),
        CALL + "\n<tool_call>unfinished",
    ],
)
def test_no_recovery_prefilled_reasoning_still_rejects_malformed_call(no_backend, call):
    """Recognizing the reasoning boundary must not repair malformed calls."""
    text = "Discuss <parameter=command> syntax.</think>\n" + call

    with pytest.raises(parser.InvalidToolCallError, match="invalid_tool_json"):
        _parse(text, reasoning_prefilled=True)
    no_backend.assert_not_called()


@pytest.mark.parametrize(
    "value", ["pwd", "literal </think> in code", "<think>literal</think>"]
)
def test_no_recovery_prefilled_unclosed_reasoning_preserves_complete_call(
    no_backend, value
):
    """Keep existing support for Qwen calls emitted without a think close."""
    prefix = "Inspect first.\n"

    calls, remaining, reason = _parse(
        prefix + CALL.replace("pwd", value), reasoning_prefilled=True
    )

    assert calls[0].function.name == "Bash"
    assert json.loads(calls[0].function.arguments)["command"] == value
    assert remaining == prefix
    assert reason == "tool_calls"
    no_backend.assert_not_called()


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("assistant\n<think>\n\n", True),
        ("assistant\n<think>" + " \n" * 40, True),
        ("assistant\n<think>\n</think>\n", False),
        ("assistant\n", False),
        ("", False),
    ],
)
def test_prompt_prefilled_reasoning_uses_actual_token_suffix(prompt, expected):
    """Inspect the rendered prompt, including default-enabled thinking."""
    tokenizer = SimpleNamespace(decode=lambda ids: "".join(map(chr, ids)))

    assert (
        client_module._prompt_prefills_reasoning(tokenizer, list(map(ord, prompt)))
        is expected
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "  keep spaces  ",
        '<div a="b">x</div>',
        "a < b && c > d",
        "<think>literal</think>",
        "literal </think> in code",
    ],
)
def test_no_recovery_preserves_string_arguments(no_backend, value):
    """XML-like code and whitespace are not confused with protocol syntax."""
    calls, _, _ = _parse(CALL.replace("pwd", value))
    assert json.loads(calls[0].function.arguments)["command"] == value


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Task complete.\n",
        "<think>unfinished <tool_call>",
        "<think>reasoning</think>Done.",
    ],
)
def test_no_recovery_no_calls_passthrough(no_backend, text):
    """Ordinary answers and reasoning-only output need no backend parser."""
    assert _parse(text) == (None, text, "stop")


def test_default_recovery_preserves_repairs_and_backend_fallback(monkeypatch):
    """Old callers keep both AReaL's typo repair and SGLang fallback."""
    backend = Mock(return_value=(None, "backend result", "stop"))
    monkeypatch.setattr(parser, "_process_tool_calls_sglang", backend)
    kwargs = dict(
        tools=TOOLS,
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        finish_reason="stop",
    )
    calls, _, _ = parser.process_tool_calls(
        CALL.replace("<function=Bash>", "<parameter=Bash>"), **kwargs
    )
    assert calls[0].function.name == "Bash"
    backend.assert_not_called()
    assert parser.process_tool_calls("<tool_call>broken", **kwargs) == (
        None,
        "backend result",
        "stop",
    )
    backend.assert_called_once()


def test_no_recovery_unsupported_parser_rejected():
    """An unsupported parser cannot silently retain permissive behavior."""
    with pytest.raises(ValueError, match="requires.*qwen3_coder"):
        parser.process_tool_calls(
            "plain",
            TOOLS,
            "qwen25",
            None,
            "stop",
            tool_call_parser_allow_recovery=False,
        )
    with pytest.raises(ValueError, match="requires.*qwen3_coder"):
        AgentConfig(agent_cls_path="test.Agent", tool_call_parser_allow_recovery=False)
    with pytest.raises(ValueError, match="requires.*qwen3_coder"):
        ArealOpenAI(engine=None, tokenizer=None, tool_call_parser_allow_recovery=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "stream", "responses"])
@pytest.mark.parametrize("malformed", [False, True])
async def test_no_recovery_client_preserves_response_and_training_cache(
    monkeypatch, no_backend, api, malformed
):
    """Malformed generations raise but remain cached for zero-reward training."""
    raw = CALL + ("<tool_call>unfinished" if malformed else "")
    tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=0, decode=lambda ids: raw)
    model_response = ModelResponse(
        input_tokens=[1],
        output_tokens=[2, 0],
        output_logprobs=[-0.1, -0.1],
        output_versions=[0, 0],
        tokenizer=tokenizer,
    )
    engine = SimpleNamespace(agenerate=AsyncMock(return_value=model_response))
    monkeypatch.setattr(
        "areal.experimental.openai.client.apply_chat_template", lambda *a, **kw: [1]
    )
    async with ArealOpenAI(
        engine=engine,
        tokenizer=tokenizer,
        api_key="test-key",
        tool_call_parser="qwen3_coder",
        reasoning_parser="",
        tool_call_parser_allow_recovery=False,
    ) as client:
        assert not client.chat.completions.tool_call_parser_allow_recovery
        assert not client.responses.tool_call_parser_allow_recovery

        async def request():
            if api == "responses":
                return await client.responses.create(
                    model="test",
                    input="Run command",
                    max_output_tokens=8,
                    tools=[{"type": "function", **tool["function"]} for tool in TOOLS],
                )
            return await client.chat.completions.create(
                model="test",
                messages=[{"role": "user", "content": "Run command"}],
                tools=TOOLS,
                max_completion_tokens=8,
                stream=api == "stream",
            )

        if malformed:
            with pytest.raises(parser.InvalidToolCallError, match="invalid_tool_json"):
                await request()
            interaction_id = next(reversed(client._cache))
            interaction = client.get_interaction(interaction_id)
            content = interaction.output_message_list[0]["content"]
            assert content == raw or content[0]["text"] == raw
        else:
            response = await request()
            if api == "responses":
                interaction_id = response.id
                assert response.output[0].type == "function_call"
            elif api == "stream":
                chunks = [chunk async for chunk in response]
                interaction_id = chunks[0].id
                assert any(c.choices[0].delta.tool_calls for c in chunks)
            else:
                interaction_id = response.id
                assert response.choices[0].message.tool_calls
        interaction = client.get_interaction(interaction_id)
        assert interaction.model_response is model_response
        assert interaction.model_response.output_tokens == [2, 0]
        assert interaction.reward is None


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "stream", "responses"])
@pytest.mark.parametrize("prefilled", [False, True])
@pytest.mark.parametrize("malformed", [False, True])
async def test_no_recovery_client_uses_rendered_reasoning_boundary(
    monkeypatch, no_backend, api, prefilled, malformed
):
    """Both APIs preserve thinking and raw tokens while validating real calls."""
    prompt = "<|im_start|>assistant\n" + ("<think>\n\n" if prefilled else "")
    prefix = "Discuss <tool_call> and <parameter=command> syntax.</think>\n"
    raw = ("" if prefilled else "<think>") + prefix + CALL
    if malformed:
        raw += "\n<tool_call>unfinished"
    tokenizer = SimpleNamespace(
        eos_token_id=0,
        pad_token_id=0,
        decode=lambda ids: "".join(map(chr, ids)),
        encode=lambda text, **kwargs: list(map(ord, text)),
    )
    output_tokens = [*map(ord, raw), 0]
    model_response = ModelResponse(
        input_tokens=list(map(ord, prompt)),
        output_tokens=output_tokens.copy(),
        output_logprobs=[-0.1] * len(output_tokens),
        output_versions=[0] * len(output_tokens),
        stop_reason="stop",
        tokenizer=tokenizer,
    )
    engine = SimpleNamespace(agenerate=AsyncMock(return_value=model_response))
    monkeypatch.setattr(
        client_module,
        "apply_chat_template",
        lambda *args, **kwargs: list(map(ord, prompt)),
    )
    reasoning_states = []

    def split_reasoning(text, reasoning_parser, force_reasoning=None):
        reasoning_states.append(force_reasoning)
        if force_reasoning or text.startswith("<think>"):
            body = text.removeprefix("<think>")
            return body.split("</think>", 1) if "</think>" in body else (body, "")
        return "", text

    monkeypatch.setattr(client_module, "split_reasoning", split_reasoning)

    async with ArealOpenAI(
        engine=engine,
        tokenizer=tokenizer,
        api_key="test-key",
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        tool_call_parser_allow_recovery=False,
    ) as client:

        async def request():
            if api == "responses":
                return await client.responses.create(
                    model="test",
                    input="Run command",
                    max_output_tokens=len(output_tokens),
                    tools=[{"type": "function", **tool["function"]} for tool in TOOLS],
                )
            return await client.chat.completions.create(
                model="test",
                messages=[{"role": "user", "content": "Run command"}],
                tools=TOOLS,
                max_completion_tokens=len(output_tokens),
                stream=api == "stream",
            )

        if malformed:
            with pytest.raises(parser.InvalidToolCallError, match="invalid_tool_json"):
                await request()
        else:
            response = await request()
            if api == "responses":
                assert any(item.type == "function_call" for item in response.output)
            elif api == "stream":
                chunks = [chunk async for chunk in response]
                assert any(chunk.choices[0].delta.tool_calls for chunk in chunks)
            else:
                assert response.choices[0].message.tool_calls[0].function.name == "Bash"
                assert (
                    "Discuss <tool_call>"
                    in response.choices[0].message.reasoning_content
                )

        interaction = next(reversed(client._cache.values()))
        assert (
            interaction.metadata["response_parse"]["tool_call_parse_failed"]
            is malformed
        )
        assert interaction.model_response.output_tokens == output_tokens
        assert reasoning_states == [prefilled]
        no_backend.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_recovery", [False, True])
async def test_proxy_config_propagates_parser_recovery_option(
    monkeypatch, allow_recovery
):
    """The rollout config reaches both API clients via proxy initialization."""
    from areal.experimental.openai.proxy import proxy_rollout_server as srv

    config = AgentConfig(
        agent_cls_path="test.Agent",
        admin_api_key="test-admin-key",
        tool_call_parser="qwen3_coder",
        tool_call_parser_allow_recovery=allow_recovery,
    )
    engine = SimpleNamespace(
        config=SimpleNamespace(agent=config, tokenizer_path="unused", lora_name="")
    )
    for name in (
        "_openai_client",
        "_admin_api_key",
        "_session_timeout_seconds",
        "_message_preprocessors",
        "_prefix_matcher",
        "_prm_runner",
        "_deterministic_sampling",
    ):
        monkeypatch.setattr(srv, name, getattr(srv, name))
    monkeypatch.setattr(srv, "_engine", engine)
    monkeypatch.setattr(
        srv, "load_hf_processor_and_tokenizer", lambda path: (None, object())
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    srv._setup_openai_client()
    async with srv._openai_client as client:
        assert client.chat.completions.tool_call_parser_allow_recovery == allow_recovery
        assert client.responses.tool_call_parser_allow_recovery == allow_recovery


@pytest.mark.asyncio
async def test_proxy_returns_non_retryable_invalid_tool_json(monkeypatch):
    """The model failure is not mislabeled as a retryable proxy failure."""
    from areal.experimental.openai.proxy import proxy_rollout_server as srv

    session_id = "invalid-tool-session"
    session = srv.SessionData(session_id=session_id)
    monkeypatch.setattr(srv, "_openai_client", object())
    monkeypatch.setattr(srv, "_session_cache", {session_id: session})

    async def invalid_create(**kwargs):
        del kwargs
        raise parser.InvalidToolCallError(
            "invalid_tool_json: model emitted a malformed tool call"
        )

    with pytest.raises(parser.InvalidToolCallError) as exc_info:
        await srv._call_client_create(
            create_fn=invalid_create,
            request={"messages": [{"role": "user", "content": "test"}]},
            session_id=session_id,
        )

    response = await srv.invalid_tool_call_error_response(None, exc_info.value)
    assert response.status_code == 422
    assert json.loads(response.body) == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "code": "invalid_tool_json",
            "message": "invalid_tool_json: model emitted a malformed tool call",
        },
    }
    assert response.headers["x-should-retry"] == "false"
    assert not session.system_error
