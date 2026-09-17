from types import SimpleNamespace

import pytest

from areal.experimental.openai.proxy import proxy_rollout_server as server
from areal.experimental.openai.proxy.server import SessionData


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [None, False, "off"])
async def test_proxy_preserves_template_defaults_and_request_overrides(
    monkeypatch, override
):
    defaults = {
        "enable_thinking": True,
        "reasoning_effort": "medium",
        "thinking_option": None,
    }
    monkeypatch.setattr(
        server,
        "_engine",
        SimpleNamespace(
            config=SimpleNamespace(agent=SimpleNamespace(chat_template_kwargs=defaults))
        ),
    )
    monkeypatch.setattr(server, "_openai_client", object())
    monkeypatch.setattr(server, "_session_cache", {"test": SessionData("test")})
    monkeypatch.setattr(server, "_message_preprocessors", [])
    monkeypatch.setattr(server, "_deterministic_sampling", False)
    request = {"messages": [], "extra_body": {"other": "kept"}}
    if override is not None:
        request["chat_template_kwargs"] = (
            {"thinking_option": "off"}
            if override == "off"
            else {"enable_thinking": override}
        )

    async def create(messages, extra_body, temperature, top_p, areal_cache):
        return extra_body

    result = await server._call_client_create(create, request, "test")
    if override == "off":
        assert result["chat_template_kwargs"]["thinking_option"] == "off"
        assert "enable_thinking" not in result["chat_template_kwargs"]
    else:
        assert result["chat_template_kwargs"]["enable_thinking"] is (
            True if override is None else override
        )
    assert result["chat_template_kwargs"]["reasoning_effort"] == "medium"
    assert result["other"] == "kept"
    assert defaults["enable_thinking"] is True
    assert request["extra_body"] == {"other": "kept"}
