# SPDX-License-Identifier: Apache-2.0
"""Score-centering extension transport through the OpenAI HTTP proxy."""

from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy.server import SessionData


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(srv, "_openai_client", object())
    monkeypatch.setattr(srv, "_engine", None)
    monkeypatch.setattr(
        srv, "_session_cache", {"sc-test": SessionData(session_id="sc-test")}
    )
    return srv._session_cache["sc-test"]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_form", ["flat", "nested", "both", "default"])
@pytest.mark.parametrize("pydantic", [False, True])
async def test_head_width_survives_sdk_flattening_and_template_options(
    session, wire_form, pydantic
):
    request = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    if wire_form in ("flat", "both"):
        request["score_centering_top_k"] = 128
    if wire_form in ("nested", "both"):
        request["extra_body"]["score_centering_top_k"] = 128
    if wire_form == "default":
        session.metadata["generation_args"] = {
            "extra_body": {"score_centering_top_k": 128}
        }
    before = deepcopy(request)
    if pydantic:

        class Request(BaseModel):
            model_config = ConfigDict(extra="allow")
            temperature: float | None = None

        request = Request(**request)

    async def create_fn(extra_body=None, **kwargs):
        return extra_body

    result = await srv._call_client_create(create_fn, request, "sc-test")
    assert result == {
        "score_centering_top_k": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if not pydantic:
        assert request == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"presence_penalty": 1},
        {"logit_bias": {"1": 10}},
        {"response_format": {"type": "json_object"}},
        {"temperature": 0.6},
        {"top_p": 0.9},
        {"top_k": 10},
        {"extra_body": {"score_centering_top_k": 4}},
    ],
)
async def test_invalid_score_centering_requests_fail_before_argument_filtering(
    session, options
):
    request = {"score_centering_top_k": 128, **options}

    async def create_fn(extra_body=None, **kwargs):
        pytest.fail("Invalid request reached generation")

    with pytest.raises(HTTPException) as error:
        await srv._call_client_create(create_fn, request, "sc-test")
    assert error.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [0, 64])
@pytest.mark.parametrize("flat", [False, True])
async def test_session_head_width_cannot_be_disabled_or_changed(session, width, flat):
    session.metadata["generation_args"] = {"extra_body": {"score_centering_top_k": 128}}
    request = (
        {"score_centering_top_k": width}
        if flat
        else {"extra_body": {"score_centering_top_k": width}}
    )

    async def create_fn(extra_body=None, **kwargs):
        pytest.fail("Conflicting width reached generation")

    with pytest.raises(HTTPException, match="cannot override"):
        await srv._call_client_create(create_fn, request, "sc-test")


@pytest.mark.asyncio
@pytest.mark.parametrize("flat", [False, True])
async def test_structural_tag_cannot_disappear_during_proxy_filtering(session, flat):
    request = {"score_centering_top_k": 2}
    constraint = {"structural_tag": {"format": "json"}}
    request.update(constraint if flat else {"extra_body": constraint})

    async def create_fn(extra_body=None, **kwargs):
        pytest.fail("Constrained request reached generation")

    with pytest.raises(HTTPException, match="constrained decoding"):
        await srv._call_client_create(create_fn, request, "sc-test")


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["chat/completions", "responses"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("invalid", [None, "constraint", "session_override"])
async def test_http_routes_preserve_extensions_before_validation(
    session, monkeypatch, endpoint, nested, invalid
):
    async def create_fn(extra_body=None, **kwargs):
        return JSONResponse({"extra_body": extra_body})

    monkeypatch.setattr(
        srv,
        "_openai_client",
        SimpleNamespace(
            responses=SimpleNamespace(create=create_fn),
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_fn)),
        ),
    )
    monkeypatch.setitem(
        srv.app.dependency_overrides, srv._require_session_key, lambda: "sc-test"
    )
    extensions = {"score_centering_top_k": 128}
    if invalid == "constraint":
        extensions["structural_tag"] = {"format": "json"}
    elif invalid == "session_override":
        session.metadata["generation_args"] = {
            "extra_body": {"score_centering_top_k": 64}
        }
    request = {"model": "test"}
    if endpoint == "responses":
        request["input"] = "hello"
    else:
        request["messages"] = [{"role": "user", "content": "hello"}]
    request.update({"extra_body": extensions} if nested else extensions)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=srv.app), base_url="http://test"
    ) as client:
        path = (
            srv.RESPONSES_PATHNAME
            if endpoint == "responses"
            else srv.OPENAI_CHAT_COMPLETIONS_PATHNAME
        )
        response = await client.post(f"/{path}", json=request)
    if invalid:
        assert response.status_code == 400, response.text
    else:
        assert response.status_code == 200, response.text
        assert response.json()["extra_body"]["score_centering_top_k"] == 128
