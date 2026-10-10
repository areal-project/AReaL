"""Active Arena routes must survive long periods without generation requests."""

import asyncio
import json
from contextlib import suppress
from types import SimpleNamespace

import httpx
import pytest

from examples.swe.arena_agent import ArenaStreamAgentWorkflow
from examples.swe.arena_client import ArenaAPIError, ArenaOpenAPIClient


@pytest.mark.parametrize(
    "status,name",
    [(200, "stream-areal-test"), (404, None), (200, "wrong"), (403, None)],
)
def test_renewal_preserves_route_and_validates_response(monkeypatch, status, name):
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")

    async def run():
        def handler(request):
            assert request.method == "PATCH"
            assert request.url.path.endswith("/llm/models/stream-areal-test")
            assert json.loads(request.content) == {}
            return httpx.Response(status, json={"model_name": name})

        client = ArenaOpenAPIClient(base_url="https://arena.example")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            if status == 403 or name == "wrong":
                with pytest.raises(ArenaAPIError):
                    await client.renew_llm_proxy_async("stream-areal-test", client=http)
            else:
                assert await client.renew_llm_proxy_async(
                    "stream-areal-test", client=http
                ) == (status == 200)

    asyncio.run(run())


@pytest.mark.parametrize(
    "first_outcome", [False, RuntimeError("temporary outage"), True]
)
def test_live_route_renews_recovers_and_stops_on_cancellation(
    monkeypatch, first_outcome
):
    monkeypatch.setattr(
        "examples.swe.arena_agent._record_arena_metrics", lambda **kw: None
    )

    async def run():
        second_renewal = asyncio.Event()
        calls = []
        restores = []

        async def renew(name, **kwargs):
            calls.append(name)
            if len(calls) == 1:
                if isinstance(first_outcome, Exception):
                    raise first_outcome
                return first_outcome
            second_renewal.set()
            await asyncio.Event().wait()

        async def register(**kwargs):
            restores.append(kwargs)
            return "https://arena.example/api", kwargs["model_name"]

        workflow = object.__new__(ArenaStreamAgentWorkflow)
        workflow.client = SimpleNamespace(
            renew_llm_proxy_async=renew, register_llm_proxy_async=register
        )
        workflow.registration_probe_interval = 0.001
        workflow.registration_timeout = 1
        async with httpx.AsyncClient() as http:
            task = asyncio.create_task(
                workflow._maintain_gateway_registration(
                    model_name="stream-areal-owned",
                    deployment_id="original-deployment",
                    proxy_base_url="http://proxy",
                    proxy_api_key="test-session-key",
                    protocol="chat_completions",
                    client=http,
                )
            )
            await asyncio.wait_for(second_renewal.wait(), timeout=2)
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            assert task.cancelled()
        assert calls == ["stream-areal-owned"] * 2
        assert len(restores) == (1 if first_outcome is False else 0)
        if restores:
            assert restores[0]["model_name"] == "stream-areal-owned"
            assert restores[0]["deployment_id"] == "original-deployment"
            assert restores[0]["upstream_api_key"] == "test-session-key"

    asyncio.run(run())
