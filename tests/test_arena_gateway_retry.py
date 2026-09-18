"""Check that transient gateway redirects preserve an already launched task."""

import asyncio

import httpx
import pytest

from examples.swe.arena_client import ArenaAPIError, ArenaOpenAPIClient


@pytest.mark.parametrize("transient", [True, False])
def test_result_redirect_retries_same_task_without_following_or_relaunching(
    monkeypatch, transient
):
    requests = []
    sleeps = []

    async def no_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("examples.swe.arena_client.asyncio.sleep", no_sleep)

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(200, json={"task_id": "task-1", "status": "RUNNING"})
        assert request.url.path == "/openapi/v1/tasks/task-1/result"
        if len(requests) == 2:
            query = "fromspanner=apigwmoe_502" if transient else "return_to=login"
            return httpx.Response(
                302, headers={"location": f"https://gateway.example/wait?{query}"}
            )
        return httpx.Response(
            200, json={"task_id": "task-1", "status": "DONE", "score": 1.0}
        )

    async def run():
        api = ArenaOpenAPIClient(
            base_url="https://arena.example", api_token="test-token", request_retries=1
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await api.launch_one_task_result(
                "stream-1",
                "data-1",
                "model-1",
                "https://proxy.example",
                "test-key",
                client=client,
            )

    if transient:
        result = asyncio.run(run())
        assert result.score == 1.0
        assert len(requests) == 3
        assert requests[1] == requests[2]
        assert sleeps == [1]
    else:
        with pytest.raises(ArenaAPIError, match="HTTP 302"):
            asyncio.run(run())
        assert len(requests) == 2
        assert not sleeps
    assert sum(method == "POST" for method, _ in requests) == 1


def test_gateway_redirect_exhausts_retry_budget(monkeypatch):
    requests = []
    monkeypatch.setattr("examples.swe.arena_client.time.sleep", lambda _: None)

    def handler(request):
        requests.append(request.url)
        return httpx.Response(
            302,
            headers={
                "location": "https://gateway.example/wait?fromspanner=apigwmoe_502"
            },
        )

    api = ArenaOpenAPIClient(
        base_url="https://arena.example", api_token="test-token", request_retries=2
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ArenaAPIError, match="HTTP 302"):
            api.list_streams(client=client)
    assert len(requests) == 3
    assert len(set(requests)) == 1
