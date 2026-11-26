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
        if request.url.path.endswith("/cancel"):
            assert request.method == "POST"
            assert request.url.path == "/openapi/v1/jobs/job-1/cancel"
            return httpx.Response(200, json={})
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"task_id": "task-1", "job_id": "job-1", "status": "RUNNING"},
            )
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
        with pytest.raises(ArenaAPIError, match="HTTP 302") as caught:
            asyncio.run(run())
        assert caught.value.arena_failure_reason == "arena_openapi_http_302"
        assert caught.value.task_id == "task-1"
        assert requests == [
            ("POST", "/openapi/v1/streams/stream-1/launch_one_task"),
            ("GET", "/openapi/v1/tasks/task-1/result"),
            ("POST", "/openapi/v1/jobs/job-1/cancel"),
        ]
        assert not sleeps
    assert sum(path.endswith("/launch_one_task") for _, path in requests) == 1


@pytest.mark.parametrize("marker", ["apigwmoe_502", "arenaonlinegateway_504"])
def test_gateway_redirect_exhausts_retry_budget(monkeypatch, marker):
    requests = []
    monkeypatch.setattr("examples.swe.arena_client.time.sleep", lambda _: None)

    def handler(request):
        requests.append(request.url)
        return httpx.Response(
            302,
            headers={"location": f"https://gateway.example/wait?fromspanner={marker}"},
        )

    api = ArenaOpenAPIClient(
        base_url="https://arena.example", api_token="test-token", request_retries=2
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ArenaAPIError, match="HTTP 302") as caught:
            api.list_streams(client=client)
    assert caught.value.arena_failure_reason == f"spanner_{marker}"
    assert len(requests) == 3
    assert len(set(requests)) == 1


@pytest.mark.parametrize(
    "failure",
    [
        "read",
        "read_timeout",
        "connect",
        "remote_protocol",
        429,
        500,
        502,
        503,
        504,
        "apigwmoe_502",
        "arenaonlinegateway_504",
    ],
)
def test_existing_task_survives_exhausted_transient_result_reads(monkeypatch, failure):
    """A temporary result outage must preserve the live task and its caller."""
    requests = []
    launched = []

    async def no_sleep(_):
        pass

    monkeypatch.setattr("examples.swe.arena_client.asyncio.sleep", no_sleep)

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(202, json={"task_id": "task-1", "status": "RUNNING"})
        assert request.url.path == "/openapi/v1/tasks/task-1/result"
        if len(requests) <= 4:
            exceptions = {
                "read": httpx.ReadError,
                "read_timeout": httpx.ReadTimeout,
                "connect": httpx.ConnectError,
                "remote_protocol": httpx.RemoteProtocolError,
            }
            if failure in exceptions:
                raise exceptions[failure]("temporary result outage", request=request)
            if failure in ("apigwmoe_502", "arenaonlinegateway_504"):
                return httpx.Response(
                    302,
                    headers={
                        "location": f"https://gateway.example/wait?fromspanner={failure}"
                    },
                )
            return httpx.Response(failure, text="temporary result outage")
        return httpx.Response(
            200, json={"task_id": "task-1", "status": "DONE", "score": 0.75}
        )

    async def run():
        api = ArenaOpenAPIClient(
            base_url="https://arena.example",
            api_token="test-token",
            request_retries=1,
            poll_interval=0,
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await api.launch_one_task_result(
                "stream-1",
                "data-1",
                "model-1",
                "https://proxy.example",
                "test-key",
                client=client,
                on_launch_success=launched.append,
            )

    result = asyncio.run(run())
    assert result.score == 0.75
    assert launched == ["task-1"]
    assert (
        requests
        == [("POST", "/openapi/v1/streams/stream-1/launch_one_task")]
        + [("GET", "/openapi/v1/tasks/task-1/result")] * 4
    )


@pytest.mark.parametrize(
    "failure", [401, 403, 404, 501, 505, "non_json", "wrong_task", "local_protocol"]
)
def test_permanent_result_read_errors_still_fail(monkeypatch, failure):
    """Do not hide authentication failures, corrupt results, or local request bugs."""
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/openapi/v1/jobs/job-1/cancel":
            return httpx.Response(200, json={"status": "CANCELLED"})
        if request.method == "POST":
            return httpx.Response(
                202,
                json={"task_id": "task-1", "job_id": "job-1", "status": "RUNNING"},
            )
        if failure == "non_json":
            return httpx.Response(200, text="invalid result")
        if failure == "wrong_task":
            return httpx.Response(
                200, json={"task_id": "other-task", "status": "RUNNING"}
            )
        if failure == "local_protocol":
            raise httpx.LocalProtocolError("invalid request header", request=request)
        return httpx.Response(failure, json={"error": "permanent failure"})

    async def run():
        api = ArenaOpenAPIClient(
            base_url="https://arena.example", api_token="test-token", request_retries=0
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

    with pytest.raises(
        ArenaAPIError, match=f"HTTP {failure}" if isinstance(failure, int) else None
    ):
        asyncio.run(asyncio.wait_for(run(), timeout=1))
    assert requests == [
        ("POST", "/openapi/v1/streams/stream-1/launch_one_task"),
        ("GET", "/openapi/v1/tasks/task-1/result"),
        ("POST", "/openapi/v1/jobs/job-1/cancel"),
    ]


@pytest.mark.asyncio
async def test_known_job_cancel_retries_gateway_504_without_relaunch(monkeypatch):
    """A safe cancellation can retry the redirect that a launch cannot replay."""
    requests = []

    async def no_sleep(_):
        pass

    monkeypatch.setattr("examples.swe.arena_client.asyncio.sleep", no_sleep)

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.url.path.endswith("/launch_one_task"):
            return httpx.Response(
                202, json={"task_id": "task-1", "job_id": "job-1", "status": "RUNNING"}
            )
        if request.url.path.endswith("/result"):
            raise TimeoutError("episode expired")
        assert request.url.path == "/openapi/v1/jobs/job-1/cancel"
        if sum(path.endswith("/cancel") for _, path in requests) == 1:
            return httpx.Response(
                302,
                headers={
                    "location": "https://gateway.example/wait?fromspanner=arenaonlinegateway_504"
                },
            )
        return httpx.Response(200, json={})

    api = ArenaOpenAPIClient(
        base_url="https://arena.example", api_token="test-token", request_retries=0
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(TimeoutError, match="episode expired"):
            await api.launch_one_task_result(
                "stream-1",
                "data-1",
                "model-1",
                "https://proxy.example",
                "test-key",
                client=client,
            )
    assert requests == [
        ("POST", "/openapi/v1/streams/stream-1/launch_one_task"),
        ("GET", "/openapi/v1/tasks/task-1/result"),
        ("POST", "/openapi/v1/jobs/job-1/cancel"),
        ("POST", "/openapi/v1/jobs/job-1/cancel"),
    ]


@pytest.mark.asyncio
async def test_known_job_cancel_gateway_retry_obeys_cleanup_deadline(monkeypatch):
    """Repeated redirects cannot extend the existing bounded cleanup window."""
    requests = []

    async def poll(*args):
        raise TimeoutError("episode expired")

    api = ArenaOpenAPIClient(
        base_url="https://arena.example",
        api_token="test-token",
        request_retries=0,
        timeout=0.02,
    )
    monkeypatch.setattr(api, "_poll_task_result", poll)

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.url.path.endswith("/launch_one_task"):
            return httpx.Response(
                202, json={"task_id": "task-1", "job_id": "job-1", "status": "RUNNING"}
            )
        return httpx.Response(
            302,
            headers={
                "location": "https://gateway.example/wait?fromspanner=arenaonlinegateway_504"
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(TimeoutError, match="episode expired"):
            await asyncio.wait_for(
                api.launch_one_task_result(
                    "stream-1",
                    "data-1",
                    "model-1",
                    "https://proxy.example",
                    "test-key",
                    client=client,
                ),
                timeout=0.5,
            )
    assert requests == [
        ("POST", "/openapi/v1/streams/stream-1/launch_one_task"),
        ("POST", "/openapi/v1/jobs/job-1/cancel"),
    ]


@pytest.mark.asyncio
async def test_result_read_retries_preserve_caller_cancellation():
    """An episode timeout must remain able to cancel a persistent read outage."""
    requests = []
    read_started = asyncio.Event()

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/openapi/v1/jobs/job-1/cancel":
            return httpx.Response(200, json={"status": "CANCELLED"})
        if request.method == "POST":
            return httpx.Response(
                202,
                json={"task_id": "task-1", "job_id": "job-1", "status": "RUNNING"},
            )
        read_started.set()
        raise httpx.ReadError("temporary result outage", request=request)

    api = ArenaOpenAPIClient(
        base_url="https://arena.example", api_token="test-token", request_retries=0
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(
            api.launch_one_task_result(
                "stream-1",
                "data-1",
                "model-1",
                "https://proxy.example",
                "test-key",
                client=client,
            )
        )
        await asyncio.wait_for(read_started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert requests == [
        ("POST", "/openapi/v1/streams/stream-1/launch_one_task"),
        ("GET", "/openapi/v1/tasks/task-1/result"),
        ("POST", "/openapi/v1/jobs/job-1/cancel"),
    ]


def test_gateway_504_redirect_during_launch_does_not_repeat_post():
    requests = []

    def handler(request):
        requests.append(request.method)
        return httpx.Response(
            302,
            headers={
                "location": "https://gateway.example/wait?fromspanner=arenaonlinegateway_504"
            },
        )

    async def run():
        api = ArenaOpenAPIClient(
            base_url="https://arena.example", api_token="test-token"
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await api.launch_one_task_result(
                "stream-1",
                "data-1",
                "model-1",
                "https://proxy.example",
                "test-key",
                client=client,
            )

    with pytest.raises(ArenaAPIError, match="HTTP 302"):
        asyncio.run(run())
    assert requests == ["POST"]


def test_gateway_504_during_proxy_delete_retries_same_model(monkeypatch):
    requests = []
    monkeypatch.setattr("examples.swe.arena_client.time.sleep", lambda _: None)

    def handler(request):
        requests.append((request.method, request.url.path))
        if len(requests) == 1:
            return httpx.Response(
                302,
                headers={
                    "location": "https://gateway.example/wait?fromspanner=arenaonlinegateway_504"
                },
            )
        return httpx.Response(404)

    api = ArenaOpenAPIClient(base_url="https://arena.example", api_token="test-token")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api.delete_llm_proxy("stream-areal-test", client=client)
    assert requests == [("DELETE", "/openapi/v1/llm/models/stream-areal-test")] * 2
