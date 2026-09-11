"""Tests for the Arena Stream dataset and proxy agent integration."""

import asyncio
import json

import httpx
import pytest

from examples.swe.arena_agent import ArenaStreamAgentWorkflow
from examples.swe.arena_client import (
    ArenaAPIError,
    ArenaOpenAPIClient,
    ArenaTaskFailedError,
    infer_llm_protocol,
    resolve_llm_protocol,
)
from examples.swe.filter_function import filter_function

from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow
from areal.infra import workflow_context
from areal.infra.workflow_context import WorkflowContext
from areal.utils import stats_tracker


def _reset_stats() -> None:
    stats_tracker.export_all(reset=True)


def test_resolve_stream_id_when_unspecified_returns_first_active(monkeypatch):
    """The first active Stream should be selected when no id is configured."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["status"] == "ACTIVE"
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(
            200,
            json={
                "items": [
                    {"stream_id": "stream-first", "status": "ACTIVE"},
                    {"stream_id": "stream-second", "status": "ACTIVE"},
                ]
            },
        )

    client = ArenaOpenAPIClient(base_url="https://arena.example")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        stream_id = client.resolve_stream_id(client=http_client)

    assert stream_id == "stream-first"


def test_llm_gateway_api_key_is_required_only_for_gateway_traffic(monkeypatch):
    """Dataset discovery may omit the key, but task launches must require it."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    monkeypatch.delenv("ARENA_LLM_API_KEY", raising=False)
    client = ArenaOpenAPIClient(base_url="https://arena.example")

    with pytest.raises(ValueError, match="ARENA_LLM_API_KEY"):
        _ = client.llm_gateway_api_key


def test_resolve_stream_when_id_is_explicit_returns_matching_metadata(monkeypatch):
    """An explicit Stream should use direct get, independent of list pagination."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/streams/stream-selected")
        return httpx.Response(
            200,
            json={
                "data": {
                    "stream": {
                        "stream_id": "stream-selected",
                        "status": "ACTIVE",
                        "default_harness_ref": {"key": "claude-code"},
                    }
                }
            },
        )

    client = ArenaOpenAPIClient(base_url="https://arena.example")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        stream = client.resolve_stream("stream-selected", client=http_client)

    assert stream["default_harness_ref"] == {"key": "claude-code"}


def test_resolve_stream_when_direct_get_returns_wrong_id(monkeypatch):
    """A stale or misrouted direct response must not select another Stream."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"stream_id": "stream-other"})

    client = ArenaOpenAPIClient(base_url="https://arena.example")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(ArenaAPIError, match="returned id"):
            client.resolve_stream("stream-selected", client=http_client)


def test_resolve_stream_async_when_id_is_explicit_uses_direct_get(monkeypatch):
    """Rollout-time resolution should not depend on Stream-list pagination."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/streams/stream-selected")
        assert "status" not in request.url.params
        return httpx.Response(
            200,
            json={
                "stream_id": "stream-selected",
                "status": "ACTIVE",
                "default_harness_ref": {"key": "claude-code"},
            },
        )

    async def resolve() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = ArenaOpenAPIClient(base_url="https://arena.example")
            return await client.resolve_stream_async(
                "stream-selected",
                client=http_client,
                timeout=10.0,
            )

    stream = asyncio.run(resolve())

    assert stream["default_harness_ref"] == {"key": "claude-code"}


@pytest.mark.parametrize(
    ("harness_key", "expected_protocol"),
    [
        ("Claude-Code-With-Skills", "anthropic"),
        ("openai-codex", "responses"),
        ("swe-agent", "chat_completions"),
        (None, "chat_completions"),
    ],
)
def test_infer_llm_protocol_from_harness_key(harness_key, expected_protocol):
    """Harness names should select their native protocol case-insensitively."""
    stream = (
        {"default_harness_ref": {"key": harness_key}} if harness_key is not None else {}
    )

    assert infer_llm_protocol(stream) == expected_protocol


def test_resolve_llm_protocol_override_bypasses_harness_inference():
    """Explicit protocol selection should bypass Harness-name inference."""
    stream = {"default_harness_ref": {"key": "claude-code"}}

    assert resolve_llm_protocol(stream, "chat_completions") == "chat_completions"


def test_resolve_llm_protocol_rejects_invalid_override():
    with pytest.raises(ValueError, match="Unsupported Arena LLM protocol override"):
        resolve_llm_protocol({}, "invalid")


def test_list_streams_transient_timeout_retries(monkeypatch):
    """Transient read timeouts should be retried before failing discovery."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("transient", request=request)
        return httpx.Response(200, json={"items": [{"stream_id": "stream-1"}]})

    client = ArenaOpenAPIClient(
        base_url="https://arena.example",
        request_retries=1,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        streams = client.list_streams(client=http_client)

    assert streams == [{"stream_id": "stream-1"}]
    assert attempts == 2


def test_list_streams_reuses_owned_client_across_retries(monkeypatch):
    """Owned sync clients should retain connection pooling between attempts."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    real_client = httpx.Client
    attempts = 0
    clients_created = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("transient", request=request)
        return httpx.Response(200, json={"items": [{"stream_id": "stream-1"}]})

    def create_client(*_args, **kwargs):
        nonlocal clients_created
        clients_created += 1
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr("examples.swe.arena_client.httpx.Client", create_client)
    client = ArenaOpenAPIClient(
        base_url="https://arena.example",
        request_retries=1,
    )

    assert client.list_streams() == [{"stream_id": "stream-1"}]
    assert attempts == 2
    assert clients_created == 1


def test_list_streams_transient_gateway_error_retries(monkeypatch):
    """Transient gateway errors should be retried before parsing the response."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(504, text="gateway timeout")
        return httpx.Response(200, json={"items": [{"stream_id": "stream-1"}]})

    client = ArenaOpenAPIClient(
        base_url="https://arena.example",
        request_retries=1,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        streams = client.list_streams(client=http_client)

    assert streams == [{"stream_id": "stream-1"}]
    assert attempts == 2


def test_get_all_dataset_rows_below_api_limit_uses_one_data_page(monkeypatch):
    """A small dataset should need one data page after the total probe."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    requested_limits: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path.endswith("/streams/stream-1/dataset")
        limit = int(request.url.params["limit"])
        requested_limits.append(limit)
        if limit == 1:
            return httpx.Response(
                200,
                json={
                    "data_ids": ["data-1"],
                    "count": 1,
                    "total": 3,
                    "offset": 0,
                    "limit": 1,
                },
            )
        return httpx.Response(
            200,
            json={
                "data_ids": ["data-1", "data-2", "data-3"],
                "items": [
                    {"data_id": "data-1", "tags": ["domain:swe"]},
                    {"data_id": "data-2", "tags": ["domain:math", "level:hard"]},
                    {"data_id": "data-3", "tags": ["level:easy"]},
                ],
                "count": 3,
                "total": 3,
                "offset": 0,
                "limit": 3,
            },
        )

    client = ArenaOpenAPIClient(base_url="https://arena.example")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        rows = client.get_all_dataset_rows("stream-1", client=http_client)

    assert requested_limits == [1, 3]
    assert rows == [
        {
            "data_id": "data-1",
            "stream_id": "stream-1",
            "llm_protocol": "chat_completions",
            "arena_task_type": "swe",
        },
        {
            "data_id": "data-2",
            "stream_id": "stream-1",
            "llm_protocol": "chat_completions",
            "arena_task_type": "math",
        },
        {
            "data_id": "data-3",
            "stream_id": "stream-1",
            "llm_protocol": "chat_completions",
            "arena_task_type": "unknown",
        },
    ]


@pytest.mark.parametrize(
    "tags",
    [
        ["domain:math", "domain:swe"],
        ["domain:swe", 1],
        {"domain": "swe"},
    ],
)
def test_get_all_dataset_rows_domain_parse_failure_uses_unknown(monkeypatch, tags):
    """Malformed or ambiguous domain metadata must not stop the experiment."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data_ids": ["data-1"],
                "items": [{"data_id": "data-1", "tags": tags}],
                "count": 1,
                "total": 1,
                "offset": 0,
                "limit": 1,
            },
        )

    client = ArenaOpenAPIClient(base_url="https://arena.example")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        rows = client.get_all_dataset_rows("stream-1", client=http_client)

    assert rows[0]["arena_task_type"] == "unknown"


def test_get_all_dataset_rows_over_api_limit_paginates(monkeypatch):
    """Dataset loading should paginate Streams larger than the API page limit."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    requests: list[tuple[int, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        limit = int(request.url.params["limit"])
        offset = int(request.url.params["offset"])
        requests.append((limit, offset))
        if limit == 1 and offset == 0 and len(requests) == 1:
            data_ids = ["data-0"]
        else:
            data_ids = [
                f"data-{index}" for index in range(offset, min(offset + limit, 1001))
            ]
        return httpx.Response(
            200,
            json={
                "data_ids": data_ids,
                "count": len(data_ids),
                "total": 1001,
                "offset": offset,
                "limit": limit,
            },
        )

    client = ArenaOpenAPIClient(base_url="https://arena.example")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        rows = client.get_all_dataset_rows("stream-1", client=http_client)

    assert requests == [(1, 0), (1000, 0), (1, 1000)]
    assert len(rows) == 1001
    assert rows[0]["data_id"] == "data-0"
    assert rows[-1]["data_id"] == "data-1000"


def test_get_all_dataset_rows_rejects_total_drift_between_pages(monkeypatch):
    """A changing Stream must not produce a mixed pagination snapshot."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    requests: list[tuple[int, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        limit = int(request.url.params["limit"])
        offset = int(request.url.params["offset"])
        requests.append((limit, offset))
        is_probe = len(requests) == 1
        page_total = 1001 if is_probe or offset == 0 else 1002
        data_ids = (
            ["data-0"]
            if is_probe
            else [f"data-{index}" for index in range(offset, min(offset + limit, 1001))]
        )
        return httpx.Response(
            200,
            json={
                "data_ids": data_ids,
                "count": len(data_ids),
                "total": page_total,
                "offset": offset,
                "limit": limit,
            },
        )

    client = ArenaOpenAPIClient(base_url="https://arena.example")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(ArenaAPIError, match="total changed during pagination"):
            client.get_all_dataset_rows("stream-1", client=http_client)


def test_get_all_dataset_rows_rejects_duplicate_ids_between_pages(monkeypatch):
    """Offset pagination must not silently repeat a row across pages."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        limit = int(request.url.params["limit"])
        offset = int(request.url.params["offset"])
        if requests == 1:
            data_ids = ["data-0"]
        elif offset == 0:
            data_ids = [f"data-{index}" for index in range(1000)]
        else:
            data_ids = ["data-999"]
        return httpx.Response(
            200,
            json={
                "data_ids": data_ids,
                "count": len(data_ids),
                "total": 1001,
                "offset": offset,
                "limit": limit,
            },
        )

    client = ArenaOpenAPIClient(base_url="https://arena.example")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(ArenaAPIError, match="pagination returned duplicate ids"):
            client.get_all_dataset_rows("stream-1", client=http_client)


@pytest.mark.parametrize(
    ("field", "wrong_value", "message"),
    [
        ("offset", 999, "pagination returned offset"),
        ("limit", 999, "pagination returned limit"),
        ("count", 999, "pagination returned count"),
    ],
)
def test_get_all_dataset_rows_rejects_inconsistent_page_metadata(
    monkeypatch, field, wrong_value, message
):
    """Incorrect page metadata must not silently produce a mixed dataset."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        limit = int(request.url.params["limit"])
        offset = int(request.url.params["offset"])
        if requests == 1:
            data_ids = ["data-0"]
        else:
            data_ids = [
                f"data-{index}" for index in range(offset, min(offset + limit, 2))
            ]
        payload = {
            "data_ids": data_ids,
            "count": len(data_ids),
            "total": 2,
            "offset": offset,
            "limit": limit,
        }
        if requests > 1:
            payload[field] = wrong_value
        return httpx.Response(200, json=payload)

    client = ArenaOpenAPIClient(base_url="https://arena.example")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(ArenaAPIError, match=message):
            client.get_all_dataset_rows("stream-1", client=http_client)


def test_register_and_delete_llm_proxy(monkeypatch):
    """OpenAPI registry calls should forward one model and endpoint."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "test-llm-key")
    deployment_id = "deployment-1"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer arena-token"
        if request.method == "POST":
            assert request.url.path.endswith("/openapi/v1/llm/models")
            payload = json.loads(request.content)
            assert payload == {
                "model_name": "stream-areal-test-1",
                "endpoints": [
                    {
                        "endpoint_id": deployment_id,
                        "upstream_model": "stream-areal-test-1",
                        "base_url": "http://rollout-proxy",
                        "api_key": "session-key",
                        "inbound_protos": ["chat"],
                        "enabled": True,
                    }
                ],
                "enabled": True,
                "metadata": {"deployment_id": deployment_id},
            }
            return httpx.Response(
                200,
                json={"model_name": "stream-areal-test-1"},
            )
        assert request.method == "DELETE"
        assert request.url.path.endswith("/openapi/v1/llm/models/stream-areal-test-1")
        return httpx.Response(204)

    client = ArenaOpenAPIClient(base_url="https://arena.example")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        registered_url, returned_id = client.register_llm_proxy(
            model_name="stream-areal-test-1",
            upstream_base_url="http://rollout-proxy",
            upstream_api_key="session-key",
            deployment_id=deployment_id,
            client=http_client,
        )
        client.delete_llm_proxy(returned_id, client=http_client)

    assert registered_url == "https://arena.example/api"
    assert returned_id == "stream-areal-test-1"
    assert len(requests) == 2


def test_probe_llm_proxy_distinguishes_present_and_garbage_collected(monkeypatch):
    """A precise model GET should expose Arena GC as a clean false result."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")

    async def handler(request: httpx.Request) -> httpx.Response:
        model_name = request.url.path.rsplit("/", 1)[-1]
        if model_name == "stream-areal-present":
            return httpx.Response(200, json={"model_name": model_name})
        if model_name == "stream-areal-collected":
            return httpx.Response(404, json={"detail": "not found"})
        return httpx.Response(200, json={"model_name": "stream-areal-wrong"})

    async def probe() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = ArenaOpenAPIClient(base_url="https://arena.example")
            assert await client.llm_proxy_exists_async(
                "stream-areal-present", client=http_client
            )
            assert not await client.llm_proxy_exists_async(
                "stream-areal-collected", client=http_client
            )
            with pytest.raises(ArenaAPIError, match="unexpected model_name"):
                await client.llm_proxy_exists_async(
                    "stream-areal-mismatch", client=http_client
                )

    asyncio.run(probe())


@pytest.mark.parametrize(
    "protocol",
    ["anthropic", "responses", "chat_completions"],
)
def test_register_llm_proxy_always_advertises_openai_chat(monkeypatch, protocol):
    """All Harness protocols should use the AReaL proxy's native Chat API."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "test-llm-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer arena-token"
        payload = json.loads(request.content)
        assert payload["endpoints"] == [
            {
                "endpoint_id": "deployment-1",
                "upstream_model": "stream-areal-test-1",
                "base_url": "http://rollout-proxy",
                "api_key": "session-key",
                "inbound_protos": ["chat"],
                "enabled": True,
            }
        ]
        return httpx.Response(200, json={"model_name": "stream-areal-test-1"})

    client = ArenaOpenAPIClient(base_url="https://arena.example")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client.register_llm_proxy(
            model_name="stream-areal-test-1",
            upstream_base_url="http://rollout-proxy",
            upstream_api_key="session-key",
            deployment_id="deployment-1",
            protocol=protocol,
            client=http_client,
        )


def test_agent_launches_task_through_proxy_and_returns_reward(monkeypatch):
    """The Arena agent should forward proxy credentials and return task reward."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "test-llm-key")
    _reset_stats()

    result_polls = 0
    registered_model_name = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal registered_model_name, result_polls
        if request.method == "POST" and request.url.path.endswith(
            "/openapi/v1/llm/models"
        ):
            assert request.headers["Authorization"] == "Bearer arena-token"
            payload = json.loads(request.content)
            registered_model_name = payload["model_name"]
            assert registered_model_name.startswith("stream-areal-")
            assert payload["endpoints"] == [
                {
                    "endpoint_id": payload["metadata"]["deployment_id"],
                    "upstream_model": registered_model_name,
                    "base_url": "http://rollout-proxy",
                    "api_key": "session-key",
                    "inbound_protos": ["chat"],
                    "enabled": True,
                }
            ]
            return httpx.Response(200, json={"model_name": registered_model_name})
        if request.url.path.endswith("/streams/stream-1/launch_one_task"):
            assert request.headers["Authorization"] == "Bearer arena-token"
            assert json.loads(request.content) == {
                "data_id": "data-1",
                "model_name": registered_model_name,
                "base_url": "https://arena.example/api",
                "api_key": "test-llm-key",
                "harness": "claude-code-with-skills@5.0.1",
                "envs": {
                    "MODEL_NAME": registered_model_name,
                    "BASE_URL": "https://arena.example/api",
                    "API_KEY": "test-llm-key",
                    "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
                },
            }
            return httpx.Response(
                202,
                json={"accepted": True, "task_id": "task-1", "status": "PENDING"},
            )
        if request.method == "DELETE":
            assert request.headers["Authorization"] == "Bearer arena-token"
            assert request.url.path.endswith(
                f"/openapi/v1/llm/models/{registered_model_name}"
            )
            return httpx.Response(204)
        assert request.headers["Authorization"] == "Bearer arena-token"
        assert request.url.path.endswith("/tasks/task-1/result")
        result_polls += 1
        if result_polls == 1:
            return httpx.Response(200, json={"status": "AGENT_RUNNING"})
        return httpx.Response(200, json={"status": "OK", "score": 0.75})

    async def run_agent() -> float:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            workflow = ArenaStreamAgentWorkflow(
                econfig={
                    "arena_base_url": "https://arena.example",
                    "arena_poll_interval": 0.0,
                    "arena_harness": "claude-code-with-skills@5.0.1",
                    "arena_task_envs": {"CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1"},
                    "timeout": 10.0,
                }
            )
            return await workflow.run(
                {
                    "stream_id": "stream-1",
                    "data_id": "data-1",
                    "llm_protocol": "anthropic",
                    "arena_task_type": "swe",
                },
                base_url="http://rollout-proxy",
                api_key="session-key",
                arena_http_client=http_client,
            )

    reward = asyncio.run(run_agent())
    assert reward == 0.75
    assert result_polls == 2
    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/arena/call_success"] == 1.0
    assert stats["rollout/arena/registration_success"] == 1.0
    assert stats["rollout/arena/launch_success"] == 1.0
    assert stats["rollout/arena/terminal_success"] == 1.0
    assert "rollout/reward" not in stats
    assert "rollout/swe/reward" not in stats
    assert stats["rollout/arena/cleanup_success"] == 1.0
    assert stats["rollout/arena/call_success__count"] == 1


def test_agent_direct_route_uses_session_credentials_without_registration(monkeypatch):
    """Direct mode should preserve per-session isolation without registry churn."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.delenv("ARENA_LLM_API_KEY", raising=False)
    _reset_stats()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path.endswith("/streams/stream-1/launch_one_task")
        payload = json.loads(request.content)
        assert payload["session_id"] == "proxy-session-1"
        assert payload["base_url"] == "http://rollout-proxy"
        assert payload["api_key"] == "session-key"
        assert payload["model_name"].startswith("stream-areal-")
        assert payload["envs"] == {
            "MODEL_NAME": payload["model_name"],
            "BASE_URL": "http://rollout-proxy",
            "API_KEY": "session-key",
            "ANTHROPIC_BASE_URL": "http://rollout-proxy",
            "ANTHROPIC_API_KEY": "session-key",
            "OPENAI_BASE_URL": "http://rollout-proxy",
            "OPENAI_API_KEY": "session-key",
            "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
            "NO_PROXY": "rollout-proxy",
            "no_proxy": "rollout-proxy",
        }
        return httpx.Response(
            200,
            json={"task_id": "task-direct", "status": "DONE", "score": 0.25},
        )

    async def run_agent() -> float:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            workflow = ArenaStreamAgentWorkflow(
                econfig={
                    "arena_base_url": "https://arena.example",
                    "arena_llm_route_mode": "direct",
                    "arena_harness": "claude-code@1",
                    "arena_task_envs": {
                        "ANTHROPIC_BASE_URL": "https://arena.example/api",
                        "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
                    },
                }
            )
            return await workflow.run(
                {
                    "stream_id": "stream-1",
                    "data_id": "data-1",
                    "llm_protocol": "anthropic",
                },
                base_url="http://rollout-proxy",
                api_key="session-key",
                session_id="proxy-session-1",
                arena_http_client=http_client,
            )

    assert asyncio.run(run_agent()) == pytest.approx(0.25)
    assert len(requests) == 1
    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/arena/direct_route"] == 1.0
    assert stats["rollout/arena/call_success"] == 1.0
    assert "rollout/arena/registration_success" not in stats
    assert "rollout/arena/cleanup_success" not in stats


def test_agent_direct_route_preserves_configured_no_proxy_hosts(monkeypatch):
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["envs"]["NO_PROXY"] == "localhost,rollout-proxy"
        assert payload["envs"]["no_proxy"] == "localhost,rollout-proxy"
        return httpx.Response(
            200,
            json={"task_id": "task-direct", "status": "DONE", "score": 0.0},
        )

    async def run_agent() -> float:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            workflow = ArenaStreamAgentWorkflow(
                econfig={
                    "arena_base_url": "https://arena.example",
                    "arena_llm_route_mode": "direct",
                    "arena_task_envs": {"NO_PROXY": "localhost"},
                }
            )
            return await workflow.run(
                {"stream_id": "stream-1", "data_id": "data-1"},
                base_url="http://rollout-proxy:1234",
                api_key="session-key",
                session_id="proxy-session-1",
                arena_http_client=http_client,
            )

    assert asyncio.run(run_agent()) == 0.0


def test_agent_direct_route_requires_proxy_session_id(monkeypatch):
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_llm_route_mode": "direct",
        }
    )

    with pytest.raises(ValueError, match="session_id is required"):
        asyncio.run(
            workflow.run(
                {"stream_id": "stream-1", "data_id": "data-1"},
                base_url="http://rollout-proxy",
                api_key="session-key",
            )
        )


def test_agent_session_gateway_reuses_worker_registration_and_binds_sessions(
    monkeypatch,
):
    """Concurrent sessions share one route while retaining explicit isolation."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "arena-gateway-key")
    _reset_stats()
    registration_count = 0
    deletion_count = 0
    launched_sessions: set[str] = set()
    both_launched = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal registration_count, deletion_count
        if request.method == "POST" and request.url.path.endswith(
            "/openapi/v1/llm/models"
        ):
            registration_count += 1
            payload = json.loads(request.content)
            assert payload["model_name"].startswith("stream-areal-session-")
            assert payload["endpoints"] == [
                {
                    "endpoint_id": payload["metadata"]["deployment_id"],
                    "upstream_model": payload["model_name"],
                    "base_url": "http://rollout-proxy",
                    "api_key": "proxy-gateway-key",
                    "inbound_protos": ["chat"],
                    "enabled": True,
                }
            ]
            return httpx.Response(201, json={"model_name": payload["model_name"]})
        if request.method == "POST" and request.url.path.endswith(
            "/streams/stream-1/launch_one_task"
        ):
            payload = json.loads(request.content)
            session_id = payload["session_id"]
            launched_sessions.add(session_id)
            assert payload["base_url"] == "https://arena.example/api"
            assert payload["api_key"] == "arena-gateway-key"
            assert payload["envs"]["ANTHROPIC_CUSTOM_HEADERS"] == (
                f"x-keepalive-enable: true\nX-Session-Id: {session_id}\n"
                f"X-Session-Token: token-{session_id}"
            )
            if len(launched_sessions) == 2:
                both_launched.set()
            await asyncio.wait_for(both_launched.wait(), timeout=1.0)
            return httpx.Response(
                200,
                json={
                    "task_id": f"task-{session_id}",
                    "status": "DONE",
                    "score": 0.5,
                },
            )
        if request.method == "DELETE":
            deletion_count += 1
            return httpx.Response(204)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def run_agents() -> list[float]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            workflow = ArenaStreamAgentWorkflow(
                econfig={
                    "arena_base_url": "https://arena.example",
                    "arena_llm_route_mode": "session_gateway",
                    "arena_task_envs": {
                        "ANTHROPIC_CUSTOM_HEADERS": "x-keepalive-enable: true"
                    },
                }
            )

            async def run_one(session_id: str, data_id: str) -> float:
                return await workflow.run(
                    {"stream_id": "stream-1", "data_id": data_id},
                    base_url="http://rollout-proxy",
                    api_key=f"key-{session_id}",
                    proxy_gateway_api_key="proxy-gateway-key",
                    proxy_session_token=f"token-{session_id}",
                    session_id=session_id,
                    arena_http_client=http_client,
                )

            return await asyncio.gather(
                run_one("proxy-session-1", "data-1"),
                run_one("proxy-session-2", "data-2"),
            )

    assert asyncio.run(run_agents()) == [0.5, 0.5]
    assert launched_sessions == {"proxy-session-1", "proxy-session-2"}
    assert registration_count == 1
    assert deletion_count == 1
    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/arena/session_gateway_route"] == 1.0
    assert stats["rollout/arena/session_gateway_route__count"] == 2


def test_agent_session_gateway_registration_lives_across_prompt_workflows(monkeypatch):
    """Separate prompt workflows on one worker must reuse one Arena model."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "arena-gateway-key")
    _reset_stats()
    registrations: list[str] = []
    async_deletions: list[str] = []
    cleanup_deletions: list[str] = []
    launches: list[tuple[str, str, str]] = []

    class WorkerRuntime:
        def __init__(self) -> None:
            self.callbacks = {}

        def register_destroy_callback(self, key, callback) -> None:
            self.callbacks.setdefault(key, callback)

        def destroy(self) -> None:
            callbacks = list(self.callbacks.values())
            self.callbacks.clear()
            for callback in callbacks:
                callback()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith(
            "/openapi/v1/llm/models"
        ):
            payload = json.loads(request.content)
            registrations.append(payload["model_name"])
            return httpx.Response(201, json={"model_name": payload["model_name"]})
        if request.method == "POST" and request.url.path.endswith(
            "/streams/stream-1/launch_one_task"
        ):
            payload = json.loads(request.content)
            launches.append(
                (
                    payload["model_name"],
                    payload["session_id"],
                    payload["envs"]["ANTHROPIC_CUSTOM_HEADERS"],
                )
            )
            return httpx.Response(
                200,
                json={
                    "task_id": f"task-{payload['session_id']}",
                    "status": "DONE",
                    "score": 0.5,
                },
            )
        if request.method == "DELETE":
            async_deletions.append(request.url.path)
            return httpx.Response(204)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    worker_runtime = WorkerRuntime()
    monkeypatch.setattr(
        ArenaOpenAPIClient,
        "delete_llm_proxy",
        lambda self, model_id, **kwargs: cleanup_deletions.append(model_id),
    )

    async def run_prompts() -> list[float]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            workflows = [
                ArenaStreamAgentWorkflow(
                    econfig={
                        "arena_base_url": "https://arena.example",
                        "arena_llm_route_mode": "session_gateway",
                    }
                )
                for _ in range(2)
            ]
            rewards = []
            for index, workflow in enumerate(workflows, start=1):
                session_id = f"proxy-session-{index}"
                rewards.append(
                    await workflow.run(
                        {
                            "stream_id": "stream-1",
                            "data_id": f"data-{index}",
                        },
                        base_url="http://rollout-proxy",
                        api_key=f"session-key-{index}",
                        proxy_gateway_api_key="proxy-gateway-key",
                        proxy_session_token=f"token-{index}",
                        session_id=session_id,
                        worker_runtime=worker_runtime,
                        arena_http_client=http_client,
                    )
                )
            return rewards

    assert asyncio.run(run_prompts()) == [0.5, 0.5]
    assert len(registrations) == 1
    assert async_deletions == []
    assert [launch[0] for launch in launches] == registrations * 2
    assert [launch[1] for launch in launches] == [
        "proxy-session-1",
        "proxy-session-2",
    ]
    assert "X-Session-Token: token-1" in launches[0][2]
    assert "X-Session-Token: token-2" in launches[1][2]
    assert len(worker_runtime.callbacks) == 1

    worker_runtime.destroy()
    assert cleanup_deletions == registrations


def test_agent_session_gateway_restores_gc_route_before_next_task(monkeypatch):
    """A stale cached route should be probed and restored under its original name."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "arena-gateway-key")
    _reset_stats()
    requests: list[tuple[str, str]] = []
    registrations: list[str] = []
    route_exists = True

    class WorkerRuntime:
        def __init__(self) -> None:
            self.callbacks = {}

        def register_destroy_callback(self, key, callback) -> None:
            self.callbacks.setdefault(key, callback)

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal route_exists
        if request.method == "POST" and request.url.path.endswith(
            "/openapi/v1/llm/models"
        ):
            model_name = json.loads(request.content)["model_name"]
            registrations.append(model_name)
            requests.append(("register", model_name))
            route_exists = True
            return httpx.Response(201, json={"model_name": model_name})
        if request.method == "GET" and "/openapi/v1/llm/models/" in request.url.path:
            model_name = request.url.path.rsplit("/", 1)[-1]
            requests.append(("probe", model_name))
            if route_exists:
                return httpx.Response(200, json={"model_name": model_name})
            return httpx.Response(404, json={"detail": "not found"})
        if request.method == "POST" and request.url.path.endswith(
            "/streams/stream-1/launch_one_task"
        ):
            model_name = json.loads(request.content)["model_name"]
            requests.append(("launch", model_name))
            return httpx.Response(
                200,
                json={"task_id": "task-1", "status": "DONE", "score": 0.5},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def run_tasks() -> None:
        nonlocal route_exists
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            workflow = ArenaStreamAgentWorkflow(
                econfig={
                    "arena_base_url": "https://arena.example",
                    "arena_llm_route_mode": "session_gateway",
                    "arena_registration_probe_interval": 0.01,
                }
            )
            worker_runtime = WorkerRuntime()

            async def run_one(index: int) -> float:
                return await workflow.run(
                    {"stream_id": "stream-1", "data_id": f"data-{index}"},
                    base_url="http://rollout-proxy",
                    api_key=f"session-key-{index}",
                    proxy_gateway_api_key="proxy-gateway-key",
                    proxy_session_token=f"token-{index}",
                    session_id=f"proxy-session-{index}",
                    worker_runtime=worker_runtime,
                    arena_http_client=http_client,
                )

            assert await run_one(1) == 0.5
            route_exists = False
            await asyncio.sleep(0.02)
            assert await run_one(2) == 0.5

    asyncio.run(run_tasks())

    assert len(registrations) == 2
    assert registrations[0] == registrations[1]
    assert requests[-3:] == [
        ("probe", registrations[0]),
        ("register", registrations[0]),
        ("launch", registrations[0]),
    ]


def test_agent_session_gateway_throttles_concurrent_prelaunch_probes(monkeypatch):
    """Concurrent task acquisition should singleflight one probe per worker route."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "arena-gateway-key")
    _reset_stats()
    registration_count = 0
    probe_count = 0

    class WorkerRuntime:
        def __init__(self) -> None:
            self.callbacks = {}

        def register_destroy_callback(self, key, callback) -> None:
            self.callbacks.setdefault(key, callback)

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal registration_count, probe_count
        if request.method == "POST" and request.url.path.endswith(
            "/openapi/v1/llm/models"
        ):
            registration_count += 1
            model_name = json.loads(request.content)["model_name"]
            return httpx.Response(201, json={"model_name": model_name})
        if request.method == "GET" and "/openapi/v1/llm/models/" in request.url.path:
            probe_count += 1
            await asyncio.sleep(0.02)
            model_name = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={"model_name": model_name})
        if request.method == "POST" and request.url.path.endswith(
            "/streams/stream-1/launch_one_task"
        ):
            payload = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "task_id": f"task-{payload['session_id']}",
                    "status": "DONE",
                    "score": 0.5,
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def run_tasks() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            workflow = ArenaStreamAgentWorkflow(
                econfig={
                    "arena_base_url": "https://arena.example",
                    "arena_llm_route_mode": "session_gateway",
                    "arena_registration_probe_interval": 1.0,
                }
            )
            worker_runtime = WorkerRuntime()

            async def run_one(index: int) -> float:
                return await workflow.run(
                    {"stream_id": "stream-1", "data_id": f"data-{index}"},
                    base_url="http://rollout-proxy",
                    api_key=f"session-key-{index}",
                    proxy_gateway_api_key="proxy-gateway-key",
                    proxy_session_token=f"token-{index}",
                    session_id=f"proxy-session-{index}",
                    worker_runtime=worker_runtime,
                    arena_http_client=http_client,
                )

            assert await run_one(0) == 0.5
            registry = getattr(worker_runtime, "_arena_session_gateway_registry_v1")
            entry = next(iter(registry._targets.values()))
            entry.last_probe_at -= 2.0
            rewards = await asyncio.gather(*(run_one(i) for i in range(1, 33)))
            assert rewards == [0.5] * 32

    asyncio.run(run_tasks())

    assert registration_count == 1
    assert probe_count == 1


def test_agent_session_gateway_throttles_slow_probe_failures(monkeypatch):
    """A slow failed probe should throttle from completion, not request start."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "arena-gateway-key")
    _reset_stats()
    probe_count = 0

    class WorkerRuntime:
        def __init__(self) -> None:
            self.callbacks = {}

        def register_destroy_callback(self, key, callback) -> None:
            self.callbacks.setdefault(key, callback)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith(
            "/openapi/v1/llm/models"
        ):
            model_name = json.loads(request.content)["model_name"]
            return httpx.Response(201, json={"model_name": model_name})
        if request.method == "POST" and request.url.path.endswith(
            "/streams/stream-1/launch_one_task"
        ):
            payload = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "task_id": f"task-{payload['session_id']}",
                    "status": "DONE",
                    "score": 0.5,
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def run_tasks() -> None:
        nonlocal probe_count
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            workflow = ArenaStreamAgentWorkflow(
                econfig={
                    "arena_base_url": "https://arena.example",
                    "arena_llm_route_mode": "session_gateway",
                    "arena_registration_probe_interval": 0.05,
                }
            )
            worker_runtime = WorkerRuntime()

            async def run_one(index: int) -> float:
                return await workflow.run(
                    {"stream_id": "stream-1", "data_id": f"data-{index}"},
                    base_url="http://rollout-proxy",
                    api_key=f"session-key-{index}",
                    proxy_gateway_api_key="proxy-gateway-key",
                    proxy_session_token=f"token-{index}",
                    session_id=f"proxy-session-{index}",
                    worker_runtime=worker_runtime,
                    arena_http_client=http_client,
                )

            assert await run_one(0) == 0.5
            registry = getattr(worker_runtime, "_arena_session_gateway_registry_v1")
            entry = next(iter(registry._targets.values()))
            entry.last_probe_at -= 1.0

            async def fail_slow_probe(*args, **kwargs) -> bool:
                nonlocal probe_count
                probe_count += 1
                await asyncio.sleep(0.06)
                raise ArenaAPIError("temporary probe failure")

            monkeypatch.setattr(
                workflow.client, "llm_proxy_exists_async", fail_slow_probe
            )
            assert await run_one(1) == 0.5
            assert await run_one(2) == 0.5

    asyncio.run(run_tasks())

    assert probe_count == 1


def test_agent_session_gateway_does_not_launch_known_missing_route(monkeypatch):
    """A failed restore must reject cached acquires instead of launching a 404 route."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "arena-gateway-key")
    _reset_stats()
    launch_count = 0
    probe_count = 0
    restore_count = 0

    class WorkerRuntime:
        def __init__(self) -> None:
            self.callbacks = {}

        def register_destroy_callback(self, key, callback) -> None:
            self.callbacks.setdefault(key, callback)

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal launch_count
        if request.method == "POST" and request.url.path.endswith(
            "/openapi/v1/llm/models"
        ):
            model_name = json.loads(request.content)["model_name"]
            return httpx.Response(201, json={"model_name": model_name})
        if request.method == "POST" and request.url.path.endswith(
            "/streams/stream-1/launch_one_task"
        ):
            launch_count += 1
            payload = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "task_id": f"task-{payload['session_id']}",
                    "status": "DONE",
                    "score": 0.5,
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def run_tasks() -> None:
        nonlocal probe_count, restore_count
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            workflow = ArenaStreamAgentWorkflow(
                econfig={
                    "arena_base_url": "https://arena.example",
                    "arena_llm_route_mode": "session_gateway",
                    "arena_registration_probe_interval": 60.0,
                }
            )
            worker_runtime = WorkerRuntime()

            async def run_one(index: int) -> float:
                return await workflow.run(
                    {"stream_id": "stream-1", "data_id": f"data-{index}"},
                    base_url="http://rollout-proxy",
                    api_key=f"session-key-{index}",
                    proxy_gateway_api_key="proxy-gateway-key",
                    proxy_session_token=f"token-{index}",
                    session_id=f"proxy-session-{index}",
                    worker_runtime=worker_runtime,
                    arena_http_client=http_client,
                )

            assert await run_one(0) == 0.5
            registry = getattr(worker_runtime, "_arena_session_gateway_registry_v1")
            entry = next(iter(registry._targets.values()))
            entry.last_probe_at -= 61.0

            async def missing_route(*args, **kwargs) -> bool:
                nonlocal probe_count
                probe_count += 1
                return False

            async def fail_restore(*args, **kwargs):
                nonlocal restore_count
                restore_count += 1
                raise ArenaAPIError("temporary restore failure")

            monkeypatch.setattr(
                workflow.client, "llm_proxy_exists_async", missing_route
            )
            monkeypatch.setattr(
                workflow.client, "register_llm_proxy_async", fail_restore
            )
            with pytest.raises(ArenaAPIError, match="temporary restore failure"):
                await run_one(1)
            with pytest.raises(ArenaAPIError, match="restore retries are throttled"):
                await run_one(2)

    asyncio.run(run_tasks())

    assert launch_count == 1
    assert probe_count == 2
    assert restore_count == 1


@pytest.mark.parametrize("probe_interval", [0.0, -1.0, float("nan"), float("inf")])
def test_agent_session_gateway_rejects_invalid_probe_interval(probe_interval):
    """Invalid intervals must not disable probes or create a busy loop."""
    with pytest.raises(ValueError, match="finite and positive"):
        ArenaStreamAgentWorkflow(
            econfig={"arena_registration_probe_interval": probe_interval}
        )


def test_agent_session_gateway_restores_gc_route_while_task_is_queued(monkeypatch):
    """The background probe should protect tasks waiting in Arena's queue."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "arena-gateway-key")
    _reset_stats()
    registrations: list[str] = []
    restored = asyncio.Event()
    route_exists = True

    class WorkerRuntime:
        def __init__(self) -> None:
            self.callbacks = {}

        def register_destroy_callback(self, key, callback) -> None:
            self.callbacks.setdefault(key, callback)

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal route_exists
        if request.method == "POST" and request.url.path.endswith(
            "/openapi/v1/llm/models"
        ):
            model_name = json.loads(request.content)["model_name"]
            registrations.append(model_name)
            route_exists = True
            if len(registrations) == 2:
                restored.set()
            return httpx.Response(201, json={"model_name": model_name})
        if request.method == "GET" and "/openapi/v1/llm/models/" in request.url.path:
            model_name = request.url.path.rsplit("/", 1)[-1]
            if route_exists:
                return httpx.Response(200, json={"model_name": model_name})
            return httpx.Response(404, json={"detail": "not found"})
        if request.method == "POST" and request.url.path.endswith(
            "/streams/stream-1/launch_one_task"
        ):
            route_exists = False
            await asyncio.wait_for(restored.wait(), timeout=1.0)
            return httpx.Response(
                200,
                json={"task_id": "task-queued", "status": "DONE", "score": 0.5},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def run_task() -> float:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            workflow = ArenaStreamAgentWorkflow(
                econfig={
                    "arena_base_url": "https://arena.example",
                    "arena_llm_route_mode": "session_gateway",
                    "arena_registration_probe_interval": 0.01,
                }
            )
            return await workflow.run(
                {"stream_id": "stream-1", "data_id": "data-1"},
                base_url="http://rollout-proxy",
                api_key="session-key",
                proxy_gateway_api_key="proxy-gateway-key",
                proxy_session_token="session-token",
                session_id="proxy-session-1",
                worker_runtime=WorkerRuntime(),
                arena_http_client=http_client,
            )

    assert asyncio.run(run_task()) == 0.5
    assert len(registrations) == 2
    assert registrations[0] == registrations[1]


def test_agent_session_gateway_cleans_uncertain_registration(monkeypatch):
    """A lost POST response must not leak the pre-generated model name."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "arena-gateway-key")
    _reset_stats()
    posted_names: list[str] = []
    deleted_names: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith(
            "/openapi/v1/llm/models"
        ):
            posted_names.append(json.loads(request.content)["model_name"])
            raise httpx.ReadTimeout("lost registration response", request=request)
        if request.method == "DELETE":
            deleted_names.append(request.url.path.rsplit("/", 1)[-1])
            return httpx.Response(404)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def run_agent() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            workflow = ArenaStreamAgentWorkflow(
                econfig={
                    "arena_base_url": "https://arena.example",
                    "arena_llm_route_mode": "session_gateway",
                    "arena_request_retries": 0,
                }
            )
            await workflow.run(
                {"stream_id": "stream-1", "data_id": "data-1"},
                base_url="http://rollout-proxy",
                api_key="session-key",
                proxy_gateway_api_key="proxy-gateway-key",
                proxy_session_token="session-token",
                session_id="proxy-session-1",
                arena_http_client=http_client,
            )

    with pytest.raises(ArenaAPIError, match="request failed"):
        asyncio.run(run_agent())

    assert len(posted_names) == 1
    assert deleted_names == posted_names
    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/arena/registration_success"] == 0.0
    assert stats["rollout/arena/cleanup_success"] == 1.0


def test_agent_session_gateway_requires_routing_capability(monkeypatch):
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_llm_route_mode": "session_gateway",
        }
    )

    with pytest.raises(ValueError, match="proxy_gateway_api_key.*proxy_session_token"):
        asyncio.run(
            workflow.run(
                {"stream_id": "stream-1", "data_id": "data-1"},
                base_url="http://rollout-proxy",
                api_key="session-key",
                session_id="proxy-session-1",
            )
        )


@pytest.mark.parametrize(
    ("raw_reward", "expected_reward"),
    [(0.98, 1.0), (0.9799, 0.0), (1.0, 1.0)],
)
def test_agent_applies_configured_reward_threshold(
    monkeypatch, raw_reward, expected_reward
):
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "test-llm-key")
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_reward_threshold": 0.98,
        }
    )

    async def fake_register(**_kwargs):
        return "http://arena.example/proxy", "model-1"

    async def fake_launch(**_kwargs):
        return raw_reward

    async def fake_delete(*_args, **_kwargs):
        return None

    monkeypatch.setattr(workflow.client, "register_llm_proxy_async", fake_register)
    monkeypatch.setattr(workflow.client, "launch_one_task_result", fake_launch)
    monkeypatch.setattr(workflow.client, "delete_llm_proxy_async", fake_delete)

    async def run_agent():
        return await workflow.run(
            {"stream_id": "stream-1", "data_id": "data-1"},
            base_url="http://rollout-proxy",
            api_key="session-key",
        )

    assert asyncio.run(run_agent()) == expected_reward


@pytest.mark.parametrize(
    ("raw_reward", "expected_reward"),
    [
        (0.0, 0.0),
        (0.2, 0.02),
        (0.75, 0.075),
        (0.9799, 0.09799),
        (0.98, 1.0),
        (1.0, 1.0),
    ],
)
def test_astra_reward_transform_scales_partial_rewards(raw_reward, expected_reward):
    """Partial scores should stay continuous at one tenth strength."""
    from examples.swe.reward_transforms import astra_partial_reward

    assert astra_partial_reward(raw_reward, {}) == pytest.approx(expected_reward)


@pytest.mark.parametrize(
    ("raw_reward", "expected_reward"),
    [(0.8999, 0.08999), (0.9, 1.0), (0.95, 1.0)],
)
def test_agent_passes_configured_threshold_to_astra_transform(
    monkeypatch, raw_reward, expected_reward
):
    """The configured Arena threshold should control Astra reward shaping."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_reward_threshold": 0.9,
            "arena_reward_transform_fn": (
                "examples.swe.reward_transforms.astra_partial_reward"
            ),
        }
    )

    assert workflow._transform_reward(raw_reward, {}) == pytest.approx(expected_reward)


@pytest.mark.parametrize("raw_reward", [-0.01, 1.01, float("nan"), float("inf")])
def test_astra_reward_transform_rejects_invalid_raw_reward(raw_reward):
    """The transform should fail fast on non-probability rewards."""
    from examples.swe.reward_transforms import astra_partial_reward

    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        astra_partial_reward(raw_reward, {})


@pytest.mark.parametrize(
    ("rewards", "expected"),
    [
        ([0.0, 0.0], False),
        ([1.0, 1.0], False),
        ([0.1, 0.1], False),
        ([0.0, 1.0], True),
        ([0.0, 0.1], True),
        ([0.1, 1.0], True),
    ],
)
def test_arena_filter_rejects_all_equal_shaped_groups(rewards, expected):
    """Mean centering gives all-equal groups zero learning signal."""
    _reset_stats()

    assert filter_function({"original_rewards": rewards}) is expected


def test_arena_filter_uses_shaped_original_rewards_before_centering():
    """Centered rewards must not hide variance in shaped original rewards."""
    _reset_stats()
    sample = {
        "rewards": [-0.01, 0.01],
        "original_rewards": [0.02, 0.04],
    }

    assert filter_function(sample) is True


def test_arena_filter_classifies_uniform_partial_rewards_as_all_wrong(monkeypatch):
    """Uniform partial progress is not a solved group in rejection metrics."""
    metrics = {}
    tracker = type(
        "Tracker",
        (),
        {"scalar": lambda _, **values: metrics.update(values)},
    )()
    monkeypatch.setattr(stats_tracker, "get", lambda _: tracker)

    assert filter_function({"original_rewards": [0.05, 0.05]}) is False
    assert metrics["rejected_by_all_correct"] == 0
    assert metrics["rejected_by_all_wrong"] == 1


@pytest.mark.parametrize(
    ("rewards", "expected_all_correct"),
    [
        ([0.999, 0.999], 1),
        ([0.9989, 0.9989], 0),
    ],
)
def test_arena_filter_uses_tolerance_for_all_correct_metrics(
    monkeypatch, rewards, expected_all_correct
):
    """Near-one floating rewards should be classified with a stable tolerance."""
    metrics = {}
    tracker = type(
        "Tracker",
        (),
        {"scalar": lambda _, **values: metrics.update(values)},
    )()
    monkeypatch.setattr(stats_tracker, "get", lambda _: tracker)

    assert filter_function({"original_rewards": rewards}) is False
    assert metrics["rejected_by_all_correct"] == expected_all_correct
    assert metrics["rejected_by_all_wrong"] == 1 - expected_all_correct


def test_agent_passes_configured_threshold_to_custom_reward_transform(monkeypatch):
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "test-llm-key")

    def transform(reward, data, *, reward_threshold):
        assert reward == 0.25
        assert data["data_id"] == "data-1"
        assert reward_threshold == 0.98
        return 0.875

    monkeypatch.setattr(
        "examples.swe.arena_agent.import_from_string", lambda _: transform
    )
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_reward_threshold": 0.98,
            "arena_reward_transform_fn": "tests.test_swe_arena.transform",
        }
    )

    async def fake_register(**_kwargs):
        return "http://arena.example/proxy", "model-1"

    async def fake_launch(**_kwargs):
        return 0.25

    async def fake_delete(*_args, **_kwargs):
        return None

    monkeypatch.setattr(workflow.client, "register_llm_proxy_async", fake_register)
    monkeypatch.setattr(workflow.client, "launch_one_task_result", fake_launch)
    monkeypatch.setattr(workflow.client, "delete_llm_proxy_async", fake_delete)

    async def run_agent():
        return await workflow.run(
            {"stream_id": "stream-1", "data_id": "data-1"},
            base_url="http://rollout-proxy",
            api_key="session-key",
        )

    assert asyncio.run(run_agent()) == 0.875


def test_agent_records_global_and_domain_raw_reward_after_transform(monkeypatch):
    """Reward metrics should expose values on both sides of the transform."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "test-llm-key")
    _reset_stats()

    def transform(reward, _data):
        return reward * 0.1

    monkeypatch.setattr(
        "examples.swe.arena_agent.import_from_string", lambda _: transform
    )
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_reward_transform_fn": "tests.test_swe_arena.transform",
        }
    )

    async def fake_register(**_kwargs):
        return "http://arena.example/proxy", "model-1"

    async def fake_launch(**_kwargs):
        return 0.75

    async def fake_delete(*_args, **_kwargs):
        return None

    monkeypatch.setattr(workflow.client, "register_llm_proxy_async", fake_register)
    monkeypatch.setattr(workflow.client, "launch_one_task_result", fake_launch)
    monkeypatch.setattr(workflow.client, "delete_llm_proxy_async", fake_delete)

    data = {
        "stream_id": "stream-1",
        "data_id": "data-1",
        "arena_task_type": "swe",
    }

    async def run_and_record_metrics():
        async with httpx.AsyncClient(trust_env=False) as http_client:
            reward = await workflow.run(
                data,
                base_url="http://rollout-proxy",
                api_key="session-key",
                arena_http_client=http_client,
            )
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)
        workflow.record_episode_metrics(data, reward)
        return reward

    assert asyncio.run(run_and_record_metrics()) == pytest.approx(0.075)
    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/reward"] == pytest.approx(0.075)
    assert stats["rollout/raw_reward"] == pytest.approx(0.75)
    assert stats["rollout/swe/reward"] == pytest.approx(0.075)
    assert stats["rollout/swe/raw_reward"] == pytest.approx(0.75)
    assert stats["debug/task/data-1/raw_reward"] == pytest.approx(0.75)
    assert "rollout/task/data-1/raw_reward" not in stats


def test_agent_records_global_and_domain_pass_at_k(monkeypatch):
    """A group passes when at least one of its k Arena rollouts succeeds."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    _reset_stats()
    workflow = ArenaStreamAgentWorkflow(
        econfig={"arena_base_url": "https://arena.example"}
    )

    workflow.record_group_metrics(
        {"arena_task_type": "swe"},
        [0.0, None, 1.0, 0.0],
        group_size=4,
    )

    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/pass@k"] == 1.0
    assert stats["rollout/swe/pass@k"] == 1.0


def test_agent_records_failed_pass_at_k_group(monkeypatch):
    """A group fails pass@k when none of its rollouts has positive reward."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    _reset_stats()
    workflow = ArenaStreamAgentWorkflow(
        econfig={"arena_base_url": "https://arena.example"}
    )

    workflow.record_group_metrics(
        {"arena_task_type": "math"},
        [0.0, None, -1.0],
        group_size=3,
    )

    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/pass@k"] == 0.0
    assert stats["rollout/math/pass@k"] == 0.0


def test_agent_does_not_count_middle_bucket_as_pass(monkeypatch):
    """Partial bucket rewards should not inflate Arena pass@k metrics."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    _reset_stats()
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_reward_threshold": 0.98,
            "arena_reward_transform_fn": "examples.swe.reward_transforms.astra_partial_reward",
        }
    )

    workflow.record_group_metrics(
        {"arena_task_type": "swe"},
        [0.0, 0.05, 0.09],
        group_size=3,
    )

    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/pass@k"] == 0.0
    assert stats["rollout/swe/pass@k"] == 0.0


def test_agent_counts_transformed_success_without_counting_partial_reward(monkeypatch):
    """The metric threshold distinguishes transformed success from partial credit."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    _reset_stats()
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_reward_threshold": 0.98,
            "arena_reward_transform_fn": "examples.swe.reward_transforms.astra_partial_reward",
        }
    )

    workflow.record_group_metrics(
        {"arena_task_type": "swe"},
        [0.09799, 1.0],
        group_size=2,
    )

    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/pass@k"] == 1.0
    assert stats["rollout/group_pass_1"] == 1.0


@pytest.mark.parametrize(
    ("threshold", "rewards", "expected_pass"),
    [(2.0, [1.0, 0.0], 1.0), (0.0, [0.0, 0.0], 0.0)],
)
def test_agent_threshold_only_pass_metrics_use_binary_reward(
    monkeypatch, threshold, rewards, expected_pass
):
    """Threshold-only transforms should classify their already-binary output."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    _reset_stats()
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_reward_threshold": threshold,
        }
    )

    workflow.record_group_metrics({}, rewards, group_size=2)

    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/pass@k"] == expected_pass


def test_agent_group_metrics_follow_eval_scope(monkeypatch):
    """Arena metrics should use eval-rollout during evaluation."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    _reset_stats()
    workflow = ArenaStreamAgentWorkflow(
        econfig={"arena_base_url": "https://arena.example"}
    )
    workflow_context.set(WorkflowContext(is_eval=True))
    try:
        workflow.record_group_metrics(
            {"arena_task_type": "swe"},
            [0.0, 1.0],
            group_size=2,
        )
    finally:
        workflow_context.set(WorkflowContext())

    stats = stats_tracker.export_all(reset=True)
    assert stats["eval-rollout/pass@k"] == 1.0
    assert stats["eval-rollout/swe/pass@k"] == 1.0
    assert "rollout/pass@k" not in stats


def test_proxy_records_domain_reward_after_export_boundary(monkeypatch):
    """The proxy callback should record domain reward exactly once in eval scope."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    _reset_stats()
    agent = ArenaStreamAgentWorkflow(
        econfig={"arena_base_url": "https://arena.example"}
    )
    proxy = OpenAIProxyWorkflow(mode="inline", agent=agent)
    workflow_context.set(WorkflowContext(is_eval=True))
    try:
        proxy.record_episode_metrics({"arena_task_type": "swe"}, 0.75)
    finally:
        workflow_context.set(WorkflowContext())

    stats = stats_tracker.export_all(reset=True)
    assert stats["eval-rollout/swe/reward"] == 0.75
    assert stats["eval-rollout/swe/reward__count"] == 1
    assert "rollout/swe/reward" not in stats


@pytest.mark.parametrize("arena_task_type", [None, "", "   ", 1])
def test_agent_invalid_task_type_records_unknown_metrics(monkeypatch, arena_task_type):
    """Invalid task types should be aggregated under unknown instead of failing."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    _reset_stats()
    workflow = ArenaStreamAgentWorkflow(
        econfig={"arena_base_url": "https://arena.example"}
    )

    workflow.record_group_metrics(
        {"arena_task_type": arena_task_type},
        [0.0, 1.0],
        group_size=2,
    )

    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/unknown/pass@k"] == 1.0


def test_agent_records_group_pass_count_distribution(monkeypatch):
    """Each group contributes one-hot values across every pass-count bucket."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    _reset_stats()
    workflow = ArenaStreamAgentWorkflow(
        econfig={"arena_base_url": "https://arena.example"}
    )
    data = {"arena_task_type": "swe"}

    for rewards in (
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ):
        workflow.record_group_metrics(data, rewards, group_size=4)

    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/group_pass_0"] == 0.25
    assert stats["rollout/group_pass_1"] == 0.25
    assert stats["rollout/group_pass_2"] == 0.25
    assert stats["rollout/group_pass_3"] == 0.0
    assert stats["rollout/group_pass_4"] == 0.25
    assert stats["rollout/group_pass_1__count"] == 4


def test_agent_records_arena_terminal_failure_metrics(monkeypatch):
    """Arena terminal failures should be counted separately from reward=0."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "test-llm-key")
    _reset_stats()

    registered_model_name = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal registered_model_name
        if request.method == "POST" and request.url.path.endswith(
            "/openapi/v1/llm/models"
        ):
            registered_model_name = json.loads(request.content)["model_name"]
            return httpx.Response(200, json={"model_name": registered_model_name})
        if request.url.path.endswith("/streams/stream-1/launch_one_task"):
            return httpx.Response(202, json={"task_id": "task-1"})
        if request.method == "DELETE":
            return httpx.Response(204)
        assert request.url.path.endswith("/tasks/task-1/result")
        return httpx.Response(200, json={"status": "COLLECT_FAILED", "score": 0})

    async def run_agent() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            workflow = ArenaStreamAgentWorkflow(
                econfig={
                    "arena_base_url": "https://arena.example",
                    "arena_poll_interval": 0.0,
                    "timeout": 10.0,
                }
            )
            await workflow.run(
                {"stream_id": "stream-1", "data_id": "data-1"},
                base_url="http://rollout-proxy",
                api_key="session-key",
                arena_http_client=http_client,
            )

    with pytest.raises(ArenaAPIError, match="COLLECT_FAILED"):
        asyncio.run(run_agent())

    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/arena/call_success"] == 0.0
    assert stats["rollout/arena/registration_success"] == 1.0
    assert stats["rollout/arena/launch_success"] == 1.0
    assert stats["rollout/arena/terminal_success"] == 0.0
    assert stats["rollout/arena/cleanup_success"] == 1.0
    assert stats["rollout/arena/call_success__count"] == 1


def test_agent_timeout_logs_task_and_data_ids(monkeypatch):
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "test-llm-key")
    _reset_stats()
    workflow = ArenaStreamAgentWorkflow(
        econfig={"arena_base_url": "https://arena.example", "timeout": 0.01}
    )
    warnings: list[tuple[str, tuple[object, ...]]] = []

    async def register_proxy(**_kwargs):
        return (
            "https://arena.example/api",
            "stream-areal-test",
        )

    async def launch_task(*, on_launch_success, **_kwargs):
        on_launch_success("task-123")
        await asyncio.Event().wait()

    async def delete_proxy(*_args, **_kwargs):
        return None

    monkeypatch.setattr(workflow.client, "register_llm_proxy_async", register_proxy)
    monkeypatch.setattr(workflow.client, "launch_one_task_result", launch_task)
    monkeypatch.setattr(workflow.client, "delete_llm_proxy_async", delete_proxy)
    monkeypatch.setattr(
        "examples.swe.arena_agent.logger.warning",
        lambda message, *args: warnings.append((message, args)),
    )

    async def run_agent() -> None:
        async with httpx.AsyncClient() as http_client:
            await workflow.run(
                {"stream_id": "stream-1", "data_id": "data-456"},
                base_url="http://rollout-proxy",
                api_key="session-key",
                arena_http_client=http_client,
            )

    with pytest.raises(TimeoutError):
        asyncio.run(run_agent())

    assert warnings == [
        (
            "Arena task wait timed out: task_id=%s data_id=%s timeout_seconds=%s",
            ("task-123", "data-456", 0.01),
        )
    ]


def test_agent_cleanup_failure_does_not_mask_registration_failure(monkeypatch):
    """Registration errors should survive a second failure during cleanup."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "test-llm-key")
    _reset_stats()
    workflow = ArenaStreamAgentWorkflow(
        econfig={"arena_base_url": "https://arena.example"}
    )
    deleted_models: list[str] = []

    async def fail_registration(**_kwargs):
        raise ArenaAPIError("registration failed")

    async def fail_cleanup(model_name, **_kwargs):
        deleted_models.append(model_name)
        raise ArenaAPIError("cleanup failed")

    monkeypatch.setattr(workflow.client, "register_llm_proxy_async", fail_registration)
    monkeypatch.setattr(workflow.client, "delete_llm_proxy_async", fail_cleanup)

    async def run_agent() -> None:
        async with httpx.AsyncClient() as http_client:
            await workflow.run(
                {"stream_id": "stream-1", "data_id": "data-1"},
                base_url="http://rollout-proxy",
                api_key="session-key",
                arena_http_client=http_client,
            )

    with pytest.raises(ArenaAPIError, match="registration failed"):
        asyncio.run(run_agent())
    assert len(deleted_models) == 1
    assert deleted_models[0].startswith("stream-areal-")
    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/arena/call_success"] == 0.0
    assert stats["rollout/arena/registration_success"] == 0.0
    assert stats["rollout/arena/cleanup_success"] == 0.0


def test_agent_cleanup_failure_preserves_valid_task_reward(monkeypatch):
    """Cleanup failure must not invalidate an already-computed Arena reward."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "test-llm-key")
    _reset_stats()
    workflow = ArenaStreamAgentWorkflow(
        econfig={"arena_base_url": "https://arena.example"}
    )

    async def register(**_kwargs):
        return "http://arena.example/proxy", "model-1"

    async def launch(**_kwargs):
        return 1.0

    async def fail_cleanup(*_args, **_kwargs):
        raise ArenaAPIError("cleanup failed")

    monkeypatch.setattr(workflow.client, "register_llm_proxy_async", register)
    monkeypatch.setattr(workflow.client, "launch_one_task_result", launch)
    monkeypatch.setattr(workflow.client, "delete_llm_proxy_async", fail_cleanup)

    async def run_agent() -> float:
        return await workflow.run(
            {
                "stream_id": "stream-1",
                "data_id": "data-1",
                "arena_task_type": "swe",
            },
            base_url="http://rollout-proxy",
            api_key="session-key",
        )

    assert asyncio.run(run_agent()) == 1.0

    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/arena/cleanup_success"] == 0.0
    assert "rollout/reward" not in stats
    assert "rollout/raw_reward" not in stats
    assert "rollout/swe/reward" not in stats
    assert "rollout/swe/raw_reward" not in stats
    assert "debug/task/data-1/raw_reward" not in stats


def test_launch_one_task_failed_result_raises(monkeypatch):
    """Terminal infrastructure failures must not silently become zero reward."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                202,
                json={"task_id": "task-1", "status": "PENDING"},
            )
        return httpx.Response(200, json={"status": "HARNESS_FAILED"})

    async def launch_task() -> None:
        client = ArenaOpenAPIClient(
            base_url="https://arena.example",
            poll_interval=0.0,
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await client.launch_one_task(
                stream_id="stream-1",
                data_id="data-1",
                model_name="deployment-1",
                proxy_base_url="http://rollout-proxy/v1",
                proxy_api_key="session-key",
                client=http_client,
            )

    with pytest.raises(ArenaAPIError, match="HARNESS_FAILED"):
        asyncio.run(launch_task())


def test_launch_one_task_pending_launch_score_polls_result(monkeypatch):
    """Accepted launch envelopes with placeholder scores must still be polled."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    get_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal get_count
        if request.method == "POST":
            return httpx.Response(
                202,
                json={"task_id": "task-1", "status": "PENDING", "score": 0},
            )
        get_count += 1
        return httpx.Response(200, json={"status": "DONE", "score": 1})

    async def launch_task() -> float:
        client = ArenaOpenAPIClient(
            base_url="https://arena.example",
            poll_interval=0.0,
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            return await client.launch_one_task(
                stream_id="stream-1",
                data_id="data-1",
                model_name="deployment-1",
                proxy_base_url="http://rollout-proxy/v1",
                proxy_api_key="session-key",
                client=http_client,
            )

    result = asyncio.run(launch_task())
    assert result == 1.0
    assert get_count == 1


def test_launch_one_task_failed_launch_score_raises(monkeypatch):
    """Failed launch envelopes with scores must not become zero rewards."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(
            202,
            json={"task_id": "task-1", "status": "HARNESS_FAILED", "score": 0},
        )

    async def launch_task() -> None:
        client = ArenaOpenAPIClient(
            base_url="https://arena.example",
            poll_interval=0.0,
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await client.launch_one_task(
                stream_id="stream-1",
                data_id="data-1",
                model_name="deployment-1",
                proxy_base_url="http://rollout-proxy/v1",
                proxy_api_key="session-key",
                client=http_client,
            )

    with pytest.raises(ArenaTaskFailedError, match="HARNESS_FAILED"):
        asyncio.run(launch_task())
