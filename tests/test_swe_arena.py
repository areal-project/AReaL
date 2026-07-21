"""Tests for the Arena Stream dataset and proxy agent integration."""

import asyncio
import json

import httpx
import pytest
import torch

from examples.swe.arena_agent import ArenaStreamAgentWorkflow
from examples.swe.arena_client import (
    ArenaAPIError,
    ArenaOpenAPIClient,
    infer_llm_protocol,
)
from examples.swe.arena_rollout_only import _run_rollout_tasks, _trajectory_reward


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


def test_resolve_stream_when_id_is_explicit_returns_matching_metadata(monkeypatch):
    """An explicit Stream should still be discovered so its Harness is available."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "items": [
                    {"stream_id": "stream-first"},
                    {
                        "stream_id": "stream-selected",
                        "default_harness_ref": {"key": "claude-code"},
                    },
                ]
            },
        )

    client = ArenaOpenAPIClient(base_url="https://arena.example")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        stream = client.resolve_stream("stream-selected", client=http_client)

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


def test_get_all_dataset_rows_uses_total_as_limit(monkeypatch):
    """Dataset loading should probe total then request all rows in one page."""
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
        },
        {
            "data_id": "data-2",
            "stream_id": "stream-1",
            "llm_protocol": "chat_completions",
        },
        {
            "data_id": "data-3",
            "stream_id": "stream-1",
            "llm_protocol": "chat_completions",
        },
    ]


def test_get_all_dataset_rows_over_api_limit_raises(monkeypatch):
    """The initial implementation should reject Streams that require pagination."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data_ids": ["data-1"],
                "count": 1,
                "total": 1001,
                "offset": 0,
                "limit": 1,
            },
        )

    client = ArenaOpenAPIClient(base_url="https://arena.example")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(ArenaAPIError, match="pagination is not implemented"):
            client.get_all_dataset_rows("stream-1", client=http_client)


def test_register_and_delete_llm_proxy(monkeypatch):
    """Registry calls should forward one deployment id and the proxy session."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "test-llm-key")
    deployment_id = "deployment-1"
    registered_model_id = "registered-model-1"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer test-llm-key"
        payload = json.loads(request.content)
        if request.url.path.endswith("/llm/model/new"):
            assert payload == {
                "model_name": "stream-areal-test-1",
                "litellm_params": {
                    "model": "openai/stream-areal-test-1",
                    "api_base": "http://rollout-proxy",
                    "api_key": "session-key",
                    "custom_llm_provider": "openai",
                },
                "model_info": {
                    "id": deployment_id,
                    "mode": "chat",
                    "disable_background_health_check": True,
                },
            }
            return httpx.Response(
                200,
                json={
                    "model_id": registered_model_id,
                    "model_name": "stream-areal-test-1",
                    "litellm_params": {},
                    "model_info": {"id": deployment_id},
                },
            )
        assert request.url.path.endswith("/llm/model/delete")
        assert payload == {"id": registered_model_id}
        return httpx.Response(200, json="deleted")

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

    assert registered_url == "https://arena.example/llm"
    assert returned_id == registered_model_id
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("protocol", "expected_model", "expected_provider", "expected_mode"),
    [
        (
            "anthropic",
            "anthropic/stream-areal-test-1",
            "anthropic",
            "chat",
        ),
        ("responses", "openai/stream-areal-test-1", "openai", "responses"),
    ],
)
def test_register_llm_proxy_selects_native_protocol(
    monkeypatch,
    protocol,
    expected_model,
    expected_provider,
    expected_mode,
):
    """Registration should route Claude and Codex Harnesses natively."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "test-llm-key")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["litellm_params"]["model"] == expected_model
        assert payload["litellm_params"]["custom_llm_provider"] == expected_provider
        assert payload["model_info"]["mode"] == expected_mode
        assert payload["model_info"]["disable_background_health_check"] is True
        return httpx.Response(200, json="https://arena.example/llm")

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

    result_polls = 0
    deployment_id = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal deployment_id, result_polls
        if request.url.path.endswith("/llm/model/new"):
            assert request.headers["Authorization"] == "Bearer test-llm-key"
            payload = json.loads(request.content)
            assert payload["model_name"].startswith("stream-areal-")
            assert payload["litellm_params"] == {
                "model": f"anthropic/{payload['model_name']}",
                "api_base": "http://rollout-proxy",
                "api_key": "session-key",
                "custom_llm_provider": "anthropic",
            }
            assert payload["model_info"]["mode"] == "chat"
            assert payload["model_info"]["disable_background_health_check"] is True
            deployment_id = payload["model_info"]["id"]
            return httpx.Response(
                200,
                json={
                    "model_id": deployment_id,
                    "model_name": payload["model_name"],
                    "litellm_params": {},
                    "model_info": {"id": deployment_id},
                },
            )
        if request.url.path.endswith("/streams/stream-1/launch_one_task"):
            assert request.headers["Authorization"] == "Bearer arena-token"
            assert json.loads(request.content) == {
                "data_id": "data-1",
                "model_name": deployment_id,
                "base_url": "https://arena.example/llm",
                "api_key": "test-llm-key",
                "envs": {
                    "MODEL_NAME": deployment_id,
                    "BASE_URL": "https://arena.example/llm",
                    "API_KEY": "test-llm-key",
                    "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
                },
            }
            return httpx.Response(
                202,
                json={"accepted": True, "task_id": "task-1", "status": "PENDING"},
            )
        if request.url.path.endswith("/llm/model/delete"):
            assert request.headers["Authorization"] == "Bearer test-llm-key"
            assert json.loads(request.content) == {"id": deployment_id}
            return httpx.Response(200, json="deleted")
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
                    "arena_task_envs": {"CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1"},
                    "timeout": 10.0,
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
                arena_http_client=http_client,
            )

    reward = asyncio.run(run_agent())
    assert reward == 0.75
    assert result_polls == 2


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


def test_trajectory_reward_sums_tensor_rewards():
    """Rollout-only logging should recover the episode reward tensor."""
    reward = _trajectory_reward({"rewards": torch.tensor([0.0, 0.75])})

    assert reward == 0.75


def test_trajectory_reward_sums_string_interactions():
    """External string trajectories should also produce one episode reward."""
    reward = _trajectory_reward(
        {
            "interactions": [
                {"reward": 0.0},
                {"reward": 1.0},
            ]
        }
    )

    assert reward == 1.0


def test_run_rollout_tasks_keeps_successes_when_one_task_is_rejected():
    """A failed harness task should not discard successful batch results."""

    class FakeController:
        def __init__(self) -> None:
            self.rows: dict[int, dict[str, str]] = {}

        def submit(self, data, **_kwargs):
            task_id = len(self.rows)
            self.rows[task_id] = data
            return task_id

        def wait_for_task(self, task_id):
            if self.rows[task_id]["data_id"] == "data-failed":
                return None
            return {"rewards": torch.tensor([1.0])}

    completed, failed = _run_rollout_tasks(
        controller=FakeController(),
        rows=[
            {"stream_id": "stream-1", "data_id": "data-success"},
            {"stream_id": "stream-1", "data_id": "data-failed"},
        ],
        workflow_kwargs={},
    )

    assert completed == [("data-success", 1.0)]
    assert failed == ["data-failed"]
