"""Arena jobs follow workflow cancellation during worker shutdown."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import torch

from examples.swe.arena_client import ArenaOpenAPIClient, ArenaTaskFailedError

from areal.api.cli_args import InferenceEngineConfig
from areal.infra.async_task_runner import AsyncTaskRunner
from areal.infra.controller.rollout_controller import RolloutController
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.trainer.rl_trainer import PPOTrainer


def _client() -> ArenaOpenAPIClient:
    return ArenaOpenAPIClient(
        base_url="https://arena.example",
        api_token="test-token",
        poll_interval=0,
        request_retries=0,
    )


async def _launch(client, http_client):
    return await client.launch_one_task_result(
        stream_id="stream-1",
        data_id="data-1",
        model_name="shared-worker-model",
        proxy_base_url="http://proxy/v1",
        proxy_api_key="session-key",
        client=http_client,
    )


@pytest.fixture
def running_arena_job():
    """Keep an accepted Arena task pending in the real workflow runner."""
    events = []
    polling = threading.Event()

    async def handler(request):
        assert request.headers["Authorization"] == "Bearer test-token"
        if request.url.path.endswith("/launch_one_task"):
            return httpx.Response(
                202,
                json={"job_id": "owned-job", "task_id": "task-1", "status": "PENDING"},
            )
        if request.url.path.endswith("/result"):
            polling.set()
            await asyncio.Event().wait()
        assert request.url.path == "/openapi/v1/jobs/owned-job/cancel"
        assert request.method == "POST"
        events.append("job-cancel")
        return httpx.Response(200, json={"cancelled_tasks": 1})

    async def workflow():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            try:
                await _launch(_client(), http_client)
            finally:
                events.append("workflow-finally")
        events.append("http-close")

    runner = AsyncTaskRunner(max_queue_size=2, poll_sleep_time=0.01)
    runner.initialize()
    engine = RemoteInfEngine(InferenceEngineConfig(setup_timeout=60), SimpleNamespace())
    engine._workflow_executor = SimpleNamespace(destroy=runner.destroy)
    engine.register_destroy_callback("proxy", lambda: events.append("proxy-delete"))
    try:
        runner.submit(workflow, task_id=1)
        assert polling.wait(timeout=5)
        yield SimpleNamespace(engine=engine, runner=runner, events=events)
    finally:
        runner.destroy()


def test_engine_destroy_cancels_owned_job_before_proxy_cleanup(running_arena_job):
    """The actual runner drains Arena cancellation before destroy callbacks."""
    engine, runner, events = (
        running_arena_job.engine,
        running_arena_job.runner,
        running_arena_job.events,
    )
    try:
        engine.destroy()
        assert not runner.thread.is_alive()
        assert events == ["job-cancel", "workflow-finally", "proxy-delete"]
        engine.destroy()
        assert events.count("job-cancel") == 1
    finally:
        runner.destroy()


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("actor update failed"),
        MemoryError("host allocation failed"),
        torch.OutOfMemoryError("CUDA allocation failed"),
        KeyboardInterrupt(),
        SystemExit(1),
    ],
    ids=["runtime", "host-memory", "cuda-oom", "interrupt", "system-exit"],
)
def test_trainer_exception_cancels_pending_arena_job(running_arena_job, failure):
    """Python exceptions escaping the trainer reach actual workflow cleanup."""
    trainer = object.__new__(PPOTrainer)
    trainer.rollout = running_arena_job.engine
    with pytest.raises(type(failure)) as caught:
        with trainer:
            raise failure
    assert caught.value is failure
    assert not running_arena_job.runner.thread.is_alive()
    assert running_arena_job.events == [
        "job-cancel",
        "workflow-finally",
        "proxy-delete",
    ]


@pytest.mark.parametrize(
    "failure", [MemoryError("init host OOM"), torch.OutOfMemoryError("init CUDA OOM")]
)
def test_trainer_constructor_oom_cancels_started_rollout(
    running_arena_job, failure, monkeypatch
):
    """Partial construction must close an already-created rollout worker."""

    def initialize(trainer, *args):
        trainer.rollout = running_arena_job.engine
        raise failure

    monkeypatch.setattr(PPOTrainer, "_init_impl", initialize)
    with pytest.raises(type(failure)) as caught:
        PPOTrainer(SimpleNamespace())
    assert caught.value is failure
    assert running_arena_job.events == [
        "job-cancel",
        "workflow-finally",
        "proxy-delete",
    ]


def test_cleanup_failure_keeps_other_workers_reachable(running_arena_job):
    """An OOM while finalizing another component cannot skip Arena cleanup."""
    trainer = object.__new__(PPOTrainer)
    trainer.rollout = running_arena_job.engine

    def fail():
        raise MemoryError("cleanup allocation failed")

    trainer.saver = SimpleNamespace(finalize=fail)
    trainer.eval_rollout = SimpleNamespace(destroy=fail)
    trainer.actor = SimpleNamespace(
        destroy=lambda: running_arena_job.events.append("actor")
    )
    error = torch.OutOfMemoryError("original CUDA OOM")
    with pytest.raises(torch.OutOfMemoryError) as caught:
        with trainer:
            raise error
    assert caught.value is error
    assert running_arena_job.events == [
        "job-cancel",
        "workflow-finally",
        "proxy-delete",
        "actor",
    ]


def test_tracer_failure_does_not_mask_oom_or_skip_cleanup(
    running_arena_job, monkeypatch
):
    """Even the final tracer flush may fail while unwinding an allocator OOM."""
    from areal.trainer import rl_trainer

    def fail(**kwargs):
        raise MemoryError("tracer allocation failed")

    monkeypatch.setattr(rl_trainer.perf_tracer, "save", fail)
    trainer = object.__new__(PPOTrainer)
    trainer.rollout = running_arena_job.engine
    original = torch.OutOfMemoryError("original CUDA OOM")
    with pytest.raises(torch.OutOfMemoryError) as caught:
        with trainer:
            raise original
    assert caught.value is original
    assert running_arena_job.events == [
        "job-cancel",
        "workflow-finally",
        "proxy-delete",
    ]


@pytest.mark.parametrize("during_init", [False, True])
def test_unexpected_close_failure_preserves_original_oom(during_init, monkeypatch):
    """A secondary failure in close must not replace the original traceback."""
    original = torch.OutOfMemoryError("original CUDA OOM")

    def fail_init(*args):
        raise original

    def fail_close(*args):
        raise RuntimeError("secondary close failure")

    monkeypatch.setattr(PPOTrainer, "close", fail_close)
    with pytest.raises(torch.OutOfMemoryError) as caught:
        if during_init:
            monkeypatch.setattr(PPOTrainer, "_init_impl", fail_init)
            PPOTrainer(SimpleNamespace())
        else:
            with object.__new__(PPOTrainer):
                raise original
    assert caught.value is original


@pytest.mark.parametrize("failed_component", ["gateway", "dispatcher", "callback"])
def test_early_rollout_cleanup_failure_still_cancels_arena_job(
    running_arena_job, failed_component
):
    """A preliminary teardown OOM cannot skip worker RPCs and job cancellation."""
    events = running_arena_job.events

    def step(component):
        def cleanup():
            events.append(component)
            if component == failed_component:
                raise MemoryError(f"{component} OOM")

        return cleanup

    controller = object.__new__(RolloutController)
    controller._stop_proxy_gateway = step("gateway")
    controller._dispatcher = SimpleNamespace(destroy=step("dispatcher"))
    controller._stop_callback_server = step("callback")
    controller._proxy_started = True
    controller.proxy_workers = [SimpleNamespace(id="proxy/0")]
    controller.proxy_addrs = ["proxy"]
    controller._proxy_engine_name = lambda rank: f"proxy/{rank}"
    controller._worker_role = "rollout"
    controller.workers = [SimpleNamespace(id="rollout/0")]
    controller._futures_lock = threading.Lock()
    controller._pending_futures = {"pending": object()}
    controller._backend_router = None

    def destroy_proxy(**kwargs):
        assert kwargs["method"] == "destroy"
        events.append("proxy-rpc")

    controller.scheduler = SimpleNamespace(
        async_call_engine=AsyncMock(side_effect=destroy_proxy),
        delete_workers=lambda role: events.append(f"delete-{role}"),
    )

    def destroy_rollout(method, **kwargs):
        assert method == "destroy"
        running_arena_job.engine.destroy()

    controller._collective_rpc = destroy_rollout
    with pytest.raises(RuntimeError, match=f"{failed_component} OOM"):
        controller.destroy()
    assert events == [
        "gateway",
        "dispatcher",
        "callback",
        "proxy-rpc",
        "delete-proxy-rollout",
        "job-cancel",
        "workflow-finally",
        "proxy-delete",
        "delete-rollout",
    ]
    assert not controller._pending_futures


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "timeout", "transport"])
@pytest.mark.parametrize("cleanup_status", [200, 403, 409, 500])
async def test_abandoned_poll_cleans_exact_job_and_preserves_original_error(
    failure, cleanup_status, monkeypatch
):
    """Cleanup failure must not change a task's original failure semantics."""
    error = {
        "cancel": asyncio.CancelledError(),
        "timeout": TimeoutError("poll deadline"),
        "transport": RuntimeError("poll transport failed"),
    }[failure]
    calls = []

    async def poll(*args):
        raise error

    async def no_sleep(_):
        return None

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/launch_one_task"):
            return httpx.Response(
                202,
                json={"job_id": "owned-job", "task_id": "task-1", "status": "PENDING"},
            )
        return httpx.Response(cleanup_status, json={})

    client = _client()
    monkeypatch.setattr(client, "_poll_task_result", poll)
    monkeypatch.setattr("examples.swe.arena_client.asyncio.sleep", no_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(type(error)) as caught:
            await _launch(client, http_client)
    assert caught.value is error
    assert calls == [("POST", "/openapi/v1/streams/stream-1/launch_one_task")] + [
        ("POST", "/openapi/v1/jobs/owned-job/cancel")
    ] * (5 if cleanup_status == 500 else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["DONE", "HARNESS_FAILED", "CANCELLED"])
async def test_terminal_task_preserves_receipt_without_cancelling_job(status):
    """Normal completion and remote terminal failures need no cancellation."""
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/launch_one_task"):
            return httpx.Response(
                202,
                json={"job_id": "owned-job", "task_id": "task-1", "status": "PENDING"},
            )
        assert request.url.path.endswith("/tasks/task-1/result")
        return httpx.Response(200, json={"status": status, "score": 0.75})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        if status == "DONE":
            result = await _launch(_client(), http_client)
            assert result.score == 0.75
        else:
            with pytest.raises(ArenaTaskFailedError) as caught:
                await _launch(_client(), http_client)
            assert caught.value.status == status
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_legacy_launch_resolves_job_using_exact_task(monkeypatch):
    """Resolve old task-only envelopes without scanning shared models."""
    calls = []

    async def poll(*args):
        raise asyncio.CancelledError

    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/launch_one_task"):
            return httpx.Response(202, json={"task_id": "task-1", "status": "PENDING"})
        if request.url.path.endswith("/tasks/task-1"):
            return httpx.Response(
                200, json={"task_id": "task-1", "job_id": "owned-job"}
            )
        assert request.url.path == "/openapi/v1/jobs/owned-job/cancel"
        return httpx.Response(200, json={})

    client = _client()
    monkeypatch.setattr(client, "_poll_task_result", poll)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(asyncio.CancelledError):
            await _launch(client, http_client)
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_cleanup_retries_transient_arena_failure(monkeypatch):
    """An idempotent cancellation can recover within the existing retry budget."""
    cancellations = 0
    original = asyncio.CancelledError()

    async def poll(*args):
        raise original

    def handler(request):
        nonlocal cancellations
        if request.url.path.endswith("/launch_one_task"):
            return httpx.Response(
                202,
                json={"job_id": "owned-job", "task_id": "task-1", "status": "PENDING"},
            )
        assert request.url.path == "/openapi/v1/jobs/owned-job/cancel"
        cancellations += 1
        return httpx.Response(500 if cancellations == 1 else 200, json={})

    client = _client()
    client.request_retries = 1
    monkeypatch.setattr(client, "_poll_task_result", poll)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(asyncio.CancelledError) as caught:
            await _launch(client, http_client)
    assert cancellations == 2
    assert caught.value is original


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("transport_loss", [False, True])
@pytest.mark.parametrize("oom", [False, True])
async def test_cleanup_recovers_gateway_outage_without_increasing_launch_retries(
    monkeypatch, legacy, transport_loss, oom
):
    """Known-job cleanup can outlive the separate zero-retry launch budget."""
    original = (
        torch.cuda.OutOfMemoryError("injected allocator OOM")
        if oom
        else asyncio.CancelledError()
    )
    calls = []
    recoveries = 0

    async def poll(*args):
        raise original

    async def no_sleep(_):
        return None

    def handler(request):
        nonlocal recoveries
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/launch_one_task"):
            payload = {"task_id": "task-1", "status": "PENDING"}
            if not legacy:
                payload["job_id"] = "owned-job"
            return httpx.Response(202, json=payload)
        recovery_path = (
            "/openapi/v1/tasks/task-1"
            if legacy
            else "/openapi/v1/jobs/owned-job/cancel"
        )
        if request.url.path == recovery_path:
            recoveries += 1
            if recoveries <= 3:
                if transport_loss:
                    raise httpx.ReadError("transient cleanup connection loss")
                return httpx.Response(503, json={})
            if legacy:
                return httpx.Response(
                    200, json={"task_id": "task-1", "job_id": "owned-job"}
                )
        assert request.url.path == "/openapi/v1/jobs/owned-job/cancel"
        assert request.method == "POST"
        return httpx.Response(200, json={})

    client = _client()
    monkeypatch.setattr(client, "_poll_task_result", poll)
    monkeypatch.setattr("examples.swe.arena_client.asyncio.sleep", no_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(type(original)) as caught:
            await _launch(client, http_client)
    assert caught.value is original
    assert recoveries == 4
    assert client.request_retries == 0
    assert calls.count(("POST", "/openapi/v1/streams/stream-1/launch_one_task")) == 1
    assert all(
        path
        in {
            "/openapi/v1/streams/stream-1/launch_one_task",
            "/openapi/v1/tasks/task-1",
            "/openapi/v1/jobs/owned-job/cancel",
        }
        for _, path in calls
    )


@pytest.mark.asyncio
async def test_cleanup_retry_backoff_obeys_existing_exit_deadline(monkeypatch):
    """A repeated gateway 503 cannot extend shutdown's existing time limit."""
    original = torch.cuda.OutOfMemoryError("injected allocator OOM")
    cancellations = 0

    async def poll(*args):
        raise original

    def handler(request):
        nonlocal cancellations
        if request.url.path.endswith("/launch_one_task"):
            return httpx.Response(
                202,
                json={"task_id": "task-1", "job_id": "owned-job", "status": "PENDING"},
            )
        assert request.url.path == "/openapi/v1/jobs/owned-job/cancel"
        cancellations += 1
        return httpx.Response(503, json={})

    client = _client()
    client.timeout = 0.02
    monkeypatch.setattr(client, "_poll_task_result", poll)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
            await asyncio.wait_for(_launch(client, http_client), timeout=1)
    assert caught.value is original
    assert cancellations == 1


@pytest.mark.asyncio
async def test_outer_workflow_deadline_cancels_job_before_raising_timeout():
    """Real deadline cancellation is cleaned up before conversion to TimeoutError."""
    events = []

    async def handler(request):
        if request.url.path.endswith("/launch_one_task"):
            return httpx.Response(
                202,
                json={"job_id": "owned-job", "task_id": "task-1", "status": "PENDING"},
            )
        if request.url.path.endswith("/result"):
            events.append("poll")
            await asyncio.Event().wait()
        assert request.url.path == "/openapi/v1/jobs/owned-job/cancel"
        events.append("cancel")
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.02):
                await _launch(_client(), http_client)
    assert events == ["poll", "cancel"]


@pytest.mark.asyncio
async def test_slow_job_cleanup_is_bounded_and_preserves_cancellation(monkeypatch):
    """An unresponsive Arena API cannot hold shutdown past its cleanup budget."""
    cancel_entered = asyncio.Event()
    original = asyncio.CancelledError()

    async def poll(*args):
        raise original

    async def handler(request):
        if request.url.path.endswith("/launch_one_task"):
            return httpx.Response(
                202,
                json={"job_id": "owned-job", "task_id": "task-1", "status": "PENDING"},
            )
        cancel_entered.set()
        await asyncio.Event().wait()

    client = _client()
    client.timeout = 0.01
    monkeypatch.setattr(client, "_poll_task_result", poll)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(_launch(client, http_client), timeout=1)
    assert cancel_entered.is_set()
    assert caught.value is original
