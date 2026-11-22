"""Exercise the RPC guard's real signal handlers with pending Arena work."""

import asyncio
import multiprocessing
import os
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import httpx
import pytest

from examples.swe.arena_client import ArenaOpenAPIClient

from areal.api.cli_args import InferenceEngineConfig
from areal.infra.async_task_runner import AsyncTaskRunner
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.infra.rpc.guard import engine_blueprint
from areal.infra.rpc.guard.app import GuardState, create_app, run_server


def _run_guard(audit_path: str, names_path: str, ready) -> None:
    audit = Path(audit_path)

    def record(event: str) -> None:
        with audit.open("a") as output:
            output.write(event + "\n")

    polling = threading.Event()

    async def handle(request):
        if request.url.path.endswith("/launch_one_task"):
            return httpx.Response(
                202,
                json={"job_id": "owned-job", "task_id": "task-1", "status": "PENDING"},
            )
        if request.url.path.endswith("/result"):
            polling.set()
            await asyncio.Event().wait()
        assert request.url.path == "/openapi/v1/jobs/owned-job/cancel"
        record("job-cancel")
        return httpx.Response(200, json={})

    async def workflow():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            try:
                await ArenaOpenAPIClient(
                    base_url="https://arena.example", api_token="test-token"
                ).launch_one_task_result(
                    "stream",
                    "data",
                    "model",
                    "http://proxy/v1",
                    "session",
                    client=client,
                )
            finally:
                record("workflow-finally")

    runner = AsyncTaskRunner(max_queue_size=2, poll_sleep_time=0.01)
    runner.initialize()
    engine = RemoteInfEngine(InferenceEngineConfig(setup_timeout=60), SimpleNamespace())
    engine._workflow_executor = SimpleNamespace(destroy=runner.destroy)
    engine.register_destroy_callback("proxy", lambda: record("proxy-cleanup"))
    runner.submit(workflow, task_id=1)
    assert polling.wait(5)
    record("task-waiting")

    state = GuardState()
    state.experiment_name = "arena-cleanup-test"
    state.trial_name = f"signal-{os.getpid()}"
    state.role = "rollout"
    state.worker_index = 0
    state.name_resolve_type = "nfs"
    state.nfs_record_root = names_path
    app = create_app(state)
    app.register_blueprint(engine_blueprint.engine_bp)
    engine_blueprint._engines = {"rollout/0": engine}
    engine_blueprint.register_engine_hooks(state)

    def report_port():
        deadline = time.monotonic() + 5
        while not state.server_port and time.monotonic() < deadline:
            time.sleep(0.01)
        ready.send(state.server_port)

    threading.Thread(target=report_port, daemon=True).start()
    run_server(state, app, "127.0.0.1", 0)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="requires POSIX fork"
)
@pytest.mark.parametrize("termination", [signal.SIGTERM, signal.SIGKILL])
def test_guard_signal_cleanup_distinguishes_graceful_and_forced_exit(
    tmp_path, termination
):
    """SIGTERM drains jobs; SIGKILL/OOM-killer termination cannot run Python cleanup."""
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    audit = tmp_path / "events.txt"
    worker = context.Process(
        target=_run_guard,
        args=(str(audit), str(tmp_path / "names"), child),
    )
    worker.start()
    try:
        assert parent.poll(10), "guard did not publish its port"
        port = parent.recv()
        # Serving a request proves the SIGTERM handler is installed, not just
        # that the listening socket has been bound.
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            assert response.status == 200
        os.kill(worker.pid, termination)
        worker.join(timeout=10)
        assert not worker.is_alive()
        events = audit.read_text().splitlines()
        if termination == signal.SIGTERM:
            assert worker.exitcode == 0
            assert events == [
                "task-waiting",
                "job-cancel",
                "workflow-finally",
                "proxy-cleanup",
            ]
        else:
            assert worker.exitcode == -signal.SIGKILL
            assert events == ["task-waiting"]
    finally:
        if worker.is_alive():
            worker.kill()
        worker.join(timeout=5)
        parent.close()
        child.close()
