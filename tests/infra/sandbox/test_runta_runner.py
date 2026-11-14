# SPDX-License-Identifier: Apache-2.0

"""Unit and integration tests for the Runta sandbox runners."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

import areal.infra.sandbox.runta as runta_module
from areal.infra.sandbox import AsyncRuntaRunner, RuntaRunner, RuntaRunResult


def _has_runta_credentials() -> bool:
    return bool(os.environ.get("RUNTA_TOKEN"))


class FakeApiError(Exception):
    def __init__(self, message="", *, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _response(exit_code=0, stdout="", stderr=""):
    return SimpleNamespace(exit_code=exit_code, stdout_text=stdout, stderr_text=stderr)


class FakeRuntime:
    def __init__(self, state):
        self._state = state
        state.created += 1
        self.id = f"runtime-{state.created}"
        state.runtimes[self.id] = self
        self.egress = SimpleNamespace(
            set_policy=state.policies.append,
            set_tcp_allowlist=state.tcp_allowlists.append,
        )

    def wait_until_ready(self, timeout=None):
        self._state.ready_timeouts.append(timeout)
        if self._state.ready_error is not None:
            raise self._state.ready_error

    def status(self):
        outcome = self._state.statuses.pop(0) if self._state.statuses else "running"
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def exec(self, argv, timeout=None, stdin=None, env=None):
        self._state.execs.append(
            SimpleNamespace(argv=argv, timeout=timeout, stdin=stdin, env=env)
        )
        outcome = self._state.outcomes.pop(0) if self._state.outcomes else _response()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def update_timeout_policies(
        self, *, auto_shutdown_after_secs=None, auto_delete_after_secs=None
    ):
        # Like the real SDK: saving the shutdown policy restarts its clock, and
        # a revision conflict surfaces as ApiError.
        assert auto_delete_after_secs is None
        error = (
            self._state.renewal_errors.pop(0) if self._state.renewal_errors else None
        )
        if error is not None:
            raise error
        self._state.renewals.append((self.id, auto_shutdown_after_secs))
        self._state.renewed.set()

    def delete(self):
        self._state.deletes += 1
        if self._state.delete_errors:
            raise self._state.delete_errors.pop(0)
        error = self._state.delete_error
        # Like the real SDK, deleting an already deleted runtime (404) succeeds.
        if error is not None and getattr(error, "status_code", None) != 404:
            raise error


class FakeAsyncRuntime(FakeRuntime):
    def __init__(self, state):
        super().__init__(state)

        async def set_policy(policy):
            state.policies.append(policy)

        async def set_tcp_allowlist(hosts):
            state.tcp_allowlists.append(hosts)

        self.egress = SimpleNamespace(
            set_policy=set_policy, set_tcp_allowlist=set_tcp_allowlist
        )

    async def wait_until_ready(self, timeout=None):
        if self._state.ready_gate is not None:
            await self._state.ready_gate.wait()
        super().wait_until_ready(timeout=timeout)

    async def status(self):
        return super().status()

    async def exec(self, argv, timeout=None, stdin=None, env=None):
        return super().exec(argv, timeout=timeout, stdin=stdin, env=env)

    async def update_timeout_policies(self, **kwargs):
        super().update_timeout_policies(**kwargs)

    async def delete(self):
        super().delete()


@pytest.fixture
def sdk(monkeypatch):
    """Replace the lazily imported SDK with a recording fake."""
    state = SimpleNamespace(
        client_kwargs=[],
        create_calls=[],
        create_error=None,
        create_gate=None,
        created=0,
        ready_timeouts=[],
        ready_error=None,
        ready_gate=None,
        statuses=[],
        execs=[],
        outcomes=[],
        policies=[],
        tcp_allowlists=[],
        deletes=0,
        delete_error=None,
        delete_errors=[],
        closes=0,
        runtimes={},
        renewals=[],
        renewal_errors=[],
        renewed=threading.Event(),
    )

    def create(name, **kwargs):
        state.create_calls.append((name, kwargs))
        if state.create_error is not None:
            raise state.create_error

    class Runta:
        def __init__(self, **kwargs):
            state.client_kwargs.append(kwargs)
            self.runtimes = SimpleNamespace(create=self._create, get=self._get)

        def _get(self, id):
            # A second handle to an existing runtime, as the heartbeat's own
            # client resolves it.
            return state.runtimes[id]

        def _create(self, name, **kwargs):
            create(name, **kwargs)
            runtime = FakeRuntime(state)
            # Like the real SDK: the runtime exists before the wait, so a failed
            # wait loses the only handle to it.
            if kwargs.get("wait", True):
                runtime.wait_until_ready(timeout=kwargs.get("timeout"))
            return runtime

        def close(self):
            state.closes += 1

    class AsyncRunta:
        def __init__(self, **kwargs):
            state.client_kwargs.append(kwargs)
            self.runtimes = SimpleNamespace(create=self._create)

        async def _create(self, name, **kwargs):
            create(name, **kwargs)
            runtime = FakeAsyncRuntime(state)
            if state.create_gate is not None:
                await state.create_gate.wait()
            if kwargs.get("wait", True):
                await runtime.wait_until_ready(timeout=kwargs.get("timeout"))
            return runtime

        async def close(self):
            state.closes += 1

    fake = SimpleNamespace(
        Runta=Runta,
        AsyncRunta=AsyncRunta,
        ApiError=FakeApiError,
        EgressPolicy=lambda **kwargs: kwargs,
        EgressMode=SimpleNamespace(ALLOWLIST="allowlist"),
        RuntimeStatus=SimpleNamespace(
            SHUTDOWN="shutdown", DELETING="deleting", ERROR="error", CRASHED="crashed"
        ),
    )
    monkeypatch.setattr(runta_module, "_load_runta_sdk", lambda: fake)
    return state


@pytest.fixture
def warnings(monkeypatch):
    """Messages the runner logs at WARNING; its logger does not reach caplog."""
    messages: list[str] = []
    monkeypatch.setattr(
        runta_module.logger,
        "warning",
        lambda message, *args: messages.append(message % args),
    )
    return messages


def test_run_result_defaults_describe_success():
    result = RuntaRunResult(stdout="x")

    assert result.stdout == "x"
    assert result.stderr == ""
    assert result.exit_code == 0
    assert result.error is None


def test_runner_raises_clear_error_when_sdk_missing(monkeypatch):
    def raise_import_error():
        raise ImportError("runta-sdk is not installed")

    monkeypatch.setattr(runta_module, "_load_runta_sdk", raise_import_error)

    with pytest.raises(ImportError, match="runta-sdk"):
        RuntaRunner()


def test_runner_rejects_image_with_checkpoint(sdk):
    with pytest.raises(ValueError, match="either checkpoint_id or image"):
        RuntaRunner(image="python:3.12", checkpoint_id="ckpt")


def test_run_creates_runtime_lazily_with_lease_backstop(sdk):
    runner = RuntaRunner(vcpus=2, memory_mib=2048, max_lifetime=900)
    assert sdk.create_calls == []

    sdk.outcomes = [_response(stdout="45\n")]
    result = runner.run("print(sum(range(10)))", timeout=5)

    assert result == RuntaRunResult(stdout="45\n")
    name, kwargs = sdk.create_calls[0]
    assert name.startswith("areal-")
    assert kwargs == {
        "vcpus": 2,
        "memory_mib": 2048,
        "auto_shutdown_after_secs": 600,  # the lease, not the 900s lifetime
        "auto_delete_after_secs": 600,
        "wait": False,
    }
    assert sdk.ready_timeouts == [240.0]
    call = sdk.execs[0]
    assert call.argv == ["python3", "-"]
    assert call.stdin == "print(sum(range(10)))"
    assert call.timeout == 5
    runner.close()


def test_run_reuses_runtime_across_calls(sdk):
    with RuntaRunner() as runner:
        runner.run("print(1)")
        runner.run("print(2)")

    assert len(sdk.create_calls) == 1
    assert sdk.deletes == 1
    assert sdk.closes == 1


def test_run_uses_default_timeout_and_env_vars_on_base_runtime(sdk):
    with RuntaRunner(default_timeout=7, env_vars={"SEED": "3"}) as runner:
        runner.run("print(1)")

    assert sdk.execs[0].timeout == 7
    assert sdk.execs[0].env == {"SEED": "3"}


def test_image_runs_code_inside_container(sdk):
    with RuntaRunner(image="python:3.12-slim", env_vars={"SEED": "3"}) as runner:
        runner.run("print(1)")

    start, run = sdk.execs
    assert start.argv == [
        "docker",
        "run",
        "-d",
        "--name",
        "areal-sandbox",
        "--network",
        "host",
        "python:3.12-slim",
        "sleep",
        "infinity",
    ]
    assert start.timeout == 240.0
    pidfile = run.argv[-1]
    assert re.fullmatch(r"/tmp/areal-exec-[0-9a-f]{12}\.pid", pidfile)
    assert run.argv == [
        "docker",
        "exec",
        "-i",
        "-e",
        "SEED=3",
        "areal-sandbox",
        "sh",
        "-c",
        runta_module._TRACK,
        "areal-exec",
        pidfile,
    ]
    assert run.stdin == "print(1)"
    assert run.env is None


def test_checkpoint_id_is_forwarded_to_create(sdk):
    with RuntaRunner(checkpoint_id="ckpt-1"):
        pass

    assert sdk.create_calls[0][1]["checkpoint_id"] == "ckpt-1"


def test_checkpoint_restore_does_not_send_resources_and_warns_if_set(sdk, warnings):
    with RuntaRunner(checkpoint_id="ckpt-1", vcpus=4, memory_mib=8192):
        pass

    kwargs = sdk.create_calls[0][1]
    assert "vcpus" not in kwargs
    assert "memory_mib" not in kwargs
    assert "ignored with checkpoint_id" in "\n".join(warnings)


def test_checkpoint_restore_with_default_resources_does_not_warn(sdk, warnings):
    with RuntaRunner(checkpoint_id="ckpt-1"):
        pass

    assert warnings == []


def test_network_block_all_clears_tcp_allowlist_inherited_from_checkpoint(sdk):
    with RuntaRunner(checkpoint_id="ckpt-1", network_block_all=True):
        assert sdk.policies == [{"mode": "allowlist", "allowed_hosts": []}]
        assert sdk.tcp_allowlists == [[]]


def test_network_block_all_on_fresh_runtime_leaves_tcp_allowlist_alone(sdk):
    # A fresh runtime already denies raw TCP; only a restore inherits a list.
    with RuntaRunner(network_block_all=True):
        assert len(sdk.policies) == 1
        assert sdk.tcp_allowlists == []


def test_network_block_all_applies_empty_allowlist_after_image_start(sdk):
    with RuntaRunner(image="python:3.12-slim", network_block_all=True):
        assert sdk.policies == [{"mode": "allowlist", "allowed_hosts": []}]
        assert len(sdk.execs) == 1  # the image was pulled before egress closed


def test_failed_image_start_deletes_runtime_and_raises(sdk):
    sdk.outcomes = [_response(exit_code=125, stderr="manifest unknown")]

    with pytest.raises(RuntimeError, match="manifest unknown"):
        RuntaRunner(image="missing:tag").start()

    assert sdk.deletes == 1
    assert sdk.closes == 1


@pytest.mark.parametrize(
    "error", [TimeoutError("not ready"), FakeApiError("boom", status_code=500)]
)
def test_failed_readiness_wait_deletes_runtime_and_reports_failure(sdk, error):
    sdk.ready_error = error
    runner = RuntaRunner()

    result = runner.run("print(1)")

    assert result.exit_code == (124 if isinstance(error, TimeoutError) else 1)
    assert sdk.created == 1
    assert sdk.deletes == 1  # the runtime existed server-side and was not leaked
    assert sdk.closes == 1
    assert sdk.execs == []
    runner.close()


def test_runtime_entering_error_state_is_deleted_and_raises(sdk):
    sdk.ready_error = RuntimeError("runtime runtime-1 entered error state")

    with pytest.raises(RuntimeError, match="entered error state"):
        RuntaRunner().start()

    assert sdk.deletes == 1
    assert sdk.closes == 1


def test_cleanup_failure_does_not_replace_create_failure(sdk):
    sdk.ready_error = TimeoutError("not ready")
    sdk.delete_error = RuntimeError("delete failed")

    with pytest.raises(TimeoutError, match="not ready"):
        RuntaRunner().start()

    assert sdk.closes == 1


def test_nonzero_exit_reports_last_stderr_line_as_error(sdk):
    sdk.outcomes = [
        _response(
            exit_code=1,
            stdout="partial\n",
            stderr="Traceback (most recent call last):\nValueError: boom\n",
        )
    ]
    seen: list[str] = []

    with RuntaRunner() as runner:
        result = runner.run("raise ValueError('boom')", on_stderr=seen.append)

    assert result.exit_code == 1
    assert result.stdout == "partial\n"
    assert result.error == "ValueError: boom"
    assert seen == ["Traceback (most recent call last):\nValueError: boom\n"]


def test_nonzero_exit_without_stderr_reports_exit_code(sdk):
    sdk.outcomes = [_response(exit_code=3)]

    with RuntaRunner() as runner:
        assert runner.run("exit(3)").error == "Process exited with code 3"


def test_timeout_exit_code_is_preserved(sdk):
    sdk.outcomes = [
        _response(exit_code=124, stderr="Command timed out after 1 seconds.")
    ]

    with RuntaRunner() as runner:
        result = runner.run("import time; time.sleep(60)", timeout=1)

    assert result.exit_code == 124
    assert result.error == "Command timed out after 1 seconds."
    assert len(sdk.execs) == 1  # without an image Runta kills the program itself


def test_api_error_on_running_runtime_becomes_failed_result(sdk, warnings):
    sdk.outcomes = [FakeApiError("runtime unavailable", status_code=503)]

    with RuntaRunner() as runner:
        result = runner.run("print(1)")

    assert result == RuntaRunResult(
        stderr="runtime unavailable", exit_code=1, error="runtime unavailable"
    )
    assert sdk.created == 1  # still running, so it is kept and not retried
    assert len(sdk.execs) == 1
    assert "not a program error: runtime unavailable" in "\n".join(warnings)


def test_create_failure_becomes_logged_failed_result_and_closes_client(sdk, warnings):
    sdk.create_error = FakeApiError("quota exceeded", status_code=429)
    runner = RuntaRunner()

    result = runner.run("print(1)")

    assert result.exit_code == 1
    assert result.error == "quota exceeded"
    assert sdk.closes == 1
    assert "not a program error: quota exceeded" in "\n".join(warnings)
    runner.close()


@pytest.mark.parametrize(
    "status",
    ["shutdown", "crashed", FakeApiError("runtime not found", status_code=404)],
)
def test_run_recreates_runtime_that_stopped_running(sdk, warnings, status):
    runner = RuntaRunner().start()
    sdk.outcomes = [
        FakeApiError("runtime is not running", status_code=409),
        _response(stdout="2\n"),
    ]
    sdk.statuses = [status]

    result = runner.run("print(2)")

    assert result == RuntaRunResult(stdout="2\n")
    assert sdk.created == 2
    assert sdk.deletes == 1  # the stale runtime
    assert [call.stdin for call in sdk.execs] == ["print(2)", "print(2)"]
    assert "runtime-1 is no longer running" in "\n".join(warnings)

    runner.run("print(3)")
    assert sdk.created == 2  # the replacement is kept warm
    runner.close()
    assert sdk.deletes == 2
    assert sdk.closes == 2


def test_run_retries_on_a_fresh_runtime_only_once(sdk):
    runner = RuntaRunner()
    sdk.outcomes = [
        FakeApiError("runtime is not running", status_code=409),
        FakeApiError("runtime is not running again", status_code=409),
    ]
    sdk.statuses = ["shutdown", "shutdown"]

    result = runner.run("print(1)")

    assert result.exit_code == 1
    assert result.error == "runtime is not running again"
    assert sdk.created == 2
    assert len(sdk.execs) == 2
    runner.close()


def test_close_is_idempotent_and_tolerates_deleted_runtime(sdk):
    runner = RuntaRunner().start()
    sdk.delete_error = FakeApiError("not found", status_code=404)

    runner.close()
    runner.close()

    assert sdk.deletes == 1
    assert sdk.closes == 1


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("delete runtime exceeded its operation deadline"),
        RuntimeError("client is closed"),
        FakeApiError("server error", status_code=500),
    ],
)
def test_close_logs_and_swallows_delete_errors(sdk, warnings, error):
    runner = RuntaRunner().start()
    sdk.delete_error = error

    runner.close()

    assert sdk.deletes == 1
    assert sdk.closes == 1  # the client is still released
    assert "could not delete its runtime" in "\n".join(warnings)


def test_run_after_close_raises(sdk):
    runner = RuntaRunner()
    runner.close()

    with pytest.raises(RuntimeError, match="closed"):
        runner.run("print(1)")


def test_endpoint_is_forwarded_to_client(sdk):
    with RuntaRunner(endpoint="https://api.example.test"):
        pass

    assert sdk.client_kwargs == [{"endpoint": "https://api.example.test"}]


def _timed_out():
    return _response(exit_code=124, stderr="Command timed out after 1 seconds.")


def test_image_timeout_kills_program_left_running_in_container(sdk):
    sdk.outcomes = [_response(), _timed_out()]

    with RuntaRunner(image="python:3.12-slim") as runner:
        result = runner.run("while True: pass", timeout=1)

    assert result.exit_code == 124
    assert result.error == "Command timed out after 1 seconds."
    _, run, kill = sdk.execs
    assert kill.argv == [
        "docker",
        "exec",
        "-u",
        "0",
        "areal-sandbox",
        "sh",
        "-c",
        runta_module._KILL,
        "areal-kill",
        run.argv[-1],  # the PID file the timed-out program was started with
    ]
    assert kill.stdin is None
    assert kill.timeout == 30


@pytest.mark.parametrize("exit_code", [0, 1, 137])
def test_image_program_that_ends_in_time_is_not_killed(sdk, exit_code):
    sdk.outcomes = [_response(), _response(exit_code=exit_code)]

    with RuntaRunner(image="python:3.12-slim") as runner:
        result = runner.run("print(1)")

    assert result.exit_code == exit_code
    assert len(sdk.execs) == 2  # container start and the program, no kill


def test_image_timeout_kill_failure_is_logged_and_keeps_result(sdk, warnings):
    sdk.outcomes = [
        _response(),
        _timed_out(),
        FakeApiError("exec unavailable", status_code=503),
    ]

    with RuntaRunner(image="python:3.12-slim") as runner:
        result = runner.run("while True: pass", timeout=1)

    assert result.exit_code == 124
    assert len(sdk.execs) == 3
    assert "Could not stop the timed-out program" in "\n".join(warnings)


def _run_tracked(tmp_path, **popen_kwargs):
    """Start the real wrapper, as `docker exec -i` would; the program goes to stdin."""
    pidfile = tmp_path / "program.pid"
    process = subprocess.Popen(
        ["sh", "-c", runta_module._TRACK, "areal-exec", str(pidfile)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **popen_kwargs,
    )
    return process, pidfile


needs_posix_shell = pytest.mark.skipif(
    os.name != "posix" or not shutil.which("sh") or not shutil.which("python3"),
    reason="needs a POSIX sh and python3 on PATH",
)


@needs_posix_shell
def test_wrapper_feeds_stdin_to_python_and_keeps_exit_code_and_streams(tmp_path):
    code = (
        "import sys\n"
        "print(sys.argv, __name__)\n"
        "print('oops', file=sys.stderr)\n"
        "sys.exit(3)\n"
    )

    process, pidfile = _run_tracked(tmp_path)
    stdout, stderr = process.communicate(code, timeout=30)

    assert stdout == "['-'] __main__\n"  # exactly what a bare `python3 -` prints
    assert stderr == "oops\n"
    assert process.returncode == 3
    assert not pidfile.exists()


@needs_posix_shell
def test_kill_script_stops_program_recorded_by_wrapper(tmp_path):
    # Its own session, like a process started by `docker exec`.
    process, pidfile = _run_tracked(tmp_path, start_new_session=True)
    try:
        process.stdin.write("import time\nprint('up', flush=True)\ntime.sleep(600)\n")
        process.stdin.close()
        assert process.stdout.readline() == "up\n"
        assert pidfile.read_text().strip().isdigit()

        kill = ["sh", "-c", runta_module._KILL, "areal-kill", str(pidfile)]
        assert subprocess.run(kill, timeout=30).returncode == 0

        assert process.wait(timeout=30) != 0
        assert process.stdout.read() == ""  # the program died with its group
        assert not pidfile.exists()
        # Nothing left to kill: the script is a no-op, not an error.
        assert subprocess.run(kill, timeout=30).returncode == 0
    finally:
        process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()


def test_runner_rejects_lease_that_would_disable_auto_shutdown(sdk):
    with pytest.raises(ValueError, match="lease_seconds"):
        RuntaRunner(lease_seconds=0)


def test_lease_seconds_sets_initial_shutdown_and_renewal_interval(sdk):
    with RuntaRunner(lease_seconds=120) as runner:
        assert runner._lease_interval == 40
        assert runner._heartbeat._thread.daemon  # never keeps the interpreter alive

    assert sdk.create_calls[0][1]["auto_shutdown_after_secs"] == 120


def test_lease_longer_than_max_lifetime_is_capped_and_never_renewed(sdk):
    with RuntaRunner(max_lifetime=300) as runner:
        assert runner._heartbeat is None

    assert sdk.create_calls[0][1]["auto_shutdown_after_secs"] == 300


def test_heartbeat_renews_full_lease_through_its_own_client(sdk):
    with RuntaRunner(endpoint="https://api.example.test") as runner:
        assert runner._heartbeat.beat() is True
        assert runner._heartbeat.beat() is True

    assert sdk.renewals == [("runtime-1", 600), ("runtime-1", 600)]
    # One client for run(), one for the heartbeat thread, reused across beats.
    assert sdk.client_kwargs == [{"endpoint": "https://api.example.test"}] * 2


def test_heartbeat_renewal_is_capped_by_remaining_lifetime_then_stops(sdk):
    with RuntaRunner() as runner:
        heartbeat = runner._heartbeat
        heartbeat.deadline = time.monotonic() + 100.5

        assert heartbeat.beat() is False  # ends at the hard cap: nothing to add

        heartbeat.deadline = time.monotonic() + 0.5
        assert heartbeat.beat() is False

    assert sdk.renewals == [("runtime-1", 100)]


def test_heartbeat_tolerates_failed_renewal_and_retries_next_beat(sdk, warnings):
    sdk.renewal_errors = [
        FakeApiError("runtime revision conflict: expected 3", status_code=409),
        RuntimeError("connection reset"),
    ]

    with RuntaRunner() as runner:
        heartbeat = runner._heartbeat
        assert heartbeat.beat() is True
        assert heartbeat.beat() is True
        assert sdk.renewals == []
        assert runner.run("print(1)").exit_code == 0  # run() is unaffected

        assert heartbeat.beat() is True

    assert sdk.renewals == [("runtime-1", 600)]
    assert "\n".join(warnings).count("Could not renew the lease") == 2
    assert "runtime revision conflict" in warnings[0]


def test_heartbeat_thread_renews_on_its_interval_and_stops_on_close(sdk):
    runner = RuntaRunner()
    runner._lease_interval = 0.01  # injected: never wait for a real lease
    runner.start()
    heartbeat = runner._heartbeat

    assert sdk.renewed.wait(timeout=10)
    runner.close()
    heartbeat._thread.join(timeout=10)

    assert not heartbeat._thread.is_alive()
    assert runner._heartbeat is None
    assert sdk.renewals[0] == ("runtime-1", 600)
    assert sdk.deletes == 1
    assert sdk.closes == 2  # the runner's client and the heartbeat's own


def test_heartbeat_is_stopped_on_close_without_waiting_for_the_interval(sdk):
    runner = RuntaRunner().start()
    heartbeat = runner._heartbeat
    assert heartbeat._thread.is_alive()

    runner.close()
    heartbeat._thread.join(timeout=10)

    assert not heartbeat._thread.is_alive()
    assert sdk.renewals == []


def test_heartbeat_is_replaced_when_runtime_is_discarded(sdk):
    runner = RuntaRunner().start()
    stale = runner._heartbeat
    sdk.outcomes = [FakeApiError("runtime is not running", status_code=409)]
    sdk.statuses = ["shutdown"]

    runner.run("print(1)")
    stale._thread.join(timeout=10)

    assert not stale._thread.is_alive()
    fresh = runner._heartbeat
    assert fresh is not stale and fresh._thread.is_alive()
    assert fresh.beat() is True
    assert sdk.renewals == [("runtime-2", 600)]  # only the live runtime is renewed
    runner.close()
    fresh._thread.join(timeout=10)
    assert not fresh._thread.is_alive()


def test_failed_renewal_after_stop_is_not_reported(sdk, warnings):
    runner = RuntaRunner().start()
    heartbeat = runner._heartbeat
    runner.close()
    sdk.renewal_errors = [FakeApiError("runtime not found", status_code=404)]

    assert heartbeat.beat() is True  # a beat that was already in flight

    assert warnings == []


def test_close_retries_delete_that_conflicts_with_a_lease_renewal(sdk, warnings):
    runner = RuntaRunner().start()
    sdk.delete_errors = [
        FakeApiError("runtime revision conflict: expected 3", status_code=409)
    ]

    runner.close()

    assert sdk.deletes == 2
    assert warnings == []


@pytest.mark.asyncio
async def test_async_runner_runs_and_cleans_up(sdk):
    sdk.outcomes = [_response(stdout="hi\n")]

    async with AsyncRuntaRunner(max_lifetime=900) as runner:
        result = await runner.arun("print('hi')", timeout=5)

    assert result == RuntaRunResult(stdout="hi\n")
    assert sdk.execs[0].argv == ["python3", "-"]
    assert sdk.deletes == 1
    assert sdk.closes == 1


@pytest.mark.asyncio
async def test_async_runner_failed_image_start_deletes_runtime(sdk):
    sdk.outcomes = [_response(exit_code=125, stderr="pull access denied")]

    with pytest.raises(RuntimeError, match="pull access denied"):
        await AsyncRuntaRunner(image="private:tag").astart()

    assert sdk.deletes == 1
    assert sdk.closes == 1


@pytest.mark.asyncio
async def test_async_runner_api_error_becomes_failed_result(sdk):
    sdk.outcomes = [FakeApiError("stream closed", status_code=502)]

    async with AsyncRuntaRunner() as runner:
        result = await runner.arun("print(1)")

    assert result.exit_code == 1
    assert result.error == "stream closed"


@pytest.mark.asyncio
async def test_async_runner_failed_readiness_wait_deletes_runtime(sdk):
    sdk.ready_error = TimeoutError("not ready")
    runner = AsyncRuntaRunner()

    result = await runner.arun("print(1)")

    assert result.exit_code == 124
    assert sdk.created == 1
    assert sdk.deletes == 1
    assert sdk.closes == 1
    await runner.aclose()


@pytest.mark.asyncio
async def test_async_runner_cancelled_while_waiting_for_readiness_deletes_runtime(sdk):
    sdk.ready_gate = asyncio.Event()
    runner = AsyncRuntaRunner()
    task = asyncio.create_task(runner.arun("print(1)"))
    while sdk.created == 0:
        await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sdk.deletes == 1
    assert sdk.closes == 1
    await runner.aclose()


@pytest.mark.asyncio
async def test_async_runner_cancelled_during_create_request_deletes_runtime(sdk):
    sdk.create_gate = asyncio.Event()
    runner = AsyncRuntaRunner()
    task = asyncio.create_task(runner.arun("print(1)"))
    while sdk.created == 0:
        await asyncio.sleep(0)

    # The runtime exists server-side but the create call has not returned yet.
    task.cancel()
    await asyncio.sleep(0)
    sdk.create_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sdk.deletes == 1
    assert sdk.closes == 1
    await runner.aclose()


@pytest.mark.asyncio
async def test_async_runner_recreates_runtime_that_stopped_running(sdk, warnings):
    runner = await AsyncRuntaRunner().astart()
    sdk.outcomes = [
        FakeApiError("runtime is not running", status_code=409),
        _response(stdout="2\n"),
    ]
    sdk.statuses = ["shutdown"]

    result = await runner.arun("print(2)")
    await runner.aclose()

    assert result == RuntaRunResult(stdout="2\n")
    assert sdk.created == 2
    assert sdk.deletes == 2
    assert sdk.closes == 2
    assert "runtime-1 is no longer running" in "\n".join(warnings)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("delete runtime exceeded its operation deadline"),
        RuntimeError("AsyncRunta is closed"),
        FakeApiError("server error", status_code=500),
    ],
)
async def test_async_close_logs_and_swallows_delete_errors(sdk, warnings, error):
    runner = await AsyncRuntaRunner().astart()
    sdk.delete_error = error

    await runner.aclose()

    assert sdk.deletes == 1
    assert sdk.closes == 1
    assert "could not delete its runtime" in "\n".join(warnings)


@pytest.mark.asyncio
async def test_async_close_waits_for_in_flight_create_and_deletes_it(sdk):
    sdk.ready_gate = asyncio.Event()
    runner = AsyncRuntaRunner()
    running = asyncio.create_task(runner.arun("print(1)"))
    while sdk.created == 0:
        await asyncio.sleep(0)

    closing = asyncio.create_task(runner.aclose())
    await asyncio.sleep(0)
    assert sdk.deletes == 0  # close has not run past the create
    sdk.ready_gate.set()
    await asyncio.gather(running, closing, return_exceptions=True)

    assert sdk.created == 1
    assert sdk.deletes == 1
    assert sdk.closes == 1
    assert runner._runtime is None


@pytest.mark.asyncio
async def test_async_run_after_close_raises(sdk):
    runner = AsyncRuntaRunner()
    await runner.aclose()

    with pytest.raises(RuntimeError, match="closed"):
        await runner.arun("print(1)")


@pytest.mark.asyncio
async def test_async_lease_sets_initial_shutdown_and_caps_it_by_max_lifetime(sdk):
    async with AsyncRuntaRunner(lease_seconds=120) as leased:
        assert leased._lease_interval == 40
        assert not leased._heartbeat.done()
    async with AsyncRuntaRunner(max_lifetime=300) as short_lived:
        assert short_lived._heartbeat is None

    shutdowns = [kwargs["auto_shutdown_after_secs"] for _, kwargs in sdk.create_calls]
    assert shutdowns == [120, 300]


@pytest.mark.asyncio
async def test_async_beat_renews_full_lease_then_caps_it_by_remaining_lifetime(sdk):
    async with AsyncRuntaRunner() as runner:
        runtime = runner._runtime

        assert await runner._abeat(runtime, time.monotonic() + 3600) is True
        assert await runner._abeat(runtime, time.monotonic() + 100.5) is False
        assert await runner._abeat(runtime, time.monotonic() + 0.5) is False

    assert sdk.renewals == [("runtime-1", 600), ("runtime-1", 100)]
    assert len(sdk.client_kwargs) == 1  # same loop, so the runner's own client


@pytest.mark.asyncio
async def test_async_beat_tolerates_failed_renewal_and_retries_next_beat(sdk, warnings):
    sdk.renewal_errors = [
        FakeApiError("runtime revision conflict: expected 3", status_code=409)
    ]

    async with AsyncRuntaRunner() as runner:
        deadline = time.monotonic() + 3600
        assert await runner._abeat(runner._runtime, deadline) is True
        assert sdk.renewals == []
        assert (await runner.arun("print(1)")).exit_code == 0  # arun() is unaffected

        assert await runner._abeat(runner._runtime, deadline) is True

    assert sdk.renewals == [("runtime-1", 600)]
    assert "\n".join(warnings).count("Could not renew the lease") == 1


@pytest.mark.asyncio
async def test_async_heartbeat_task_renews_on_its_interval_and_stops_on_close(sdk):
    runner = AsyncRuntaRunner()
    runner._lease_interval = 0.001  # injected: never wait for a real lease
    await runner.astart()
    heartbeat = runner._heartbeat

    while not sdk.renewals:
        await asyncio.sleep(0.001)
    await runner.aclose()

    assert heartbeat.cancelled()
    assert runner._heartbeat is None
    assert sdk.renewals[0] == ("runtime-1", 600)
    renewed = len(sdk.renewals)
    await asyncio.sleep(0.01)
    assert len(sdk.renewals) == renewed  # nothing renews a deleted runtime


@pytest.mark.asyncio
async def test_async_heartbeat_is_replaced_when_runtime_is_discarded(sdk):
    runner = await AsyncRuntaRunner().astart()
    stale = runner._heartbeat
    sdk.outcomes = [FakeApiError("runtime is not running", status_code=409)]
    sdk.statuses = ["shutdown"]

    await runner.arun("print(1)")

    assert stale.cancelled()
    fresh = runner._heartbeat
    assert fresh is not stale and not fresh.done()
    await runner.aclose()
    assert fresh.cancelled()
    assert sdk.renewals == []


@pytest.mark.asyncio
async def test_async_close_retries_delete_that_conflicts_with_a_lease_renewal(
    sdk, warnings
):
    runner = await AsyncRuntaRunner().astart()
    sdk.delete_errors = [
        FakeApiError("runtime revision conflict: expected 3", status_code=409)
    ]

    await runner.aclose()

    assert sdk.deletes == 2
    assert warnings == []


@pytest.mark.asyncio
async def test_async_image_timeout_kills_program_left_running_in_container(sdk):
    sdk.outcomes = [_response(), _timed_out()]

    async with AsyncRuntaRunner(image="python:3.12-slim") as runner:
        result = await runner.arun("while True: pass", timeout=1)

    assert result.exit_code == 124
    _, run, kill = sdk.execs
    assert run.argv[-5:-1] == ["sh", "-c", runta_module._TRACK, "areal-exec"]
    assert kill.argv[:5] == ["docker", "exec", "-u", "0", "areal-sandbox"]
    assert kill.argv[5:] == ["sh", "-c", runta_module._KILL, "areal-kill", run.argv[-1]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("image", "exit_code", "expected_execs"),
    [
        ("python:3.12-slim", 1, 2),  # ended in time: container start + program
        (None, 124, 1),  # no image: Runta kills the program itself
    ],
)
async def test_async_kill_is_issued_only_for_timeouts_in_image_mode(
    sdk, image, exit_code, expected_execs
):
    program = _response(exit_code=exit_code, stderr="boom")
    sdk.outcomes = [_response(), program] if image else [program]

    async with AsyncRuntaRunner(image=image) as runner:
        result = await runner.arun("print(1)")

    assert result.exit_code == exit_code
    assert len(sdk.execs) == expected_execs


@pytest.mark.integration
@pytest.mark.skipif(not _has_runta_credentials(), reason="RUNTA_TOKEN is required")
def test_runner_runs_python_code_live():
    with RuntaRunner(max_lifetime=600) as runner:
        ok = runner.run("print('hello')")
        failed = runner.run("raise ValueError('boom')")
        runner.run("open('/tmp/state.txt', 'w').write('kept')")
        kept = runner.run("print(open('/tmp/state.txt').read())")

    assert ok.exit_code == 0
    assert ok.stdout.strip() == "hello"
    assert failed.exit_code == 1
    assert failed.error == "ValueError: boom"
    assert kept.stdout.strip() == "kept"


@pytest.mark.integration
@pytest.mark.skipif(not _has_runta_credentials(), reason="RUNTA_TOKEN is required")
def test_runner_reports_timeout_live():
    with RuntaRunner(max_lifetime=600) as runner:
        result = runner.run("import time; time.sleep(60)", timeout=3)

    assert result.exit_code == 124
