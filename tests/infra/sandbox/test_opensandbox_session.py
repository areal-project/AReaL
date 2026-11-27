# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from datetime import timedelta

import pytest

from tests.infra.sandbox._fake_opensandbox import (
    FakeConnectionConfig,
    FakeExecution,
    FakeExecutionError,
    FakeSandboxException,
    install,
)

import areal.infra.sandbox._opensandbox_session as session_module


@pytest.fixture
def fake_sdk(monkeypatch):
    return install(monkeypatch)


def _repl_payload(stdout="", error=None):
    return FakeExecution(
        stdout=json.dumps({"stdout": stdout, "stderr": "", "error": error}), exit_code=0
    )


@pytest.mark.asyncio
async def test_session_defaults_to_python_image(fake_sdk):
    session = session_module.OpenSandboxSession()
    await session.arun_script("print('hi')", timeout=5)

    assert fake_sdk.state.create_calls[0]["image"] == "python:3.11"
    assert "snapshot_id" not in fake_sdk.state.create_calls[0]


@pytest.mark.asyncio
async def test_session_passes_ttl_env_resource_metadata(fake_sdk):
    session = session_module.OpenSandboxSession(
        image="ubuntu:24.04",
        cpu="2",
        memory="4Gi",
        env_vars={"A": "1"},
        metadata={"team": "x"},
        ttl_seconds=600,
    )
    await session.arun_script("print('hi')", timeout=5)

    kwargs = fake_sdk.state.create_calls[0]
    assert kwargs["image"] == "ubuntu:24.04"
    assert kwargs["timeout"] == timedelta(seconds=600)
    assert kwargs["env"] == {"A": "1"}
    assert kwargs["resource"] == {"cpu": "2", "memory": "4Gi"}
    assert kwargs["metadata"] == {"areal-sandbox": "true", "team": "x"}


@pytest.mark.asyncio
async def test_session_network_block_maps_to_deny_policy(fake_sdk):
    session = session_module.OpenSandboxSession(network_block_all=True)
    await session.arun_script("print('hi')", timeout=5)

    policy = fake_sdk.state.create_calls[0]["network_policy"]
    assert policy.default_action == "deny"
    assert policy.egress == []


@pytest.mark.asyncio
async def test_session_ttl_none_requires_explicit_cleanup(fake_sdk):
    session = session_module.OpenSandboxSession(ttl_seconds=None)
    await session.arun_script("print('hi')", timeout=5)

    assert "timeout" not in fake_sdk.state.create_calls[0]


@pytest.mark.asyncio
async def test_session_connection_config_uses_explicit_credentials(fake_sdk):
    session = session_module.OpenSandboxSession(api_key="k", domain="d.example")
    await session.arun_script("print('hi')", timeout=5)

    config = fake_sdk.state.create_calls[0]["connection_config"]
    assert isinstance(config, FakeConnectionConfig)
    assert config.kwargs == {"api_key": "k", "domain": "d.example"}


@pytest.mark.asyncio
async def test_session_wires_create_timeout_to_ready_timeout(fake_sdk):
    session = session_module.OpenSandboxSession(create_timeout=45)
    await session.arun_script("print('hi')", timeout=5)

    assert fake_sdk.state.create_calls[0]["ready_timeout"] == timedelta(seconds=45)


@pytest.mark.asyncio
async def test_session_default_timeout_used_when_none(fake_sdk):
    session = session_module.OpenSandboxSession(default_timeout=7)
    fake_sdk.state.queue_execution(FakeExecution(stdout="hi\n", exit_code=0))

    result = await session.arun_script("print('hi')")

    assert result.exit_code == 0
    # The exec-level timeout is the backstop beyond the in-sandbox one.
    assert fake_sdk.state.last_opts.timeout == timedelta(seconds=7 + 15)


@pytest.mark.asyncio
async def test_concurrent_runs_create_one_sandbox(fake_sdk):
    fake_sdk.state.create_delays = [0.05, 0.05]
    fake_sdk.state.queue_execution(FakeExecution(stdout="a\n", exit_code=0))
    fake_sdk.state.queue_execution(FakeExecution(stdout="b\n", exit_code=0))
    session = session_module.OpenSandboxSession()

    results = await asyncio.gather(
        session.arun_script("print('a')", timeout=5),
        session.arun_script("print('b')", timeout=5),
    )

    assert [r.stdout for r in results] == ["a", "b"]
    assert len(fake_sdk.state.create_calls) == 1


@pytest.mark.asyncio
async def test_arun_script_recreates_when_write_file_reports_gone(fake_sdk):
    session = session_module.OpenSandboxSession()
    sandbox = await session.astart()
    fake_sdk.state.write_errors.append(FakeSandboxException("not found"))
    sandbox.get_info_results.append(FakeSandboxException("not found"))
    fake_sdk.state.queue_execution(FakeExecution(stdout="ok\n", exit_code=0))

    result = await session.arun_script("print('ok')", timeout=5)

    assert result.stdout == "ok"
    assert len(fake_sdk.state.create_calls) == 2


@pytest.mark.asyncio
async def test_arun_cell_recreates_when_repl_write_reports_gone(fake_sdk):
    session = session_module.OpenSandboxSession()
    sandbox = await session.astart()
    fake_sdk.state.write_errors.append(FakeSandboxException("not found"))
    sandbox.get_info_results.append(FakeSandboxException("not found"))
    fake_sdk.state.queue_execution(FakeExecution(stdout="0\n", exit_code=0))

    result = await session.arun_cell("x = 1", timeout=5)

    # The cell was never delivered; the result reports the state loss and a
    # fresh sandbox with a fresh interpreter is already running.
    assert result.error is not None
    assert "no longer exists" in result.error.value
    assert len(fake_sdk.state.create_calls) == 2


@pytest.mark.asyncio
async def test_arun_cell_sends_small_cells_on_argv(fake_sdk):
    session = session_module.OpenSandboxSession()
    fake_sdk.state.queue_execution(FakeExecution(stdout="0\n", exit_code=0))
    fake_sdk.state.queue_execution(_repl_payload("2\n"))

    result = await session.arun_cell("print(2)", timeout=5)

    assert result.stdout == "2\n"
    assert not fake_sdk.state.executed_commands[-1][-1].startswith("@")


@pytest.mark.asyncio
async def test_arun_cell_sends_large_cells_via_file(fake_sdk, monkeypatch):
    monkeypatch.setattr(session_module, "_MAX_INLINE_CELL_CHARS", 8)
    session = session_module.OpenSandboxSession()
    fake_sdk.state.queue_execution(FakeExecution(stdout="0\n", exit_code=0))
    fake_sdk.state.queue_execution(_repl_payload("done\n"))

    result = await session.arun_cell("print('a large cell payload')", timeout=5)

    assert result.stdout == "done\n"
    call_command = fake_sdk.state.executed_commands[-1]
    assert call_command[-1].startswith("@/tmp/.areal-sandbox/cell-")
    assert call_command[-1].endswith(".b64")
    # The transfer file is cleaned up after the call.
    assert not [p for p in fake_sdk.state.files if "cell-" in p]


@pytest.mark.asyncio
async def test_session_rejects_bad_ttl():
    with pytest.raises(ValueError, match="ttl_seconds"):
        session_module.OpenSandboxSession(ttl_seconds=0)


@pytest.mark.asyncio
async def test_session_rejects_image_and_snapshot():
    with pytest.raises(ValueError, match="not both"):
        session_module.OpenSandboxSession(image="a", snapshot_id="b")


@pytest.mark.asyncio
async def test_arun_script_returns_stdout_and_exit_code(fake_sdk):
    session = session_module.OpenSandboxSession()
    fake_sdk.state.queue_execution(FakeExecution(stdout="hi\n", exit_code=0))

    result = await session.arun_script("print('hi')", timeout=5)

    assert result.exit_code == 0
    assert result.stdout == "hi"
    # The code file was removed again after the run.
    written = [
        p for p in fake_sdk.state.files if p.startswith("/tmp/.areal-sandbox/code-")
    ]
    assert written == []


@pytest.mark.asyncio
async def test_arun_script_maps_timeout_exit_code(fake_sdk):
    session = session_module.OpenSandboxSession()
    fake_sdk.state.queue_execution(FakeExecution(stderr="Terminated\n", exit_code=124))

    result = await session.arun_script("import time; time.sleep(99)", timeout=1)

    assert result.exit_code == 124
    # The exec-level timeout is the backstop beyond the in-sandbox one.
    assert fake_sdk.state.last_opts.timeout == timedelta(seconds=16)


@pytest.mark.asyncio
async def test_arun_script_reports_nonzero_exit(fake_sdk):
    session = session_module.OpenSandboxSession()
    error = FakeExecutionError("ValueError", "x", exit_code=1)
    fake_sdk.state.queue_execution(
        FakeExecution(stderr="boom\n", exit_code=1, error=error)
    )

    result = await session.arun_script("raise ValueError('x')", timeout=5)

    assert result.exit_code == 1
    assert result.error == "ValueError: x"


@pytest.mark.asyncio
async def test_arun_script_recreates_gone_sandbox(fake_sdk):
    session = session_module.OpenSandboxSession()
    fake_sdk.state.queue_execution(FakeSandboxException("not found"))
    fake_sdk.state.queue_execution(FakeExecution(stdout="ok\n", exit_code=0))
    sandbox = await session.astart()
    sandbox.get_info_results.append(FakeSandboxException("not found"))

    result = await session.arun_script("print('ok')", timeout=5)

    assert result.stdout == "ok"
    assert len(fake_sdk.state.create_calls) == 2


@pytest.mark.asyncio
async def test_arun_script_does_not_recreate_healthy_sandbox(fake_sdk):
    session = session_module.OpenSandboxSession()
    fake_sdk.state.queue_execution(FakeSandboxException("exec failed"))
    await session.astart()

    with pytest.raises(FakeSandboxException):
        await session.arun_script("print('x')", timeout=5)

    assert len(fake_sdk.state.create_calls) == 1


@pytest.mark.asyncio
async def test_lease_renewed_after_run(fake_sdk):
    session = session_module.OpenSandboxSession(ttl_seconds=600)
    await session.arun_script("print('hi')", timeout=5)

    sandbox = fake_sdk.state.sandboxes[0]
    assert sandbox.renew_calls == [timedelta(seconds=600)]


@pytest.mark.asyncio
async def test_lease_renew_failure_is_swallowed(fake_sdk):
    session = session_module.OpenSandboxSession(ttl_seconds=600)
    sandbox = await session.astart()

    async def broken_renew(timeout):
        raise FakeSandboxException("renew unavailable")

    sandbox.renew = broken_renew
    fake_sdk.state.queue_execution(FakeExecution(stdout="hi\n", exit_code=0))

    result = await session.arun_script("print('hi')", timeout=5)
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_arun_cell_keeps_state_across_calls(fake_sdk):
    session = session_module.OpenSandboxSession()
    fake_sdk.state.queue_execution(FakeExecution(stdout="0\n", exit_code=0))
    fake_sdk.state.queue_execution(_repl_payload("5\n"))
    fake_sdk.state.queue_execution(_repl_payload("5\n"))

    first = await session.arun_cell("x = 5", timeout=5)
    second = await session.arun_cell("print(x)", timeout=5)

    assert first.error is None
    assert second.stdout == "5\n"
    # One sandbox, one interpreter start, two interpreter calls.
    assert len(fake_sdk.state.create_calls) == 1
    repl_starts = [
        c for c in fake_sdk.state.executed_commands if c[:2] == ["bash", "-c"]
    ]
    assert len(repl_starts) == 1


@pytest.mark.asyncio
async def test_arun_cell_reports_restarted_interpreter(fake_sdk):
    session = session_module.OpenSandboxSession()
    fake_sdk.state.queue_execution(FakeExecution(stdout="0\n", exit_code=0))  # start
    fake_sdk.state.queue_execution(
        FakeExecution(stderr="crashed\n", exit_code=1)
    )  # call
    fake_sdk.state.queue_execution(FakeExecution(stdout="0\n", exit_code=0))  # restart

    result = await session.arun_cell("x = 1", timeout=5)

    assert result.error is not None
    assert "restarted" in result.error.value
    assert "earlier variables are gone" in result.error.value


@pytest.mark.asyncio
async def test_arun_cell_recreates_gone_sandbox(fake_sdk):
    session = session_module.OpenSandboxSession()
    fake_sdk.state.queue_execution(FakeSandboxException("not found"))  # repl start exec
    sandbox = await session.astart()
    sandbox.get_info_results.append(FakeSandboxException("not found"))

    result = await session.arun_cell("x = 1", timeout=5)

    # The cell was never delivered; the result reports the state loss and a
    # fresh sandbox is already running with a new interpreter.
    assert result.error is not None
    assert "no longer exists" in result.error.value
    assert result.stdout == ""
    assert len(fake_sdk.state.create_calls) == 2


@pytest.mark.asyncio
async def test_adelete_destroys_and_tolerates_failure(fake_sdk):
    session = session_module.OpenSandboxSession()
    sandbox = await session.astart()

    await session.adelete()
    assert sandbox.destroy_calls == 1
    assert session.sandbox_id is None

    async def broken_destroy():
        raise FakeSandboxException("already gone")

    sandbox.destroy = broken_destroy
    session._sandbox = sandbox
    session.sandbox_id = sandbox.id
    await session.adelete()
    assert session.sandbox_id is None


@pytest.mark.asyncio
async def test_cancelled_create_is_deleted(fake_sdk):
    fake_sdk.state.create_delays = [0.1]
    session = session_module.OpenSandboxSession()
    task = asyncio.create_task(session.astart())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Let the pending create finish and the cleanup run.
    for _ in range(40):
        await asyncio.sleep(0.01)
        if fake_sdk.state.destroyed_ids:
            break

    assert len(fake_sdk.state.destroyed_ids) == 1
    assert session.sandbox_id is None
