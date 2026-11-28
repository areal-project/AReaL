# SPDX-License-Identifier: Apache-2.0

"""Integration and unit tests for the synchronous OpenSandbox runner."""

from __future__ import annotations

import importlib.util
import os
from datetime import timedelta

import pytest

from tests.infra.sandbox._fake_opensandbox import (
    FakeConnectionConfigSync,
    FakeExecution,
    FakeSandboxException,
    install,
)

import areal.infra.sandbox.opensandbox_runner as runner_module
from areal.infra.sandbox import OpenSandboxRunner, OpenSandboxRunResult


def _has_opensandbox_sdk() -> bool:
    return importlib.util.find_spec("opensandbox") is not None


def _has_opensandbox_endpoint() -> bool:
    return _has_opensandbox_sdk() and bool(os.environ.get("OPEN_SANDBOX_DOMAIN"))


@pytest.fixture
def fake_sdk(monkeypatch):
    return install(monkeypatch)


def test_run_result_fields_round_trip():
    result = OpenSandboxRunResult(
        stdout="x",
        stderr="",
        exit_code=0,
        error=None,
    )

    assert result.stdout == "x"
    assert result.stderr == ""
    assert result.exit_code == 0
    assert result.error is None


def test_runner_raises_clear_error_when_opensandbox_missing(monkeypatch):
    def raise_import_error():
        raise ImportError("opensandbox is not installed")

    monkeypatch.setattr(runner_module, "load_opensandbox_sdk", raise_import_error)

    with pytest.raises(ImportError, match="opensandbox"):
        OpenSandboxRunner()


def test_runner_rejects_image_and_snapshot(fake_sdk):
    with pytest.raises(ValueError, match="not both"):
        OpenSandboxRunner(image="a", snapshot_id="b")


def test_runner_rejects_bad_ttl(fake_sdk):
    with pytest.raises(ValueError, match="ttl_seconds"):
        OpenSandboxRunner(ttl_seconds=0)


def test_runner_runs_code_and_removes_script(fake_sdk):
    fake_sdk.state.queue_execution(FakeExecution(stdout="hello\n", exit_code=0))

    with OpenSandboxRunner() as runner:
        result = runner.run("print('hello')")

    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert fake_sdk.state.executed_commands[0][0] == "timeout"
    assert fake_sdk.state.files == {}


def test_runner_timeout_surfaces_as_error(fake_sdk):
    fake_sdk.state.queue_execution(FakeExecution(stderr="Terminated\n", exit_code=124))

    with OpenSandboxRunner() as runner:
        result = runner.run("import time; time.sleep(10)", timeout=1)

    assert result.exit_code == 124
    assert result.error


def test_runner_creates_sandbox_with_ttl(fake_sdk):
    fake_sdk.state.queue_execution(FakeExecution(stdout="", exit_code=0))

    runner = OpenSandboxRunner(ttl_seconds=600, network_block_all=True)
    try:
        runner.run("print('x')")
    finally:
        runner.close()

    kwargs = fake_sdk.state.create_calls[0]

    assert kwargs["timeout"] == timedelta(seconds=600)
    assert kwargs["network_policy"].default_action == "deny"


def test_runner_merges_default_metadata(fake_sdk):
    fake_sdk.state.queue_execution(FakeExecution(stdout="", exit_code=0))

    runner = OpenSandboxRunner(metadata={"team": "x"})
    try:
        runner.run("print('x')")
    finally:
        runner.close()

    assert fake_sdk.state.create_calls[0]["metadata"] == {
        "areal-sandbox": "true",
        "team": "x",
    }


def test_runner_uses_sync_connection_config_and_ready_timeout(fake_sdk):
    fake_sdk.state.queue_execution(FakeExecution(stdout="", exit_code=0))

    runner = OpenSandboxRunner(api_key="k", create_timeout=77)
    try:
        runner.run("print('x')")
    finally:
        runner.close()

    kwargs = fake_sdk.state.create_calls[0]
    assert isinstance(kwargs["connection_config"], FakeConnectionConfigSync)
    assert kwargs["ready_timeout"] == timedelta(seconds=77)


def test_runner_run_after_close_raises_and_does_not_leak(fake_sdk):
    fake_sdk.state.queue_execution(FakeExecution(stdout="", exit_code=0))
    runner = OpenSandboxRunner()
    runner.start()
    sandbox_id = runner.sandbox_id
    runner.close()

    with pytest.raises(RuntimeError, match="OpenSandboxRunner is closed"):
        runner.run("print('x')")

    # No extra sandbox was created, the id is cleared, and the original
    # sandbox was destroyed.
    assert len(fake_sdk.state.create_calls) == 1
    assert runner.sandbox_id is None
    assert fake_sdk.state.destroyed_ids == [sandbox_id]


def test_runner_close_clears_sandbox_id(fake_sdk):
    runner = OpenSandboxRunner()
    runner.start()
    assert runner.sandbox_id is not None

    runner.close()

    assert runner.sandbox_id is None


def test_runner_recreates_gone_sandbox(fake_sdk):
    fake_sdk.state.queue_execution(FakeSandboxException("not found"))
    fake_sdk.state.queue_execution(FakeExecution(stdout="ok\n", exit_code=0))

    runner = OpenSandboxRunner()
    runner.start()
    runner._sandbox.get_info_results.append(FakeSandboxException("not found"))

    result = runner.run("print('ok')")

    assert result.stdout == "ok"
    assert len(fake_sdk.state.create_calls) == 2
    runner.close()


def test_runner_close_destroys_sandbox(fake_sdk):
    runner = OpenSandboxRunner()
    runner.start()
    sandbox = fake_sdk.state.sandboxes[0]

    runner.close()

    assert sandbox.destroy_calls == 1
    # Closing twice is a no-op.
    runner.close()
    assert sandbox.destroy_calls == 1


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_opensandbox_endpoint(), reason="OPEN_SANDBOX_DOMAIN is required"
)
def test_runner_runs_python_code():
    runner = OpenSandboxRunner()

    try:
        result = runner.run("print('hello')")
    finally:
        runner.close()

    assert result.exit_code == 0
    assert "hello" in result.stdout


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_opensandbox_endpoint(), reason="OPEN_SANDBOX_DOMAIN is required"
)
def test_runner_context_manager_cleans_up():
    with OpenSandboxRunner() as runner:
        result = runner.run("print('context manager')")

    assert result.exit_code == 0
    assert "context manager" in result.stdout


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_opensandbox_endpoint(), reason="OPEN_SANDBOX_DOMAIN is required"
)
def test_runner_captures_runtime_error():
    with OpenSandboxRunner() as runner:
        result = runner.run("raise ValueError('x')")

    assert result.exit_code != 0
    assert result.error


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_opensandbox_endpoint(), reason="OPEN_SANDBOX_DOMAIN is required"
)
def test_runner_timeout_surfaces_as_error_live():
    with OpenSandboxRunner() as runner:
        result = runner.run("import time; time.sleep(10)", timeout=1)

    assert result.exit_code == 124
    assert result.error
