# SPDX-License-Identifier: Apache-2.0

"""Integration and unit tests for the synchronous boxd runner."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("boxd")

from boxd.errors import BoxdError  # noqa: E402

import areal.infra.sandbox._boxd_client as client_module  # noqa: E402
import areal.infra.sandbox.boxd_runner as runner_module  # noqa: E402
from areal.infra.sandbox import BoxdRunner, BoxdRunResult  # noqa: E402

from ._fake_boxd import FakeAsyncBoxd, FakeExecResult  # noqa: E402


def _has_boxd_credentials() -> bool:
    return bool(os.environ.get("BOXD_API_KEY"))


@pytest.fixture
def fake_boxd(monkeypatch):
    clients: list[FakeAsyncBoxd] = []

    def make_client(**kwargs):
        client = FakeAsyncBoxd(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(client_module, "AsyncBoxd", make_client, raising=False)
    return clients


def test_run_result_fields_round_trip():
    result = BoxdRunResult(stdout="x", stderr="", exit_code=0, error=None)

    assert result.stdout == "x"
    assert result.stderr == ""
    assert result.exit_code == 0
    assert result.error is None


def test_runner_raises_clear_error_when_boxd_missing(monkeypatch):
    def raise_import_error():
        raise ImportError("boxd is not installed")

    monkeypatch.setattr(runner_module, "_load_boxd_client_sdk", raise_import_error)

    with pytest.raises(ImportError, match="boxd"):
        BoxdRunner()


def test_runner_returns_stdout_and_deletes_machine_on_close(fake_boxd):
    with BoxdRunner() as runner:
        machines = fake_boxd[0].machines
        machines.exec_handler = lambda command, kwargs: FakeExecResult(stdout="45\n")
        result = runner.run("print(sum(range(10)))")

    assert result == BoxdRunResult(stdout="45\n", stderr="", exit_code=0, error=None)
    assert machines.created[0]["isolated"] is True
    assert machines.deleted == ["vm-1"]
    assert fake_boxd[0].close_calls == 1


def test_runner_reports_last_stderr_line_as_error(fake_boxd):
    seen: list[str] = []
    with BoxdRunner() as runner:
        fake_boxd[0].machines.exec_handler = lambda command, kwargs: FakeExecResult(
            stderr="Traceback (most recent call last):\nValueError: x\n",
            exit_code=1,
        )
        result = runner.run("raise ValueError('x')", on_stderr=seen.append)

    assert result.exit_code == 1
    assert result.error == "ValueError: x"
    assert seen == [result.stderr]


def test_runner_maps_timeout_exit_code(fake_boxd):
    with BoxdRunner(default_timeout=3) as runner:
        fake_boxd[0].machines.exec_handler = lambda command, kwargs: FakeExecResult(
            exit_code=124
        )
        result = runner.run("import time; time.sleep(10)")

    assert result.exit_code == 124
    assert result.error == "Execution timed out after 3s"
    assert fake_boxd[0].machines.execs[-1]["command"][3] == "3"


def test_runner_maps_sdk_errors_to_result(fake_boxd):
    def raise_error(command, kwargs):
        raise BoxdError("exec timed out after 18s")

    with BoxdRunner() as runner:
        fake_boxd[0].machines.exec_handler = raise_error
        result = runner.run("print(1)")

    assert result.exit_code == 124
    assert result.error == "exec timed out after 18s"


def test_runner_sets_a_delete_after_timer_by_default(fake_boxd):
    with BoxdRunner() as runner:
        runner.run("print(1)")

    assert fake_boxd[0].machines.created[0]["delete_after"] == 6 * 3600


def test_runner_returns_an_error_when_the_machine_is_not_ready_in_time(fake_boxd):
    seen: list[str] = []
    runner = BoxdRunner()
    try:
        runner._run(runner._aget_client())
        machines = fake_boxd[0].machines
        machines.ready_errors.append(
            TimeoutError("machine vm-1 did not reach 'running' within 120.0s")
        )
        machines.exec_handler = lambda command, kwargs: FakeExecResult(stdout="ok\n")

        result = runner.run("print('ok')", on_stderr=seen.append)
        after = runner.run("print('ok')")
    finally:
        runner.close()

    assert result.exit_code == 1
    assert result.error == "machine vm-1 did not reach 'running' within 120.0s"
    assert seen == [result.error]
    # The machine that never became ready is deleted, and the next run gets a new one.
    assert after == BoxdRunResult(stdout="ok\n", stderr="", exit_code=0, error=None)
    assert machines.deleted == ["vm-1", "vm-2"]


def test_runner_close_without_start_creates_nothing(fake_boxd):
    runner = BoxdRunner()
    runner.close()

    assert fake_boxd == []
    with pytest.raises(RuntimeError, match="closed"):
        runner.run("print(1)")


@pytest.mark.integration
@pytest.mark.skipif(not _has_boxd_credentials(), reason="BOXD_API_KEY is required")
def test_boxd_runner_runs_python_code():
    with BoxdRunner() as runner:
        result = runner.run("print('hello')")

    assert result.exit_code == 0
    assert "hello" in result.stdout


@pytest.mark.integration
@pytest.mark.skipif(not _has_boxd_credentials(), reason="BOXD_API_KEY is required")
def test_boxd_runner_captures_runtime_error():
    with BoxdRunner() as runner:
        result = runner.run("raise ValueError('x')")

    assert result.exit_code != 0
    assert result.error == "ValueError: x"


@pytest.mark.integration
@pytest.mark.skipif(not _has_boxd_credentials(), reason="BOXD_API_KEY is required")
def test_boxd_runner_timeout_surfaces_as_error():
    with BoxdRunner() as runner:
        result = runner.run("import time; time.sleep(10)", timeout=1)
        after = runner.run("print('still usable')")

    assert result.exit_code == 124
    assert result.error
    assert "still usable" in after.stdout


@pytest.mark.integration
@pytest.mark.skipif(not _has_boxd_credentials(), reason="BOXD_API_KEY is required")
def test_boxd_runner_env_vars_and_packages():
    with BoxdRunner(env_vars={"AREAL_FLAG": "on"}, python_packages=["sympy"]) as runner:
        result = runner.run(
            "import os, sympy\nprint(os.environ['AREAL_FLAG'], sympy.sqrt(16))"
        )

    assert result.exit_code == 0
    assert result.stdout.strip() == "on 4"
