# SPDX-License-Identifier: Apache-2.0

"""Integration and unit tests for the synchronous Daytona runner."""

from __future__ import annotations

import os

import pytest

import areal.infra.sandbox.runner as runner_module
from areal.infra.sandbox import DaytonaRunner, DaytonaRunResult


def _has_daytona_credentials() -> bool:
    return bool(os.environ.get("DAYTONA_API_KEY"))


def test_run_result_fields_round_trip():
    result = DaytonaRunResult(
        stdout="x",
        stderr="",
        exit_code=0,
        charts=[],
        error=None,
    )

    assert result.stdout == "x"
    assert result.stderr == ""
    assert result.exit_code == 0
    assert result.charts == []
    assert result.error is None


def test_runner_raises_clear_error_when_daytona_missing(monkeypatch):
    def raise_import_error():
        raise ImportError("daytona is not installed")

    monkeypatch.setattr(runner_module, "_load_daytona_sdk", raise_import_error)

    with pytest.raises(ImportError, match="daytona"):
        DaytonaRunner()


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_daytona_credentials(), reason="DAYTONA_API_KEY is required"
)
def test_runner_runs_python_code():
    runner = DaytonaRunner()

    try:
        result = runner.run("print('hello')")
    finally:
        runner.close()

    assert result.exit_code == 0
    assert "hello" in result.stdout

    runner.close()


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_daytona_credentials(), reason="DAYTONA_API_KEY is required"
)
def test_runner_context_manager_cleans_up():
    with DaytonaRunner() as runner:
        result = runner.run("print('context manager')")

    assert result.exit_code == 0
    assert "context manager" in result.stdout


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_daytona_credentials(), reason="DAYTONA_API_KEY is required"
)
def test_runner_captures_runtime_error():
    with DaytonaRunner() as runner:
        result = runner.run("raise ValueError('x')")

    assert result.exit_code != 0
    assert result.error


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_daytona_credentials(), reason="DAYTONA_API_KEY is required"
)
def test_runner_extracts_matplotlib_charts():
    code = """
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [3, 2, 1])
plt.title('chart')
plt.show()
"""

    with DaytonaRunner() as runner:
        result = runner.run(code)

    assert result.charts


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_daytona_credentials(), reason="DAYTONA_API_KEY is required"
)
def test_runner_timeout_surfaces_as_error():
    with DaytonaRunner() as runner:
        result = runner.run("import time; time.sleep(10)", timeout=1)

    assert result.exit_code != 0
    assert result.error
