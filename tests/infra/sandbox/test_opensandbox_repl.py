# SPDX-License-Identifier: Apache-2.0

"""Direct tests for the interpreter script that runs inside an OpenSandbox sandbox.

The script is started as a real subprocess, so these tests exercise the same
wire protocol the session uses, without a sandbox.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from areal.infra.sandbox._opensandbox_repl import call

pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(signal, "SIGALRM"),
    reason="uses AF_UNIX sockets and SIGALRM",
)


def _wait_for_socket(path: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return
        time.sleep(0.05)
    raise RuntimeError(f"interpreter socket {path} never appeared")


def _start_server(tmp_path: Path) -> tuple[subprocess.Popen, str]:
    from areal.infra.sandbox import _opensandbox_repl

    socket_path = str(tmp_path / f"repl-{uuid.uuid4().hex[:8]}.sock")
    process = subprocess.Popen(
        [sys.executable, "-u", _opensandbox_repl.__file__, "serve", socket_path]
    )
    _wait_for_socket(socket_path)
    return process, socket_path


def _encode(code: str) -> str:
    return base64.b64encode(code.encode("utf-8")).decode("ascii")


def _run_cell(socket_path: str, code: str, timeout: float) -> dict:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        call(socket_path, str(timeout), _encode(code))
    return json.loads(buffer.getvalue())


def test_interpreter_keeps_state_and_captures_output(tmp_path):
    process, socket_path = _start_server(tmp_path)
    try:
        first = _run_cell(socket_path, "x = 5\nprint('set')", timeout=5)
        second = _run_cell(socket_path, "print(x * 2)", timeout=5)
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert first["error"] is None
    assert first["stdout"] == "set\n"
    assert second["error"] is None
    assert second["stdout"] == "10\n"


def test_interpreter_reports_error_with_cell_traceback(tmp_path):
    process, socket_path = _start_server(tmp_path)
    try:
        result = _run_cell(socket_path, "raise ValueError('boom')", timeout=5)
        after = _run_cell(socket_path, "print('alive')", timeout=5)
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert result["error"]["name"] == "ValueError"
    assert result["error"]["value"] == "boom"
    assert "<cell>" in result["error"]["traceback"]
    # The namespace survives a failed cell.
    assert after["stdout"] == "alive\n"


def test_interpreter_timeout_returns_error(tmp_path):
    process, socket_path = _start_server(tmp_path)
    try:
        result = _run_cell(socket_path, "import time; time.sleep(5)", timeout=1)
        after = _run_cell(socket_path, "print('back')", timeout=5)
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert result["error"] is not None
    assert "timed out" in result["error"]["value"].lower()
    # The server still answers after a timed-out cell.
    assert after["stdout"] == "back\n"


def test_call_reads_code_from_file(tmp_path):
    process, socket_path = _start_server(tmp_path)
    try:
        code_file = tmp_path / "cell.b64"
        code_file.write_text(_encode("print('from-file')"))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            call(socket_path, "5", f"@{code_file}")
        payload = json.loads(buffer.getvalue())
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert payload["stdout"] == "from-file\n"
    assert not code_file.exists()
