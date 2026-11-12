# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the boxd machine session and the in-machine interpreter."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("boxd")

from boxd.errors import BoxdError, ConflictError, NotFoundError  # noqa: E402

from areal.infra.sandbox import _boxd_session as session_module  # noqa: E402
from areal.infra.sandbox._boxd_session import (  # noqa: E402
    DEFAULT_DELETE_AFTER,
    REPL_PATH,
    BoxdSession,
)

from ._fake_boxd import FakeAsyncBoxd, FakeExecResult  # noqa: E402

REPL_SOURCE = Path(session_module.__file__).with_name("_boxd_repl.py")


def _repl_reply(stdout="", stderr="", error=None) -> FakeExecResult:
    return FakeExecResult(
        stdout=json.dumps({"stdout": stdout, "stderr": stderr, "error": error})
    )


@pytest.mark.asyncio
async def test_astart_creates_isolated_machine_with_options():
    client = FakeAsyncBoxd()
    session = BoxdSession(
        name_prefix="rl",
        org="my-org",
        region="fra",
        snapshot="py-tools",
        vcpu=4,
        memory="16G",
        env_vars={"A": "1"},
        labels={"run": "exp1"},
    )

    machine_id = await session.astart(client)

    created = client.machines.created[0]
    assert machine_id == "vm-1"
    assert created["name"].startswith("rl-")
    assert created["isolated"] is True
    assert created["org"] == "my-org"
    assert created["region"] == "fra"
    assert created["from_snapshot"] == "py-tools"
    assert created["vcpu"] == 4
    assert created["memory"] == "16G"
    assert created["labels"] == {"areal-sandbox": "true", "run": "exp1"}
    # A snapshot restore takes no env; env_vars go with every command instead.
    assert "env" not in created
    await session.aexec(client, "true", timeout=5)
    assert client.machines.execs[-1]["env"] == {"A": "1"}


@pytest.mark.asyncio
async def test_astart_passes_env_and_delete_after_without_snapshot():
    client = FakeAsyncBoxd()
    session = BoxdSession(env_vars={"A": "1"}, delete_after=3600)

    await session.astart(client)

    created = client.machines.created[0]
    assert created["env"] == {"A": "1"}
    assert created["delete_after"] == 3600
    assert created["labels"] == {"areal-sandbox": "true"}


@pytest.mark.asyncio
async def test_astart_sets_a_delete_after_timer_by_default():
    client = FakeAsyncBoxd()

    await BoxdSession().astart(client)

    assert DEFAULT_DELETE_AFTER == 6 * 3600
    assert client.machines.created[0]["delete_after"] == DEFAULT_DELETE_AFTER


@pytest.mark.asyncio
async def test_astart_sets_no_timer_when_delete_after_is_none():
    client = FakeAsyncBoxd()

    await BoxdSession(delete_after=None).astart(client)

    assert "delete_after" not in client.machines.created[0]


@pytest.mark.parametrize("delete_after", [0, -5])
def test_non_positive_delete_after_is_rejected(delete_after):
    with pytest.raises(ValueError, match="delete_after must be positive or None"):
        BoxdSession(delete_after=delete_after)


@pytest.mark.asyncio
async def test_astart_deletes_machine_when_cancelled_during_create():
    client = FakeAsyncBoxd()
    release = asyncio.Event()
    original_create = client.machines.create

    async def slow_create(name, **kwargs):
        await release.wait()
        return await original_create(name, **kwargs)

    client.machines.create = slow_create
    session = BoxdSession()
    task = asyncio.create_task(session.astart(client))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert client.machines.deleted == ["vm-1"]
    assert session.machine_id is None


@pytest.mark.asyncio
async def test_astart_is_idempotent():
    client = FakeAsyncBoxd()
    session = BoxdSession()

    await session.astart(client)
    await session.astart(client)

    assert len(client.machines.created) == 1


@pytest.mark.asyncio
async def test_astart_installs_python_packages_into_a_venv():
    client = FakeAsyncBoxd()
    session = BoxdSession(python_packages=["numpy", "sympy>=1.12"])

    await session.astart(client)

    command = client.machines.execs[0]["command"]
    assert command[:2] == ["bash", "-c"]
    assert "uv pip install" in command[2]
    assert "numpy 'sympy>=1.12'" in command[2]
    assert session.python.endswith("/venv/bin/python")


@pytest.mark.asyncio
async def test_astart_deletes_machine_when_package_install_fails():
    client = FakeAsyncBoxd()
    client.machines.exec_handler = lambda command, kwargs: FakeExecResult(
        stderr="no such package", exit_code=1
    )
    session = BoxdSession(python_packages=["does-not-exist"])

    with pytest.raises(RuntimeError, match="no such package"):
        await session.astart(client)

    assert client.machines.deleted == ["vm-1"]
    assert session.machine_id is None


@pytest.mark.asyncio
async def test_arun_script_wraps_code_in_timeout():
    client = FakeAsyncBoxd()
    session = BoxdSession(env_vars={"A": "1"})

    await session.arun_script(client, "print(1)", timeout=5)

    call = client.machines.execs[-1]
    assert call["command"] == ["timeout", "-k", "2", "5", "python3", "-c", "print(1)"]
    assert call["timeout"] > 5
    assert call["env"] == {"A": "1"}


@pytest.mark.asyncio
async def test_arun_script_uploads_large_code():
    client = FakeAsyncBoxd()
    session = BoxdSession()
    code = "x = 1\n" * 50_000

    await session.arun_script(client, code, timeout=5)

    command = client.machines.execs[-1]["command"]
    assert command[-1].endswith(".py")
    assert client.machines.uploads[command[-1]] == code


@pytest.mark.asyncio
async def test_arun_script_uploads_code_whose_quoting_is_too_long():
    client = FakeAsyncBoxd()
    session = BoxdSession()
    code = "print('a')\n" * 8_000  # 88 KB raw, much longer once shell-quoted

    await session.arun_script(client, code, timeout=5)

    command = client.machines.execs[-1]["command"]
    assert command[-1].endswith(".py")
    assert client.machines.uploads[command[-1]] == code
    assert client.machines.removed == [command[-1]]


@pytest.mark.asyncio
async def test_arun_script_measures_the_command_in_bytes():
    client = FakeAsyncBoxd()
    session = BoxdSession()
    code = "x = '" + "\u00e9" * 60_000 + "'"  # 60 K characters, 120 KB as UTF-8

    await session.arun_script(client, code, timeout=5)

    command = client.machines.execs[-1]["command"]
    assert command[-1].endswith(".py")
    assert client.machines.uploads[command[-1]] == code


@pytest.mark.asyncio
async def test_fits_inline_measures_env_vars_in_bytes():
    session = BoxdSession(env_vars={"NOTE": "\u00e9" * 60_000})

    assert not session._fits_inline(["true"])


@pytest.mark.asyncio
async def test_arun_script_uses_a_new_machine_when_the_old_one_is_gone():
    client = FakeAsyncBoxd()
    client.machines.exec_handler = lambda command, kwargs: FakeExecResult(stdout="1\n")
    session = BoxdSession()
    await session.arun_script(client, "print(1)", timeout=5)
    client.machines.gone.add("vm-1")  # its delete_after time passed

    result = await session.arun_script(client, "print(1)", timeout=5)

    assert result.stdout == "1\n"
    assert session.machine_id == "vm-2"
    assert [call["id"] for call in client.machines.execs] == ["vm-1", "vm-2"]


@pytest.mark.asyncio
async def test_arun_script_keeps_the_machine_on_an_unrelated_not_found():
    client = FakeAsyncBoxd()
    session = BoxdSession()
    await session.astart(client)

    def raise_not_found(command, kwargs):
        raise NotFoundError("no such file")

    client.machines.exec_handler = raise_not_found

    with pytest.raises(NotFoundError):
        await session.arun_script(client, "print(1)", timeout=5)

    assert session.machine_id == "vm-1"
    assert len(client.machines.created) == 1


@pytest.mark.asyncio
async def test_arun_cell_reports_lost_state_when_the_machine_is_gone():
    client = FakeAsyncBoxd()
    client.machines.exec_handler = lambda command, kwargs: (
        _repl_reply() if command[1:3] == [REPL_PATH, "call"] else FakeExecResult()
    )
    session = BoxdSession()
    await session.arun_cell(client, "x = 5", timeout=5)
    client.machines.gone.add("vm-1")  # its delete_after time passed

    result = await session.arun_cell(client, "print(x)", timeout=5)
    after = await session.arun_cell(client, "y = 1", timeout=5)

    assert result.error is not None
    assert "new one was created" in result.error.value
    assert "earlier variables are gone" in result.error.value
    assert session.machine_id == "vm-2"
    # The cell that found the machine gone is not replayed in the new one.
    sent = [
        base64.b64decode(call["command"][-1]).decode()
        for call in client.machines.execs
        if call["command"][1:3] == [REPL_PATH, "call"]
    ]
    assert sent == ["x = 5", "y = 1"]
    assert after.error is None


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1])
async def test_non_positive_timeout_is_rejected(timeout):
    session = BoxdSession()

    with pytest.raises(ValueError, match="timeout must be positive"):
        await session.arun_script(FakeAsyncBoxd(), "print(1)", timeout=timeout)
    with pytest.raises(ValueError, match="timeout must be positive"):
        await session.arun_cell(FakeAsyncBoxd(), "print(1)", timeout=timeout)


@pytest.mark.asyncio
async def test_aexec_does_not_retry_conflict_on_running_machine():
    client = FakeAsyncBoxd()
    session = BoxdSession()
    await session.astart(client)

    def conflict(command, kwargs):
        raise ConflictError("aborted")

    client.machines.exec_handler = conflict

    with pytest.raises(ConflictError):
        await session.aexec(client, "true", timeout=5)

    assert client.machines.started == []
    assert len(client.machines.execs) == 1


@pytest.mark.asyncio
async def test_arun_cell_is_not_replayed_after_machine_restart():
    client = FakeAsyncBoxd()
    session = BoxdSession()
    calls = []

    def handler(command, kwargs):
        if command[0] == "bash":
            return FakeExecResult()
        calls.append(command)
        # After the restart the old interpreter is gone.
        return FakeExecResult(stderr="ConnectionRefusedError", exit_code=1)

    client.machines.exec_handler = handler
    await session.arun_cell(client, "x = 1", timeout=5)
    client.machines.stopped.add("vm-1")

    result = await session.arun_cell(client, "x += 1", timeout=5)

    assert client.machines.started == ["vm-1"]
    assert result.error is not None
    assert "earlier variables are gone" in result.error.value


@pytest.mark.asyncio
async def test_adelete_retries_and_keeps_id_until_success():
    client = FakeAsyncBoxd()
    session = BoxdSession()
    await session.astart(client)
    client.machines.delete_errors = [BoxdError("unavailable")] * 3

    with pytest.raises(BoxdError):
        await session.adelete(client)
    assert session.machine_id == "vm-1"

    client.machines.delete_errors = [BoxdError("unavailable")]
    await session.adelete(client)
    assert session.machine_id is None
    assert client.machines.deleted == ["vm-1"]


@pytest.mark.asyncio
async def test_aexec_starts_a_stopped_machine_and_retries():
    client = FakeAsyncBoxd()
    session = BoxdSession()
    await session.astart(client)
    client.machines.stopped.add("vm-1")

    result = await session.aexec(client, "true", timeout=5)

    assert result.exit_code == 0
    assert client.machines.started == ["vm-1"]


@pytest.mark.asyncio
async def test_arun_cell_starts_interpreter_once_and_parses_result():
    client = FakeAsyncBoxd()

    def handler(command, kwargs):
        if command[0] == "bash":
            return FakeExecResult()
        return _repl_reply(stdout="10\n")

    client.machines.exec_handler = handler
    session = BoxdSession()

    first = await session.arun_cell(client, "x = 5", timeout=5)
    second = await session.arun_cell(client, "print(x * 2)", timeout=5)

    starts = [c for c in client.machines.execs if c["command"][0] == "bash"]
    assert len(starts) == 1
    assert client.machines.uploads[REPL_PATH] == REPL_SOURCE.read_text()
    assert first.error is None
    assert second.stdout == "10\n"
    call = client.machines.execs[-1]["command"]
    assert call[2] == "call"
    assert base64.b64decode(call[-1]).decode() == "print(x * 2)"


@pytest.mark.asyncio
async def test_arun_cell_maps_error_payload():
    client = FakeAsyncBoxd()
    error = {"name": "ValueError", "value": "x", "traceback": "Traceback ...\n"}
    client.machines.exec_handler = lambda command, kwargs: (
        FakeExecResult() if command[0] == "bash" else _repl_reply(error=error)
    )
    session = BoxdSession()

    result = await session.arun_cell(client, "raise ValueError('x')", timeout=5)

    assert result.error is not None
    assert result.error.name == "ValueError"


@pytest.mark.asyncio
async def test_arun_cell_restarts_interpreter_when_it_died():
    client = FakeAsyncBoxd()
    client.machines.exec_handler = lambda command, kwargs: (
        FakeExecResult()
        if command[0] == "bash"
        else FakeExecResult(stderr="ConnectionRefusedError", exit_code=1)
    )
    session = BoxdSession()

    result = await session.arun_cell(client, "import os; os._exit(1)", timeout=5)

    starts = [c for c in client.machines.execs if c["command"][0] == "bash"]
    assert len(starts) == 2
    assert result.error is not None
    assert "earlier variables are gone" in result.error.value


@pytest.mark.asyncio
async def test_arun_cell_restarts_interpreter_when_exec_times_out():
    client = FakeAsyncBoxd()

    def handler(command, kwargs):
        if command[0] == "bash":
            return FakeExecResult()
        raise BoxdError("exec timed out after 20.0s")

    client.machines.exec_handler = handler
    session = BoxdSession()

    result = await session.arun_cell(client, "while True: pass", timeout=5)

    assert result.error is not None
    assert result.error.name == "TimeoutError"


@pytest.mark.asyncio
async def test_adelete_tolerates_missing_machine():
    client = FakeAsyncBoxd()
    session = BoxdSession()
    await session.astart(client)
    client.machines.gone.add("vm-1")

    await session.adelete(client)

    assert session.machine_id is None


def _call_repl(socket_path: Path, code: str, timeout: float = 5) -> dict:
    encoded = base64.b64encode(code.encode()).decode()
    out = subprocess.run(
        [
            sys.executable,
            str(REPL_SOURCE),
            "call",
            str(socket_path),
            str(timeout),
            encoded,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(out.stdout)


@pytest.mark.skipif(sys.platform == "win32", reason="Unix sockets required")
def test_repl_keeps_state_and_reports_errors(tmp_path):
    socket_path = tmp_path / "repl.sock"
    server = subprocess.Popen(
        [sys.executable, str(REPL_SOURCE), "serve", str(socket_path)]
    )
    try:
        for _ in range(100):
            if socket_path.exists():
                break
            time.sleep(0.05)

        assert _call_repl(socket_path, "x = 5")["error"] is None
        assert _call_repl(socket_path, "print(x * 2)")["stdout"] == "10\n"

        syntax = _call_repl(socket_path, "def f(: pass")
        assert syntax["error"]["name"] == "SyntaxError"

        timed_out = _call_repl(socket_path, "import time; time.sleep(5)", timeout=0.5)
        assert timed_out["error"]["name"] == "TimeoutError"
        assert _call_repl(socket_path, "print(x)")["stdout"] == "5\n"

        big = "y = 0\n" + "y += 1\n" * 20_000 + "print(y)"
        code_file = tmp_path / "cell.b64"
        code_file.write_text(base64.b64encode(big.encode()).decode())
        out = subprocess.run(
            [
                sys.executable,
                str(REPL_SOURCE),
                "call",
                str(socket_path),
                "5",
                f"@{code_file}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(out.stdout)["stdout"] == "20000\n"
    finally:
        server.kill()
        server.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="Unix sockets required")
def test_repl_survives_a_caller_that_goes_away(tmp_path):
    socket_path = tmp_path / "repl.sock"
    server = subprocess.Popen(
        [sys.executable, str(REPL_SOURCE), "serve", str(socket_path)]
    )
    try:
        for _ in range(100):
            if socket_path.exists():
                break
            time.sleep(0.05)
        assert _call_repl(socket_path, "kept = 'yes'")["error"] is None

        code = base64.b64encode(b"import time; time.sleep(1)").decode()
        caller = subprocess.Popen(
            [sys.executable, str(REPL_SOURCE), "call", str(socket_path), "5", code]
        )
        time.sleep(0.3)
        caller.kill()
        caller.wait()
        time.sleep(1.0)

        assert server.poll() is None
        assert _call_repl(socket_path, "print(kept)")["stdout"] == "yes\n"

        code_file = tmp_path / "cell.b64"
        code_file.write_text(base64.b64encode(b"print(kept)").decode())
        out = subprocess.run(
            [
                sys.executable,
                str(REPL_SOURCE),
                "call",
                str(socket_path),
                "5",
                f"@{code_file}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(out.stdout)["stdout"] == "yes\n"
        assert not os.path.exists(code_file)
    finally:
        server.kill()
        server.wait()
