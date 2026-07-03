from __future__ import annotations

from unittest.mock import Mock, patch

from areal.infra.utils.proc import build_streaming_log_cmd, run_with_streaming_logs


def _target_command(shell_command: str) -> str:
    return shell_command.split(" 2>&1", maxsplit=1)[0]


@patch("areal.infra.utils.proc.shutil.which", return_value="/usr/bin/stdbuf")
def test_build_streaming_log_cmd_regular_env_uses_stdbuf(mock_which):
    """Regular target commands retain line buffering and quote env values."""
    command = build_streaming_log_cmd(
        ["python", "worker.py"],
        "/tmp/worker.log",
        "/tmp/merged.log",
        "actor",
        env_vars={"WORKER_LABEL": "actor one"},
    )

    assert (
        _target_command(command)
        == "WORKER_LABEL='actor one' stdbuf -oL python worker.py"
    )
    mock_which.assert_called_once_with("stdbuf")


@patch("areal.infra.utils.proc.shutil.which", return_value="/usr/bin/stdbuf")
def test_build_streaming_log_cmd_explicit_ld_preload_skips_target_stdbuf(
    mock_which,
):
    """An explicit preload reaches the target unchanged while sed stays buffered."""
    command = build_streaming_log_cmd(
        ["python", "worker.py"],
        "/tmp/worker.log",
        "/tmp/merged.log",
        "actor",
        env_vars={
            "LD_PRELOAD": "/opt/tms hooks/tms preload.so",
            "WORKER_LABEL": "actor one",
        },
    )

    assert _target_command(command) == (
        "LD_PRELOAD='/opt/tms hooks/tms preload.so' "
        "WORKER_LABEL='actor one' python worker.py"
    )
    assert "stdbuf -oL sed" in command
    mock_which.assert_called_once_with("stdbuf")


@patch("areal.infra.utils.proc.subprocess.Popen")
@patch("areal.infra.utils.proc.shutil.which", return_value="/usr/bin/stdbuf")
def test_run_with_streaming_logs_inherited_ld_preload_skips_target_stdbuf(
    mock_which,
    mock_popen,
):
    """A preload supplied through Popen's environment is not changed by stdbuf."""
    mock_popen.return_value = Mock()
    child_env = {
        "LD_PRELOAD": "/opt/tms/torch_memory_saver.so",
        "WORKER_LABEL": "actor",
    }

    run_with_streaming_logs(
        ["python", "worker.py"],
        "/tmp/worker.log",
        "/tmp/merged.log",
        "actor",
        env=child_env,
    )

    command = mock_popen.call_args.args[0]
    assert _target_command(command) == "python worker.py"
    assert "stdbuf -oL sed" in command
    assert mock_popen.call_args.kwargs["env"] is child_env
    mock_which.assert_called_once_with("stdbuf")


@patch("areal.infra.utils.proc.subprocess.Popen")
@patch("areal.infra.utils.proc.shutil.which", return_value="/usr/bin/stdbuf")
def test_run_with_streaming_logs_parent_ld_preload_skips_target_stdbuf(
    mock_which,
    mock_popen,
    monkeypatch,
):
    """Popen's default inherited environment is considered before wrapping."""
    mock_popen.return_value = Mock()
    monkeypatch.setenv("LD_PRELOAD", "/opt/tms/torch_memory_saver.so")

    run_with_streaming_logs(
        ["python", "worker.py"],
        "/tmp/worker.log",
        "/tmp/merged.log",
        "actor",
    )

    command = mock_popen.call_args.args[0]
    assert _target_command(command) == "python worker.py"
    assert "stdbuf -oL sed" in command
    assert mock_popen.call_args.kwargs["env"] is None
    mock_which.assert_called_once_with("stdbuf")


@patch("areal.infra.utils.proc.subprocess.Popen")
@patch("areal.infra.utils.proc.shutil.which", return_value="/usr/bin/stdbuf")
def test_run_with_streaming_logs_explicit_empty_env_uses_target_stdbuf(
    mock_which,
    mock_popen,
    monkeypatch,
):
    """An explicit empty Popen environment does not inherit the parent's preload."""
    mock_popen.return_value = Mock()
    monkeypatch.setenv("LD_PRELOAD", "/opt/tms/torch_memory_saver.so")
    child_env = {}

    run_with_streaming_logs(
        ["python", "worker.py"],
        "/tmp/worker.log",
        "/tmp/merged.log",
        "actor",
        env=child_env,
    )

    command = mock_popen.call_args.args[0]
    assert _target_command(command) == "stdbuf -oL python worker.py"
    assert mock_popen.call_args.kwargs["env"] is child_env
    mock_which.assert_called_once_with("stdbuf")
