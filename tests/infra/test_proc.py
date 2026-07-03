from __future__ import annotations

from unittest.mock import patch

from areal.infra.utils.proc import build_streaming_log_cmd


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
