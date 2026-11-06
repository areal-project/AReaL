from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from areal.infra.utils.launcher import get_env_vars, run_post_exit_hook


def _config(post_exit_hook: str):
    return SimpleNamespace(
        post_exit_hook=post_exit_hook,
        cluster=SimpleNamespace(fileroot="/tmp/areal-test"),
        experiment_name="exp",
        trial_name="trial",
    )


def test_run_post_exit_hook_empty_hook_skips_subprocess():
    with patch("areal.infra.utils.launcher.subprocess.run") as mock_run:
        run_post_exit_hook(_config(""))

    mock_run.assert_not_called()


def test_run_post_exit_hook_command_injects_log_dir():
    with patch("areal.infra.utils.launcher.getpass.getuser", return_value="tester"):
        with patch("areal.infra.utils.launcher.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args="cleanup",
                returncode=0,
                stdout="ok",
                stderr="",
            )

            run_post_exit_hook(_config(" cleanup "))

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert mock_run.call_args.args[0] == "cleanup"
    assert kwargs["shell"] is True
    assert kwargs["timeout"] == 600
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["env"]["LOG_DIR"] == "/tmp/areal-test/logs/tester/exp/trial"


def test_run_post_exit_hook_timeout_does_not_raise():
    with patch("areal.infra.utils.launcher.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd="cleanup",
            timeout=600,
        )

        run_post_exit_hook(_config("cleanup"))

    mock_run.assert_called_once()


def test_get_env_vars_mapping_keeps_commas_and_equals_in_values():
    env = get_env_vars(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
            "X": "a=b",
        }
    )

    assert (
        env["PYTORCH_CUDA_ALLOC_CONF"]
        == "expandable_segments:True,max_split_size_mb:128"
    )
    assert env["X"] == "a=b"


def test_get_env_vars_string_splits_on_first_equals_only():
    env = get_env_vars("A=1,B=x=y")

    assert env["A"] == "1"
    assert env["B"] == "x=y"


def test_get_env_vars_without_additional_vars():
    assert "PYTHONPATH" in get_env_vars()
    assert "PYTHONPATH" in get_env_vars(None)
    assert "PYTHONPATH" in get_env_vars({})
