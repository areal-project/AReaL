from __future__ import annotations

import subprocess
from unittest import mock

import pytest

from areal.api.cli_args import InferenceEngineConfig
from areal.infra.remote_inf_engine import RemoteInfEngine


def test_wait_for_server_dead_process_raises_runtime_error_with_returncode():
    # Arrange
    config = InferenceEngineConfig(setup_timeout=5.0)
    engine = RemoteInfEngine(config, backend=mock.Mock())
    process = mock.Mock(spec=subprocess.Popen)
    process.pid = 4242
    process.poll.return_value = 7
    process.returncode = 7

    # Act / Assert
    with pytest.raises(RuntimeError, match="exited with code 7") as exc_info:
        engine._wait_for_server("localhost:30000", process=process)
    assert "pid=4242" in str(exc_info.value)


@pytest.mark.parametrize(
    "failure", [TimeoutError("timed out"), RuntimeError("process died")]
)
def test_launch_server_shuts_down_the_server_on_any_launch_failure(failure):
    # Arrange
    config = InferenceEngineConfig(setup_timeout=5.0)
    backend = mock.Mock()
    backend.launch_server.return_value = mock.Mock(spec=subprocess.Popen)
    engine = RemoteInfEngine(config, backend=backend)

    # Act / Assert
    with (
        mock.patch.object(engine, "_wait_for_server", side_effect=failure),
        mock.patch.object(engine, "_shutdown_one_server") as shutdown,
        mock.patch(
            "areal.infra.remote_inf_engine.find_free_ports", return_value=[30000]
        ),
        mock.patch("areal.infra.remote_inf_engine.gethostip", return_value="127.0.0.1"),
        pytest.raises(type(failure)),
    ):
        engine.launch_server({})

    shutdown.assert_called_once()
    assert engine.local_server_processes == []
