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
