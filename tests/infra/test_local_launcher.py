from __future__ import annotations

from collections import defaultdict
from unittest.mock import Mock, patch

import pytest

from areal.infra.launcher.local import LocalLauncher


@pytest.fixture
def launcher(tmp_path):
    """Create a LocalLauncher without probing host GPU devices."""
    local_launcher = object.__new__(LocalLauncher)
    local_launcher.experiment_name = "experiment"
    local_launcher.trial_name = "trial"
    local_launcher.fileroot = str(tmp_path)
    local_launcher._jobs = {}
    local_launcher._job_counter = defaultdict(int)
    local_launcher._job_states = {}
    local_launcher._gpu_counter = 0
    local_launcher._gpu_devices = []
    local_launcher.wait = Mock()
    yield local_launcher
    local_launcher._jobs.clear()
    local_launcher._job_counter.clear()


def _submitted_command(launcher: LocalLauncher, env_vars: dict[str, str]) -> str:
    process = Mock(pid=1234)
    with patch(
        "areal.infra.launcher.local.subprocess.Popen", return_value=process
    ) as mock_popen:
        launcher.submit(
            job_name="trainer",
            cmd="python train.py",
            env_vars=env_vars,
        )

    return mock_popen.call_args.args[0]


def test_local_launcher_regular_env_uses_stdbuf_and_quotes_value(launcher):
    """Regular local jobs retain stdbuf and shell-safe environment values."""
    command = _submitted_command(launcher, {"WORKER_LABEL": "actor one"})

    assert command.startswith(
        "WORKER_LABEL='actor one' stdbuf -oL python train.py 2>&1 | tee -a "
    )


def test_local_launcher_explicit_ld_preload_skips_stdbuf_and_quotes_value(launcher):
    """TMS preloads are passed directly to the target local process."""
    command = _submitted_command(
        launcher,
        {
            "LD_PRELOAD": "/opt/tms hooks/tms preload.so",
            "WORKER_LABEL": "actor one",
        },
    )

    assert command.startswith(
        "LD_PRELOAD='/opt/tms hooks/tms preload.so' "
        "WORKER_LABEL='actor one' python train.py 2>&1 | tee -a "
    )
    assert "stdbuf" not in command.split(" 2>&1", maxsplit=1)[0]
