# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from areal.api.cli_args import SchedulingSpec
from areal.infra.scheduler.slurm import (
    _resolve_fork_python_executable,
    _resolve_srun_additional_args,
)


@pytest.mark.parametrize("command", ["", "  ", "'", "env python -m worker", "python"])
def test_fork_python_unrecognized_command_uses_default(command):
    assert (
        _resolve_fork_python_executable(SchedulingSpec(cmd=command)) == sys.executable
    )


def test_fork_python_uses_worker_interpreter():
    spec = SchedulingSpec(
        cmd="/opt/worker/bin/python3.12 -u -m areal.infra.rpc.rpc_server"
    )
    assert _resolve_fork_python_executable(spec) == "/opt/worker/bin/python3.12"
    assert _resolve_fork_python_executable(None) == sys.executable


def test_srun_role_override_preserves_global_default():
    assert _resolve_srun_additional_args(SchedulingSpec(), "--mpi=none") == "--mpi=none"
    spec = SchedulingSpec(srun_additional_args="--mpi=pmi2 --unbuffered")
    assert (
        _resolve_srun_additional_args(spec, "--mpi=none") == spec.srun_additional_args
    )
