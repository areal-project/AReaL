# SPDX-License-Identifier: Apache-2.0

import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from areal.api.cli_args import NameResolveConfig, SchedulingSpec
from areal.infra.scheduler.slurm import (
    SlurmScheduler,
    _resolve_fork_python_executable,
    _resolve_srun_additional_args,
    _supervise_rpc_command,
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


@pytest.mark.parametrize("container_type", ["native", "apptainer"])
def test_sbatch_worker_runs_supervisor_with_selected_python(tmp_path, container_type):
    """Supervision is inside the container and follows runtime environment setup."""
    scheduler = object.__new__(SlurmScheduler)
    scheduler._n_gpus_per_node = 8
    scheduler.experiment_name = "exp"
    scheduler.trial_name = "trial"
    scheduler.fileroot = str(tmp_path)
    scheduler._slurm_name = lambda role: f"exp-trial-{role}"
    scheduler.name_resolve_config = NameResolveConfig()
    scheduler.container_type = container_type
    scheduler.container_mounts = ""
    scheduler.srun_additional_args = "--mpi=none"
    scheduler._log_path_of = lambda role: tmp_path / f"{role}.log"
    scheduler._merged_log_path = lambda: tmp_path / "merged.log"
    spec = SchedulingSpec(
        gpu=0,
        cpu=1,
        mem=1,
        cmd="/opt/worker/bin/python3.12 -m areal.infra.rpc.rpc_server",
        additional_bash_cmds=["export WORKER_VALUE=from_setup"],
    )
    script = scheduler._generate_sbatch_script(
        role="actor",
        replicas=1,
        nodes=1,
        total_gpus=0,
        cpus_per_task=1,
        mem_per_task=1024,
        schedulings=[spec],
        nodelist=None,
        exclude=None,
    )
    tokens = shlex.split(script[script.index("srun ") :])
    setup = tokens[tokens.index("bash") + 2]
    supervisor = shlex.split(setup.split(";\n")[-1])
    assert supervisor[:2] == ["exec", "/opt/worker/bin/python3.12"]
    assert "process_supervisor.py" in supervisor[3]
    assert supervisor[4:7] == ["--", "bash", "-c"]

    # Execute the generated worker setup without requiring the custom interpreter
    # or RPC service; command quoting and exit propagation remain exercised.
    probe = "import os,sys; sys.stdout.write(os.environ['WORKER_VALUE']); sys.exit(6)"
    worker = shlex.join([sys.executable, "-c", probe])
    supervised = shlex.join([sys.executable, *supervisor[2:7], worker])
    result = subprocess.run(
        ["bash", "-c", f"export WORKER_VALUE=from_setup; exec {supervised}"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 6, result.stderr
    assert result.stdout == "from_setup"

    # A successful tee must not turn a failed worker step into COMPLETED.
    step = tmp_path / "srun"
    step.write_text("#!/bin/sh\nexit 17\n")
    step.chmod(0o755)
    failed_step = subprocess.run(
        ["bash"],
        input=f"export PATH={shlex.quote(str(tmp_path))}:$PATH\n" + script,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert failed_step.returncode == 17, failed_step.stderr


@pytest.mark.parametrize("with_env", [False, True])
@pytest.mark.parametrize("interpreter", ["absolute", "variable", "quoted_variable"])
def test_supervision_preserves_python_script_and_env_wrapper(
    tmp_path, with_env, interpreter
):
    # A missing bare python catches accidental fallback from the explicit runtime.
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import os,sys; sys.stdout.write(os.environ['LABEL']); sys.exit(4)"
    )
    executable = shlex.quote(sys.executable)
    if interpreter != "absolute":
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "python3").symlink_to(sys.executable)
        executable = "$WORKER_RUNTIME/python3"
        if interpreter == "quoted_variable":
            executable = '"$WORKER_RUNTIME/python3"'
    command = executable + " " + shlex.quote(str(worker))
    if with_env:
        command = 'env LABEL="$EXPECTED_LABEL" ' + command
    supervised = _supervise_rpc_command(command)
    result = subprocess.run(
        ["bash", "-c", supervised],
        env={
            "PATH": "/usr/bin:/bin",
            "LABEL": "inherited",
            "EXPECTED_LABEL": "with spaces ' quote",
            "WORKER_RUNTIME": str(tmp_path / "runtime"),
        },
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 4, result.stderr
    assert result.stdout == ("with spaces ' quote" if with_env else "inherited")
