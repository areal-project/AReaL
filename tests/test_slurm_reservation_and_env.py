# SPDX-License-Identifier: Apache-2.0
"""Tests for reservation/exclusive sbatch options and env-var precedence."""

import os
import shlex
import subprocess
from unittest import mock

import pytest

from areal.api.cli_args import NameResolveConfig, SchedulingSpec
from areal.infra.scheduler.slurm import SlurmScheduler


class TestSchedulingSpecSlurmOptions:
    def test_reservation_and_exclusive_default_to_off(self):
        spec = SchedulingSpec()

        assert spec.reservation is None
        assert spec.exclusive is False

    @pytest.mark.parametrize(
        "reservation,exclusive,present,absent",
        [
            (None, False, [], ["--reservation", "--exclusive"]),
            ("shanghai", False, ["--reservation=shanghai"], ["--exclusive"]),
            (None, True, ["--exclusive"], ["--reservation"]),
            ("shanghai", True, ["--reservation=shanghai", "--exclusive"], []),
        ],
    )
    def test_options_reach_the_sbatch_script(
        self, reservation, exclusive, present, absent
    ):
        scheduler = object.__new__(SlurmScheduler)
        scheduler._n_gpus_per_node = 8
        scheduler.experiment_name = "exp"
        scheduler.trial_name = "trial"
        scheduler.fileroot = "/tmp/areal-test"
        scheduler._slurm_name = lambda role: f"exp-trial-{role}"
        scheduler.name_resolve_config = NameResolveConfig(
            type="nfs", nfs_record_root="/tmp/areal-test/nr"
        )
        scheduler.container_type = "apptainer"
        scheduler.container_mounts = "/storage:/storage"
        scheduler.srun_additional_args = "--unbuffered --mpi=pmi2"
        spec = SchedulingSpec(
            gpu=8, cpu=4, mem=32, reservation=reservation, exclusive=exclusive
        )

        script = SlurmScheduler._generate_sbatch_script(
            scheduler,
            role="actor",
            replicas=8,
            nodes=1,
            total_gpus=8,
            cpus_per_task=4,
            mem_per_task=32768,
            schedulings=[spec],
            nodelist=None,
            exclude=None,
        )

        for token in present:
            assert token in script, f"{token!r} missing from sbatch script"
        for token in absent:
            assert token not in script, f"{token!r} unexpectedly in sbatch script"


@pytest.mark.parametrize("container_type", ["native", "apptainer"])
@pytest.mark.parametrize(
    "device_var", ["CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"]
)
@pytest.mark.parametrize(
    "visible,local_rank,gpus,expected",
    [
        ("3", 0, 1, "3"),
        ("2,5,7,9", 0, 2, "2,5"),
        ("2,5,7,9", 1, 2, "7,9"),
        ("0", 0, 1, "0"),
        ("GPU-abc,MIG-GPU-def/1/0", 1, 1, "MIG-GPU-def/1/0"),
        (None, 0, 1, None),
        ("", 0, 1, None),
        ("-1", 0, 1, None),
        ("2,,3", 0, 1, None),
        ("2,", 0, 1, None),
        ("2", 1, 1, None),
    ],
)
def test_worker_gpu_assignment_preserves_slurm_allocation(
    container_type, device_var, visible, local_rank, gpus, expected
):
    scheduler = object.__new__(SlurmScheduler)
    scheduler._n_gpus_per_node = 4
    scheduler.experiment_name = "exp"
    scheduler.trial_name = "trial"
    scheduler.fileroot = "/tmp/areal-test"
    scheduler._slurm_name = lambda role: f"exp-trial-{role}"
    scheduler.name_resolve_config = NameResolveConfig()
    scheduler.container_type = container_type
    scheduler.container_mounts = ""
    scheduler.srun_additional_args = "--mpi=none"
    spec = SchedulingSpec(gpu=gpus, cpu=1, mem=1, cmd="true")
    script = scheduler._generate_sbatch_script(
        role="actor",
        replicas=4 // gpus,
        nodes=1,
        total_gpus=4,
        cpus_per_task=1,
        mem_per_task=1024,
        schedulings=[spec],
        nodelist=None,
        exclude=None,
    )
    tokens = shlex.split(script[script.index("stdbuf -oL srun") :])
    command = tokens[tokens.index("bash") + 2]
    command = command[: command.rindex("true --experiment-name")]
    command += f'printf "%s" "${{{device_var}}}"'
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"}
    }
    env.update(SLURM_LOCALID=str(local_rank), SLURM_STEP_GPUS="6")
    other_var = (
        "ASCEND_RT_VISIBLE_DEVICES"
        if device_var == "CUDA_VISIBLE_DEVICES"
        else "CUDA_VISIBLE_DEVICES"
    )
    env[other_var] = ""
    if visible is not None:
        env[device_var] = visible
    result = subprocess.run(
        ["bash", "-c", command], env=env, capture_output=True, text=True
    )
    if expected is None:
        assert result.returncode != 0
        assert result.stdout == ""
        assert "Slurm-visible device" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout == expected


class TestUserEnvPrecedence:
    def test_user_env_wins_over_framework_defaults(self):
        scheduler = object.__new__(SlurmScheduler)
        scheduler.enable_tms_offload = False
        spec = SchedulingSpec(cpu=4, env_vars={"OMP_NUM_THREADS": "13"})

        with (
            mock.patch(
                "areal.infra.scheduler.slurm.get_env_vars",
                return_value={"OMP_NUM_THREADS": "1", "AREAL_X": "1"},
            ),
            mock.patch(
                "areal.infra.scheduler.slurm.get_thread_env_vars",
                return_value={"OMP_NUM_THREADS": "4"},
            ),
        ):
            out = SlurmScheduler._prepare_worker_specs(scheduler, "actor", 1, [spec])

        assert out[0].env_vars["OMP_NUM_THREADS"] == "13", (
            "framework defaults overrode an explicit scheduling_spec env var"
        )
        assert out[0].env_vars["AREAL_X"] == "1"

    def test_roles_keep_independent_env_vars(self):
        scheduler = object.__new__(SlurmScheduler)
        scheduler.enable_tms_offload = False
        actor = SchedulingSpec(cpu=4, env_vars={"PYTORCH_CUDA_ALLOC_CONF": "a:1"})
        rollout = SchedulingSpec(cpu=4, env_vars={})

        with (
            mock.patch("areal.infra.scheduler.slurm.get_env_vars", return_value={}),
            mock.patch(
                "areal.infra.scheduler.slurm.get_thread_env_vars", return_value={}
            ),
        ):
            a = SlurmScheduler._prepare_worker_specs(scheduler, "actor", 1, [actor])
            r = SlurmScheduler._prepare_worker_specs(scheduler, "rollout", 1, [rollout])

        assert a[0].env_vars["PYTORCH_CUDA_ALLOC_CONF"] == "a:1"
        assert "PYTORCH_CUDA_ALLOC_CONF" not in r[0].env_vars


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
