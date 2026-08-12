# SPDX-License-Identifier: Apache-2.0
"""Tests for reservation/exclusive sbatch options and env-var precedence."""

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
