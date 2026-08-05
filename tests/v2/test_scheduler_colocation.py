# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from areal.api import Job, Worker
from areal.api.cli_args import (
    SchedulingSpec,
    SchedulingStrategy,
    SchedulingStrategyType,
    is_colocation_strategy,
)
from areal.infra.scheduler.exceptions import WorkerCreationError
from areal.infra.scheduler.ray import RayScheduler
from areal.infra.scheduler.slurm import SlurmScheduler, SlurmWorkerInfo


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (None, False),
        (SchedulingStrategy(), False),
        (SchedulingStrategy(type="colocation", target="rollout"), True),
        (
            SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="rollout"
            ),
            True,
        ),
    ],
)
def test_is_colocation_strategy(strategy, expected):
    assert is_colocation_strategy(strategy) is expected


def _slurm_worker(worker_id, ip, gpu):
    return SlurmWorkerInfo(
        worker=Worker(id=worker_id, ip=ip, worker_ports=["10000"]),
        role="rollout-inf",
        slurm_job_id=11,
        task_index=int(worker_id.rsplit("/", 1)[1]),
        spec=SchedulingSpec(cpu=1, gpu=gpu, mem=1),
    )


def test_slurm_v2_guard_colocation_forks_inside_target_allocation(tmp_path):
    scheduler = object.__new__(SlurmScheduler)
    scheduler._n_gpus_per_node = 8
    scheduler._workers = {
        "rollout-inf": [
            _slurm_worker("rollout-inf/0", "10.0.0.1", gpu=8),
            _slurm_worker("rollout-inf/1", "10.0.0.2", gpu=8),
        ]
    }
    scheduler._jobs = {"rollout-inf": 11}
    scheduler._colocated_roles = {}
    scheduler._v2_guard_parents = {}
    scheduler.enable_tms_offload = False
    scheduler.experiment_name = "exp"
    scheduler.trial_name = "trial"
    scheduler.fileroot = str(tmp_path)

    job = Job(
        role="actor-guard",
        replicas=16,
        tasks=[
            SchedulingSpec(
                cpu=1,
                gpu=1,
                mem=1,
                cmd="python -m areal.v2.training_service.guard",
            )
        ],
        scheduling_strategy=SchedulingStrategy(type="colocation", target="rollout"),
    )

    with (
        patch(
            "areal.infra.scheduler.slurm.run_async_task",
            return_value=[f"actor-guard/{rank}" for rank in range(16)],
        ) as run_async,
    ):
        worker_ids = scheduler.create_workers(job)

    assert len(worker_ids) == 16
    parent_workers = run_async.call_args.args[3]
    assert [worker.worker.ip for worker in parent_workers] == [
        *("10.0.0.1" for _ in range(8)),
        *("10.0.0.2" for _ in range(8)),
    ]
    assert run_async.call_args.args[4] == "areal.v2.training_service.guard"
    assert run_async.call_args.args[5] == [job.tasks[0].env_vars] * 16


def test_slurm_v2_guard_colocation_requires_full_node_topology(tmp_path):
    scheduler = object.__new__(SlurmScheduler)
    scheduler._n_gpus_per_node = 8
    scheduler._workers = {
        "rollout-inf": [_slurm_worker("rollout-inf/0", "10.0.0.1", gpu=4)]
    }
    scheduler._colocated_roles = {}
    scheduler._v2_guard_parents = {}
    scheduler.enable_tms_offload = False

    job = Job(
        role="actor-guard",
        replicas=4,
        tasks=[
            SchedulingSpec(
                cpu=1,
                gpu=1,
                mem=1,
                cmd="python -m areal.v2.training_service.guard",
            )
        ],
        scheduling_strategy=SchedulingStrategy(type="colocation", target="rollout"),
    )

    with pytest.raises(WorkerCreationError, match="full-node target allocation"):
        scheduler.create_workers(job)


def test_ray_v2_guard_colocation_fails_clearly(tmp_path):
    scheduler = object.__new__(RayScheduler)
    scheduler._workers = {}
    scheduler.enable_tms_offload = False

    job = Job(
        role="actor-guard",
        replicas=1,
        tasks=[SchedulingSpec(cpu=1, gpu=1, mem=1)],
        scheduling_strategy=SchedulingStrategy(type="colocation", target="rollout"),
    )

    with pytest.raises(
        NotImplementedError,
        match="ray colocation for v2 guard jobs is not yet supported",
    ):
        scheduler.create_workers(job)
