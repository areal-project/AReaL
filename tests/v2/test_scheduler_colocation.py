# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from areal.api import Job, Worker
from areal.api.cli_args import (
    SchedulingSpec,
    SchedulingStrategy,
    SchedulingStrategyType,
    is_colocation_strategy,
)
from areal.infra.scheduler.colocation import is_v2_training_guard_colocation
from areal.infra.scheduler.exceptions import WorkerCreationError
from areal.infra.scheduler.local import LocalScheduler, WorkerInfo
from areal.infra.scheduler.ray import RayScheduler, RayWorkerInfo
from areal.infra.scheduler.slurm import SlurmScheduler, SlurmWorkerInfo
from areal.infra.utils.concurrent import run_async_task


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


def test_v2_training_guard_colocation_requires_training_guard_module():
    strategy = SchedulingStrategy(type="colocation", target="rollout")

    assert is_v2_training_guard_colocation(
        "actor-guard",
        strategy,
        "python -m areal.v2.training_service.guard",
    )
    assert not is_v2_training_guard_colocation(
        "data-guard",
        strategy,
        "python -m areal.infra.data_service.guard",
    )
    assert not is_v2_training_guard_colocation(
        "actor-guard", strategy, "python worker.py"
    )


def _slurm_worker(worker_id, ip, gpu):
    return SlurmWorkerInfo(
        worker=Worker(id=worker_id, ip=ip, worker_ports=["10000"]),
        role="rollout-inf",
        slurm_job_id=11,
        task_index=int(worker_id.rsplit("/", 1)[1]),
        spec=SchedulingSpec(cpu=1, gpu=gpu, mem=1),
    )


def _local_worker(worker_id, gpu_devices):
    return WorkerInfo(
        worker=Worker(id=worker_id, ip="127.0.0.1", worker_ports=["10000"]),
        process=None,
        role="rollout-inf",
        gpu_devices=gpu_devices,
        created_at=0.0,
        log_file="rollout-inf.log",
    )


def test_local_v2_guard_colocation_expands_inference_group_per_gpu(tmp_path):
    scheduler = object.__new__(LocalScheduler)
    parent = _local_worker("rollout-inf/0", [4, 5])
    scheduler.gpu_devices = [4, 5]
    scheduler._workers = {"rollout-inf": [parent]}
    scheduler._colocated_roles = {}
    scheduler._v2_guard_parents = {}
    scheduler._allocated_ports = set()
    scheduler.enable_tms_offload = False
    scheduler.experiment_name = "exp"
    scheduler.trial_name = "trial"
    scheduler.fileroot = str(tmp_path)
    scheduler.get_workers = MagicMock(return_value=[parent.worker])

    job = Job(
        role="actor-guard",
        replicas=2,
        tasks=[
            SchedulingSpec(
                cpu=1,
                gpu=1,
                mem=1,
                cmd="python -m areal.v2.training_service.guard",
                env_vars={"KEEP": "1"},
            )
        ],
        scheduling_strategy=SchedulingStrategy(type="colocation", target="rollout"),
    )

    with patch(
        "areal.infra.scheduler.local.run_async_task",
        return_value=["actor-guard/0", "actor-guard/1"],
    ) as run_async:
        worker_ids = scheduler.create_workers(job)

    assert worker_ids == ["actor-guard/0", "actor-guard/1"]
    assert run_async.call_args.args[3] == [parent, parent]
    assert run_async.call_args.args[4] == "areal.v2.training_service.guard"
    fork_envs = run_async.call_args.args[5]
    assert [env["CUDA_VISIBLE_DEVICES"] for env in fork_envs] == ["4", "5"]
    assert all(env["KEEP"] == "1" for env in fork_envs)
    assert all(env["TMS_INIT_ENABLE"] == "0" for env in fork_envs)
    assert run_async.call_args.args[6] == [[4], [5]]
    assert scheduler._v2_guard_parents["actor-guard"] == [parent, parent]


def test_local_v2_guard_cleanup_uses_saved_parent_after_target_deleted():
    scheduler = object.__new__(LocalScheduler)
    parent = _local_worker("rollout-inf/0", [0, 1])
    child = _local_worker("actor-guard/0", [0])
    child.role = "actor-guard"
    scheduler._workers = {}

    with patch.object(scheduler, "_kill_forked_worker", new=AsyncMock()) as kill_worker:
        run_async_task(
            scheduler._cleanup_forked_workers_async,
            "actor-guard",
            "rollout-inf",
            [child],
            [parent],
        )

    kill_worker.assert_awaited_once()
    assert kill_worker.await_args.args[3] is parent


def test_slurm_v2_guard_colocation_forks_inside_target_allocation(tmp_path):
    scheduler = object.__new__(SlurmScheduler)
    scheduler._n_gpus_per_node = 8
    scheduler._workers = {
        "rollout-inf": [
            _slurm_worker("rollout-inf/0", "192.0.2.1", gpu=4),
            _slurm_worker("rollout-inf/1", "192.0.2.2", gpu=4),
            _slurm_worker("rollout-inf/2", "192.0.2.1", gpu=4),
            _slurm_worker("rollout-inf/3", "192.0.2.2", gpu=4),
        ]
    }
    scheduler._jobs = {"rollout-inf": 11}
    scheduler._colocated_roles = {}
    scheduler._v2_guard_parents = {}
    scheduler.enable_tms_offload = False
    scheduler.experiment_name = "exp"
    scheduler.trial_name = "trial"
    scheduler.fileroot = str(tmp_path)
    scheduler.get_workers = MagicMock(
        return_value=[worker.worker for worker in scheduler._workers["rollout-inf"]]
    )

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
    assert [worker.worker.id for worker in parent_workers] == [
        *("rollout-inf/0" for _ in range(4)),
        *("rollout-inf/1" for _ in range(4)),
        *("rollout-inf/2" for _ in range(4)),
        *("rollout-inf/3" for _ in range(4)),
    ]
    assert run_async.call_args.args[4] == "areal.v2.training_service.guard"
    fork_envs = run_async.call_args.args[5]
    assert [env["CUDA_VISIBLE_DEVICES"] for env in fork_envs] == [
        *map(str, range(4)),
        *map(str, range(4)),
        *map(str, range(4, 8)),
        *map(str, range(4, 8)),
    ]
    assert all(env["TMS_INIT_ENABLE"] == "0" for env in fork_envs)
    assert all(env["LD_PRELOAD"] == "" for env in fork_envs)


def test_slurm_v2_guard_colocation_requires_full_node_topology(tmp_path):
    scheduler = object.__new__(SlurmScheduler)
    scheduler._n_gpus_per_node = 8
    scheduler._workers = {
        "rollout-inf": [_slurm_worker("rollout-inf/0", "192.0.2.1", gpu=4)]
    }
    scheduler._colocated_roles = {}
    scheduler._v2_guard_parents = {}
    scheduler.enable_tms_offload = False
    scheduler.get_workers = MagicMock(
        return_value=[worker.worker for worker in scheduler._workers["rollout-inf"]]
    )

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


def _ray_worker(worker_id, ip, gpu, gpu_devices):
    return RayWorkerInfo(
        worker=Worker(id=worker_id, ip=ip, worker_ports=["10000"]),
        role="rollout-inf",
        task_index=int(worker_id.rsplit("/", 1)[1]),
        spec=SchedulingSpec(cpu=1, gpu=gpu, mem=1),
        gpu_devices=tuple(map(str, gpu_devices)),
    )


@pytest.mark.parametrize(
    "worker_order",
    [
        (0, 1, 2, 3),
        (3, 1, 2, 0),
        (2, 0, 3, 1),
    ],
)
def test_ray_v2_guard_colocation_uses_canonical_physical_gpu_order(
    tmp_path, worker_order
):
    scheduler = object.__new__(RayScheduler)
    scheduler._n_gpus_per_node = 8
    target_workers = [
        _ray_worker("rollout-inf/0", "192.0.2.1", 4, range(4)),
        _ray_worker("rollout-inf/1", "192.0.2.1", 4, range(4, 8)),
        _ray_worker("rollout-inf/2", "192.0.2.2", 4, range(4)),
        _ray_worker("rollout-inf/3", "192.0.2.2", 4, range(4, 8)),
    ]
    scheduler._workers = {
        "rollout-inf": [target_workers[index] for index in worker_order]
    }
    scheduler._colocated_roles = {}
    scheduler._v2_guard_parents = {}
    scheduler.enable_tms_offload = False
    scheduler.startup_timeout = 30.0
    scheduler.health_check_interval = 0.01

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
        patch.object(scheduler, "get_workers") as get_workers,
        patch(
            "areal.infra.scheduler.ray.run_async_task",
            return_value=[f"actor-guard/{rank}" for rank in range(16)],
        ) as run_async,
    ):
        worker_ids = scheduler.create_workers(job)

    assert len(worker_ids) == 16
    get_workers.assert_called_once_with(role="rollout-inf", timeout=30.0)
    parent_workers = run_async.call_args.args[3]
    assert [worker.worker.id for worker in parent_workers] == [
        *("rollout-inf/0" for _ in range(4)),
        *("rollout-inf/2" for _ in range(4)),
        *("rollout-inf/1" for _ in range(4)),
        *("rollout-inf/3" for _ in range(4)),
    ]
    assert scheduler._v2_guard_parents["actor-guard"] == parent_workers
    assert run_async.call_args.args[4] == "areal.v2.training_service.guard"
    fork_envs = run_async.call_args.args[5]
    assert [env["CUDA_VISIBLE_DEVICES"] for env in fork_envs] == [
        *(str(rank) for rank in range(4)),
        *(str(rank) for rank in range(4)),
        *(str(rank) for rank in range(4, 8)),
        *(str(rank) for rank in range(4, 8)),
    ]
    assert all(env["LD_PRELOAD"] == "" for env in fork_envs)
    assert all(env["TMS_INIT_ENABLE"] == "0" for env in fork_envs)
    assert all(env["TMS_INIT_ENABLE_CPU_BACKUP"] == "0" for env in fork_envs)


def test_ray_v2_guard_colocation_requires_full_node_topology(tmp_path):
    scheduler = object.__new__(RayScheduler)
    scheduler._n_gpus_per_node = 8
    scheduler._workers = {
        "rollout-inf": [_ray_worker("rollout-inf/0", "192.0.2.1", 4, range(4))]
    }
    scheduler._colocated_roles = {}
    scheduler._v2_guard_parents = {}
    scheduler.enable_tms_offload = False
    scheduler.startup_timeout = 30.0
    scheduler.health_check_interval = 0.01

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

    with (
        patch.object(scheduler, "get_workers"),
        pytest.raises(WorkerCreationError, match="full-node target allocation"),
    ):
        scheduler.create_workers(job)


@pytest.mark.parametrize(
    ("target_workers", "error"),
    [
        (
            [_ray_worker("rollout-inf/0", "192.0.2.1", 4, [])],
            "reports 0 assigned GPUs but declares 4",
        ),
        (
            [
                _ray_worker("rollout-inf/0", "192.0.2.1", 4, range(4)),
                _ray_worker("rollout-inf/1", "192.0.2.1", 4, range(4)),
            ],
            "belongs to multiple groups",
        ),
        (
            [
                _ray_worker("rollout-inf/0", "192.0.2.1", 4, [0, 1, 3, 4]),
                _ray_worker("rollout-inf/1", "192.0.2.1", 4, range(4, 8)),
            ],
            "must be contiguous in local-rank order",
        ),
    ],
)
def test_ray_v2_guard_colocation_rejects_ambiguous_gpu_metadata(target_workers, error):
    scheduler = object.__new__(RayScheduler)
    scheduler._n_gpus_per_node = 8
    scheduler._workers = {"rollout-inf": target_workers}
    scheduler._colocated_roles = {}
    scheduler._v2_guard_parents = {}
    scheduler.enable_tms_offload = False
    scheduler.startup_timeout = 30.0
    scheduler.health_check_interval = 0.01
    job = Job(
        role="actor-guard",
        replicas=8,
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
        patch.object(scheduler, "get_workers"),
        pytest.raises(WorkerCreationError, match=error),
    ):
        scheduler.create_workers(job)


def test_ray_v2_guard_colocation_cleanup_uses_expanded_parent_mapping():
    scheduler = object.__new__(RayScheduler)
    target_workers = [
        _ray_worker("rollout-inf/0", "192.0.2.1", 4, range(4)),
        _ray_worker("rollout-inf/1", "192.0.2.1", 4, range(4, 8)),
    ]
    parent_workers = [
        *(target_workers[0] for _ in range(4)),
        *(target_workers[1] for _ in range(4)),
    ]
    actor_workers = [
        RayWorkerInfo(
            worker=Worker(
                id=f"actor-guard/{rank}",
                ip="192.0.2.1",
                worker_ports=[str(11000 + rank)],
            ),
            role="actor-guard",
            task_index=rank,
        )
        for rank in range(8)
    ]
    scheduler._workers = {
        "rollout-inf": target_workers,
        "actor-guard": actor_workers,
    }
    scheduler._colocated_roles = {"actor-guard": "rollout-inf"}
    scheduler._v2_guard_parents = {"actor-guard": parent_workers}

    with patch("areal.infra.scheduler.ray.run_async_task") as run_async:
        scheduler.delete_workers("actor-guard")

    cleanup_calls = [
        call
        for call in run_async.call_args_list
        if len(call.args) > 1 and call.args[1] == "actor-guard"
    ]
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0].args[1:] == (
        "actor-guard",
        "rollout-inf",
        actor_workers,
        parent_workers,
    )
    assert "actor-guard" not in scheduler._workers
    assert "actor-guard" not in scheduler._colocated_roles
    assert "actor-guard" not in scheduler._v2_guard_parents


def test_ray_fork_partial_failure_cleans_children_with_parent_mapping():
    scheduler = object.__new__(RayScheduler)
    scheduler.exp_config = None
    parent_workers = [
        _ray_worker("rollout-inf/0", "192.0.2.1", 4, range(4)),
        _ray_worker("rollout-inf/0", "192.0.2.1", 4, range(4)),
        _ray_worker("rollout-inf/1", "192.0.2.1", 4, range(4, 8)),
    ]

    async def fork_one(_session, role, idx, *_args, **_kwargs):
        if idx == 1:
            raise RuntimeError("fork failed")
        return RayWorkerInfo(
            worker=Worker(
                id=f"{role}/{idx}",
                ip="192.0.2.1",
                worker_ports=[str(12000 + idx)],
            ),
            role=role,
            task_index=idx,
        )

    cleanup = AsyncMock()
    with (
        patch.object(scheduler, "_fork_single_worker", side_effect=fork_one),
        patch.object(scheduler, "_cleanup_forked_workers_async", cleanup),
        pytest.raises(WorkerCreationError, match="Failed to fork 1 out of 3 workers"),
    ):
        asyncio.run(
            scheduler._create_forked_workers_async(
                "actor-guard",
                "rollout-inf",
                parent_workers,
                "areal.v2.training_service.guard",
                [{}, {}, {}],
            )
        )

    cleanup.assert_awaited_once()
    assert cleanup.await_args.args[0:2] == ("actor-guard", "rollout-inf")
    assert [worker.worker.id for worker in cleanup.await_args.args[2]] == [
        "actor-guard/0",
        "actor-guard/2",
    ]
    assert cleanup.await_args.args[3] == parent_workers
