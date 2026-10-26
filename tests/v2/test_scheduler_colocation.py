# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from areal.api import Job, Worker
from areal.api.cli_args import (
    SchedulingSpec,
    SchedulingStrategy,
)
from areal.infra.scheduler.colocation import (
    is_v2_training_guard_colocation,
    validate_colocated_device_namespace,
)
from areal.infra.scheduler.exceptions import WorkerCreationError
from areal.infra.scheduler.local import LocalScheduler, WorkerInfo
from areal.infra.scheduler.ray import RayScheduler
from areal.infra.scheduler.slurm import SlurmScheduler, SlurmWorkerInfo
from areal.infra.utils.concurrent import run_async_task


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
        role=worker_id.rsplit("/", 1)[0],
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
    scheduler._allocated_ports = {10000}
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


@pytest.mark.parametrize("node_order", [(0, 1), (1, 0), (0, 1, 2, 3), (3, 1, 0, 2)])
@pytest.mark.parametrize("target", ["rollout", "actor"])
def test_slurm_v2_guard_colocation_forks_inside_target_allocation(
    tmp_path, node_order, target
):
    parent_role = "rollout-inf" if target == "rollout" else "actor-guard"
    child_role = "actor-guard" if target == "rollout" else "ref-guard"
    parent_gpus = 4 if target == "rollout" else 1
    groups_per_node = 8 // parent_gpus
    num_nodes = len(node_order)
    world_size = num_nodes * 8
    scheduler = object.__new__(SlurmScheduler)
    scheduler._n_gpus_per_node = 8
    scheduler._workers = {
        parent_role: [
            _slurm_worker(
                f"{parent_role}/{group * num_nodes + node}",
                f"192.0.2.{node + 1}",
                gpu=parent_gpus,
            )
            for group in range(groups_per_node)
            for node in node_order
        ]
    }
    scheduler._jobs = {parent_role: 11}
    scheduler._colocated_roles = {}
    scheduler._v2_guard_parents = {}
    scheduler.enable_tms_offload = False
    scheduler.experiment_name = "exp"
    scheduler.trial_name = "trial"
    scheduler.fileroot = str(tmp_path)
    scheduler.get_workers = MagicMock(
        return_value=[worker.worker for worker in scheduler._workers[parent_role]]
    )

    job = Job(
        role=child_role,
        replicas=world_size,
        tasks=[
            SchedulingSpec(
                cpu=1,
                gpu=1,
                mem=1,
                cmd="python -m areal.v2.training_service.guard",
            )
        ],
        scheduling_strategy=SchedulingStrategy(type="colocation", target=target),
    )

    health_responses = [
        MagicMock(
            json=lambda base=base: {
                "cuda_visible_devices": ",".join(
                    str(i) for i in range(base, base + parent_gpus)
                )
            }
        )
        for base in range(0, 8, parent_gpus)
        for _ in node_order
    ]
    with (
        patch("areal.infra.scheduler.slurm.requests.get", side_effect=health_responses),
        patch(
            "areal.infra.scheduler.slurm.run_async_task",
            return_value=[f"{child_role}/{rank}" for rank in range(world_size)],
        ) as run_async,
    ):
        worker_ids = scheduler.create_workers(job)

    assert len(worker_ids) == world_size
    parent_workers = run_async.call_args.args[3]
    assert [worker.worker.id for worker in parent_workers] == [
        f"{parent_role}/{group * num_nodes + node}"
        for node in range(num_nodes)
        for group in range(groups_per_node)
        for _ in range(parent_gpus)
    ]
    assert run_async.call_args.args[4] == "areal.v2.training_service.guard"
    fork_envs = run_async.call_args.args[5]
    assert [env["CUDA_VISIBLE_DEVICES"] for env in fork_envs] == [
        str(device) for _ in range(num_nodes) for device in range(8)
    ]
    slots = [
        (parent.worker.ip, int(env["CUDA_VISIBLE_DEVICES"]))
        for parent, env in zip(parent_workers, fork_envs, strict=True)
    ]
    assert len(set(slots)) == world_size
    for node in range(num_nodes):
        assert slots[8 * node : 8 * node + 8] == [
            (f"192.0.2.{node + 1}", device) for device in range(8)
        ]
        # DP=num_nodes, PP=1, TP=2, CP=4: rank = 8*dp + 2*cp + tp.
        for cp in range(4):
            assert len({slots[8 * node + 2 * cp + tp][0] for tp in range(2)}) == 1
        for tp in range(2):
            assert len({slots[8 * node + 2 * cp + tp][0] for cp in range(4)}) == 1
    assert scheduler._v2_guard_parents[child_role] == parent_workers
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

    with (
        patch(
            "areal.infra.scheduler.slurm.requests.get",
            return_value=MagicMock(json=lambda: {"cuda_visible_devices": "0,1,2,3"}),
        ),
        pytest.raises(WorkerCreationError, match="full-node target allocation"),
    ):
        scheduler.create_workers(job)


def test_ray_v2_guard_colocation_fails_clearly(tmp_path):
    scheduler = object.__new__(RayScheduler)
    scheduler._workers = {}
    scheduler.enable_tms_offload = False

    job = Job(
        role="actor-guard",
        replicas=1,
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

    with pytest.raises(
        NotImplementedError,
        match="ray colocation for v2 guard jobs is not yet supported",
    ):
        scheduler.create_workers(job)


@pytest.mark.parametrize(
    "visible", [None, "4,5,6,7", "3,2,1,0", "GPU-a,GPU-b,GPU-c,GPU-d"]
)
def test_colocated_namespace_rejects_unsupported_allocation(visible):
    with pytest.raises(ValueError, match="contiguous numeric"):
        validate_colocated_device_namespace(visible, base=0, count=4)


@pytest.mark.parametrize("service", ["inference_service", "training_service"])
def test_guard_reports_only_visible_device_namespace(monkeypatch, service):
    import importlib

    app = importlib.import_module(f"areal.v2.{service}.guard.app").app

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    response = app.test_client().get("/health")

    assert response.status_code == 200
    assert response.get_json()["cuda_visible_devices"] == "4,5,6,7"
