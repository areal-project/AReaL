import asyncio
import time
from unittest.mock import Mock, patch

import pytest
from ray.util.state import summarize_actors

from areal.api import Job, Worker
from areal.api.cli_args import BaseExperimentConfig, SchedulingSpec, SchedulingStrategy
from areal.infra.scheduler.ray import RayScheduler, RayWorkerInfo, ray_resource_type
from areal.infra.utils.ray_placement_group import _create_bundle_specs_split

pytestmark = pytest.mark.skip(
    reason=(
        "Ray scheduler tests will only run if ray environment is explicitly initialized\n"
        "To run this test:\n"
        "1. Set up the ray cluster with `ray start --head` and ensure that there are enough resources;\n"
        "2. Comment this skip mark."
    ),
)


class TestRaySchedulerInitialization:
    def test_init(self):
        scheduler = RayScheduler(
            startup_timeout=60.0, exp_config=BaseExperimentConfig()
        )
        assert scheduler.startup_timeout == 60.0


class TestWorkerCreationAndDeletion:
    def test_create_delete_workers(self):
        config = BaseExperimentConfig()

        scheduler = RayScheduler(startup_timeout=60.0, exp_config=config)

        job = Job(
            replicas=2,
            role="train",
            tasks=[
                SchedulingSpec(cpu=1, mem=1, gpu=1, ray_placement_strategy="shared"),
                SchedulingSpec(cpu=1, mem=1, gpu=1, ray_placement_strategy="shared"),
            ],
        )

        # create workers
        worker_ids = scheduler.create_workers(job)
        assert len(worker_ids) == 2
        assert len(scheduler._workers["train"]) == 2

        actor_summary = summarize_actors()

        assert (
            actor_summary["cluster"]["summary"]["RayRPCServer"]["state_counts"]["ALIVE"]
            == 2
        )
        assert len(scheduler.get_workers("train")) == 2

        scheduler._ping_workers("train")

        # delete workers
        scheduler.delete_workers()
        assert len(scheduler._workers["train"]) == 0

        actor_summary = summarize_actors()
        assert (
            actor_summary["cluster"]["summary"]["RayRPCServer"]["state_counts"]["DEAD"]
            == 2
        )


class TestWorkerCallEngine:
    def test_create_call_engine(self):
        # to simulate an awaitable None
        async def async_none(*args, **kwargs):
            return None

        def sync_none(*args, **kwargs):
            return None

        config = BaseExperimentConfig()

        scheduler = RayScheduler(startup_timeout=60.0, exp_config=config)
        ray_actor_handle = Mock()
        ray_actor_handle.create_engine.remote = async_none

        worker = RayWorkerInfo(
            worker=Worker(id="test/0", ip="0.0.0.0"),
            actor=ray_actor_handle,
            role="test",
            placement_group=None,
            bundle_index=0,
            created_at=0,
        )

        scheduler._workers["test"] = [worker]
        scheduler._worker_info_by_id[worker.worker.id] = worker

        # create engine
        result = asyncio.run(
            scheduler.create_engine(
                worker.worker.id, "test_engines.DummyEngine", name="TestEngine"
            )
        )
        assert result is None

        # sync
        ray_actor_handle.call.remote = sync_none
        with patch("areal.infra.scheduler.ray.ray.get", return_value=None):
            result = scheduler.call_engine(
                worker.worker.id, "test_fn", engine_name="test/0"
            )
        assert result is None

        # async
        ray_actor_handle.call.remote = async_none
        result = asyncio.run(
            scheduler.async_call_engine(
                worker.worker.id, "test_fn", engine_name="test/0"
            )
        )
        assert result is None


class TestUtilityFunctions:
    def test_utilities(self):
        _num_gpu_per_node = 16
        config = BaseExperimentConfig()

        config.cluster.n_gpus_per_node = _num_gpu_per_node

        scheduler = RayScheduler(startup_timeout=60.0, exp_config=config)

        schedulings = [
            SchedulingSpec(
                cpu=1,
                mem=1,
                gpu=1,
            ),
            SchedulingSpec(
                cpu=1,
                mem=1,
                gpu=1,
            ),
        ]

        new_schedulings = scheduler._prepare_worker_specs("train", 2, schedulings)
        assert len(new_schedulings) == 2
        for spec in new_schedulings:
            assert spec.cpu == 1
            assert spec.mem == 1
            assert spec.gpu == 1

        # case where only 1 spec is passed but multiple workers
        new_schedulings = scheduler._prepare_worker_specs("train", 2, schedulings[0:])
        assert len(new_schedulings) == 2
        for spec in new_schedulings:
            assert spec.cpu == 1
            assert spec.mem == 1
            assert spec.gpu == 1

        bundle_list = _create_bundle_specs_split(16, 1, 24, 1024)
        assert len(bundle_list) == 2
        for bundle in bundle_list:
            assert bundle[ray_resource_type()] <= _num_gpu_per_node


class TestForkColocation:
    """Tests for fork-based colocation in RayScheduler."""

    def test_fork_creates_workers_on_same_placement_group(self):
        """Test that forked workers are created on the same placement group."""
        config = BaseExperimentConfig()

        scheduler = RayScheduler(startup_timeout=60.0, exp_config=config)

        # First create target workers (each gets its own PG with bundle_index=0)
        actor_job = Job(
            replicas=2,
            role="actor",
            tasks=[
                SchedulingSpec(cpu=1, mem=1, gpu=1),
                SchedulingSpec(cpu=1, mem=1, gpu=1),
            ],
        )

        # Create actor workers
        actor_worker_ids = scheduler.create_workers(actor_job)
        assert len(actor_worker_ids) == 2
        assert len(scheduler._workers["actor"]) == 2

        # Create forked ref workers
        ref_job = Job(
            replicas=2,
            role="ref",
            tasks=[SchedulingSpec(cpu=1, mem=1, gpu=1)],
            scheduling_strategy=SchedulingStrategy(
                type="colocation", target="actor", fork=True
            ),
        )

        ref_worker_ids = scheduler.create_workers(ref_job)

        # Verify forked workers were created
        assert len(ref_worker_ids) == 2
        assert "ref" in scheduler._workers
        assert len(scheduler._workers["ref"]) == 2

        # Verify forked workers use same placement groups
        for i in range(2):
            actor_pg = scheduler._workers["actor"][i].placement_group
            ref_pg = scheduler._workers["ref"][i].placement_group
            assert actor_pg == ref_pg, "Forked worker should use same placement group"
            # Verify both have the same bundle index as their parent
            assert (
                scheduler._workers["actor"][i].bundle_index
                == scheduler._workers["ref"][i].bundle_index
            )

        # Verify ref role is tracked as colocated
        assert "ref" in scheduler._colocated_roles
        assert scheduler._colocated_roles["ref"] == "actor"

        # Verify actors summary shows 4 actors (2 actor + 2 ref)
        actor_summary = summarize_actors()
        assert (
            actor_summary["cluster"]["summary"]["RayRPCServer"]["state_counts"]["ALIVE"]
            == 4
        )

        # Clean up
        scheduler.delete_workers()

    def test_fork_get_workers_returns_forked_workers(self):
        """Test that get_workers returns forked workers directly."""
        config = BaseExperimentConfig()
        scheduler = RayScheduler(startup_timeout=60.0, exp_config=config)

        # Create target workers
        actor_job = Job(
            replicas=2,
            role="actor",
            tasks=[SchedulingSpec(cpu=1, mem=1, gpu=1)],
        )
        scheduler.create_workers(actor_job)

        # Create forked workers
        ref_job = Job(
            replicas=2,
            role="ref",
            tasks=[SchedulingSpec(cpu=1, mem=1, gpu=1)],
            scheduling_strategy=SchedulingStrategy(
                type="colocation", target="actor", fork=True
            ),
        )
        scheduler.create_workers(ref_job)

        # Get workers for ref role should return ref workers, not actor workers
        ref_workers = scheduler.get_workers("ref")
        assert len(ref_workers) == 2
        assert all(w.id.startswith("ref/") for w in ref_workers)

        # Clean up
        scheduler.delete_workers()

    def test_fork_delete_cleans_up_forked_actors(self):
        """Test that deleting forked role cleans up its actors."""
        config = BaseExperimentConfig()
        scheduler = RayScheduler(startup_timeout=60.0, exp_config=config)

        # Create target workers
        actor_job = Job(
            replicas=2,
            role="actor",
            tasks=[SchedulingSpec(cpu=1, mem=1, gpu=1)],
        )
        scheduler.create_workers(actor_job)

        # Create forked workers
        ref_job = Job(
            replicas=2,
            role="ref",
            tasks=[SchedulingSpec(cpu=1, mem=1, gpu=1)],
            scheduling_strategy=SchedulingStrategy(
                type="colocation", target="actor", fork=True
            ),
        )
        scheduler.create_workers(ref_job)

        # Verify we have 4 actors total
        actor_summary = summarize_actors()
        assert (
            actor_summary["cluster"]["summary"]["RayRPCServer"]["state_counts"]["ALIVE"]
            == 4
        )

        # Delete only the ref role
        scheduler.delete_workers("ref")

        # Verify ref role is cleaned up
        assert "ref" not in scheduler._workers
        assert "ref" not in scheduler._colocated_roles

        # Verify actor workers still exist
        assert "actor" in scheduler._workers
        assert len(scheduler._workers["actor"]) == 2

        # Verify only 2 actors remain alive
        # Wait for actors to be terminated (actor.destroy.remote() is async)

        deadline = time.time() + 10
        while time.time() < deadline:
            actor_summary = summarize_actors()
            alive_count = (
                actor_summary["cluster"]["summary"]
                .get("RayRPCServer", {})
                .get("state_counts", {})
                .get("ALIVE", 0)
            )
            if alive_count == 2:
                break
            time.sleep(1)

        actor_summary = summarize_actors()
        assert (
            actor_summary["cluster"]["summary"]["RayRPCServer"]["state_counts"]["ALIVE"]
            == 2
        )

        # Clean up
        scheduler.delete_workers()

    def test_non_fork_colocation_reuses_workers(self):
        """Test that non-fork colocation reuses target workers."""
        config = BaseExperimentConfig()
        scheduler = RayScheduler(startup_timeout=60.0, exp_config=config)

        # Create target workers
        actor_job = Job(
            replicas=2,
            role="actor",
            tasks=[SchedulingSpec(cpu=1, mem=1, gpu=1)],
        )
        scheduler.create_workers(actor_job)

        # Create colocated workers without fork
        ref_job = Job(
            replicas=2,
            role="ref",
            tasks=[SchedulingSpec(cpu=1, mem=1, gpu=1)],
            scheduling_strategy=SchedulingStrategy(
                type="colocation", target="actor", fork=False
            ),
        )
        ref_worker_ids = scheduler.create_workers(ref_job)

        # Ref should reuse actor worker IDs
        assert all(w.startswith("actor/") for w in ref_worker_ids)

        # Ref role should NOT have its own workers
        assert "ref" not in scheduler._workers

        # But ref should be tracked as colocated
        assert "ref" in scheduler._colocated_roles

        # get_workers for ref should return actor workers
        ref_workers = scheduler.get_workers("ref")
        assert all(w.id.startswith("actor/") for w in ref_workers)

        # Only 2 actors total (no new actors for ref)
        actor_summary = summarize_actors()
        assert (
            actor_summary["cluster"]["summary"]["RayRPCServer"]["state_counts"]["ALIVE"]
            == 2
        )

        # Clean up
        scheduler.delete_workers()
