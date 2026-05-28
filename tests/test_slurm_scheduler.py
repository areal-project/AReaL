import os
import shutil
import time
from pathlib import Path

import pytest

from areal.api import Job
from areal.api.cli_args import (
    BaseExperimentConfig,
    SchedulingSpec,
    SchedulingStrategy,
    SchedulingStrategyType,
)
from areal.infra.scheduler.exceptions import (
    EngineCreationError,
    WorkerCreationError,
    WorkerNotFoundError,
    WorkerTimeoutError,
)
from areal.infra.scheduler.slurm import SlurmScheduler

# Check if 'srun' exists in the system PATH
if shutil.which("srun") is None:
    pytest.skip(
        "srun executable not found, skipping integration tests", allow_module_level=True
    )

ROOT_NFS_PATH = Path("/storage/openpsi/")
# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def scheduler(tmp_path):
    """Create SlurmScheduler for testing."""
    config = BaseExperimentConfig(
        experiment_name="test_slurm_scheduler",
        trial_name=f"test_{int(time.time())}",
    )
    config.cluster.n_gpus_per_node = 8
    config.cluster.fileroot = str(ROOT_NFS_PATH / str(tmp_path).lstrip("/"))
    name_resolve_root = os.path.join(config.cluster.fileroot, "name_resolve")
    config.cluster.name_resolve.nfs_record_root = name_resolve_root

    scheduler = SlurmScheduler(
        n_gpus_per_node=8,
        experiment_name=config.experiment_name,
        trial_name=config.trial_name,
        startup_timeout=300.0,
        exp_config=config,
    )

    yield scheduler

    # Cleanup: delete all workers
    try:
        scheduler.delete_workers()
    except Exception as e:
        print(f"Cleanup warning: {e}")


@pytest.fixture
def simple_job():
    """Simple job with 2 workers."""
    return Job(
        role="test_worker",
        replicas=2,
        tasks=[
            SchedulingSpec(
                cpu=2,
                gpu=4,
                mem=4,  # 4 GB
                port_count=2,
            )
        ],
    )


# ============================================================================
# Basic Worker Lifecycle Tests
# ============================================================================


def test_create_and_get_workers(scheduler, simple_job):
    """Test basic worker creation and discovery."""
    # Create workers
    worker_ids = scheduler.create_workers(simple_job)
    assert len(worker_ids) == 2
    assert worker_ids[0] == "test_worker/0"
    assert worker_ids[1] == "test_worker/1"

    # Get workers (wait for startup)
    workers = scheduler.get_workers("test_worker", timeout=180)
    assert len(workers) == 2

    # Verify worker properties
    for worker in workers:
        assert worker.ip != ""
        assert len(worker.worker_ports) >= 1
        assert worker.worker_ports[0].isdigit()

        worker_info = scheduler._find_worker_by_id(worker.id)
        assert worker_info.discovered is True
        assert worker_info.worker.ip == worker.ip
        assert worker_info.worker.worker_ports == worker.worker_ports
        assert scheduler._is_worker_ready(worker_info) is True

    # First status check
    scheduler._check_job_status("test_worker")

    # Second check should use cache (within 5s TTL)
    start = time.time()
    scheduler._check_job_status("test_worker")
    duration = time.time() - start

    # Should be very fast due to caching
    assert duration < 0.1, "Cached check should be instant"

    # check that creating duplicate workers will fail
    with pytest.raises(WorkerCreationError, match="already exists"):
        scheduler.create_workers(simple_job)

    # Cleanup
    scheduler.delete_workers("test_worker")

    # Verify workers are gone
    with pytest.raises(WorkerNotFoundError):
        scheduler.get_workers("test_worker", timeout=5)

    # Verify internal state cleaned up
    assert "test_worker" not in scheduler._workers
    assert "test_worker" not in scheduler._jobs


def test_worker_timeout(scheduler):
    """Test timeout when workers don't start."""
    job = Job(
        role="timeout_test",
        replicas=1,
        tasks=[SchedulingSpec(cpu=1, gpu=8, mem=1)],
    )

    scheduler.create_workers(job)

    # Use very short timeout to trigger timeout error
    with pytest.raises(WorkerTimeoutError):
        scheduler.get_workers("timeout_test", timeout=1)

    scheduler.delete_workers("timeout_test")


# ============================================================================
# RPC Communication Tests
# ============================================================================


@pytest.mark.asyncio
async def test_set_worker_env(scheduler, simple_job):
    """Test setting environment variables on workers."""
    scheduler.create_workers(simple_job)
    workers = scheduler.get_workers("test_worker", timeout=180)

    # Set environment variables
    env_vars = {
        "TEST_VAR": "test_value",
        "RANK": "0",
        "WORLD_SIZE": "2",
    }

    await scheduler.set_worker_env(workers[0].id, env_vars)

    # Environment is set - we can't directly verify without engine,
    # but no exception means success

    scheduler.delete_workers("test_worker")


@pytest.mark.asyncio
async def test_create_engine_invalid_import(scheduler, simple_job):
    """Test engine creation with invalid import path."""
    scheduler.create_workers(simple_job)
    workers = scheduler.get_workers("test_worker", timeout=180)

    # Try to create engine with invalid path
    with pytest.raises(EngineCreationError):
        await scheduler.create_engine(
            workers[0].id,
            "nonexistent.module.Engine",
        )

    scheduler.delete_workers("test_worker")


@pytest.mark.asyncio
async def test_create_engine_not_string(scheduler, simple_job):
    """Test that engine parameter must be string."""
    scheduler.create_workers(simple_job)
    workers = scheduler.get_workers("test_worker", timeout=180)

    # Try to create engine with non-string
    with pytest.raises(EngineCreationError, match="must be a string"):
        await scheduler.create_engine(workers[0].id, 123)

    scheduler.delete_workers("test_worker")


@pytest.mark.asyncio
async def test_rpc_on_nonexistent_worker(scheduler):
    """Test RPC calls on nonexistent workers fail properly."""
    with pytest.raises(WorkerNotFoundError):
        await scheduler.set_worker_env("nonexistent/0", {"TEST": "value"})

    with pytest.raises(WorkerNotFoundError):
        await scheduler.create_engine("nonexistent/0", "some.Engine")


# ============================================================================
# Job Status Monitoring Tests
# ============================================================================


def test_worker_discovery_timeout(scheduler):
    """Test that discovery handles workers that never write markers."""
    # Create job but it will timeout before workers write discovery markers
    job = Job(
        role="discovery_timeout",
        replicas=1,
        tasks=[SchedulingSpec(cpu=1, gpu=8, mem=1)],
    )

    scheduler.create_workers(job)

    # Very short timeout - workers won't have time to start
    with pytest.raises(WorkerTimeoutError):
        scheduler.get_workers("discovery_timeout", timeout=2)

    scheduler.delete_workers("discovery_timeout")


# ============================================================================
# Edge Cases and Error Conditions
# ============================================================================


def test_empty_scheduling_spec_fails(scheduler):
    """Test that job with no scheduling specs fails."""
    job = Job(role="empty_spec", replicas=2, tasks=[])

    with pytest.raises(ValueError, match="No scheduling specs"):
        scheduler.create_workers(job)


def test_mismatched_spec_count_fails(scheduler):
    """Test that mismatched spec count fails."""
    job = Job(
        role="mismatched",
        replicas=3,
        tasks=[
            SchedulingSpec(cpu=1, gpu=8, mem=1),
            SchedulingSpec(cpu=2, gpu=8, mem=2),
        ],
    )

    with pytest.raises(ValueError, match="must be 1 or match"):
        scheduler.create_workers(job)


def test_zero_replicas_fails(scheduler):
    """Test that zero replicas fails."""
    job = Job(
        role="zero_replicas",
        replicas=0,
        tasks=[SchedulingSpec(cpu=1, gpu=1, mem=1)],
    )

    with pytest.raises(WorkerCreationError, match="replicas must be greater than 0"):
        scheduler.create_workers(job)


def test_delete_all_workers(scheduler, simple_job):
    """Test deleting all workers at once."""
    # Create multiple roles
    job1 = Job(role="role1", replicas=1, tasks=[SchedulingSpec(cpu=1, gpu=8, mem=1)])
    job2 = Job(role="role2", replicas=1, tasks=[SchedulingSpec(cpu=1, gpu=8, mem=1)])

    scheduler.create_workers(job1)
    scheduler.create_workers(job2)

    scheduler.get_workers("role1", timeout=180)
    scheduler.get_workers("role2", timeout=180)

    # Delete all workers
    scheduler.delete_workers()

    assert len(scheduler._workers) == 0
    assert len(scheduler._jobs) == 0


# ============================================================================
# Scheduler Configuration Tests
# ============================================================================


def test_scheduler_initialization_from_config(tmp_path):
    """Test scheduler initialization from config."""
    config = BaseExperimentConfig(
        experiment_name="test_init",
        trial_name="trial_001",
    )
    config.cluster.n_gpus_per_node = 8
    config.cluster.fileroot = str(ROOT_NFS_PATH / str(tmp_path).lstrip("/"))
    name_resolve_root = os.path.join(config.cluster.fileroot, "name_resolve")
    config.cluster.name_resolve.nfs_record_root = name_resolve_root

    scheduler = SlurmScheduler(exp_config=config)

    assert scheduler.n_gpus_per_node == 8
    assert scheduler.experiment_name == "test_init"
    assert scheduler.trial_name == "trial_001"
    assert str(tmp_path) in scheduler.fileroot


def test_scheduler_initialization_with_overrides(tmp_path):
    """Test scheduler parameter overrides."""
    config = BaseExperimentConfig(
        experiment_name="test_override",
        trial_name="trial_001",
    )
    config.cluster.n_gpus_per_node = 4

    scheduler = SlurmScheduler(
        n_gpus_per_node=8,  # Override
        experiment_name="custom_exp",  # Override
        trial_name="custom_trial",  # Override
        fileroot=str(ROOT_NFS_PATH / str(tmp_path).lstrip("/")),
        exp_config=config,
    )

    assert scheduler.n_gpus_per_node == 4
    assert scheduler.experiment_name == "test_override"
    assert scheduler.trial_name == "trial_001"


def test_scheduler_no_config_no_gpus_fails():
    """Test that initialization fails without config or n_gpus_per_node."""
    with pytest.raises(
        ValueError, match="experiment_name and trial_name must be provided"
    ):
        SlurmScheduler()


# ============================================================================
# Colocation Tests
# ============================================================================


def test_create_workers_with_colocate_strategy(scheduler):
    """Test colocation reuses existing workers from target role."""
    # Create target role first
    actor_job = Job(
        role="actor",
        replicas=2,
        tasks=[SchedulingSpec(cpu=2, gpu=4, mem=4, port_count=2)],
    )
    scheduler.create_workers(actor_job)
    scheduler.get_workers("actor", timeout=180)

    # Create colocated role
    ref_job = Job(
        role="ref",
        replicas=2,
        tasks=[SchedulingSpec(cpu=2, gpu=4, mem=4, port_count=2)],
        scheduling_strategy=SchedulingStrategy(
            type=SchedulingStrategyType.colocation, target="actor"
        ),
    )
    ref_ids = scheduler.create_workers(ref_job)

    # Verify colocated role with fork=True returns NEW worker IDs (not same as target)
    # Forked workers get their own IDs like "ref/0", "ref/1"
    assert ref_ids == ["ref/0", "ref/1"]

    # Verify colocation tracking is set up correctly
    assert "ref" in scheduler._colocated_roles
    assert scheduler._colocated_roles["ref"] == "actor"
    # Forked workers have their own entries in _workers
    assert "ref" in scheduler._workers

    # Cleanup
    scheduler.delete_workers()


def test_get_workers_for_colocated_role_delegates_to_target(scheduler):
    """Test that get_workers for forked colocated role returns forked workers."""
    # Create target role
    actor_job = Job(
        role="actor",
        replicas=2,
        tasks=[SchedulingSpec(cpu=2, gpu=4, mem=4, port_count=2)],
    )
    scheduler.create_workers(actor_job)
    actor_workers = scheduler.get_workers("actor", timeout=180)

    # Create colocated role (with fork=True by default)
    ref_job = Job(
        role="ref",
        replicas=2,
        tasks=[SchedulingSpec(cpu=2, gpu=4, mem=4, port_count=2)],
        scheduling_strategy=SchedulingStrategy(
            type=SchedulingStrategyType.colocation, target="actor"
        ),
    )
    scheduler.create_workers(ref_job)
    ref_workers = scheduler.get_workers("ref", timeout=60)

    # Forked workers have their own IDs and ports, but same IP (same node)
    assert len(ref_workers) == len(actor_workers)
    for ref_w, actor_w in zip(ref_workers, actor_workers):
        # Forked workers have different IDs (ref/0 vs actor/0)
        assert ref_w.id != actor_w.id
        assert ref_w.id.startswith("ref/")
        # Same IP since they're on the same node
        assert ref_w.ip == actor_w.ip
        # Different ports since they're separate processes
        assert ref_w.worker_ports != actor_w.worker_ports

    # Cleanup
    scheduler.delete_workers()


def test_delete_colocated_role_does_not_kill_processes(scheduler):
    """Test deleting colocated role only removes mapping, not actual workers."""
    # Create target role
    actor_job = Job(
        role="actor",
        replicas=2,
        tasks=[SchedulingSpec(cpu=2, gpu=4, mem=4, port_count=2)],
    )
    scheduler.create_workers(actor_job)
    scheduler.get_workers("actor", timeout=180)

    # Create colocated role
    ref_job = Job(
        role="ref",
        replicas=2,
        tasks=[SchedulingSpec(cpu=2, gpu=4, mem=4, port_count=2)],
        scheduling_strategy=SchedulingStrategy(
            type=SchedulingStrategyType.colocation, target="actor"
        ),
    )
    scheduler.create_workers(ref_job)

    # Delete colocated role
    scheduler.delete_workers("ref")

    # Verify colocated role is removed from tracking
    assert "ref" not in scheduler._colocated_roles
    assert "ref" not in scheduler._workers

    # Verify target role workers are still available
    actor_workers = scheduler.get_workers("actor", timeout=60)
    assert len(actor_workers) == 2

    # Cleanup
    scheduler.delete_workers()


def test_colocation_replica_mismatch_raises_error(scheduler):
    """Test that colocation fails if replica count doesn't match target."""
    # Create target role
    actor_job = Job(
        role="actor",
        replicas=2,
        tasks=[SchedulingSpec(cpu=2, gpu=4, mem=4, port_count=2)],
    )
    scheduler.create_workers(actor_job)
    scheduler.get_workers("actor", timeout=180)

    # Try to create colocated role with different replica count
    ref_job = Job(
        role="ref",
        replicas=3,  # Mismatch!
        tasks=[SchedulingSpec(cpu=2, gpu=4, mem=4, port_count=2)],
        scheduling_strategy=SchedulingStrategy(
            type=SchedulingStrategyType.colocation, target="actor"
        ),
    )

    with pytest.raises(WorkerCreationError, match="Replica count mismatch"):
        scheduler.create_workers(ref_job)

    # Cleanup
    scheduler.delete_workers()


def test_colocation_target_not_found_raises_error(scheduler):
    """Test that colocation fails if target role doesn't exist."""
    # Create colocated role without target
    ref_job = Job(
        role="ref",
        replicas=2,
        tasks=[SchedulingSpec(cpu=2, gpu=4, mem=4, port_count=2)],
        scheduling_strategy=SchedulingStrategy(
            type=SchedulingStrategyType.colocation, target="nonexistent"
        ),
    )

    with pytest.raises(WorkerNotFoundError):
        scheduler.create_workers(ref_job)
