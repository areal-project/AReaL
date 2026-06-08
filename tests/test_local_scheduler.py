import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, Mock, call, patch

import aiohttp
import psutil
import pytest
import requests

from areal.api import Job, Worker
from areal.api.cli_args import (
    SchedulingSpec,
    SchedulingStrategy,
    SchedulingStrategyType,
)
from areal.infra.scheduler.exceptions import (
    EngineCallError,
    EngineCreationError,
    EngineImportError,
    GPUAllocationError,
    PortAllocationError,
    RPCConnectionError,
    WorkerCreationError,
    WorkerFailedError,
    WorkerNotFoundError,
    WorkerTimeoutError,
)
from areal.infra.scheduler.local import LocalScheduler, WorkerInfo
from areal.infra.utils.proc import kill_process_tree

# Skip all tests in this module by default - run manually only
pytestmark = pytest.mark.skip(
    reason="LocalScheduler tests have unexpected behavior on GCP CI machines. "
    "Run manually with: pytest tests/test_local_scheduler.py"
)

# ============================================================================
# Fixtures and Helper Functions
# ============================================================================


@pytest.fixture
def scheduler(tmp_path):
    """Create a LocalScheduler instance with default configuration."""
    fileroot = tmp_path / "fileroot"
    fileroot.mkdir()
    name_resolve_root = tmp_path / "name_resolve"
    name_resolve_root.mkdir()
    return LocalScheduler(
        gpu_devices=[0],
        log_dir=str(tmp_path),
        experiment_name="test_exp",
        trial_name="test_trial",
        fileroot=str(fileroot),
        nfs_record_root=str(name_resolve_root),
    )


@pytest.fixture
def multi_gpu_scheduler(tmp_path):
    """Create a LocalScheduler instance with multiple GPUs."""
    fileroot = tmp_path / "fileroot"
    fileroot.mkdir()
    name_resolve_root = tmp_path / "name_resolve"
    name_resolve_root.mkdir()
    return LocalScheduler(
        gpu_devices=[0, 1, 2],
        log_dir=str(tmp_path),
        experiment_name="test_exp",
        trial_name="test_trial",
        fileroot=str(fileroot),
        nfs_record_root=str(name_resolve_root),
    )


@pytest.fixture(autouse=True)
def mock_kill_process_tree():
    """Automatically mock kill_process_tree to prevent LocalScheduler.__del__ from killing fake PIDs."""
    with patch("areal.infra.scheduler.local.kill_process_tree") as mock_kill:
        yield mock_kill


def create_mock_process(pid=1234, is_alive=True, exit_code=None):
    """Create a mock subprocess.Popen process.

    Args:
        pid: Process ID
        is_alive: Whether process is still running
        exit_code: Exit code if process has terminated

    Returns:
        Mock process object
    """
    mock_proc = Mock()
    mock_proc.pid = pid
    mock_proc.poll.return_value = None if is_alive else exit_code
    if not is_alive:
        mock_proc.returncode = exit_code
    return mock_proc


def create_scheduler(tmp_path, gpu_devices=None, **kwargs):
    """Create a LocalScheduler instance with proper directory setup.

    Args:
        tmp_path: Pytest tmp_path fixture
        gpu_devices: List of GPU device indices (default: [0])
        **kwargs: Additional arguments to pass to LocalScheduler

    Returns:
        LocalScheduler instance with fileroot and name_resolve_root configured
    """
    fileroot = tmp_path / "fileroot"
    fileroot.mkdir(exist_ok=True)
    name_resolve_root = tmp_path / "name_resolve"
    name_resolve_root.mkdir(exist_ok=True)

    defaults = {
        "gpu_devices": gpu_devices or [0],
        "log_dir": str(tmp_path),
        "experiment_name": "test_exp",
        "trial_name": "test_trial",
        "fileroot": str(fileroot),
        "nfs_record_root": str(name_resolve_root),
    }
    defaults.update(kwargs)
    return LocalScheduler(**defaults)


def create_worker_info(
    worker_id="test/0",
    role="test",
    ip="127.0.0.1",
    ports=None,
    gpu_devices=None,
    log_file="/tmp/test.log",
    process=None,
):
    """Create a WorkerInfo instance with sensible defaults.

    Args:
        worker_id: Worker identifier
        role: Worker role name
        ip: IP address
        ports: List of port strings
        gpu_devices: List of GPU device IDs
        log_file: Path to log file
        process: Mock process object (created if not provided)

    Returns:
        WorkerInfo instance
    """
    if ports is None:
        ports = ["8000"]
    if gpu_devices is None:
        gpu_devices = [0]
    if process is None:
        process = create_mock_process()

    return WorkerInfo(
        worker=Worker(id=worker_id, ip=ip, worker_ports=ports, engine_ports=[]),
        process=process,
        role=role,
        gpu_devices=gpu_devices,
        created_at=time.time(),
        log_file=log_file,
    )


def create_mock_http_response(status_code=200, json_data=None):
    """Create a mock HTTP response.

    Args:
        status_code: HTTP status code
        json_data: Dictionary to return from response.json()

    Returns:
        Mock response object
    """
    mock_response = Mock()
    mock_response.status_code = status_code
    if json_data is not None:
        mock_response.json.return_value = json_data
    return mock_response


class TestLocalSchedulerInitialization:
    """Test LocalScheduler initialization and GPU detection."""

    def test_init_with_explicit_gpu_devices(self, tmp_path):
        """Should initialize with explicitly provided GPU devices."""
        scheduler = create_scheduler(
            tmp_path,
            gpu_devices=[0, 1, 2],
            startup_timeout=60.0,
            health_check_interval=2.0,
        )

        assert scheduler.gpu_devices == [0, 1, 2]
        assert scheduler.log_dir == tmp_path
        assert scheduler.startup_timeout == 60.0
        assert scheduler.health_check_interval == 2.0
        assert scheduler._gpu_counter == 0
        assert len(scheduler._allocated_ports) == 0
        assert len(scheduler._workers) == 0
        assert tmp_path.exists()

    def test_init_without_gpu_devices_uses_cuda_visible_devices(self, tmp_path):
        """Should detect GPUs from CUDA_VISIBLE_DEVICES environment variable."""
        # We need to patch the current_platform module-level attribute used in local.py
        # by patching the import in the local module directly
        mock_platform = Mock()
        mock_platform.device_control_env_var = "CUDA_VISIBLE_DEVICES"

        # Set up directories for LocalScheduler
        fileroot = tmp_path / "fileroot"
        fileroot.mkdir()
        name_resolve_root = tmp_path / "name_resolve"
        name_resolve_root.mkdir()

        with (
            patch("areal.infra.scheduler.local.current_platform", mock_platform),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,3"}),
        ):
            # Don't use create_scheduler helper here as it defaults gpu_devices to [0]
            # We want to test the auto-detection from CUDA_VISIBLE_DEVICES
            scheduler = LocalScheduler(
                log_dir=str(tmp_path),
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=str(fileroot),
                nfs_record_root=str(name_resolve_root),
            )
            assert scheduler.gpu_devices == [0, 1, 3]

    def test_init_creates_log_directory(self, tmp_path):
        """Should create log directory if it doesn't exist."""
        log_dir = tmp_path / "nested" / "log" / "dir"
        assert not log_dir.exists()

        scheduler = create_scheduler(tmp_path, log_dir=str(log_dir))

        assert log_dir.exists()
        assert scheduler.log_dir == log_dir


class TestGPUAllocation:
    """Test GPU allocation strategies."""

    def test_allocate_gpus_round_robin(self, tmp_path):
        """Should allocate GPUs in round-robin fashion."""
        scheduler = create_scheduler(tmp_path, gpu_devices=[0, 1, 2])

        # First allocation
        gpus1 = scheduler._allocate_gpus(2)
        assert gpus1 == [0, 1]

        # Second allocation (wraps around)
        gpus2 = scheduler._allocate_gpus(3)
        assert gpus2 == [2, 0, 1]

        # Third allocation
        gpus3 = scheduler._allocate_gpus(1)
        assert gpus3 == [2]

    def test_allocate_gpus_exceeds_available(self, tmp_path):
        """Should raise GPUAllocationError when requesting more GPUs than available."""
        scheduler = create_scheduler(tmp_path, gpu_devices=[0, 1])

        with pytest.raises(GPUAllocationError) as exc_info:
            scheduler._allocate_gpus(3)

        assert "Requested 3 GPUs but only 2 available" in str(exc_info.value)

    def test_allocate_gpus_single_gpu_multiple_times(self, scheduler):
        """Should allow multiple workers to share a single GPU via round-robin."""
        # Multiple allocations should all get GPU 0
        for _ in range(5):
            gpus = scheduler._allocate_gpus(1)
            assert gpus == [0]

    def test_get_colocated_gpus_success(self, multi_gpu_scheduler):
        """Should return GPU devices from target worker for colocation."""
        # Create mock workers for target role
        worker1 = create_worker_info(
            worker_id="actor/0", role="actor", ports=["8000"], gpu_devices=[0, 1]
        )
        worker2 = create_worker_info(
            worker_id="actor/1", role="actor", ports=["8001"], gpu_devices=[2]
        )
        multi_gpu_scheduler._workers["actor"] = [worker1, worker2]

        # Get colocated GPUs
        gpus = multi_gpu_scheduler._get_colocated_gpus("actor", 0)
        assert gpus == [0, 1]

        gpus = multi_gpu_scheduler._get_colocated_gpus("actor", 1)
        assert gpus == [2]

    def test_get_colocated_gpus_role_not_found(self, scheduler):
        """Should raise WorkerNotFoundError when target role doesn't exist."""
        with pytest.raises(WorkerNotFoundError) as exc_info:
            scheduler._get_colocated_gpus("nonexistent", 0)

        assert "Cannot colocate with role 'nonexistent' - role not found" in str(
            exc_info.value
        )

    def test_get_colocated_gpus_worker_index_out_of_range(self, scheduler):
        """Should raise ValueError when worker index is out of range."""
        # Create only one worker for target role
        worker = create_worker_info(worker_id="actor/0", role="actor", gpu_devices=[0])
        scheduler._workers["actor"] = [worker]

        with pytest.raises(ValueError) as exc_info:
            scheduler._get_colocated_gpus("actor", 5)

        assert "only 1 workers exist" in str(exc_info.value)


class TestPortAllocation:
    """Test port allocation and tracking."""

    def test_allocate_ports_success(self, tmp_path):
        """Should allocate requested number of free ports."""
        with patch("areal.infra.scheduler.local.find_free_ports") as mock_find_ports:
            mock_find_ports.return_value = [8000, 8001, 8002]

            scheduler = create_scheduler(tmp_path)
            ports = scheduler._allocate_ports(3)

            assert ports == [8000, 8001, 8002]
            assert scheduler._allocated_ports == {8000, 8001, 8002}
            mock_find_ports.assert_called_once_with(3, exclude_ports=set())

    def test_allocate_ports_excludes_already_allocated(self, tmp_path):
        """Should exclude already allocated ports from search."""
        with patch("areal.infra.scheduler.local.find_free_ports") as mock_find_ports:
            mock_find_ports.side_effect = [
                [8000, 8001],
                [8002, 8003],
            ]

            scheduler = create_scheduler(tmp_path)

            # First allocation
            ports1 = scheduler._allocate_ports(2)
            assert ports1 == [8000, 8001]

            # Second allocation should exclude previously allocated ports
            ports2 = scheduler._allocate_ports(2)
            assert ports2 == [8002, 8003]
            assert scheduler._allocated_ports == {8000, 8001, 8002, 8003}

            # Verify excluded ports were passed
            calls = mock_find_ports.call_args_list
            assert calls[0] == call(2, exclude_ports=set())
            assert calls[1] == call(2, exclude_ports={8000, 8001})

    def test_allocate_ports_failure(self, tmp_path):
        """Should raise PortAllocationError when port allocation fails."""
        with patch("areal.infra.scheduler.local.find_free_ports") as mock_find_ports:
            mock_find_ports.side_effect = ValueError("No free ports available")

            scheduler = create_scheduler(tmp_path)

            with pytest.raises(PortAllocationError) as exc_info:
                scheduler._allocate_ports(5)

            assert "No free ports available" in str(exc_info.value)


class TestWorkerCreation:
    """Test worker creation with various configurations."""

    @patch("areal.infra.scheduler.local.gethostip")
    @patch("areal.infra.scheduler.local.subprocess.Popen")
    @patch("areal.infra.scheduler.local.find_free_ports")
    def test_create_workers_with_default_spec(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should create workers with default spec (1 GPU, 2 ports) when no specs provided."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.side_effect = [[8000, 8001], [8002, 8003]]

        # Mock process
        mock_process1 = Mock()
        mock_process1.pid = 1234
        mock_process1.poll.return_value = None
        mock_process2 = Mock()
        mock_process2.pid = 1235
        mock_process2.poll.return_value = None
        mock_popen.side_effect = [mock_process1, mock_process2]

        scheduler = create_scheduler(tmp_path, gpu_devices=[0, 1])

        with patch.object(scheduler, "_configure_worker", return_value=None):
            job = Job(replicas=2, role="rollout")
            worker_ids = scheduler.create_workers(job)

            assert worker_ids == ["rollout/0", "rollout/1"]
            assert "rollout" in scheduler._workers
            assert len(scheduler._workers["rollout"]) == 2

            # Verify default spec was used
            assert mock_popen.call_count == 2

        # Clean up workers while mock is still active
        scheduler.delete_workers(None)

    @patch("areal.infra.scheduler.local.gethostip")
    @patch("areal.infra.scheduler.local.subprocess.Popen")
    @patch("areal.infra.scheduler.local.find_free_ports")
    def test_create_workers_with_single_spec_for_all(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should use single spec for all workers when specs length is 1."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.side_effect = [[8000, 8001, 8002]] * 3

        # Mock processes
        mock_processes = []
        for i in range(3):
            mock_proc = Mock()
            mock_proc.pid = 1000 + i
            mock_proc.poll.return_value = None
            mock_processes.append(mock_proc)
        mock_popen.side_effect = mock_processes

        scheduler = create_scheduler(tmp_path, gpu_devices=[0, 1, 2])

        job = Job(
            replicas=3,
            role="actor",
            tasks=[
                SchedulingSpec(
                    cpu=1,
                    mem=1024,
                    gpu=2,
                    port_count=3,
                    cmd="python -m areal.infra.rpc.rpc_server",
                )
            ],
        )
        with patch.object(scheduler, "_configure_worker", return_value=None):
            worker_ids = scheduler.create_workers(job)

        assert len(worker_ids) == 3
        assert mock_popen.call_count == 3

        # All workers should use the same spec
        for worker_info in scheduler._workers["actor"]:
            assert len(worker_info.worker.worker_ports) == 3

        # Clean up workers while mock is still active
        scheduler.delete_workers(None)

    @patch("areal.infra.scheduler.local.gethostip")
    @patch("areal.infra.scheduler.local.subprocess.Popen")
    @patch("areal.infra.scheduler.local.find_free_ports")
    def test_create_workers_with_per_worker_specs(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should use individual specs when specs length equals replicas."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.side_effect = [[8000], [8001, 8002]]

        # Mock processes
        mock_proc1 = Mock()
        mock_proc1.pid = 1000
        mock_proc1.poll.return_value = None
        mock_proc2 = Mock()
        mock_proc2.pid = 1001
        mock_proc2.poll.return_value = None
        mock_popen.side_effect = [mock_proc1, mock_proc2]

        scheduler = create_scheduler(tmp_path, gpu_devices=[0, 1])

        job = Job(
            replicas=2,
            role="critic",
            tasks=[
                SchedulingSpec(
                    cpu=1,
                    mem=1024,
                    gpu=1,
                    port_count=1,
                    cmd="python -m areal.infra.rpc.rpc_server",
                ),
                SchedulingSpec(
                    cpu=1,
                    mem=1024,
                    gpu=1,
                    port_count=2,
                    cmd="python -m areal.infra.rpc.rpc_server",
                ),
            ],
        )
        with patch.object(scheduler, "_configure_worker", return_value=None):
            worker_ids = scheduler.create_workers(job)

        assert len(worker_ids) == 2
        assert len(scheduler._workers["critic"][0].worker.worker_ports) == 1
        assert len(scheduler._workers["critic"][1].worker.worker_ports) == 2

        # Clean up workers while mock is still active
        scheduler.delete_workers(None)

    @patch("areal.infra.scheduler.local.gethostip")
    @patch("areal.infra.scheduler.local.subprocess.Popen")
    @patch("areal.infra.scheduler.local.find_free_ports")
    def test_create_workers_with_custom_command(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should use custom command from spec when provided."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = [8000, 8001]

        mock_proc = Mock()
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        scheduler = create_scheduler(tmp_path)

        job = Job(
            replicas=1,
            role="custom",
            tasks=[
                SchedulingSpec(
                    cpu=1,
                    mem=1024,
                    gpu=1,
                    port_count=2,
                    cmd="python my_custom_server.py",
                )
            ],
        )
        with patch.object(scheduler, "_configure_worker", return_value=None):
            worker_ids = scheduler.create_workers(job)

        assert len(worker_ids) == 1

        # Verify custom command was used
        # The command is passed as a string to subprocess.Popen with shell=True
        popen_call = mock_popen.call_args
        cmd_str = popen_call[0][0]
        assert isinstance(cmd_str, str), f"Expected string, got {type(cmd_str)}"
        assert "my_custom_server.py --port 8000" in cmd_str
        # Verify shell=True is used since cmd is a string
        assert popen_call[1]["shell"] is True
        # Verify that subprocess.Popen was called
        mock_popen.assert_called_once()

        # Clean up workers while mock is still active
        scheduler.delete_workers(None)

    @patch("areal.infra.scheduler.local.gethostip")
    @patch("areal.infra.scheduler.local.subprocess.Popen")
    @patch("areal.infra.scheduler.local.find_free_ports")
    def test_create_workers_with_environment_variables(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should merge environment variables from spec into worker environment."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = [8000, 8001]

        mock_proc = Mock()
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        # Mock the platform to use CUDA_VISIBLE_DEVICES
        mock_platform = Mock()
        mock_platform.device_control_env_var = "CUDA_VISIBLE_DEVICES"

        with patch("areal.infra.scheduler.local.current_platform", mock_platform):
            scheduler = create_scheduler(tmp_path)

            job = Job(
                replicas=1,
                role="envtest",
                tasks=[
                    SchedulingSpec(
                        cpu=1,
                        mem=1024,
                        gpu=1,
                        port_count=2,
                        env_vars={"CUSTOM_VAR": "custom_value", "ANOTHER_VAR": "123"},
                        cmd="python -m areal.infra.rpc.rpc_server",
                    )
                ],
            )
            with patch.object(scheduler, "_configure_worker", return_value=None):
                worker_ids = scheduler.create_workers(job)

            assert len(worker_ids) == 1

            # Verify environment variables were passed
            # Environment variables are encoded into the shell command string, not passed as env parameter
            popen_call = mock_popen.call_args
            cmd_str = popen_call[0][0]
            assert isinstance(cmd_str, str), f"Expected string, got {type(cmd_str)}"
            # Verify custom environment variables are in the command string
            assert "CUSTOM_VAR=custom_value" in cmd_str
            assert "ANOTHER_VAR=123" in cmd_str
            # Verify CUDA_VISIBLE_DEVICES is set correctly
            assert "CUDA_VISIBLE_DEVICES=0" in cmd_str
            # Verify shell=True is used since cmd is a string
            assert popen_call[1]["shell"] is True

            # Clean up workers while mock is still active
            scheduler.delete_workers(None)

    @patch("areal.infra.scheduler.local.gethostip")
    @patch("areal.infra.scheduler.local.subprocess.Popen")
    @patch("areal.infra.scheduler.local.find_free_ports")
    def test_create_workers_with_colocate_strategy(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should reuse existing workers from target role when colocate strategy is used."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = [8000, 8001]

        mock_processes = []
        for i in range(2):  # Only 2 processes for actors
            mock_proc = Mock()
            mock_proc.pid = 1000 + i
            mock_proc.poll.return_value = None
            mock_processes.append(mock_proc)
        mock_popen.side_effect = mock_processes

        scheduler = create_scheduler(tmp_path, gpu_devices=[0, 1, 2, 3])

        # Create target workers (actors)
        actor_job = Job(
            replicas=2,
            role="actor",
            tasks=[
                SchedulingSpec(
                    cpu=1,
                    mem=1024,
                    gpu=2,
                    port_count=2,
                    cmd="python -m areal.infra.rpc.rpc_server",
                )
            ],
        )
        with patch.object(scheduler, "_configure_worker", return_value=None):
            actor_ids = scheduler.create_workers(actor_job)

        # Verify actors were created
        assert actor_ids == ["actor/0", "actor/1"]
        initial_popen_count = mock_popen.call_count

        # Create colocated workers (critics) with fork=False - should NOT spawn new processes
        critic_job = Job(
            replicas=2,
            role="critic",
            tasks=[
                SchedulingSpec(
                    cpu=1,
                    mem=1024,
                    gpu=2,
                    port_count=2,
                    cmd="python -m areal.infra.rpc.rpc_server",
                )
            ],
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="actor", fork=False
            ),
        )
        critic_ids = scheduler.create_workers(critic_job)

        # Verify colocated role returns the SAME worker IDs as target role
        assert critic_ids == actor_ids

        # Verify NO new processes were spawned for colocated role
        assert mock_popen.call_count == initial_popen_count

        # Verify colocation tracking is set up correctly
        assert "critic" in scheduler._colocated_roles
        assert scheduler._colocated_roles["critic"] == "actor"

        # Clean up workers while mock is still active
        scheduler.delete_workers(None)

    def test_create_workers_duplicate_role_error(self, tmp_path):
        """Should raise WorkerCreationError when attempting to create workers for existing role."""
        scheduler = create_scheduler(tmp_path)

        with (
            patch("areal.infra.scheduler.local.subprocess.Popen") as mock_popen,
            patch("areal.infra.scheduler.local.find_free_ports") as mock_find_ports,
            patch("areal.infra.scheduler.local.gethostip") as mock_gethostip,
        ):
            mock_gethostip.return_value = "127.0.0.1"
            mock_find_ports.return_value = [8000, 8001]
            mock_proc = Mock()
            mock_proc.pid = 1234
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            job = Job(replicas=1, role="test")
            with patch.object(scheduler, "_configure_worker", return_value=None):
                scheduler.create_workers(job)

            # Try to create again
            with pytest.raises(WorkerCreationError) as exc_info:
                scheduler.create_workers(job)

            assert "Worker group already exists" in str(exc_info.value)
            assert exc_info.value.worker_key == "test"

            # Clean up workers while mock is still active
            scheduler.delete_workers(None)

    def test_create_workers_zero_replicas_error(self, tmp_path):
        """Should raise WorkerCreationError when replicas is 0."""
        scheduler = create_scheduler(tmp_path)

        job = Job(replicas=0, role="test")

        with pytest.raises(WorkerCreationError) as exc_info:
            scheduler.create_workers(job)

        assert "replicas must be greater than 0" in str(exc_info.value)

    def test_create_workers_invalid_specs_length(self, tmp_path):
        """Should raise WorkerCreationError when tasks length is invalid."""
        scheduler = create_scheduler(tmp_path, gpu_devices=[0, 1])

        job = Job(
            replicas=3,
            role="test",
            tasks=[
                SchedulingSpec(
                    cpu=1,
                    mem=1024,
                    gpu=1,
                    port_count=2,
                    cmd="python -m areal.infra.rpc.rpc_server",
                ),
                SchedulingSpec(cpu=1, mem=1024, gpu=1, port_count=2),
            ],  # 2 tasks for 3 replicas
        )

        with pytest.raises(WorkerCreationError) as exc_info:
            scheduler.create_workers(job)

        assert "schedulings length (2) must be 1 or equal to replicas (3)" in str(
            exc_info.value
        )

    @patch("areal.infra.scheduler.local.gethostip")
    @patch("areal.infra.scheduler.local.subprocess.Popen")
    @patch("areal.infra.scheduler.local.find_free_ports")
    def test_create_workers_subprocess_fails_immediately(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should raise WorkerCreationError when subprocess exits immediately."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = [8000, 8001]

        # Mock process that exits immediately
        mock_proc = Mock()
        mock_proc.pid = 1234
        mock_proc.poll.return_value = 1  # Exit code 1
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc

        # Create log file with error message
        log_file = tmp_path / "test.log"
        log_file.write_text("Error: Failed to start server\n")

        scheduler = create_scheduler(tmp_path)

        job = Job(replicas=1, role="test")

        with patch.object(
            scheduler, "_read_log_tail", return_value="Error: Failed to start server"
        ):
            with pytest.raises(WorkerCreationError) as exc_info:
                scheduler.create_workers(job)

            assert "exited immediately with code 1" in str(exc_info.value)
            assert mock_popen.call_count == 1

    @patch("areal.infra.scheduler.local.gethostip")
    @patch("areal.infra.scheduler.local.subprocess.Popen")
    @patch("areal.infra.scheduler.local.find_free_ports")
    def test_create_workers_retries_immediate_port_conflict(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.side_effect = [[8000, 8001], [8002, 8003]]

        conflict_proc = Mock()
        conflict_proc.pid = 1234
        conflict_proc.poll.return_value = 1
        conflict_proc.returncode = 1

        success_proc = Mock()
        success_proc.pid = 1235
        success_proc.poll.return_value = None

        mock_popen.side_effect = [conflict_proc, success_proc]

        scheduler = create_scheduler(tmp_path)
        job = Job(replicas=1, role="test")

        with patch.object(
            scheduler,
            "_read_log_tail",
            return_value="Address already in use\nPort 8000 is in use by another program",
        ):
            worker_ids = scheduler.create_workers(job)

        assert worker_ids == ["test/0"]
        assert mock_popen.call_count == 2
        assert scheduler._workers["test"][0].worker.worker_ports == ["8002", "8003"]
        assert scheduler._allocated_ports == {8002, 8003}

    @patch("areal.infra.scheduler.local.gethostip")
    @patch("areal.infra.scheduler.local.subprocess.Popen")
    @patch("areal.infra.scheduler.local.find_free_ports")
    def test_create_workers_cleanup_on_partial_failure(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should clean up successfully created workers when a later worker fails."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.side_effect = [
            [8000, 8001],  # First worker succeeds
            ValueError("No free ports"),  # Second worker fails
        ]

        # First process succeeds
        mock_proc1 = Mock()
        mock_proc1.pid = 1234
        mock_proc1.poll.return_value = None
        mock_popen.return_value = mock_proc1

        scheduler = create_scheduler(tmp_path)

        job = Job(replicas=2, role="test")

        with patch.object(scheduler, "_cleanup_workers") as mock_cleanup:
            with pytest.raises(WorkerCreationError) as exc_info:
                scheduler.create_workers(job)

            # Verify cleanup was called
            assert mock_cleanup.called
            assert "Resource allocation failed" in str(exc_info.value)

    def test_create_workers_colocate_strategy_missing_target(self, tmp_path):
        """Should raise WorkerCreationError when colocation strategy is missing target role."""
        scheduler = create_scheduler(tmp_path)

        job = Job(
            replicas=1,
            role="test",
            tasks=[
                SchedulingSpec(
                    cpu=1,
                    mem=1024,
                    gpu=1,
                    port_count=2,
                    cmd="python -m areal.infra.rpc.rpc_server",
                )
            ],
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target=""
            ),  # Missing target
        )

        with pytest.raises(WorkerCreationError) as exc_info:
            scheduler.create_workers(job)

        assert "Colocation strategy requires target" in str(exc_info.value)


class TestGetWorkers:
    """Test getting workers and waiting for readiness."""

    def test_get_workers_role_not_found(self, scheduler):
        """Should raise WorkerNotFoundError when role doesn't exist."""
        with pytest.raises(WorkerNotFoundError) as exc_info:
            scheduler.get_workers("nonexistent")

        assert exc_info.value.worker_id == "nonexistent"

    @patch("areal.infra.scheduler.local.time.sleep")
    def test_get_workers_success(self, mock_sleep, scheduler, tmp_path):
        """Should return workers when all are ready."""
        # Create mock workers
        worker1 = create_worker_info(
            worker_id="test/0", ports=["8000"], log_file=str(tmp_path / "test.log")
        )
        worker2 = create_worker_info(
            worker_id="test/1", ports=["8001"], log_file=str(tmp_path / "test.log")
        )

        scheduler._workers["test"] = [worker1, worker2]

        with patch.object(scheduler, "_is_worker_ready", return_value=True):
            workers = scheduler.get_workers("test", timeout=10.0)

            assert len(workers) == 2
            assert workers[0].id == "test/0"
            assert workers[1].id == "test/1"

    @patch("areal.infra.scheduler.local.time.time")
    @patch("areal.infra.scheduler.local.time.sleep")
    def test_get_workers_timeout(self, mock_sleep, mock_time, scheduler, tmp_path):
        """Should raise WorkerTimeoutError when timeout is exceeded."""
        # Mock time progression - provide enough values
        mock_time.side_effect = [0.0] + [i for i in range(1, 20)]

        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        worker.created_at = 0.0

        scheduler._workers["test"] = [worker]

        # Worker never becomes ready
        with patch.object(scheduler, "_is_worker_ready", return_value=False):
            with pytest.raises(WorkerTimeoutError) as exc_info:
                scheduler.get_workers("test", timeout=5.0)

            assert exc_info.value.worker_key == "test"
            assert exc_info.value.timeout == 5.0

    def test_get_workers_process_died(self, scheduler, tmp_path):
        """Should raise WorkerFailedError when worker process dies during readiness check."""
        log_file = tmp_path / "test.log"
        log_file.write_text("Error: Connection refused\n")

        # Process dies after first check
        mock_proc = create_mock_process()
        mock_proc.poll.side_effect = [None, 1]  # None (alive), then 1 (dead)
        mock_proc.returncode = 1

        worker = create_worker_info(process=mock_proc, log_file=str(log_file))
        scheduler._workers["test"] = [worker]

        with patch.object(scheduler, "_is_worker_ready", return_value=False):
            with pytest.raises(WorkerFailedError) as exc_info:
                scheduler.get_workers("test", timeout=10.0)

            assert exc_info.value.worker_id == "test/0"
            assert exc_info.value.exit_code == 1

    @patch("areal.infra.scheduler.local.time.sleep")
    def test_get_workers_gradual_readiness(self, mock_sleep, scheduler, tmp_path):
        """Should wait for all workers to become ready gradually."""
        worker1 = create_worker_info(
            worker_id="test/0", ports=["8000"], log_file=str(tmp_path / "test.log")
        )
        worker2 = create_worker_info(
            worker_id="test/1", ports=["8001"], log_file=str(tmp_path / "test.log")
        )

        scheduler._workers["test"] = [worker1, worker2]

        # Worker 1 ready immediately, worker 2 ready on second check
        ready_calls = [True, False, True, True]
        with patch.object(scheduler, "_is_worker_ready", side_effect=ready_calls):
            workers = scheduler.get_workers("test", timeout=10.0)

            assert len(workers) == 2


class TestWorkerHealthCheck:
    """Test worker health checking functionality."""

    @pytest.mark.parametrize(
        "status_code,expected",
        [
            (200, True),  # Success
            (503, False),  # Service unavailable
            (500, False),  # Internal server error
        ],
    )
    def test_is_worker_ready_http_status(
        self, scheduler, tmp_path, status_code, expected
    ):
        """Should return appropriate result based on HTTP status code."""
        worker_info = create_worker_info(log_file=str(tmp_path / "test.log"))
        mock_response = create_mock_http_response(status_code=status_code)

        with patch.object(requests, "get", return_value=mock_response):
            assert scheduler._is_worker_ready(worker_info) is expected

    def test_is_worker_ready_connection_error(self, scheduler, tmp_path):
        """Should return False when connection to worker fails."""
        worker_info = create_worker_info(log_file=str(tmp_path / "test.log"))

        with patch.object(
            requests,
            "get",
            side_effect=requests.exceptions.ConnectionError("Connection refused"),
        ):
            assert scheduler._is_worker_ready(worker_info) is False

    def test_check_worker_health_all_healthy(self, scheduler, tmp_path):
        """Should pass when all workers are healthy."""
        worker1 = create_worker_info(
            worker_id="test/0", ports=["8000"], log_file=str(tmp_path / "test.log")
        )
        worker2 = create_worker_info(
            worker_id="test/1", ports=["8001"], log_file=str(tmp_path / "test.log")
        )

        scheduler._workers["test"] = [worker1, worker2]

        # Should not raise
        scheduler._check_worker_health("test")

    def test_check_worker_health_worker_failed(self, scheduler, tmp_path):
        """Should raise WorkerFailedError when a worker has failed."""
        log_file = tmp_path / "test.log"
        log_file.write_text("Killed by signal\n")

        mock_proc = create_mock_process(is_alive=False, exit_code=137)
        worker = create_worker_info(process=mock_proc, log_file=str(log_file))

        scheduler._workers["test"] = [worker]

        with pytest.raises(WorkerFailedError) as exc_info:
            scheduler._check_worker_health("test")

        assert exc_info.value.worker_id == "test/0"
        assert exc_info.value.exit_code == 137

    def test_check_worker_health_nonexistent_role(self, scheduler):
        """Should silently pass when role doesn't exist."""
        # Should not raise
        scheduler._check_worker_health("nonexistent")


class TestDeleteWorkers:
    """Test worker deletion and cleanup."""

    def test_delete_workers_specific_role(self, scheduler, tmp_path):
        """Should delete workers for specific role."""
        # Create mock workers for multiple roles
        worker1 = create_worker_info(
            worker_id="role1/0",
            role="role1",
            ports=["8000"],
            log_file=str(tmp_path / "role1.log"),
        )
        worker2 = create_worker_info(
            worker_id="role2/0",
            role="role2",
            ports=["8001"],
            log_file=str(tmp_path / "role2.log"),
        )

        scheduler._workers["role1"] = [worker1]
        scheduler._workers["role2"] = [worker2]
        scheduler._allocated_ports = {8000, 8001}

        scheduler.delete_workers("role1")

        # role1 should be deleted, role2 should remain
        assert "role1" not in scheduler._workers
        assert "role2" in scheduler._workers
        assert 8000 not in scheduler._allocated_ports
        assert 8001 in scheduler._allocated_ports

    def test_delete_workers_all_roles(self, scheduler, tmp_path):
        """Should delete all workers when role is None."""
        worker1 = create_worker_info(
            worker_id="role1/0",
            role="role1",
            ports=["8000"],
            log_file=str(tmp_path / "role1.log"),
        )
        worker2 = create_worker_info(
            worker_id="role2/0",
            role="role2",
            ports=["8001"],
            log_file=str(tmp_path / "role2.log"),
        )

        scheduler._workers["role1"] = [worker1]
        scheduler._workers["role2"] = [worker2]
        scheduler._allocated_ports = {8000, 8001}

        scheduler.delete_workers(None)

        # All workers should be deleted
        assert len(scheduler._workers) == 0
        assert len(scheduler._allocated_ports) == 0

    def test_delete_workers_nonexistent_role(self, scheduler):
        """Should log warning and return when role doesn't exist."""
        # Should not raise
        scheduler.delete_workers("nonexistent")

    def test_delete_workers_reverse_order(self, scheduler, tmp_path, monkeypatch):
        """With reverse_order=True, workers are cleaned up in reverse rank order.

        This protects rank-0 (owner of the global TCPStore server) from being
        torn down before non-zero ranks finish their final NCCL abort.
        """
        workers = [
            create_worker_info(
                worker_id=f"role1/{i}",
                role="role1",
                ports=[str(8000 + i)],
                log_file=str(tmp_path / f"role1-{i}.log"),
            )
            for i in range(4)
        ]
        scheduler._workers["role1"] = workers
        scheduler._allocated_ports = {8000, 8001, 8002, 8003}

        observed_order: list[str] = []

        original_cleanup = scheduler._cleanup_workers

        def spy(workers_arg):
            observed_order.extend(w.worker.id for w in workers_arg)
            original_cleanup(workers_arg)

        monkeypatch.setattr(scheduler, "_cleanup_workers", spy)

        scheduler.delete_workers("role1", reverse_order=True)

        assert observed_order == ["role1/3", "role1/2", "role1/1", "role1/0"]
        assert "role1" not in scheduler._workers

    def test_cleanup_workers_releases_ports(self, scheduler, tmp_path):
        """Should release allocated ports when cleaning up workers."""
        worker = create_worker_info(
            ports=["8000", "8001"], log_file=str(tmp_path / "test.log")
        )
        scheduler._allocated_ports = {8000, 8001, 8002}

        scheduler._cleanup_workers([worker])

        # Ports 8000 and 8001 should be released
        assert scheduler._allocated_ports == {8002}

    def test_cleanup_workers_handles_errors(
        self, scheduler, tmp_path, mock_kill_process_tree
    ):
        """Should continue cleanup even if terminating a process fails."""
        worker1 = create_worker_info(
            worker_id="test/0", ports=["8000"], log_file=str(tmp_path / "test.log")
        )
        worker2 = create_worker_info(
            worker_id="test/1", ports=["8001"], log_file=str(tmp_path / "test.log")
        )

        # First termination fails, second succeeds
        # Configure the autouse mock to raise exception on first call
        mock_kill_process_tree.side_effect = [Exception("Failed to terminate"), None]
        # Should not raise, just log error
        scheduler._cleanup_workers([worker1, worker2])


class TestProcessTermination:
    """Test process termination functionality."""

    @patch("areal.infra.utils.proc.psutil.Process")
    @patch("areal.infra.utils.proc.psutil.wait_procs")
    def test_kill_process_tree_graceful(self, mock_wait_procs, mock_process_class):
        """Should gracefully terminate process tree."""
        # Mock parent process
        mock_parent = Mock()
        mock_child1 = Mock()
        mock_child2 = Mock()

        mock_parent.children.return_value = [mock_child1, mock_child2]
        mock_process_class.return_value = mock_parent

        # All processes terminate gracefully
        mock_wait_procs.return_value = ([], [])  # (gone, alive)

        kill_process_tree(1234, timeout=3, graceful=True)

        # Verify termination sequence
        mock_child1.terminate.assert_called_once()
        mock_child2.terminate.assert_called_once()
        mock_parent.terminate.assert_called_once()

        # Should not call kill since all terminated gracefully
        mock_child1.kill.assert_not_called()
        mock_child2.kill.assert_not_called()
        mock_parent.kill.assert_not_called()

    @patch("areal.infra.utils.proc.psutil.Process")
    @patch("areal.infra.utils.proc.psutil.wait_procs")
    def test_kill_process_tree_force_kill(self, mock_wait_procs, mock_process_class):
        """Should force kill processes that don't terminate gracefully."""
        mock_parent = Mock()
        mock_child = Mock()

        mock_parent.children.return_value = [mock_child]
        mock_process_class.return_value = mock_parent

        # Child doesn't terminate gracefully
        mock_wait_procs.return_value = ([], [mock_child])  # (gone, alive)

        kill_process_tree(1234, timeout=3, graceful=True)

        # Verify force kill was called
        mock_child.terminate.assert_called_once()
        mock_child.kill.assert_called_once()

    @patch("areal.infra.utils.proc.psutil.Process")
    def test_kill_process_tree_no_such_process(self, mock_process_class):
        """Should handle gracefully when process doesn't exist."""
        mock_process_class.side_effect = psutil.NoSuchProcess(1234)

        # Should not raise
        kill_process_tree(1234, timeout=3, graceful=True)

    @patch("areal.infra.utils.proc.psutil.Process")
    def test_kill_process_tree_handles_child_no_such_process(self, mock_process_class):
        """Should handle when child process disappears during termination."""
        mock_parent = Mock()
        mock_child = Mock()
        mock_child.terminate.side_effect = psutil.NoSuchProcess(1235)

        mock_parent.children.return_value = [mock_child]
        mock_process_class.return_value = mock_parent

        # Should not raise
        kill_process_tree(1234, timeout=3, graceful=True)


class TestLogFileHandling:
    """Test log file reading and handling."""

    def test_read_log_tail_success(self, tmp_path):
        """Should read last N lines from log file."""
        scheduler = create_scheduler(tmp_path)

        log_file = tmp_path / "test.log"
        log_lines = [f"Line {i}\n" for i in range(100)]
        log_file.write_text("".join(log_lines))

        tail = scheduler._read_log_tail(str(log_file), lines=10)

        # Should contain last 10 lines
        assert "Line 90" in tail
        assert "Line 99" in tail
        assert "Line 89" not in tail

    def test_read_log_tail_file_not_found(self, tmp_path):
        """Should return error message when log file doesn't exist."""
        scheduler = create_scheduler(tmp_path)

        tail = scheduler._read_log_tail("/nonexistent/file.log")

        assert "Could not read log file" in tail

    def test_read_log_tail_fewer_lines_than_requested(self, tmp_path):
        """Should return all lines when file has fewer lines than requested."""
        scheduler = create_scheduler(tmp_path)

        log_file = tmp_path / "test.log"
        log_file.write_text("Line 1\nLine 2\nLine 3\n")

        tail = scheduler._read_log_tail(str(log_file), lines=50)

        assert "Line 1" in tail
        assert "Line 2" in tail
        assert "Line 3" in tail


class TestSetEnv:
    """Test configuring worker environment variables."""

    def test_set_env_success(self, scheduler, tmp_path):
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.post = Mock(return_value=mock_response)

        with patch(
            "areal.infra.scheduler.local.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            asyncio.run(
                scheduler.set_worker_env("test/0", {"RANK": "0", "WORLD_SIZE": "1"})
            )

            mock_session.post.assert_called_once()

    def test_set_env_worker_not_found(self, scheduler):
        with pytest.raises(WorkerNotFoundError):
            asyncio.run(
                scheduler.set_worker_env("missing/0", {"RANK": "0", "WORLD_SIZE": "1"})
            )


class TestEngineCreation:
    """Test engine creation on workers."""

    def test_create_engine_success(self, scheduler, tmp_path):
        """Should successfully create engine on worker."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        # Mock aiohttp.ClientSession and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"result": {"status": "initialized", "name": "TestEngine"}}
        )
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.post = Mock(return_value=mock_response)

        with patch(
            "areal.infra.scheduler.local.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            result = asyncio.run(
                scheduler.create_engine(
                    "test/0", "test_engines.DummyEngine", name="TestEngine", param=123
                )
            )

            assert result == {"status": "initialized", "name": "TestEngine"}

    def test_create_engine_worker_not_found(self, scheduler):
        """Should raise WorkerNotFoundError when worker doesn't exist."""
        with pytest.raises(WorkerNotFoundError) as exc_info:
            asyncio.run(
                scheduler.create_engine("nonexistent/0", "test_engines.DummyEngine")
            )

        assert exc_info.value.worker_id == "nonexistent/0"

    def test_create_engine_worker_died(self, scheduler, tmp_path):
        """Should raise WorkerFailedError when worker process has died."""
        log_file = tmp_path / "test.log"
        log_file.write_text("Worker crashed\n")

        mock_proc = create_mock_process(is_alive=False, exit_code=1)
        worker = create_worker_info(process=mock_proc, log_file=str(log_file))
        scheduler._workers["test"] = [worker]

        with pytest.raises(WorkerFailedError) as exc_info:
            asyncio.run(scheduler.create_engine("test/0", "test_engines.DummyEngine"))

        assert exc_info.value.worker_id == "test/0"
        assert exc_info.value.exit_code == 1

    def test_create_engine_invalid_engine_type(self, scheduler, tmp_path):
        """Should raise EngineCreationError when engine is not a string."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        with pytest.raises(EngineCreationError) as exc_info:
            asyncio.run(scheduler.create_engine("test/0", 123))  # Invalid type

        assert "Engine must be a string import path" in str(exc_info.value)

    def test_create_engine_import_error(self, scheduler, tmp_path):
        """Should raise EngineImportError when engine import fails."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        # Mock aiohttp.ClientSession and response
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json = AsyncMock(
            return_value={"error": "Failed to import 'nonexistent.Engine'"}
        )
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.post = Mock(return_value=mock_response)

        with patch(
            "areal.infra.scheduler.local.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            with pytest.raises(EngineImportError) as exc_info:
                asyncio.run(scheduler.create_engine("test/0", "nonexistent.Engine"))

            assert "nonexistent.Engine" in str(exc_info.value)

    def test_create_engine_initialization_error(self, scheduler, tmp_path):
        """Should raise EngineCreationError when engine initialization fails."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        # Mock aiohttp.ClientSession and response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.json = AsyncMock(
            return_value={"error": "Engine initialization failed: out of memory"}
        )
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.post = Mock(return_value=mock_response)

        with patch(
            "areal.infra.scheduler.local.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            with pytest.raises(EngineCreationError) as exc_info:
                asyncio.run(
                    scheduler.create_engine("test/0", "test_engines.DummyEngine")
                )

            assert "out of memory" in str(exc_info.value)
            assert exc_info.value.status_code == 500

    def test_create_engine_connection_error_worker_died(self, scheduler, tmp_path):
        """Should raise WorkerFailedError when connection fails and worker is dead."""
        log_file = tmp_path / "test.log"
        log_file.write_text("Worker crashed during engine creation\n")

        # First call returns None (alive), second call returns exit code (dead)
        mock_proc = create_mock_process()
        mock_proc.poll.side_effect = [None, 1]
        mock_proc.returncode = 1

        worker = create_worker_info(process=mock_proc, log_file=str(log_file))
        scheduler._workers["test"] = [worker]

        # Mock aiohttp.ClientSession to raise connection error
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.post = Mock(
            side_effect=aiohttp.ClientConnectionError("Connection refused")
        )

        with patch(
            "areal.infra.scheduler.local.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            with pytest.raises(WorkerFailedError) as exc_info:
                asyncio.run(
                    scheduler.create_engine("test/0", "test_engines.DummyEngine")
                )

            assert exc_info.value.worker_id == "test/0"

    def test_create_engine_connection_error_worker_alive(self, scheduler, tmp_path):
        """Should raise RPCConnectionError when connection fails but worker is alive."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        # Mock aiohttp.ClientSession to raise connection error
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.post = Mock(
            side_effect=aiohttp.ClientConnectionError("Connection refused")
        )

        with patch(
            "areal.infra.scheduler.local.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            with pytest.raises(RPCConnectionError) as exc_info:
                asyncio.run(
                    scheduler.create_engine("test/0", "test_engines.DummyEngine")
                )

            assert exc_info.value.worker_id == "test/0"
            assert exc_info.value.host == "127.0.0.1"
            assert exc_info.value.port == 8000

    def test_create_engine_timeout(self, scheduler, tmp_path):
        """Should raise EngineCreationError when request times out."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        # Mock aiohttp.ClientSession to raise timeout error
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.post = Mock(side_effect=TimeoutError("Request timeout"))

        with patch(
            "areal.infra.scheduler.local.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            with pytest.raises(EngineCreationError) as exc_info:
                asyncio.run(
                    scheduler.create_engine("test/0", "test_engines.DummyEngine")
                )

            assert "Request timed out" in str(exc_info.value)


class TestEngineMethodCalls:
    """Test calling methods on engines (sync and async)."""

    def test_call_engine_success(self, scheduler, tmp_path):
        """Should successfully call engine method synchronously."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        mock_response = create_mock_http_response(
            status_code=200, json_data={"result": 42}
        )

        with patch.object(requests, "post", return_value=mock_response):
            result = scheduler.call_engine("test/0", "compute", arg1=10, arg2=20)

            assert result == 42

    def test_call_engine_worker_not_found(self, scheduler):
        """Should raise WorkerNotFoundError when worker doesn't exist."""
        with pytest.raises(WorkerNotFoundError):
            scheduler.call_engine("nonexistent/0", "method")

    def test_call_engine_worker_died(self, scheduler, tmp_path):
        """Should raise WorkerFailedError when worker dies before call."""
        log_file = tmp_path / "test.log"
        log_file.write_text("Worker crashed\n")

        mock_proc = create_mock_process(is_alive=False, exit_code=1)
        worker = create_worker_info(process=mock_proc, log_file=str(log_file))
        scheduler._workers["test"] = [worker]

        with pytest.raises(WorkerFailedError):
            scheduler.call_engine("test/0", "method")

    def test_call_engine_method_error(self, scheduler, tmp_path):
        """Should raise EngineCallError when method call returns 400/500."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        mock_response = create_mock_http_response(
            status_code=400, json_data={"error": "Method 'nonexistent' not found"}
        )

        with patch.object(requests, "post", return_value=mock_response):
            with pytest.raises(EngineCallError) as exc_info:
                scheduler.call_engine("test/0", "nonexistent")

            assert "Method 'nonexistent' not found" in str(exc_info.value)

    @patch("areal.infra.scheduler.local.time.sleep")
    def test_call_engine_retry_on_503(self, mock_sleep, scheduler, tmp_path):
        """Should retry on 503 Service Unavailable."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        # First call returns 503, second call succeeds
        mock_response_503 = create_mock_http_response(status_code=503)
        mock_response_200 = create_mock_http_response(
            status_code=200, json_data={"result": "success"}
        )

        with patch.object(
            requests,
            "post",
            side_effect=[mock_response_503, mock_response_200],
        ):
            result = scheduler.call_engine("test/0", "method", max_retries=3)

            assert result == "success"
            assert mock_sleep.called

    @patch("areal.infra.scheduler.local.time.sleep")
    def test_call_engine_max_retries_exhausted(self, mock_sleep, scheduler, tmp_path):
        """Should raise EngineCallError after max retries."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        mock_response = create_mock_http_response(status_code=503)

        with patch.object(requests, "post", return_value=mock_response):
            with pytest.raises(EngineCallError) as exc_info:
                scheduler.call_engine("test/0", "method", max_retries=3)

            assert "Max retries exceeded" in str(
                exc_info.value
            ) or "Service unavailable" in str(exc_info.value)
            assert exc_info.value.attempt == 3

    @patch("areal.infra.scheduler.local.time.sleep")
    def test_call_engine_exponential_backoff(self, mock_sleep, scheduler, tmp_path):
        """Should use exponential backoff for retries."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        mock_response = create_mock_http_response(status_code=503)

        with patch.object(requests, "post", return_value=mock_response):
            try:
                scheduler.call_engine(
                    "test/0", "method", max_retries=3, retry_delay=1.0
                )
            except EngineCallError:
                pass

        # Verify exponential backoff: 1.0, 2.0
        sleep_calls = [call_args[0][0] for call_args in mock_sleep.call_args_list]
        assert sleep_calls[0] == 1.0  # First retry
        assert sleep_calls[1] == 2.0  # Second retry

    def test_async_call_engine_success(self, scheduler, tmp_path):
        """Should successfully call engine method asynchronously."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        # Mock aiohttp.ClientSession and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": 42})
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.post = Mock(return_value=mock_response)

        with patch(
            "areal.infra.scheduler.local.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            result = asyncio.run(
                scheduler.async_call_engine("test/0", "compute", arg1=10, arg2=20)
            )

            assert result == 42

    def test_async_call_engine_worker_not_found(self, scheduler):
        """Should raise WorkerNotFoundError when worker doesn't exist (async)."""
        with pytest.raises(WorkerNotFoundError):
            asyncio.run(scheduler.async_call_engine("nonexistent/0", "method"))

    def test_async_call_engine_retry_with_backoff(self, scheduler, tmp_path):
        """Should retry with exponential backoff in async mode."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        # First call returns 503, second call succeeds
        # Mock aiohttp.ClientSession and responses
        mock_response_503 = AsyncMock()
        mock_response_503.status = 503
        mock_response_503.__aenter__.return_value = mock_response_503
        mock_response_503.__aexit__.return_value = None

        mock_response_200 = AsyncMock()
        mock_response_200.status = 200
        mock_response_200.json = AsyncMock(return_value={"result": "success"})
        mock_response_200.__aenter__.return_value = mock_response_200
        mock_response_200.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.post = Mock(side_effect=[mock_response_503, mock_response_200])

        with patch(
            "areal.infra.scheduler.local.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            with patch("asyncio.sleep") as mock_async_sleep:
                result = asyncio.run(
                    scheduler.async_call_engine("test/0", "method", max_retries=3)
                )

                assert result == "success"
                assert mock_async_sleep.called


class TestFindWorkerById:
    """Test finding workers by ID."""

    def test_find_worker_by_id_success(self, scheduler, tmp_path):
        """Should find worker by ID."""
        worker1 = create_worker_info(
            worker_id="role1/0",
            role="role1",
            ports=["8000"],
            log_file=str(tmp_path / "role1.log"),
        )
        worker2 = create_worker_info(
            worker_id="role2/0",
            role="role2",
            ports=["8001"],
            log_file=str(tmp_path / "role2.log"),
        )

        scheduler._workers["role1"] = [worker1]
        scheduler._workers["role2"] = [worker2]

        found = scheduler._find_worker_by_id("role2/0")

        assert found is worker2
        assert found.worker.id == "role2/0"

    def test_find_worker_by_id_not_found(self, scheduler, tmp_path):
        """Should return None when worker ID is not found."""
        worker = create_worker_info(
            worker_id="role1/0", role="role1", log_file=str(tmp_path / "role1.log")
        )
        scheduler._workers["role1"] = [worker]

        found = scheduler._find_worker_by_id("nonexistent/99")

        assert found is None


class TestSchedulerCleanup:
    """Test scheduler cleanup and destructor."""

    def test_destructor_deletes_all_workers(self, scheduler, tmp_path):
        """Should delete all workers when scheduler is destroyed."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        with patch.object(scheduler, "delete_workers") as mock_delete:
            scheduler.__del__()

            mock_delete.assert_called_once()

    def test_destructor_handles_errors_gracefully(self, scheduler):
        """Should handle errors gracefully in destructor."""
        with patch.object(scheduler, "delete_workers", side_effect=Exception("Error")):
            # Should not raise
            scheduler.__del__()


class TestEdgeCases:
    """Test various edge cases and corner scenarios."""

    def test_gpu_counter_wraps_correctly(self, tmp_path):
        """Should correctly wrap GPU counter for round-robin allocation."""
        scheduler = create_scheduler(tmp_path, gpu_devices=[0, 1])

        # Allocate many times to ensure wrapping
        for i in range(10):
            gpus = scheduler._allocate_gpus(1)
            expected_gpu = i % 2
            assert gpus == [expected_gpu]

    def test_port_allocation_accumulates_correctly(self, tmp_path):
        """Should correctly accumulate allocated ports over multiple allocations."""
        with patch("areal.infra.scheduler.local.find_free_ports") as mock_find_ports:
            mock_find_ports.side_effect = [
                [8000, 8001],
                [8002, 8003],
                [8004, 8005, 8006],
            ]

            scheduler = create_scheduler(tmp_path)

            scheduler._allocate_ports(2)
            scheduler._allocate_ports(2)
            scheduler._allocate_ports(3)

            assert scheduler._allocated_ports == {
                8000,
                8001,
                8002,
                8003,
                8004,
                8005,
                8006,
            }

    @patch("areal.infra.scheduler.local.gethostip")
    @patch("areal.infra.scheduler.local.subprocess.Popen")
    @patch("areal.infra.scheduler.local.find_free_ports")
    def test_worker_id_format(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should create worker IDs in correct format (role/index)."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = [8000, 8001]

        mock_processes = []
        for i in range(5):
            mock_proc = Mock()
            mock_proc.pid = 1000 + i
            mock_proc.poll.return_value = None
            mock_processes.append(mock_proc)
        mock_popen.side_effect = mock_processes

        scheduler = create_scheduler(tmp_path)

        job = Job(replicas=5, role="worker")
        with patch.object(scheduler, "_configure_worker", return_value=None):
            worker_ids = scheduler.create_workers(job)

        assert worker_ids == [
            "worker/0",
            "worker/1",
            "worker/2",
            "worker/3",
            "worker/4",
        ]

        # Clean up workers while mock is still active
        scheduler.delete_workers(None)

    def test_empty_workers_dict_operations(self, tmp_path):
        """Should handle operations on empty workers dictionary gracefully."""
        scheduler = create_scheduler(tmp_path)

        # Delete all workers when none exist
        scheduler.delete_workers(None)

        # Check health of non-existent role
        scheduler._check_worker_health("nonexistent")

        # Find worker by ID when no workers exist
        assert scheduler._find_worker_by_id("any/0") is None

    def test_concurrent_gpu_allocations(self, tmp_path):
        """Should handle concurrent GPU allocations correctly."""
        scheduler = create_scheduler(tmp_path, gpu_devices=[0, 1, 2])

        # Simulate multiple workers requesting GPUs simultaneously
        results = []
        for _ in range(6):
            gpus = scheduler._allocate_gpus(1)
            results.append(gpus[0])

        # Should cycle through GPUs in order
        assert results == [0, 1, 2, 0, 1, 2]

    def test_log_directory_with_special_characters(self, tmp_path):
        """Should handle log directory paths with special characters."""
        log_dir = tmp_path / "logs with spaces" / "special-chars_123"
        scheduler = create_scheduler(tmp_path, log_dir=str(log_dir))

        assert log_dir.exists()
        assert scheduler.log_dir == log_dir


class TestColocationBehavior:
    """Test colocation-specific behavior for worker reuse."""

    @patch("areal.infra.scheduler.local.gethostip")
    @patch("areal.infra.scheduler.local.subprocess.Popen")
    @patch("areal.infra.scheduler.local.find_free_ports")
    def test_get_workers_for_colocated_role_delegates_to_target(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should return target role's workers when getting colocated role workers."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = [8000, 8001]

        mock_proc = Mock()
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        scheduler = create_scheduler(tmp_path, gpu_devices=[0, 1])

        # Create target workers
        actor_job = Job(replicas=1, role="actor")
        with patch.object(scheduler, "_configure_worker", return_value=None):
            scheduler.create_workers(actor_job)

        # Create colocated role with fork=False (reuses existing workers)
        ref_job = Job(
            replicas=1,
            role="ref",
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="actor", fork=False
            ),
        )
        scheduler.create_workers(ref_job)

        # Get workers for colocated role should return target role's workers
        with patch.object(scheduler, "_is_worker_ready", return_value=True):
            workers = scheduler.get_workers("ref")

        assert len(workers) == 1
        assert workers[0].id == "actor/0"

        # Clean up workers while mock is still active
        scheduler.delete_workers(None)

    @patch("areal.infra.scheduler.local.gethostip")
    @patch("areal.infra.scheduler.local.subprocess.Popen")
    @patch("areal.infra.scheduler.local.find_free_ports")
    def test_delete_colocated_role_does_not_kill_processes(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should only remove mapping when deleting colocated role, not kill processes."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = [8000, 8001]

        mock_proc = Mock()
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        scheduler = create_scheduler(tmp_path)

        # Create target workers
        actor_job = Job(replicas=1, role="actor")
        with patch.object(scheduler, "_configure_worker", return_value=None):
            scheduler.create_workers(actor_job)

        # Create colocated role with fork=False (reuses existing workers)
        ref_job = Job(
            replicas=1,
            role="ref",
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="actor", fork=False
            ),
        )
        scheduler.create_workers(ref_job)

        # Verify colocation is set up
        assert "ref" in scheduler._colocated_roles

        # Delete colocated role
        with patch.object(scheduler, "_cleanup_workers") as mock_cleanup:
            scheduler.delete_workers("ref")

            # _cleanup_workers should NOT be called for colocated roles
            mock_cleanup.assert_not_called()

        # Colocation mapping should be removed
        assert "ref" not in scheduler._colocated_roles

        # Target role's workers should still exist
        assert "actor" in scheduler._workers
        assert len(scheduler._workers["actor"]) == 1

        # Clean up workers while mock is still active
        scheduler.delete_workers(None)

    @patch("areal.infra.scheduler.local.gethostip")
    @patch("areal.infra.scheduler.local.subprocess.Popen")
    @patch("areal.infra.scheduler.local.find_free_ports")
    def test_colocation_replica_mismatch_raises_error(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should raise error when colocated role has different replica count."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = [8000, 8001]

        mock_processes = []
        for i in range(2):
            mock_proc = Mock()
            mock_proc.pid = 1000 + i
            mock_proc.poll.return_value = None
            mock_processes.append(mock_proc)
        mock_popen.side_effect = mock_processes

        scheduler = create_scheduler(tmp_path, gpu_devices=[0, 1])

        # Create target workers with 2 replicas
        actor_job = Job(replicas=2, role="actor")
        with patch.object(scheduler, "_configure_worker", return_value=None):
            scheduler.create_workers(actor_job)

        # Try to create colocated role with different replica count
        ref_job = Job(
            replicas=1,  # Mismatch!
            role="ref",
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="actor"
            ),
        )
        with pytest.raises(WorkerCreationError) as exc_info:
            scheduler.create_workers(ref_job)

        assert "replica count" in str(exc_info.value).lower()

        # Clean up workers while mock is still active
        scheduler.delete_workers(None)

    @patch("areal.infra.scheduler.local.gethostip")
    @patch("areal.infra.scheduler.local.subprocess.Popen")
    @patch("areal.infra.scheduler.local.find_free_ports")
    def test_colocation_target_not_found_raises_error(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should raise error when colocation target role doesn't exist."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = [8000, 8001]

        scheduler = create_scheduler(tmp_path)

        # Try to create colocated role with non-existent target
        ref_job = Job(
            replicas=1,
            role="ref",
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="nonexistent"
            ),
        )
        with pytest.raises(WorkerNotFoundError):
            scheduler.create_workers(ref_job)


class TestForkColocationBehavior:
    """Test fork colocation behavior for spawning new worker processes.

    These tests use real subprocesses and RPC servers to verify fork functionality.
    """

    @pytest.fixture
    def rpc_server_process(self, tmp_path):
        """Start a real RPC server process for testing.

        Returns tuple of (process, host, port).
        """
        import socket
        import subprocess

        host = "127.0.0.1"

        # Try to find a free port and start the server
        # Retry a few times in case of port collision
        proc = None
        port = None
        last_error = None
        for _ in range(5):
            # Find a free port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port = s.getsockname()[1]

            # Start RPC server
            cmd = [
                "python",
                "-m",
                "areal.infra.rpc.rpc_server",
                "--host",
                host,
                "--port",
                str(port),
                "--experiment-name",
                "test_fork_exp",
                "--trial-name",
                "test_fork_trial",
                "--role",
                "actor",
                "--worker-index",
                "0",
                "--fileroot",
                str(tmp_path),
            ]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            # Wait for server to be ready
            deadline = time.time() + 15
            server_ready = False
            while time.time() < deadline:
                try:
                    resp = requests.get(f"http://{host}:{port}/health", timeout=2)
                    if resp.status_code == 200:
                        server_ready = True
                        break
                except (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                ):
                    pass
                # Check if process died
                if proc.poll() is not None:
                    stdout = proc.stdout.read().decode() if proc.stdout else ""
                    last_error = (
                        f"Process died with code {proc.returncode}: {stdout[:500]}"
                    )
                    break
                time.sleep(0.5)

            if server_ready:
                break
            else:
                # Kill the failed process and retry
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
                proc = None
        else:
            raise RuntimeError(
                f"RPC server failed to start after 5 attempts on port {port}. "
                f"Last error: {last_error}"
            )

        yield proc, host, port

        # Cleanup
        kill_process_tree(proc.pid, timeout=3, graceful=True)

    def test_fork_endpoint_spawns_new_process(self, rpc_server_process):
        """Should spawn a new RPC server process when /fork is called."""
        _, host, port = rpc_server_process

        alloc_resp = requests.post(
            f"http://{host}:{port}/alloc_ports",
            json={"count": 1},
            timeout=10,
        )
        assert alloc_resp.status_code == 200
        child_port = alloc_resp.json()["ports"][0]

        raw_cmd = [
            sys.executable,
            "-m",
            "areal.infra.rpc.rpc_server",
            "--host",
            "0.0.0.0",
            "--port",
            str(child_port),
            "--experiment-name",
            "test_fork_exp",
            "--trial-name",
            "test_fork_trial",
            "--role",
            "ref",
            "--worker-index",
            "0",
        ]

        response = requests.post(
            f"http://{host}:{port}/fork",
            json={"role": "ref", "worker_index": 0, "raw_cmd": raw_cmd},
            timeout=60,
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert "host" in result
        assert "pid" in result

        forked_pid = result["pid"]

        # Verify new process exists
        assert psutil.pid_exists(forked_pid)

        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                forked_response = requests.get(
                    f"http://{result['host']}:{child_port}/health", timeout=2
                )
                if forked_response.status_code == 200:
                    break
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ):
                pass
            time.sleep(0.5)
        else:
            pytest.fail("Forked worker did not become ready")

    def test_forked_worker_inherits_environment(self, rpc_server_process):
        """Forked worker should inherit environment variables from parent."""
        _, host, port = rpc_server_process

        alloc_resp = requests.post(
            f"http://{host}:{port}/alloc_ports",
            json={"count": 1},
            timeout=10,
        )
        assert alloc_resp.status_code == 200
        child_port = alloc_resp.json()["ports"][0]

        raw_cmd = [
            sys.executable,
            "-m",
            "areal.infra.rpc.rpc_server",
            "--host",
            "0.0.0.0",
            "--port",
            str(child_port),
            "--experiment-name",
            "test_fork_exp",
            "--trial-name",
            "test_fork_trial",
            "--role",
            "ref",
            "--worker-index",
            "0",
        ]

        response = requests.post(
            f"http://{host}:{port}/fork",
            json={"role": "ref", "worker_index": 0, "raw_cmd": raw_cmd},
            timeout=60,
        )

        assert response.status_code == 200
        result = response.json()

        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                forked_response = requests.get(
                    f"http://{result['host']}:{child_port}/health", timeout=2
                )
                if forked_response.status_code == 200:
                    break
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ):
                pass
            time.sleep(0.5)
        else:
            pytest.fail("Forked worker did not become ready")

    def test_create_forked_workers_via_scheduler(self, tmp_path):
        """LocalScheduler should create forked workers through /fork endpoint."""
        import socket
        import subprocess

        # Find two free ports
        ports = []
        for _ in range(2):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                ports.append(s.getsockname()[1])

        host = "127.0.0.1"

        # Start RPC server manually
        cmd = [
            "python",
            "-m",
            "areal.infra.rpc.rpc_server",
            "--host",
            host,
            "--port",
            str(ports[0]),
            "--experiment-name",
            "test_fork_exp",
            "--trial-name",
            "test_fork_trial",
            "--role",
            "actor",
            "--worker-index",
            "0",
            "--fileroot",
            str(tmp_path),
        ]

        server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        try:
            # Wait for server to be ready
            deadline = time.time() + 30
            while time.time() < deadline:
                try:
                    resp = requests.get(f"http://{host}:{ports[0]}/health", timeout=1)
                    if resp.status_code == 200:
                        break
                except (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                ):
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError("RPC server failed to start")

            # Create scheduler and manually add the worker
            scheduler = create_scheduler(
                tmp_path,
                experiment_name="test_fork_exp",
                trial_name="test_fork_trial",
            )

            # Manually register the actor worker (simulating what create_workers does)
            actor_worker = Worker(
                id="actor/0",
                ip=host,
                worker_ports=[str(ports[0])],
                engine_ports=[],
            )
            actor_worker_info = WorkerInfo(
                worker=actor_worker,
                process=server_proc,
                role="actor",
                gpu_devices=[0],
                created_at=time.time(),
                log_file=str(tmp_path / "actor.log"),
                env_vars={},
            )
            scheduler._workers["actor"] = [actor_worker_info]

            # Now create forked workers using fork=True
            ref_job = Job(
                replicas=1,
                role="ref",
                scheduling_strategy=SchedulingStrategy(
                    type=SchedulingStrategyType.colocation, target="actor", fork=True
                ),
            )

            worker_ids = scheduler.create_workers(ref_job)

            # Verify forked workers were created
            assert worker_ids == ["ref/0"]
            assert "ref" in scheduler._workers
            assert len(scheduler._workers["ref"]) == 1

            # Verify forked role is tracked in _colocated_roles
            assert "ref" in scheduler._colocated_roles
            assert scheduler._colocated_roles["ref"] == "actor"

            # Verify forked worker has process=None (managed by parent)
            forked_worker = scheduler._workers["ref"][0]
            assert forked_worker.process is None

            # Verify forked worker is a real, responsive server
            forked_response = requests.get(
                f"http://{forked_worker.worker.ip}:{forked_worker.worker.worker_ports[0]}/health",
                timeout=5,
            )
            assert forked_response.status_code == 200

            # Cleanup via scheduler
            scheduler.delete_workers(None)

        finally:
            # Ensure cleanup
            kill_process_tree(server_proc.pid, timeout=3, graceful=True)

    def test_fork_replica_mismatch_raises_error(self, tmp_path):
        """Should raise error when forked role has different replica count."""
        import socket
        import subprocess

        # Find ports for 2 workers
        ports = []
        for _ in range(4):  # 2 ports per worker
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                ports.append(s.getsockname()[1])

        host = "127.0.0.1"
        server_procs = []

        try:
            # Start 2 RPC servers for actor role
            for i in range(2):
                cmd = [
                    "python",
                    "-m",
                    "areal.infra.rpc.rpc_server",
                    "--host",
                    host,
                    "--port",
                    str(ports[i]),
                    "--experiment-name",
                    "test_fork_exp",
                    "--trial-name",
                    "test_fork_trial",
                    "--role",
                    "actor",
                    "--worker-index",
                    str(i),
                    "--fileroot",
                    str(tmp_path),
                ]

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                server_procs.append(proc)

            # Wait for servers to be ready
            for i in range(2):
                deadline = time.time() + 30
                while time.time() < deadline:
                    try:
                        resp = requests.get(
                            f"http://{host}:{ports[i]}/health", timeout=1
                        )
                        if resp.status_code == 200:
                            break
                    except (
                        requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                    ):
                        pass
                    time.sleep(0.2)
                else:
                    raise RuntimeError(f"RPC server {i} failed to start")

            # Create scheduler and manually add workers
            scheduler = create_scheduler(
                tmp_path,
                gpu_devices=[0, 1],
                experiment_name="test_fork_exp",
                trial_name="test_fork_trial",
            )

            # Manually register actor workers
            scheduler._workers["actor"] = []
            for i in range(2):
                actor_worker = Worker(
                    id=f"actor/{i}",
                    ip=host,
                    worker_ports=[str(ports[i])],
                    engine_ports=[],
                )
                actor_worker_info = WorkerInfo(
                    worker=actor_worker,
                    process=server_procs[i],
                    role="actor",
                    gpu_devices=[i],
                    created_at=time.time(),
                    log_file=str(tmp_path / f"actor_{i}.log"),
                    env_vars={},
                )
                scheduler._workers["actor"].append(actor_worker_info)

            # Try to create forked role with different replica count
            ref_job = Job(
                replicas=1,  # Mismatch - actor has 2 replicas!
                role="ref",
                scheduling_strategy=SchedulingStrategy(
                    type=SchedulingStrategyType.colocation, target="actor", fork=True
                ),
            )

            with pytest.raises(WorkerCreationError) as exc_info:
                scheduler.create_workers(ref_job)

            assert "replica count" in str(exc_info.value).lower()

        finally:
            # Cleanup all server processes
            for proc in server_procs:
                kill_process_tree(proc.pid, timeout=3, graceful=True)

    def test_fork_target_not_found_raises_error(self, tmp_path):
        """Should raise error when fork target role doesn't exist."""
        scheduler = create_scheduler(tmp_path)

        # Try to create forked role with non-existent target
        ref_job = Job(
            replicas=1,
            role="ref",
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="nonexistent", fork=True
            ),
        )

        with pytest.raises(WorkerNotFoundError):
            scheduler.create_workers(ref_job)

    def test_delete_forked_workers_cleans_up_tracking(self, tmp_path):
        """Should remove forked role from tracking when deleted."""
        import socket
        import subprocess

        # Find a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        host = "127.0.0.1"

        # Start RPC server
        cmd = [
            "python",
            "-m",
            "areal.infra.rpc.rpc_server",
            "--host",
            host,
            "--port",
            str(port),
            "--experiment-name",
            "test_fork_exp",
            "--trial-name",
            "test_fork_trial",
            "--role",
            "actor",
            "--worker-index",
            "0",
            "--fileroot",
            str(tmp_path),
        ]

        server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        try:
            # Wait for server to be ready
            deadline = time.time() + 30
            while time.time() < deadline:
                try:
                    resp = requests.get(f"http://{host}:{port}/health", timeout=1)
                    if resp.status_code == 200:
                        break
                except (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                ):
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError("RPC server failed to start")

            # Create scheduler and manually add the worker
            scheduler = create_scheduler(
                tmp_path,
                experiment_name="test_fork_exp",
                trial_name="test_fork_trial",
            )

            # Manually register the actor worker
            actor_worker = Worker(
                id="actor/0",
                ip=host,
                worker_ports=[str(port)],
                engine_ports=[],
            )
            actor_worker_info = WorkerInfo(
                worker=actor_worker,
                process=server_proc,
                role="actor",
                gpu_devices=[0],
                created_at=time.time(),
                log_file=str(tmp_path / "actor.log"),
                env_vars={},
            )
            scheduler._workers["actor"] = [actor_worker_info]

            # Create forked workers
            ref_job = Job(
                replicas=1,
                role="ref",
                scheduling_strategy=SchedulingStrategy(
                    type=SchedulingStrategyType.colocation, target="actor", fork=True
                ),
            )

            scheduler.create_workers(ref_job)

            # Verify forked role exists
            assert "ref" in scheduler._colocated_roles
            assert "ref" in scheduler._workers

            # Delete forked role
            scheduler.delete_workers("ref")

            # Verify forked role is removed
            assert "ref" not in scheduler._colocated_roles
            assert "ref" not in scheduler._workers

            # Target role should still exist
            assert "actor" in scheduler._workers

        finally:
            # Cleanup
            kill_process_tree(server_proc.pid, timeout=3, graceful=True)
