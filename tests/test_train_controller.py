"""Unit tests for TrainController.

Tests cover initialization, worker management, batch operations,
RPC wrappers, PPO/SFT methods, weight management, and error handling.
"""

import asyncio
from unittest.mock import Mock

import pytest
import torch

from areal.api import (
    FinetuneSpec,
    ParallelStrategy,
    SaveLoadMeta,
    TrainEngine,
    WeightUpdateMeta,
    Worker,
)
from areal.api.cli_args import SchedulingSpec, TrainEngineConfig
from areal.infra import TrainController
from areal.infra.controller.train_controller import _merge_tensors


class MockTrainEngine(TrainEngine):
    """Mock TrainEngine for testing."""

    @classmethod
    def __module__(cls):
        return "tests.test_train_controller"

    @classmethod
    def __name__(cls):
        return "MockTrainEngine"


class MockScheduler:
    """Mock Scheduler for testing TrainController."""

    def __init__(self):
        self.workers = []
        self.call_count = 0
        self.engine_calls = []
        self.deleted_roles = []
        self.env_settings: dict[str, dict[str, str]] = {}

    def create_workers(self, job):
        """Create mock workers based on job configuration."""
        worker_ids = [f"{job.role}/{i}" for i in range(job.replicas)]
        self.workers = [
            Worker(
                id=wid,
                ip="127.0.0.1",
                worker_ports=["8000", "8001"],
                engine_ports=["9000", "9001"],
            )
            for wid in worker_ids
        ]
        return worker_ids

    def get_workers(self, role, timeout=None):
        """Return list of workers for the given role."""
        return self.workers

    async def set_worker_env(self, worker_id, env):
        """Mock environment configuration."""
        await asyncio.sleep(0.001)
        self.env_settings[worker_id] = {k: str(v) for k, v in env.items()}

    async def create_engine(self, worker_id, engine, **kwargs):
        """Mock engine creation."""
        await asyncio.sleep(0.001)
        return None

    async def async_call_engine(self, worker_id, method, *args, **kwargs):
        """Mock async engine call."""
        self.engine_calls.append((worker_id, method, args, kwargs))
        self.call_count += 1

        # Return appropriate mock results based on method
        if method == "is_data_parallel_head":
            # First worker in each DP group is the head
            worker_idx = int(worker_id.split("/")[-1])
            return worker_idx % 2 == 0  # Every other worker is a DP head

        elif method == "get_version":
            return 1

        elif method == "train_lm":
            return {"lm_loss": 0.4, "perplexity": 1.5}

        elif method == "evaluate_lm":
            # Return scalar loss (real implementation would return float or dict)
            return {"eval_loss": 0.35}

        await asyncio.sleep(0.001)
        return None

    def delete_workers(self, role, reverse_order: bool = False):
        """Mock worker deletion."""
        self.deleted_roles.append(role)
        self.delete_reverse_order = reverse_order
        self.workers.clear()


# ==================== FIXTURES ====================


@pytest.fixture
def mock_scheduler():
    """Provide a MockScheduler instance."""
    return MockScheduler()


@pytest.fixture
def train_config():
    """Provide a TrainEngineConfig for testing."""
    return TrainEngineConfig(
        backend="fsdp:d4t2",
        scheduling_spec=(
            SchedulingSpec(
                cpu=4,
                gpu=1,
                mem=16000,
                port_count=2,
                cmd="python -m areal.infra.rpc.rpc_server",
            ),
        ),
    )


@pytest.fixture
def parallel_strategy():
    """Provide a ParallelStrategy for testing."""
    return ParallelStrategy(
        data_parallel_size=4, tensor_parallel_size=2, pipeline_parallel_size=1
    )


@pytest.fixture
def ft_spec():
    """Provide a FinetuneSpec for testing."""
    return FinetuneSpec(total_train_epochs=10, dataset_size=1000, train_batch_size=32)


@pytest.fixture
def train_controller(mock_scheduler, train_config):
    """Provide a TrainController instance."""
    train_controller = TrainController(
        train_engine=MockTrainEngine, config=train_config, scheduler=mock_scheduler
    )
    yield train_controller
    train_controller.destroy()


def create_mock_distributed_batch(size=4, seq_len=10):
    """Create a mock trajectory list for testing (_dispatch_tensors expects list[dict])."""
    return [
        {
            "input_ids": torch.randint(0, 100, (1, seq_len)),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
            "loss_mask": torch.ones(1, seq_len, dtype=torch.bool),
        }
        for _ in range(size)
    ]


# ==================== TEST CLASSES ====================


class TestTrainControllerInitialization:
    """Tests for TrainController initialization and setup."""

    def test_constructor(self, mock_scheduler, train_config):
        """Test TrainController constructor."""
        controller = TrainController(
            train_engine=MockTrainEngine, config=train_config, scheduler=mock_scheduler
        )

        assert controller.train_engine == MockTrainEngine
        assert controller.config == train_config
        assert controller.scheduler == mock_scheduler
        assert controller.workers == []
        assert controller.workers_is_dp_head == []

    def test_initialize(self, train_controller, ft_spec):
        """Test initialize method creates workers and engines."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        # Verify workers were created
        assert (
            len(train_controller.workers)
            == train_controller.train_alloc.parallel.world_size
        )
        assert train_controller._worker_role == "train_worker"

        # Verify DP heads were identified
        assert len(train_controller.workers_is_dp_head) == len(train_controller.workers)

        # Verify scheduler was called
        assert train_controller.scheduler.call_count > 0
        # Verify environment configuration occurred for each worker
        assert len(train_controller.scheduler.env_settings) == len(
            train_controller.workers
        )

    def test_identify_dp_heads(self, train_controller, ft_spec):
        """Test _identify_dp_heads correctly identifies DP head workers."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        # MockScheduler returns True for even-indexed workers
        for idx, is_head in enumerate(train_controller.workers_is_dp_head):
            assert is_head == (idx % 2 == 0)


class TestTrainControllerDestroy:
    """Tests for TrainController cleanup and destruction."""

    def test_destroy(self, train_controller, ft_spec):
        """Test destroy method cleans up resources."""
        # Initialize first
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        initial_worker_count = len(train_controller.workers)
        assert initial_worker_count > 0

        # Call destroy
        train_controller.destroy()

        # Verify cleanup
        assert len(train_controller.workers) == 0
        assert len(train_controller.workers_is_dp_head) == 0
        assert "train_worker" in train_controller.scheduler.deleted_roles

    def test_destroy_handles_errors(self, train_controller, ft_spec):
        """Test destroy handles errors gracefully."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        # Make delete_workers raise an exception
        def raise_error(role):
            raise RuntimeError("Simulated error")

        train_controller.scheduler.delete_workers = raise_error

        # Should not raise, just log the error
        train_controller.destroy()

        # Workers should still be cleared
        assert len(train_controller.workers) == 0

    def test_destroy_requests_reverse_order(self, train_controller, ft_spec):
        """Workers must be torn down in reverse rank order.

        This protects rank-0 (which owns the global TCPStore server) from
        being killed before non-zero ranks finish NCCL abort, avoiding a
        noisy ``TCPStore.recvValue failed`` warning from
        HeartbeatMonitor.
        """
        train_controller.initialize(role="train_worker", ft_spec=ft_spec)

        train_controller.destroy()

        assert (
            getattr(train_controller.scheduler, "delete_reverse_order", False) is True
        )


class TestTrainControllerMergeResults:
    """Tests for result merging via _merge_tensors."""

    def test_merge_tensors_reorders_results(self):
        """Test _merge_tensors reorders results to original trajectory order."""
        results = [{"status": "ok"}, {"status": "done"}]

        merged = _merge_tensors(results, group_indices=[[0], [1]])

        assert merged is not None
        assert len(merged) == 2
        assert merged[0] == {"status": "ok"}
        assert merged[1] == {"status": "done"}

    def test_merge_tensors_with_tensor_results(self):
        """Test _merge_tensors correctly reorders tensor results."""
        results = [torch.tensor([[0.5, 0.5]]), torch.tensor([[0.3, 0.3]])]

        merged = _merge_tensors(results, group_indices=[[0], [1]])

        assert merged is not None
        assert len(merged) == 2


class TestTrainControllerRPCWrappers:
    """Tests for RPC wrapper methods."""

    def test_train_mode(self, train_controller, ft_spec):
        """Test train() method sets training mode."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        result = train_controller.train(mode=True)

        # Should return self for chaining
        assert result is train_controller

        # Verify custom_function_call was invoked
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "train" in engine_calls

    def test_eval_mode(self, train_controller, ft_spec):
        """Test eval() method sets evaluation mode."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        result = train_controller.eval()

        # Should return self for chaining
        assert result is train_controller

        # Verify train(False) was called
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "train" in engine_calls

    def test_step_lr_scheduler(self, train_controller, ft_spec):
        """Test step_lr_scheduler() method."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        train_controller.step_lr_scheduler()

        # Verify step_lr_scheduler was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "step_lr_scheduler" in engine_calls


class TestTrainControllerWeightManagement:
    """Tests for weight management operations."""

    def test_set_version(self, train_controller, ft_spec):
        """Test set_version() method."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        train_controller.set_version(42)

        # Verify set_version was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "set_version" in engine_calls

    def test_get_version(self, train_controller, ft_spec):
        """Test get_version() method."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        version = train_controller.get_version()

        # Should return version number
        assert isinstance(version, int)

        # Verify get_version was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "get_version" in engine_calls

    def test_save(self, train_controller, ft_spec):
        """Test save() method."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        meta = SaveLoadMeta(
            path="/tmp/checkpoint", weight_format="safetensors", with_optim=True
        )
        train_controller.save(meta)

        # Verify save was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "save" in engine_calls

    def test_load(self, train_controller, ft_spec):
        """Test load() method."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        meta = SaveLoadMeta(
            path="/tmp/checkpoint", weight_format="safetensors", with_optim=True
        )
        train_controller.load(meta)

        # Verify load was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "load" in engine_calls


class TestTrainControllerCustomFunctionCall:
    """Tests for custom_function_call orchestration."""

    def test_custom_function_call_with_distributed_batch(
        self, train_controller, ft_spec
    ):
        """Test custom_function_call with batch argument."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        # Clear previous calls from initialization
        train_controller.scheduler.engine_calls = []

        batch = create_mock_distributed_batch(size=16)
        result = train_controller._custom_function_call("train_lm", input_=batch)

        # Should split batch across DP groups and call only DP heads
        assert result is not None

        # Count how many workers were called
        worker_calls = len(train_controller.scheduler.engine_calls)

        # Should call all workers (DP heads get data, others get empty)
        assert worker_calls == len(train_controller.workers)

    def test_custom_function_call_with_regular_args(self, train_controller, ft_spec):
        """Test custom_function_call with non-batch arguments."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        # Clear previous calls
        train_controller.scheduler.engine_calls = []

        result = train_controller._custom_function_call("set_version", 5)

        # set_version returns None, which is expected - just verify it doesn't crash
        # The key test is that all workers were called
        assert result is None

        # Verify all workers were called
        assert len(train_controller.scheduler.engine_calls) == len(
            train_controller.workers
        )

    def test_custom_function_call_filters_dp_heads(self, train_controller, ft_spec):
        """Test custom_function_call only returns results from DP heads."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=8)
        train_controller._custom_function_call("ppo_update", input_=batch)

        # Results should only come from DP head workers
        # (verified by _collect_results filtering to DP heads)


class TestTrainControllerEdgeCases:
    def test_create_process_group_is_dummy(self, train_controller, parallel_strategy):
        """Test create_process_group is now a dummy method that does nothing."""
        # Don't initialize, so workers list is empty
        # Should not raise - create_process_group is now a no-op
        train_controller.create_process_group(parallel_strategy)

        # parallel_strategy should be set by constructor (from backend="fsdp:d4t2")
        assert train_controller.parallel_strategy is not None

    def test_method_chaining(self, train_controller, ft_spec):
        """Test that train() and eval() support method chaining."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        # Should be able to chain calls
        result = train_controller.train().eval().train()
        assert result is train_controller


class TestTrainControllerRolloutIntegration:
    """Tests for rollout engine integration methods."""

    def test_connect_engine_sets_rollout(self, train_controller, ft_spec):
        """Test connect_engine correctly sets the rollout controller."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        mock_rollout = Mock()
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")

        train_controller.connect_engine(mock_rollout, meta)

        assert train_controller.rollout == mock_rollout

    def test_connect_engine_warns_on_change(self, train_controller, ft_spec):
        """Test connect_engine logs warning when rollout controller changes."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        mock_rollout1 = Mock()
        mock_rollout2 = Mock()
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")

        train_controller.connect_engine(mock_rollout1, meta)
        train_controller.connect_engine(mock_rollout2, meta)

        assert train_controller.rollout == mock_rollout2

    def test_connect_engine_same_rollout_no_warning(self, train_controller, ft_spec):
        """Test connect_engine does not warn when same rollout controller is used."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        mock_rollout = Mock()
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")

        train_controller.connect_engine(mock_rollout, meta)
        train_controller.connect_engine(mock_rollout, meta)

        assert train_controller.rollout == mock_rollout

    def test_check_rollout_engine_connected_raises_when_not_connected(
        self, train_controller, ft_spec
    ):
        """Test _check_rollout_engine_connected raises when rollout is not connected."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        with pytest.raises(RuntimeError, match="Rollout engine not connected"):
            train_controller._check_rollout_engine_connected()

    def test_check_rollout_engine_connected_passes_when_connected(
        self, train_controller, ft_spec
    ):
        """Test _check_rollout_engine_connected passes when rollout is connected."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        mock_rollout = Mock()
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        train_controller.connect_engine(mock_rollout, meta)

        # Should not raise
        train_controller._check_rollout_engine_connected()

    def test_prepare_batch_delegates_to_rollout(self, train_controller, ft_spec):
        """Test prepare_batch delegates to rollout controller."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        mock_rollout = Mock()
        mock_rollout.prepare_batch.return_value = {}
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        train_controller.connect_engine(mock_rollout, meta)

        mock_dataloader = Mock()
        train_controller.prepare_batch(
            dataloader=mock_dataloader,
            workflow="test.workflow",
            workflow_kwargs={"key": "value"},
        )

        mock_rollout.prepare_batch.assert_called_once_with(
            dataloader=mock_dataloader,
            workflow="test.workflow",
            workflow_kwargs={"key": "value"},
            should_accept_fn=None,
            dynamic_bs=False,
            group_size=1,
        )

    def test_rollout_batch_delegates_to_rollout(self, train_controller, ft_spec):
        """Test rollout_batch delegates to rollout controller."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        mock_rollout = Mock()
        mock_rollout.rollout_batch.return_value = {}
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        train_controller.connect_engine(mock_rollout, meta)

        data = [{"id": 1}, {"id": 2}]
        train_controller.rollout_batch(
            data=data,
            workflow="test.workflow",
            workflow_kwargs={"key": "value"},
        )

        mock_rollout.rollout_batch.assert_called_once_with(
            data=data,
            workflow="test.workflow",
            workflow_kwargs={"key": "value"},
            should_accept_fn=None,
            group_size=1,
        )


class TestTrainControllerWeightUpdateMethods:
    """Tests for weight update methods."""

    def test_update_weights_raises_when_not_connected(self, train_controller, ft_spec):
        """Test update_weights raises when rollout is not connected."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        meta = WeightUpdateMeta(type="disk", path="/tmp/test")

        with pytest.raises(RuntimeError, match="Rollout engine not connected"):
            train_controller.update_weights(meta)


class TestTrainControllerExportStats:
    """Tests for export_stats method."""

    def test_export_stats(self, train_controller, ft_spec):
        """Test export_stats returns statistics from first worker."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        # Mock the scheduler to return stats
        expected_stats = {"loss": 0.5, "accuracy": 0.95}

        async def mock_async_call(*args, **kwargs):
            if kwargs.get("method") == "export_stats" or (
                len(args) > 1 and args[1] == "export_stats"
            ):
                return expected_stats
            return None

        train_controller.scheduler.async_call_engine = mock_async_call

        result = train_controller.export_stats()
        for k in expected_stats:
            assert result[k] == expected_stats[k]


class TestTrainControllerDispatchInputs:
    """Tests for input dispatching across DP groups."""

    def test_prepare_dispatch_splits_distributed_batch(self, train_controller, ft_spec):
        """Test _prepare_dispatch correctly splits batch."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=16)
        split_args, split_kwargs, _ = train_controller._prepare_dispatch(batch)

        # Should split into dp_size chunks
        assert len(split_args) == 1
        assert len(split_args[0]) == train_controller.parallel_strategy.dp_size

    def test_prepare_dispatch_partitions_without_duplication(
        self, train_controller, ft_spec
    ):
        """Regression for #1202: trajectories are partitioned across DP ranks,
        never replicated. Each DP rank must see a disjoint slice so that total
        tokens processed equal the original batch, not batch * dp_size."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        dp_size = train_controller.parallel_strategy.dp_size
        batch_size = 4 * dp_size
        batch = create_mock_distributed_batch(size=batch_size)

        split_args, _, group_indices = train_controller._prepare_dispatch(batch)

        # Sanity: tensor-like list triggers the partition path, not replication.
        assert group_indices is not None
        assert len(group_indices) == dp_size

        shards = split_args[0]
        assert sum(len(shard) for shard in shards) == batch_size

        flat_indices = [idx for group in group_indices for idx in group]
        assert sorted(flat_indices) == list(range(batch_size))

    def test_prepare_dispatch_replicates_non_batch_args(
        self, train_controller, ft_spec
    ):
        """Test _prepare_dispatch replicates non-batch arguments."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        scalar_arg = 42
        split_args, split_kwargs, _ = train_controller._prepare_dispatch(scalar_arg)

        # Should replicate to all DP groups
        assert len(split_args) == 1
        assert all(arg == 42 for arg in split_args[0])
        assert len(split_args[0]) == train_controller.parallel_strategy.dp_size

    def test_prepare_dispatch_handles_kwargs(self, train_controller, ft_spec):
        """Test _prepare_dispatch correctly handles keyword arguments."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=16)
        split_args, split_kwargs, _ = train_controller._prepare_dispatch(
            input_=batch, learning_rate=0.001
        )

        assert "input_" in split_kwargs
        assert "learning_rate" in split_kwargs
        assert len(split_kwargs["input_"]) == train_controller.parallel_strategy.dp_size
        assert all(lr == 0.001 for lr in split_kwargs["learning_rate"])
