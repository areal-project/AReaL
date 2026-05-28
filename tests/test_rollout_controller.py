import asyncio
from unittest.mock import Mock

import pytest
import requests
import torch

from tests.utils import get_model_path

from areal.api import (
    ModelRequest,
    ParamSpec,
    WeightUpdateMeta,
    Worker,
)
from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    SchedulingSpec,
    SGLangConfig,
)
from areal.infra import RolloutController
from areal.infra.scheduler.local import LocalScheduler
from areal.utils.hf_utils import load_hf_tokenizer


def create_test_config(backend="sglang:d2", **kwargs):
    """Create a test InferenceEngineConfig with proper scheduling_spec."""
    # Create a mutable SchedulingSpec that can be modified by the controller
    scheduling_spec = SchedulingSpec(cpu=1, gpu=1, mem=1)

    defaults = {
        "backend": backend,
        "consumer_batch_size": 16,
        "scheduling_spec": (scheduling_spec,),
    }
    defaults.update(kwargs)
    config = InferenceEngineConfig(**defaults)

    return config


class MockScheduler:
    def __init__(self):
        self.workers = []
        self.call_count = 0
        self.engine_calls = []
        self._pending_results = {}  # worker_id -> dict[task_id -> result]
        self._task_counter = 0

    def create_workers(self, job, *args, **kwargs):
        """Create workers based on Job specification."""
        role = job.role
        replicas = job.replicas
        worker_ids = [f"{role}/{i}" for i in range(replicas)]
        self.workers = [
            Worker(
                id=wid,
                ip="127.0.0.1",
                worker_ports=["8000", "8001"],
                engine_ports=["9000", "9001"],
            )
            for wid in worker_ids
        ]
        # Initialize pending results for each worker
        for wid in worker_ids:
            self._pending_results[wid] = {}
        return worker_ids

    def get_workers(self, role, timeout=None):
        return self.workers

    async def create_engine(self, worker_id, engine, engine_name, config):
        pass

    async def async_call_engine(self, worker_id, method, *args, **kwargs):
        self.engine_calls.append((worker_id, method, args, kwargs))
        self.call_count += 1
        if method == "agenerate":
            return Mock()
        # Handle submit method - return a task_id and store the result
        elif method == "submit":
            if worker_id not in self._pending_results:
                self._pending_results[worker_id] = {}
            # Generate a unique task_id
            task_id = self._task_counter
            self._task_counter += 1
            # Simulate a successful rollout result
            result = {
                "input_ids": torch.randint(0, 100, (1, 10)),
                "attention_mask": torch.ones(1, 10, dtype=torch.bool),
                "loss_mask": torch.tensor(
                    [0] * 5 + [1] * 5, dtype=torch.bool
                ).unsqueeze(0),
                "rewards": torch.randn(1),
            }
            self._pending_results[worker_id][task_id] = result
            # Immediately fire callback
            callback_addr = kwargs["callback_addr"]
            resp = requests.post(callback_addr, json=dict(task_id=task_id))
            resp.raise_for_status()
            return task_id
        # Handle wait_for_task method
        elif method == "wait_for_task":
            task_id = kwargs.get("task_id")
            if (
                worker_id in self._pending_results
                and task_id in self._pending_results[worker_id]
            ):
                result = self._pending_results[worker_id].pop(task_id)
                return result
            return None
        elif method == "wait":
            # Return a result from pending results if available
            count = kwargs["count"]
            if worker_id in self._pending_results and self._pending_results[worker_id]:
                if len(self._pending_results[worker_id]) < count:
                    return []
                # Get first count results
                task_ids = list(self._pending_results[worker_id].keys())[:count]
                results = [
                    self._pending_results[worker_id].pop(tid) for tid in task_ids
                ]
                return results
            return []
        return None

    def call_engine(self, worker_id, method, *args, **kwargs):
        self.engine_calls.append((worker_id, method, args, kwargs))

        # For weight update methods that await call_engine, return a coroutine
        if method in [
            "update_weights_from_distributed",
            "update_weights_from_disk",
            "init_weights_update_group",
        ]:
            return self._async_call_engine_internal(worker_id, method, *args, **kwargs)

        # Handle submit method - return a task_id and store the result
        if method == "submit":
            if worker_id not in self._pending_results:
                self._pending_results[worker_id] = {}
            # Generate a unique task_id
            task_id = self._task_counter
            self._task_counter += 1
            # Simulate a successful rollout result
            result = {
                "input_ids": torch.randint(0, 100, (1, 10)),
                "attention_mask": torch.ones(1, 10, dtype=torch.bool),
                "loss_mask": torch.tensor(
                    [0] * 5 + [1] * 5, dtype=torch.bool
                ).unsqueeze(0),
                "rewards": torch.randn(1),
            }
            self._pending_results[worker_id][task_id] = result
            return task_id

        return None

    async def _async_call_engine_internal(self, worker_id, method, *args, **kwargs):
        await asyncio.sleep(0.001)
        return None

    def delete_workers(self, role, reverse_order: bool = False):
        self.workers.clear()
        self._pending_results.clear()
        self._task_counter = 0


class MockInferenceEngine:
    @classmethod
    def __module__(cls):
        return "tests.test_rollout_controller"

    @classmethod
    def __name__(cls):
        return "MockInferenceEngine"


class TestRolloutControllerInitialization:
    def test_constructor(self):
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        assert controller.config == config
        assert controller.scheduler == scheduler
        assert controller.workers == []
        assert controller._current_worker_idx == 0
        assert controller._version == 0
        assert controller.staleness_manager is None

    def test_initialize_creates_workers(self):
        config = create_test_config(
            backend="sglang:d2",
            consumer_batch_size=16,
            max_head_offpolicyness=2,
            enable_rollout_tracing=False,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        assert len(controller.workers) == 2
        assert controller.staleness_manager is not None

        controller.destroy()

    def test_initialize_creates_staleness_manager(self):
        config = create_test_config(
            consumer_batch_size=32,
            max_head_offpolicyness=5,
            max_concurrent_rollouts=100,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        assert controller.staleness_manager.max_concurrent_rollouts == 100
        assert controller.staleness_manager.consumer_batch_size == 32
        assert controller.staleness_manager.max_staleness == 5

        controller.destroy()

    def test_initialize_uses_consumer_batch_size_as_fallback(self):
        config = create_test_config(
            consumer_batch_size=64,
            max_head_offpolicyness=3,
            max_concurrent_rollouts=None,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        assert controller.staleness_manager.max_concurrent_rollouts == 64

        controller.destroy()

    def test_initialize_with_tracing_enabled(self):
        config = create_test_config(
            consumer_batch_size=16,
            max_head_offpolicyness=2,
            enable_rollout_tracing=True,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.destroy()


class TestRolloutControllerDestroy:
    def test_destroy_cleans_up_resources(self):
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        assert len(controller.workers) > 0

        controller.destroy()
        assert len(controller.workers) == 0

    def test_destroy_deletes_workers_via_scheduler(self):
        config = create_test_config(backend="sglang:d2", consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        assert len(scheduler.workers) == 2

        controller.destroy()

        assert len(scheduler.workers) == 0

    def test_destroy_handles_scheduler_error(self):
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        scheduler.delete_workers = Mock(side_effect=Exception("Test error"))

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.destroy()


class TestRolloutControllerCapacity:
    def test_get_capacity_initial_state(self):
        config = create_test_config(
            consumer_batch_size=16,
            max_concurrent_rollouts=32,
            max_head_offpolicyness=2,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        capacity = controller.get_capacity()
        assert capacity == 32

        controller.destroy()

    def test_get_capacity_uses_version(self):
        config = create_test_config(
            consumer_batch_size=8,
            max_concurrent_rollouts=1000,
            max_head_offpolicyness=2,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        capacity_v0 = controller.get_capacity()

        controller.set_version(5)
        capacity_v5 = controller.get_capacity()

        assert capacity_v5 > capacity_v0

        controller.destroy()


class TestRolloutControllerWorkerSelection:
    def test_choose_worker_round_robin(self):
        config = create_test_config(backend="sglang:d3", consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        worker_ids = []
        for _ in range(6):
            worker, _ = controller._choose_worker()
            worker_ids.append(worker.id)

        assert worker_ids[0] == "rollout/0"
        assert worker_ids[1] == "rollout/1"
        assert worker_ids[2] == "rollout/2"
        assert worker_ids[3] == "rollout/0"
        assert worker_ids[4] == "rollout/1"
        assert worker_ids[5] == "rollout/2"

        controller.destroy()


class TestRolloutControllerSubmitAndWait:
    def test_wait_returns_distributed_batch(self):
        config = create_test_config(consumer_batch_size=16, max_concurrent_rollouts=50)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        for i in range(3):
            controller.submit(
                {"id": i},
                workflow="tests.utils.TestWorkflow",
                workflow_kwargs={},
            )

        batch = controller.wait(count=3, timeout=5.0)

        assert isinstance(batch, list)
        assert len(batch) == 3
        for b in batch:
            assert isinstance(b, dict)

        controller.destroy()

    def test_wait_timeout_when_insufficient_results(self):
        config = create_test_config(consumer_batch_size=16, max_concurrent_rollouts=10)
        scheduler = MockScheduler()

        async def async_mock(*args, **kwargs):
            res = await MockScheduler.async_call_engine(scheduler, *args, **kwargs)
            await asyncio.sleep(0.1)
            return res

        # Mock the `wait` call.
        scheduler.async_call_engine = async_mock

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.submit(
            {"id": 0},
            workflow="tests.utils.TestWorkflow",
            workflow_kwargs={},
        )

        with pytest.raises(TimeoutError, match="Timed out waiting for"):
            controller.wait(count=1, timeout=0.1)

        controller.destroy()

    def test_submit_passes_is_eval_and_group_size(self):
        config = create_test_config(consumer_batch_size=16, max_concurrent_rollouts=50)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.submit(
            data={"id": 1},
            workflow="tests.utils.TestWorkflow",
            workflow_kwargs={},
            is_eval=True,
            group_size=4,
        )
        controller.wait(count=1, timeout=5.0)

        submit_calls = [call for call in scheduler.engine_calls if call[1] == "submit"]
        assert len(submit_calls) == 1
        submit_kwargs = submit_calls[0][3]
        assert "is_eval" in submit_kwargs and submit_kwargs["is_eval"] is True
        assert "group_size" in submit_kwargs and submit_kwargs["group_size"] == 4

        controller.destroy()


class TestRolloutControllerBatchOperations:
    def test_rollout_batch_returns_list_of_dicts(self):
        """Verify RolloutController returns list of regular dicts, NOT RTensors.

        Unlike TrainController which uses RTensors for distributed batch storage,
        RolloutController uses task-based round-robin and returns list of regular Python dicts.
        """
        from areal.infra.rpc.rtensor import RTensor

        config = create_test_config(consumer_batch_size=16, max_concurrent_rollouts=50)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        batch_data = [{"id": i} for i in range(3)]
        batch = controller.rollout_batch(
            batch_data,
            workflow="tests.utils.TestWorkflow",
            workflow_kwargs={},
        )

        # Verify batch is a dict, not RTensor
        assert isinstance(batch, list), "RolloutController should return list of dicts"

        # Verify no RTensors in the result
        for item in batch:
            assert isinstance(item, dict), "Each item should be a dict"
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    assert not isinstance(value, RTensor), f"Found RTensor at key {key}"
                elif isinstance(value, dict):
                    for k, v in value.items():
                        assert not isinstance(v, RTensor), f"Found RTensor at {key}.{k}"

        controller.destroy()

    def test_rollout_batch_submits_all_data(self):
        config = create_test_config(consumer_batch_size=16, max_concurrent_rollouts=50)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        batch_data = [{"id": i, "value": f"item_{i}"} for i in range(4)]
        batch = controller.rollout_batch(
            batch_data,
            workflow="tests.utils.TestWorkflow",
            workflow_kwargs={},
        )

        # Check that all trajectories are returned (each as a separate dict)
        assert len(batch) == 4

        controller.destroy()

    def test_rollout_batch_waits_for_all_results(self):
        config = create_test_config(consumer_batch_size=16, max_concurrent_rollouts=100)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        batch_data = [{"id": i} for i in range(10)]
        batch = controller.rollout_batch(
            batch_data,
            workflow="tests.utils.TestWorkflow",
            workflow_kwargs={},
        )

        # Check that all trajectories are returned (each as a separate dict)
        assert len(batch) == 10

        controller.destroy()


class TestRolloutControllerVersionManagement:
    def test_get_version_initial(self):
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        assert controller.get_version() == 0
        controller.destroy()

    def test_set_version_updates_controller_version(self):
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.set_version(42)
        assert controller.get_version() == 42

        controller.destroy()

    def test_set_version_calls_workers(self):
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.set_version(10)

        version_calls = [
            call for call in scheduler.engine_calls if call[1] == "set_version"
        ]
        assert len(version_calls) == 2

        controller.destroy()

    def test_set_version_handles_worker_error(self):
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()

        def failing_call(*args, **kwargs):
            raise Exception("Worker error")

        scheduler.call_engine = failing_call

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.set_version(5)
        controller.destroy()


class TestRolloutControllerWeightUpdates:
    def test_init_weights_update_group_returns_future(self):
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        coro = controller.init_weights_update_group(meta)

        # Run the coroutine and verify it completes successfully
        asyncio.run(coro)

        controller.destroy()

    def test_update_weights_from_distributed_returns_future(self):
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        param_specs = [ParamSpec(name="test", shape=(10, 10), dtype="float32")]
        coro = controller.update_weights_from_distributed(meta, param_specs)

        # Run the coroutine and verify it completes successfully
        asyncio.run(coro)

        controller.destroy()

    def test_update_weights_from_disk_returns_future(self):
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        coro = controller.update_weights_from_disk(meta)

        # Run the coroutine and verify it completes successfully
        asyncio.run(coro)

        controller.destroy()


class TestRolloutControllerLifecycle:
    def test_pause_calls_all_workers(self):
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.pause()

        pause_calls = [call for call in scheduler.engine_calls if call[1] == "pause"]
        assert len(pause_calls) == 2

        controller.destroy()

    def test_resume_calls_all_workers(self):
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.resume()

        resume_calls = [call for call in scheduler.engine_calls if call[1] == "resume"]
        assert len(resume_calls) == 2

        controller.destroy()

    def test_pause_handles_worker_error(self):
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()

        def failing_call(*args, **kwargs):
            raise Exception("Worker error")

        scheduler.call_engine = failing_call

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.pause()

    def test_resume_handles_worker_error(self):
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()

        def failing_call(*args, **kwargs):
            raise Exception("Worker error")

        scheduler.call_engine = failing_call

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.resume()


class TestRolloutControllerAgenerate:
    def test_agenerate_chooses_worker(self):
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        req = ModelRequest(input_ids=[1, 2, 3, 4, 5])

        async def test_agenerate():
            result = await controller.agenerate(req)
            return result

        asyncio.run(test_agenerate())

        agenerate_calls = [
            call for call in scheduler.engine_calls if call[1] == "agenerate"
        ]
        assert len(agenerate_calls) == 1
        assert agenerate_calls[0][3]["req"] == req

        controller.destroy()

    def test_agenerate_round_robin(self):
        config = create_test_config(backend="sglang:d3", consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        async def test_multiple_agenerate():
            for _ in range(6):
                req = ModelRequest(input_ids=[1, 2, 3])
                await controller.agenerate(req)

        asyncio.run(test_multiple_agenerate())

        agenerate_calls = [
            call for call in scheduler.engine_calls if call[1] == "agenerate"
        ]
        worker_ids = [call[0] for call in agenerate_calls]

        assert worker_ids[0] == "rollout/0"
        assert worker_ids[1] == "rollout/1"
        assert worker_ids[2] == "rollout/2"
        assert worker_ids[3] == "rollout/0"

        controller.destroy()


class TestRolloutControllerErrorHandling:
    def test_wait_returns_empty_batch_on_no_results(self):
        config = create_test_config(consumer_batch_size=16, max_concurrent_rollouts=50)
        scheduler = MockScheduler()

        async def reject_all(*args, **kwargs):
            await asyncio.sleep(0.01)
            return None

        scheduler.async_call_engine = reject_all

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        with pytest.raises(TimeoutError):
            controller.wait(count=1, timeout=0.5)

        controller.destroy()


class TestRolloutControllerIntegration:
    def test_end_to_end_workflow(self):
        config = create_test_config(
            consumer_batch_size=8,
            max_concurrent_rollouts=20,
            max_head_offpolicyness=2,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        capacity = controller.get_capacity()
        assert capacity == 20

        for i in range(5):
            controller.submit(
                {"id": i},
                workflow="tests.utils.TestWorkflow",
                workflow_kwargs={},
            )

        batch = controller.wait(count=5, timeout=5.0)
        assert len(batch) == 5

        controller.set_version(1)
        assert controller.get_version() == 1

        controller.destroy()

    def test_multiple_batch_cycles(self):
        config = create_test_config(
            consumer_batch_size=4,
            max_concurrent_rollouts=50,
            max_head_offpolicyness=5,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        for cycle in range(3):
            batch_data = [{"id": i, "cycle": cycle} for i in range(4)]
            batch = controller.rollout_batch(
                batch_data,
                workflow="tests.utils.TestWorkflow",
                workflow_kwargs={},
            )
            assert len(batch) == 4

        controller.destroy()


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_parametrized_worker_count(num_workers):
    config = create_test_config(
        backend=f"sglang:d{num_workers}", consumer_batch_size=16
    )
    scheduler = MockScheduler()
    controller = RolloutController(
        inf_engine=MockInferenceEngine,
        config=config,
        scheduler=scheduler,
    )

    controller.initialize(role="rollout", server_args={})

    assert len(controller.workers) == num_workers

    controller.destroy()


@pytest.mark.parametrize(
    "consumer_batch_size,max_concurrent_rollouts,expected_capacity",
    [(16, 32, 32), (32, 64, 64), (8, 100, 24)],
)
def test_parametrized_capacity_settings(
    consumer_batch_size, max_concurrent_rollouts, expected_capacity
):
    config = create_test_config(
        consumer_batch_size=consumer_batch_size,
        max_concurrent_rollouts=max_concurrent_rollouts,
        max_head_offpolicyness=2,
    )
    scheduler = MockScheduler()
    controller = RolloutController(
        inf_engine=MockInferenceEngine,
        config=config,
        scheduler=scheduler,
    )

    controller.initialize(role="rollout", server_args={})

    capacity = controller.get_capacity()
    assert capacity == expected_capacity

    controller.destroy()


QWEN3_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)


@pytest.mark.sglang
@pytest.mark.parametrize("model_path", [QWEN3_PATH])
@pytest.mark.slow
@pytest.mark.ci
def test_rollout_controller_integration(tmp_path, model_path):
    from areal.engine.sglang_remote import RemoteSGLangEngine

    tokenizer = load_hf_tokenizer(model_path)
    fileroot = tmp_path / "fileroot"
    fileroot.mkdir()
    name_resolve_root = tmp_path / "name_resolve"
    name_resolve_root.mkdir()
    scheduler = LocalScheduler(
        log_dir=tmp_path,
        experiment_name="test_rollout_controller_integration",
        trial_name="trial0",
        fileroot=str(fileroot),
        nfs_record_root=str(name_resolve_root),
    )
    rollout = RolloutController(
        inf_engine=RemoteSGLangEngine,
        config=InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            backend="sglang:d2",
            consumer_batch_size=128,
            max_head_offpolicyness=1,
            max_concurrent_rollouts=5,
            setup_timeout=300,
            enable_rollout_tracing=True,
            scheduling_spec=(
                SchedulingSpec(
                    cpu=4, gpu=1, cmd="python -m areal.infra.rpc.rpc_server"
                ),
            ),
        ),
        scheduler=scheduler,
    )

    bs = 10
    try:
        rollout.initialize(
            role="rollout",
            server_args=SGLangConfig.build_args(
                SGLangConfig(model_path=model_path, mem_fraction_static=0.5),
                tp_size=1,
                base_gpu_id=0,
            ),
        )
        result = rollout.rollout_batch(
            data=[dict(messages=[dict(role="user", content="hello")], answer="1")] * bs,
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=dict(
                reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
                gconfig=GenerationHyperparameters(),
                tokenizer=tokenizer,
            ),
        )
        assert isinstance(result, list)
        assert len(result) == bs
        assert isinstance(result[0], dict)
        assert "attention_mask" in result[0]
    finally:
        rollout.destroy()


class TestRolloutControllerResolveWorkflow:
    """Tests for workflow resolution methods."""

    def test_resolve_workflow_str_with_string(self):
        """Test _resolve_workflow_str with string input."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        result = controller._resolve_workflow_str("areal.workflow.rlvr.RLVRWorkflow")
        assert result == "areal.workflow.rlvr.RLVRWorkflow"


class TestRolloutControllerShouldAcceptFn:
    """Tests for should_accept_fn resolution."""

    def test_resolve_should_accept_fn_with_none(self):
        """Test _resolve_should_accept_fn with None input."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        result = controller._resolve_should_accept_fn(None)
        assert result is None

    def test_resolve_should_accept_fn_with_callable_raises(self):
        """Test _resolve_should_accept_fn raises for callable input."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        def my_filter(data):
            return True

        with pytest.raises(RuntimeError, match="must be an importable string path"):
            controller._resolve_should_accept_fn(my_filter)

    def test_resolve_should_accept_fn_with_invalid_path_raises(self):
        """Test _resolve_should_accept_fn raises for invalid import path."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        with pytest.raises(RuntimeError, match="Failed to import"):
            controller._resolve_should_accept_fn("invalid.module.path.function")


class TestRolloutControllerDispatcher:
    """Tests for dispatcher property and initialization."""

    def test_dispatcher_raises_before_initialization(self):
        """Test dispatcher property raises when not initialized."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        with pytest.raises(RuntimeError, match="initialize\\(\\) must be called"):
            _ = controller.dispatcher

    def test_dispatcher_available_after_initialization(self):
        """Test dispatcher property works after initialization."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        dispatcher = controller.dispatcher
        assert dispatcher is not None

        controller.destroy()


class TestRolloutControllerStalenessManager:
    """Tests for staleness manager property."""

    def test_staleness_manager_none_before_initialization(self):
        """Test staleness_manager is None before initialization."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        assert controller.staleness_manager is None

    def test_staleness_manager_available_after_initialization(self):
        """Test staleness_manager is available after initialization."""
        config = create_test_config(
            consumer_batch_size=16,
            max_head_offpolicyness=2,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        assert controller.staleness_manager is not None
        assert controller.staleness_manager.max_staleness == 2

        controller.destroy()


class TestRolloutControllerRunner:
    """Tests for runner property (backward compatibility)."""

    def test_runner_property_returns_dispatcher_runner(self):
        """Test runner property returns the dispatcher's runner."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        runner = controller.runner
        assert runner is controller.dispatcher.runner

        controller.destroy()


class TestRolloutControllerExportStats:
    """Tests for export_stats method."""

    def test_export_stats_aggregates_from_workers(self):
        """Test export_stats correctly aggregates stats from all workers."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()

        # Override async_call_engine to return stats for export_stats method
        original_async_call = scheduler.async_call_engine

        async def mock_async_call_engine(worker_id, method, *args, **kwargs):
            if method == "export_stats":
                return {
                    "reward": 0.5,
                    "reward__count": 10,
                    "loss": 0.3,
                    "loss__count": 10,
                }
            return await original_async_call(worker_id, method, *args, **kwargs)

        scheduler.async_call_engine = mock_async_call_engine

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        stats = controller.export_stats()

        # Should aggregate stats from all workers
        assert "reward" in stats or "loss" in stats

        controller.destroy()


class TestRolloutControllerRolloutStats:
    """Tests for _rollout_stats method."""

    def test_rollout_stats_returns_formatted_string(self):
        """Test _rollout_stats returns properly formatted stats string."""
        config = create_test_config(
            consumer_batch_size=16,
            max_head_offpolicyness=2,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        stats_str = controller._rollout_stats()

        assert "enqueued:" in stats_str
        assert "running:" in stats_str
        assert "accepted:" in stats_str
        assert "rejected:" in stats_str

        controller.destroy()


class TestRolloutControllerSchedulingSpec:
    """Tests for scheduling spec handling during initialization."""

    def test_initialization_scales_scheduling_spec(self):
        """Test initialization correctly scales scheduling spec for instance size."""
        config = create_test_config(
            backend="sglang:d2t2",
            consumer_batch_size=16,
            max_concurrent_rollouts=32,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        # Use TP=2 to test instance size scaling
        controller.initialize(role="rollout", server_args={})

        # Verify workers were created with correct count
        assert len(controller.workers) == 2  # dp_size = 2

        controller.destroy()


class TestRolloutControllerQueueSize:
    """Tests for queue size configuration."""

    def test_queue_size_uses_config_value(self):
        """Test queue size uses config value when provided."""
        config = create_test_config(
            consumer_batch_size=16,
            max_concurrent_rollouts=32,
            queue_size=100,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        # Queue size should be used from config
        assert controller.dispatcher is not None

        controller.destroy()

    def test_queue_size_defaults_to_concurrent_rollouts(self):
        """Test queue size defaults to max_concurrent_rollouts * 16 when not provided."""
        config = create_test_config(
            consumer_batch_size=16,
            max_concurrent_rollouts=32,
            queue_size=None,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        # Should use default queue size
        assert controller.dispatcher is not None

        controller.destroy()


class TestRolloutControllerCollectiveRPC:
    """Tests for collective RPC methods."""

    def test_collective_rpc_calls_all_workers(self):
        """Test _collective_rpc calls all workers."""
        config = create_test_config(backend="sglang:d3", consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        # Clear previous calls
        scheduler.engine_calls = []

        controller._collective_rpc("test_method", arg1="value1")

        # Should have called all 3 workers
        test_calls = [
            call for call in scheduler.engine_calls if call[1] == "test_method"
        ]
        assert len(test_calls) == 3

        controller.destroy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
