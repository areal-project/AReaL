# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.test_stream_dataloader import make_loader

from areal.api.cli_args import InferenceEngineConfig
from areal.infra.controller.rollout_controller import (
    RolloutController,
    _RemoteRolloutTaskInput,
    logger,
)
from areal.infra.staleness_manager import StalenessManager
from areal.infra.workflow_executor import (
    BatchTaskDispatcher,
    WorkflowContractFailure,
    get_workflow_result_error,
)


def make_controller(*, batch_size=2, max_concurrent_rollouts=4, max_staleness=1):
    controller = RolloutController(
        inf_engine=Mock,
        config=InferenceEngineConfig(
            backend="sglang:d1",
            consumer_batch_size=batch_size,
            max_concurrent_rollouts=max_concurrent_rollouts,
            max_head_offpolicyness=max_staleness,
            request_timeout=5,
        ),
        scheduler=Mock(),
    )
    controller._worker_role = "rollout"
    controller._callback_host = "127.0.0.1"
    controller._callback_port = 12345
    controller._choose_worker = lambda: (SimpleNamespace(id="worker"), 0)
    controller._staleness_manager = StalenessManager(
        controller,
        max_concurrent_rollouts=max_concurrent_rollouts,
        consumer_batch_size=batch_size,
        max_staleness=max_staleness,
    )
    return controller


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    ["accepted", "rejected", "empty", "timeout", "error", "contract", "cancelled"],
)
async def test_completion_feedback_covers_terminal_paths(outcome):
    controller = make_controller()
    loader = make_loader()
    loader.next_rollout(0)
    loader.complete_rollout(0, accepted=False)
    data = loader.next_rollout(1)

    async def call_engine(worker_id, method, **kwargs):
        if method == "submit":
            if outcome == "timeout":
                raise TimeoutError()
            if outcome == "error":
                raise RuntimeError("test worker failure")
            if outcome == "cancelled":
                raise asyncio.CancelledError()
            controller._pending_futures[kwargs["task_id"]].set_result(None)
            return kwargs["task_id"]
        if outcome == "empty":
            return None
        if outcome == "contract":
            return WorkflowContractFailure(message="test contract violation")
        return {"trajectory": data if outcome == "accepted" else None, "metrics": None}

    controller.scheduler.async_call_engine = call_engine
    callback = controller._create_submit_callback(
        _RemoteRolloutTaskInput(
            task_id=1,
            data=data,
            workflow=None,
            workflow_kwargs={},
            should_accept_fn=None,
            stream_dataloader=loader,
        )
    )
    controller.staleness_manager.on_rollout_enqueued()
    controller.staleness_manager.on_rollout_submitted()
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await callback()
    else:
        await callback()

    assert loader.state_dict()["debt"]["a"] == int(outcome != "accepted")
    assert loader.next_rollout(2)["stream"] == ("b" if outcome == "accepted" else "a")


@pytest.mark.parametrize("dynamic_bs", [False, True])
def test_fast_stream_results_are_consumed_without_waiting_for_slow_stream(dynamic_bs):
    controller = make_controller()
    loader = make_loader(batch_size=2)
    release_slow = threading.Event()
    inputs = {}

    async def call_engine(worker_id, method, **kwargs):
        task_id = kwargs["task_id"]
        if method == "submit":
            inputs[task_id] = kwargs["data"]
            controller._pending_futures[task_id].set_result(None)
            return task_id
        data = inputs[task_id]
        if data["stream"] == "a":
            while not release_slow.is_set():
                await asyncio.sleep(0.01)
        return {"trajectory": data, "metrics": {"rollout/accepted": [1.0]}}

    controller.scheduler.async_call_engine = call_engine
    controller._dispatcher = BatchTaskDispatcher(
        max_queue_size=32,
        task_factory=controller._create_submit_callback,
        staleness_manager=controller.staleness_manager,
        consume_result=controller.consumed_metrics.consume,
        terminal_error_fn=get_workflow_result_error,
    )
    controller.dispatcher.initialize(logger=logger)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(
            controller.prepare_batch,
            loader,
            workflow="tests.utils.TestWorkflow",
            dynamic_bs=dynamic_bs,
        )
        results = future.result(timeout=10)
        # Both fast B groups reach training while the A groups remain in flight.
        assert [result["stream"] for result in results] == ["b", "b"]
        assert not release_slow.is_set()
        assert controller.consumed_metrics.export()["rollout/group_attempt_count"] == 2
    finally:
        release_slow.set()
        controller.dispatcher._set_thread_exception(
            RuntimeError("Test dispatcher stopped")
        )
        pool.shutdown(wait=True)
        controller.dispatcher.destroy()


def test_prepare_batch_replenishes_filtered_stream_and_preserves_rejection_metrics():
    controller = make_controller(batch_size=4)
    loader = make_loader(batch_size=4)
    # Give the test enough room for replacements while the main thread waits.
    inputs = {}
    rejected = False

    async def call_engine(worker_id, method, **kwargs):
        nonlocal rejected
        task_id = kwargs["task_id"]
        if method == "submit":
            inputs[task_id] = kwargs["data"]
            controller._pending_futures[task_id].set_result(None)
            return task_id
        while len(inputs) < 4:
            await asyncio.sleep(0.01)
        data = inputs[task_id]
        accepted = not (data["stream"] == "a" and not rejected)
        rejected |= not accepted
        return {
            "trajectory": data if accepted else None,
            "metrics": {"rollout/accepted": [float(accepted)]},
        }

    controller.scheduler.async_call_engine = call_engine
    controller._dispatcher = BatchTaskDispatcher(
        max_queue_size=32,
        task_factory=controller._create_submit_callback,
        staleness_manager=controller.staleness_manager,
        consume_result=controller.consumed_metrics.consume,
    )
    controller.dispatcher.initialize(logger=logger)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        result = pool.submit(
            controller.prepare_batch,
            loader,
            workflow="tests.utils.TestWorkflow",
        ).result(timeout=10)
        assert len(result) == 4
        assert inputs[4]["stream"] == "a"
        # Consuming rejection metrics must not be bypassed by stream feedback.
        assert controller.consumed_metrics.export()["rollout/group_attempt_count"] == 5
    finally:
        controller.dispatcher._set_thread_exception(
            RuntimeError("Test dispatcher stopped")
        )
        controller.dispatcher.destroy()
        pool.shutdown(wait=True)


def test_permanently_filtered_stream_does_not_starve_healthy_stream():
    controller = make_controller(
        batch_size=1, max_concurrent_rollouts=1, max_staleness=0
    )
    loader = make_loader(batch_size=1)
    inputs = {}

    async def call_engine(worker_id, method, **kwargs):
        task_id = kwargs["task_id"]
        if method == "submit":
            inputs[task_id] = kwargs["data"]
            controller._pending_futures[task_id].set_result(None)
            return task_id
        data = inputs[task_id]
        return {
            "trajectory": data if data["stream"] == "b" else None,
            "metrics": None,
        }

    controller.scheduler.async_call_engine = call_engine
    controller._dispatcher = BatchTaskDispatcher(
        max_queue_size=32,
        task_factory=controller._create_submit_callback,
        staleness_manager=controller.staleness_manager,
        consume_result=controller.consumed_metrics.consume,
    )
    controller.dispatcher.initialize(logger=logger)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        result = pool.submit(
            controller.prepare_batch,
            loader,
            workflow="tests.utils.TestWorkflow",
        ).result(timeout=10)
        assert [row["stream"] for row in result] == ["b"]
        assert [row["stream"] for row in inputs.values()] == ["a", "a", "b"]
        assert loader.state_dict()["debt"] == {"a": 1, "b": 0}
    finally:
        controller.dispatcher._set_thread_exception(
            RuntimeError("Test dispatcher stopped")
        )
        controller.dispatcher.destroy()
        pool.shutdown(wait=True)
