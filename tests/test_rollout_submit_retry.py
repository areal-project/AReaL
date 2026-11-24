# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.api import RolloutWorkflow
from areal.api.cli_args import InferenceEngineConfig
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.infra.workflow_executor import BatchTaskDispatcher, WorkflowExecutor


def test_remote_submit_retry_uses_assigned_task_id():
    config = InferenceEngineConfig()
    engine = RemoteInfEngine(config, backend=Mock())
    executor = WorkflowExecutor(config, engine)
    engine.workflow_executor = executor
    executor.initialize(logger=Mock())
    try:
        executor.pause()
        workflow = Mock(spec=RolloutWorkflow)
        address = "http://controller/callback"
        task_id = engine.submit({}, workflow, callback_addr=address)
        assert (
            engine.submit({}, workflow, task_id=task_id, callback_addr=address)
            == task_id
        )
        assert executor.staleness_manager.get_stats().enqueued == 1
        assert executor.dispatcher._task_callbacks == {task_id: address}
    finally:
        executor.destroy()


@pytest.mark.parametrize("consume_by_id", [True, False])
@pytest.mark.parametrize("result", [383, None], ids=["accepted", "rejected"])
def test_retried_submit_runs_once_and_preserves_callback(consume_by_id, result):
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    executed = []

    def task_factory(task):
        async def work():
            executed.append(task.task_id)
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.001)
            return result

        return work

    manager = Mock()
    manager.get_capacity.return_value = 5
    dispatcher = BatchTaskDispatcher(16, task_factory, manager)
    dispatcher._send_callback = Mock(side_effect=lambda *args: completed.set())
    dispatcher.initialize(Mock())
    task = SimpleNamespace(task_id=383)
    address = "http://controller/callback"
    try:
        dispatcher.pause()

        def submit(_):
            dispatcher.submit_task_input(task, callback_addr=address)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(submit, range(8)))
        manager.on_rollout_enqueued.assert_called_once_with()
        with pytest.raises(ValueError, match="Duplicate task_id"):
            dispatcher.submit_task_input(task, callback_addr="http://another/callback")

        with pytest.raises(ValueError, match="Duplicate task_id"):
            dispatcher.submit_task_input(task)

        dispatcher.resume()
        assert started.wait(5)
        submit(None)
        release.set()
        assert completed.wait(5)
        submit(None)
        if consume_by_id:
            assert dispatcher.wait_for_task(383, timeout=5) == result
        else:
            assert dispatcher.wait_results(1, timeout=5) == [result]
        dispatcher._send_callback.assert_called_once_with(address, 383, result)
        assert not dispatcher._task_callbacks
        dispatcher.submit_task_input(task)
        assert dispatcher.wait_for_task(383, timeout=5) == result
        dispatcher._send_callback.assert_called_once()
        assert executed == [383, 383]
        assert manager.on_rollout_enqueued.call_count == 2
    finally:
        release.set()
        dispatcher.destroy()


def test_cancelled_callback_preserves_result_retrieval():
    started = threading.Event()
    release = threading.Event()

    def task_factory(task):
        async def work():
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.001)
            return task.task_id

        return work

    manager = Mock()
    manager.get_capacity.return_value = 5
    dispatcher = BatchTaskDispatcher(16, task_factory, manager)
    dispatcher._send_callback = Mock()
    dispatcher.initialize(Mock())
    try:
        dispatcher.submit_task_input(
            SimpleNamespace(task_id=383), callback_addr="http://controller/callback"
        )
        assert started.wait(5)
        dispatcher.cancel_callback(383)
        release.set()
        assert dispatcher.wait_for_task(383, timeout=5) == 383
        dispatcher._send_callback.assert_not_called()
        assert not dispatcher._task_callbacks
    finally:
        release.set()
        dispatcher.destroy()
