# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.infra.workflow_executor import BatchTaskDispatcher


@pytest.mark.parametrize("consume_by_id", [True, False])
def test_retried_submit_runs_once_and_preserves_callback(consume_by_id):
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
            return task.task_id

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

        dispatcher.resume()
        assert started.wait(5)
        submit(None)
        release.set()
        assert completed.wait(5)
        submit(None)
        if consume_by_id:
            assert dispatcher.wait_for_task(383, timeout=5) == 383
        else:
            assert dispatcher.wait_results(1, timeout=5) == [383]
        dispatcher._send_callback.assert_called_once_with(address, 383, 383)
        assert not dispatcher._task_callbacks
        dispatcher.submit_task_input(SimpleNamespace(task_id=999))
        assert dispatcher.wait_for_task(999, timeout=5) == 999
        assert executed == [383, 999]
        assert manager.on_rollout_enqueued.call_count == 2
    finally:
        release.set()
        dispatcher.destroy()


@pytest.mark.parametrize("consume_by_id", [True, False])
def test_terminal_error_releases_task_id_and_callback(consume_by_id):
    completed = threading.Event()

    def task_factory(task):
        async def work():
            return task.task_id

        return work

    manager = Mock()
    manager.get_capacity.return_value = 5
    dispatcher = BatchTaskDispatcher(
        16, task_factory, manager, terminal_error_fn=lambda _: ValueError("terminal")
    )
    dispatcher._send_callback = Mock(side_effect=lambda *args: completed.set())
    dispatcher.initialize(Mock())
    task = SimpleNamespace(task_id=383)
    try:
        dispatcher.submit_task_input(task, callback_addr="http://controller/callback")
        assert completed.wait(5)
        with pytest.raises(ValueError, match="terminal"):
            if consume_by_id:
                dispatcher.wait_for_task(383, timeout=5)
            else:
                dispatcher.wait_results(1, timeout=5)
        assert not dispatcher._task_callbacks
        dispatcher.terminal_error_fn = None
        dispatcher.submit_task_input(task)
        assert dispatcher.wait_for_task(383, timeout=5) == 383
        assert dispatcher._send_callback.call_count == 1
    finally:
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
