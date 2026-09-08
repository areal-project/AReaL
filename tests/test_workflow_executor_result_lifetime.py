"""Consumed rollout payloads must not remain in the dispatcher fetch thread."""

import gc
import time
import weakref
from unittest.mock import MagicMock

import pytest

from areal.infra.async_task_runner import TimedResult
from areal.infra.workflow_executor import BatchTaskDispatcher
from areal.utils import logging


@pytest.mark.parametrize("by_task_id", [False, True])
def test_consumed_results_are_released_while_fetcher_polls(by_task_id):
    """Both consumption APIs release the fetcher's last batch without new work."""

    class Payload:
        pass

    manager = MagicMock()
    manager.get_capacity.return_value = 0
    dispatcher = BatchTaskDispatcher(
        max_queue_size=10,
        task_factory=MagicMock(),
        staleness_manager=manager,
    )
    dispatcher._active_task_ids.update(range(3))
    dispatcher.initialize(logger=logging.getLogger("WorkflowExecutor"))
    try:
        payloads = [Payload() for _ in range(3)]
        refs = [weakref.ref(payload) for payload in payloads]
        # Isolate fetch-loop ownership from the runner's task-result ownership.
        for task_id, payload in enumerate(payloads):
            dispatcher.runner.output_queue.put(
                TimedResult(
                    create_time=time.monotonic_ns(), data=payload, task_id=task_id
                )
            )
        if by_task_id:
            results = [dispatcher.wait_for_task(i, timeout=2.0) for i in range(3)]
        else:
            results = dispatcher.wait_results(count=3, timeout=2.0)
        assert {id(result) for result in results} == {
            id(payload) for payload in payloads
        }
        del payloads, payload, results

        deadline = time.monotonic() + 2.0
        while any(ref() is not None for ref in refs) and time.monotonic() < deadline:
            gc.collect()
            time.sleep(0.01)
        assert all(ref() is None for ref in refs)
        dispatcher._check_thread_exception()
    finally:
        dispatcher.destroy()
