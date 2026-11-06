# SPDX-License-Identifier: Apache-2.0

"""Abandoned rollout results must not be retained on the worker.

The rollout controller stops waiting for a task when it times out. The worker
still finishes the rollout later, and its result used to stay in
``BatchTaskDispatcher._pending_results`` for the lifetime of the process.
"""

import asyncio
import logging
import time

import pytest

from areal.infra.staleness_manager import StalenessManager
from areal.infra.workflow_executor import BatchTaskDispatcher


class _VersionProvider:
    def get_version(self) -> int:
        return 0


class _Task:
    def __init__(self, task_id: int):
        self.task_id = task_id


def _make_dispatcher(discarded: list) -> BatchTaskDispatcher:
    manager = StalenessManager(
        version_provider=_VersionProvider(),
        max_concurrent_rollouts=4,
        consumer_batch_size=4,
        max_staleness=1,
    )

    def task_factory(task: _Task):
        async def _run():
            await asyncio.sleep(0.01)
            return task.task_id

        return _run

    dispatcher = BatchTaskDispatcher(
        max_queue_size=16,
        task_factory=task_factory,
        staleness_manager=manager,
        discard_result=discarded.append,
    )
    dispatcher.initialize(logging.getLogger("test-abandoned-rollout"))
    return dispatcher


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_abandon_before_result_drops_it():
    discarded: list = []
    dispatcher = _make_dispatcher(discarded)
    try:
        dispatcher.submit_task_input(_Task(7))
        assert dispatcher.abandon_task(7) is True

        assert _wait_until(lambda: discarded == [7])
        assert 7 not in dispatcher._pending_results
        assert 7 not in dispatcher._abandoned_task_ids
        # The task is no longer fetchable.
        with pytest.raises(ValueError):
            dispatcher.wait_for_task(7, timeout=0.1)
    finally:
        dispatcher.destroy()


def test_abandon_after_result_drops_it():
    discarded: list = []
    dispatcher = _make_dispatcher(discarded)
    try:
        dispatcher.submit_task_input(_Task(9))
        assert _wait_until(lambda: 9 in dispatcher._pending_results)

        assert dispatcher.abandon_task(9) is True

        assert _wait_until(lambda: discarded == [9])
        assert 9 not in dispatcher._pending_results
    finally:
        dispatcher.destroy()


def test_abandon_unknown_task_is_a_noop():
    discarded: list = []
    dispatcher = _make_dispatcher(discarded)
    try:
        assert dispatcher.abandon_task(123) is False
        # Unknown ids must not accumulate.
        assert 123 not in dispatcher._abandoned_task_ids
    finally:
        dispatcher.destroy()


def test_unabandoned_result_still_reaches_the_caller():
    discarded: list = []
    dispatcher = _make_dispatcher(discarded)
    try:
        dispatcher.submit_task_input(_Task(11))
        assert dispatcher.wait_for_task(11, timeout=5.0) == 11
        assert discarded == []
    finally:
        dispatcher.destroy()


def test_controller_tells_the_worker_to_abandon_the_task():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from areal.infra.controller.rollout_controller import RolloutController

    controller = object.__new__(RolloutController)
    controller.scheduler = MagicMock()
    controller.scheduler.async_call_engine = AsyncMock()

    asyncio.run(
        RolloutController._abandon_rollout_task(
            controller,
            SimpleNamespace(id="rollout/0"),
            "rollout-0",
            5,
        )
    )

    controller.scheduler.async_call_engine.assert_awaited_once_with(
        "rollout/0",
        "abandon_rollout_task",
        engine_name="rollout-0",
        task_id=5,
        http_timeout=60.0,
    )
