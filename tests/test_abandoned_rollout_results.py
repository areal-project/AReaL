# SPDX-License-Identifier: Apache-2.0
"""Rollout results the controller gave up on must not stay on the worker (#1863)."""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tests.test_rollout_controller_arena_streams import make_arena_stream_controller

from areal.infra.controller.rollout_controller import _RemoteRolloutTaskInput
from areal.infra.staleness_manager import StalenessManager
from areal.infra.workflow_executor import (
    BatchTaskDispatcher,
    WorkflowContractFailure,
    WorkflowExecutor,
    _RolloutResult,
)

_logger = logging.getLogger("test_abandoned_rollout_results")


# --- Worker side: BatchTaskDispatcher.discard_task -------------------------


@dataclass
class _TaskInput:
    task_id: int


class _Version:
    def get_version(self) -> int:
        return 0


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.01)


@pytest.fixture
def dispatcher_env():
    release = threading.Event()
    released: list[object] = []

    def task_factory(task_input):
        async def run():
            while not release.is_set():
                await asyncio.sleep(0.01)
            return {"task": task_input.task_id, "payload": "x" * 1024}

        return run

    dispatcher = BatchTaskDispatcher(
        max_queue_size=8,
        task_factory=task_factory,
        staleness_manager=StalenessManager(
            _Version(),
            max_concurrent_rollouts=8,
            consumer_batch_size=8,
            max_staleness=0,
        ),
        discard_result=released.append,
    )
    sent_callbacks: list[int] = []
    dispatcher._send_callback = lambda addr, task_id, result: sent_callbacks.append(
        task_id
    )
    dispatcher.initialize(logger=_logger)
    try:
        yield SimpleNamespace(
            dispatcher=dispatcher,
            release=release,
            released=released,
            sent_callbacks=sent_callbacks,
        )
    finally:
        release.set()
        dispatcher.destroy()


def test_result_arriving_after_discard_is_dropped_not_stored(dispatcher_env):
    env = dispatcher_env
    env.dispatcher.register_callback(1, "http://controller/callback")
    env.dispatcher.submit_task_input(_TaskInput(task_id=1))
    _wait_until(lambda: env.dispatcher.runner.get_input_queue_size() == 0)

    # The controller timed out while the rollout is still running.
    env.dispatcher.discard_task(1)
    env.release.set()
    _wait_until(lambda: len(env.released) == 1)

    assert env.released == [{"task": 1, "payload": "x" * 1024}]
    assert env.dispatcher._pending_results == {}
    assert env.dispatcher._active_task_ids == set()
    assert env.dispatcher._abandoned_task_ids == set()
    # Nobody is waiting for it, so the controller is not called back either.
    assert env.sent_callbacks == []


def test_result_already_stored_is_removed_on_discard(dispatcher_env):
    env = dispatcher_env
    env.release.set()
    env.dispatcher.submit_task_input(_TaskInput(task_id=2))
    _wait_until(lambda: 2 in env.dispatcher._pending_results)

    env.dispatcher.discard_task(2)

    assert env.dispatcher._pending_results == {}
    assert env.dispatcher._active_task_ids == set()
    assert env.released == [{"task": 2, "payload": "x" * 1024}]
    with pytest.raises(ValueError, match="never submitted"):
        env.dispatcher.wait_for_task(2, timeout=0)


def test_discard_leaves_other_tasks_untouched(dispatcher_env):
    env = dispatcher_env
    env.dispatcher.submit_task_input(_TaskInput(task_id=3))
    env.dispatcher.submit_task_input(_TaskInput(task_id=4))
    _wait_until(lambda: env.dispatcher.runner.get_input_queue_size() == 0)

    env.dispatcher.discard_task(3)
    env.release.set()

    assert env.dispatcher.wait_for_task(4, timeout=5) == {
        "task": 4,
        "payload": "x" * 1024,
    }
    _wait_until(lambda: len(env.released) == 1)
    assert env.released[0]["task"] == 3


def test_discarding_unknown_or_fetched_task_is_a_no_op(dispatcher_env):
    env = dispatcher_env
    env.release.set()
    env.dispatcher.submit_task_input(_TaskInput(task_id=5))
    assert env.dispatcher.wait_for_task(5, timeout=5)["task"] == 5

    env.dispatcher.discard_task(5)
    env.dispatcher.discard_task(999)

    assert env.released == []
    assert env.dispatcher._abandoned_task_ids == set()


def test_release_failure_does_not_break_the_fetch_loop(dispatcher_env):
    env = dispatcher_env

    def failing_release(_result):
        raise RuntimeError("shard store unreachable")

    env.dispatcher.discard_result = failing_release
    env.dispatcher.submit_task_input(_TaskInput(task_id=6))
    env.dispatcher.submit_task_input(_TaskInput(task_id=7))
    _wait_until(lambda: env.dispatcher.runner.get_input_queue_size() == 0)
    env.dispatcher.discard_task(6)
    env.release.set()

    assert env.dispatcher.wait_for_task(7, timeout=5)["task"] == 7
    assert env.dispatcher._thread_exception is None


# --- Controller side: _create_submit_callback --------------------------------


def _task(task_id: int) -> _RemoteRolloutTaskInput:
    return _RemoteRolloutTaskInput(
        task_id=task_id,
        data={"prompt": "p"},
        workflow=None,
        workflow_kwargs={},
        should_accept_fn=None,
    )


async def _run_callback(controller, task_id: int):
    controller.staleness_manager.on_rollout_enqueued()
    controller.staleness_manager.on_rollout_submitted()
    return await controller._create_submit_callback(_task(task_id))()


def _discards(calls):
    return [
        (worker_id, kwargs["task_id"])
        for worker_id, method, kwargs in calls
        if method == "discard_rollout_result"
    ]


@pytest.mark.asyncio
async def test_controller_discards_worker_result_after_waiting_times_out():
    controller = make_arena_stream_controller()
    controller.config.request_timeout = 0.05
    calls = []

    async def call_engine(worker_id, method, **kwargs):
        calls.append((worker_id, method, kwargs))
        if method == "submit":
            # The worker accepts the task but never calls back in time.
            return kwargs["task_id"]
        return None

    controller.scheduler.async_call_engine = call_engine

    assert await _run_callback(controller, 11) is None
    assert _discards(calls) == [("worker", 11)]
    assert 11 not in controller._pending_futures


@pytest.mark.asyncio
async def test_controller_discards_worker_result_when_fetching_it_fails():
    controller = make_arena_stream_controller()
    calls = []

    async def call_engine(worker_id, method, **kwargs):
        calls.append((worker_id, method, kwargs))
        if method == "submit":
            controller._pending_futures[kwargs["task_id"]].set_result(None)
            return kwargs["task_id"]
        if method == "get_rollout_result":
            raise ConnectionError("worker RPC failed")
        return None

    controller.scheduler.async_call_engine = call_engine

    assert await _run_callback(controller, 12) is None
    assert _discards(calls) == [("worker", 12)]


@pytest.mark.asyncio
async def test_controller_does_not_discard_when_no_worker_was_chosen():
    controller = make_arena_stream_controller()
    calls = []

    def no_worker():
        raise RuntimeError("no rollout workers")

    async def call_engine(worker_id, method, **kwargs):
        calls.append((worker_id, method, kwargs))
        return None

    controller._choose_worker = no_worker
    controller.scheduler.async_call_engine = call_engine

    assert await _run_callback(controller, 13) is None
    assert calls == []


@pytest.mark.asyncio
async def test_controller_does_not_discard_a_fetched_result():
    controller = make_arena_stream_controller()
    calls = []

    async def call_engine(worker_id, method, **kwargs):
        calls.append((worker_id, method, kwargs))
        if method == "submit":
            controller._pending_futures[kwargs["task_id"]].set_result(None)
            return kwargs["task_id"]
        return {"trajectory": {"prompt": "p"}, "metrics": None}

    controller.scheduler.async_call_engine = call_engine

    result = await _run_callback(controller, 14)
    assert result.trajectory == {"prompt": "p"}
    assert _discards(calls) == []


@pytest.mark.asyncio
async def test_controller_survives_a_failing_discard_rpc():
    controller = make_arena_stream_controller()
    controller.config.request_timeout = 0.05

    async def call_engine(worker_id, method, **kwargs):
        if method == "submit":
            return kwargs["task_id"]
        if method == "discard_rollout_result":
            raise ConnectionError("worker is gone")
        return None

    controller.scheduler.async_call_engine = call_engine

    assert await _run_callback(controller, 15) is None
    assert controller.staleness_manager.get_stats().rejected == 1


# --- WorkflowExecutor: remote shards of a discarded trajectory ---------------


@pytest.mark.parametrize(
    "result,cleared",
    [
        (_RolloutResult(task_id=1, trajectory={"ids": "shard"}), [{"ids": "shard"}]),
        (_RolloutResult(task_id=1, trajectory=None), []),
        (WorkflowContractFailure(message="bad"), []),
        (None, []),
    ],
)
def test_discarded_trajectory_shards_are_cleared_like_rejected_ones(result, cleared):
    executor = object.__new__(WorkflowExecutor)
    seen = []

    async def clear(traj):
        seen.append(traj)

    executor._clear_rejected_trajectory = clear
    executor._release_discarded_result(result)
    assert seen == cleared
