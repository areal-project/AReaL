"""Tests for generic workflow dispatcher correctness."""

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.infra import workflow_executor as workflow_executor_module
from areal.infra.controller.rollout_controller import RolloutController
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.infra.workflow_executor import (
    BatchTaskDispatcher,
    TaskIdGenerator,
    WorkflowExecutor,
)


def _dispatcher_for_batch_wait(results):
    dispatcher = object.__new__(BatchTaskDispatcher)
    dispatcher._result_cv = threading.Condition()
    dispatcher._pending_results = {result.task_id: result for result in results}
    dispatcher._active_task_ids = {result.task_id for result in results}
    dispatcher._shutdown_event = threading.Event()
    dispatcher._check_thread_exception = lambda: None
    return dispatcher


def test_callback_registration_rejects_duplicate_task_id():
    dispatcher = object.__new__(BatchTaskDispatcher)
    dispatcher._result_cv = threading.Condition()
    dispatcher._task_callbacks = {}

    dispatcher.register_callback(7, "http://first")

    with pytest.raises(ValueError, match="already registered"):
        dispatcher.register_callback(7, "http://second")
    assert dispatcher._task_callbacks == {7: "http://first"}


def test_cancel_callback_only_removes_matching_registration():
    dispatcher = object.__new__(BatchTaskDispatcher)
    dispatcher._result_cv = threading.Condition()
    dispatcher._task_callbacks = {7: "http://current"}

    dispatcher.cancel_callback(7, "http://stale")
    assert dispatcher._task_callbacks == {7: "http://current"}

    dispatcher.cancel_callback(7, "http://current")
    assert dispatcher._task_callbacks == {}


def test_submit_task_input_rejects_duplicate_active_task_id():
    dispatcher = object.__new__(BatchTaskDispatcher)
    dispatcher._check_thread_exception = lambda: None
    dispatcher._result_cv = threading.Condition()
    dispatcher._input_cv = threading.Condition()
    dispatcher._active_task_ids = {7}
    dispatcher._pending_inputs = deque()
    dispatcher.staleness_manager = SimpleNamespace(on_rollout_enqueued=Mock())
    dispatcher.enable_tracing = False

    with pytest.raises(ValueError, match="already active"):
        dispatcher.submit_task_input(SimpleNamespace(task_id=7))

    dispatcher.staleness_manager.on_rollout_enqueued.assert_not_called()
    assert list(dispatcher._pending_inputs) == []


def test_submit_task_input_rolls_back_when_enqueue_hook_fails():
    dispatcher = object.__new__(BatchTaskDispatcher)
    dispatcher._check_thread_exception = lambda: None
    dispatcher._result_cv = threading.Condition()
    dispatcher._input_cv = threading.Condition()
    dispatcher._active_task_ids = set()
    dispatcher._pending_inputs = deque()
    dispatcher.staleness_manager = SimpleNamespace(
        on_rollout_enqueued=Mock(side_effect=RuntimeError("hook failed"))
    )
    dispatcher.enable_tracing = False
    task_input = SimpleNamespace(task_id=7)

    with pytest.raises(RuntimeError, match="hook failed"):
        dispatcher.submit_task_input(task_input)

    assert dispatcher._active_task_ids == set()
    assert list(dispatcher._pending_inputs) == []


def test_wait_results_fails_fast_when_dispatcher_is_shutting_down():
    dispatcher = _dispatcher_for_batch_wait([])
    dispatcher._shutdown_event.set()

    with pytest.raises(RuntimeError, match="shutting down"):
        dispatcher.wait_results(1, timeout=1)


def test_wait_for_task_detects_result_consumed_by_another_waiter():
    dispatcher = _dispatcher_for_batch_wait([])
    dispatcher._active_task_ids = {7}
    waiter_entered = threading.Event()
    dispatcher._check_thread_exception = waiter_entered.set

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(dispatcher.wait_for_task, 7, 1)
        assert waiter_entered.wait(timeout=1)
        with dispatcher._result_cv:
            dispatcher._active_task_ids.remove(7)
            dispatcher._result_cv.notify_all()

        with pytest.raises(RuntimeError, match="consumed by another waiter"):
            future.result(timeout=1)


def test_wait_results_removes_only_selected_results_atomically():
    results = [
        SimpleNamespace(task_id=0, create_time=3.0, data="newest"),
        SimpleNamespace(task_id=1, create_time=1.0, data="oldest"),
        SimpleNamespace(task_id=2, create_time=2.0, data="middle"),
    ]
    dispatcher = _dispatcher_for_batch_wait(results)

    selected = dispatcher.wait_results(2, timeout=0)

    assert set(selected) == {"oldest", "middle"}
    assert set(dispatcher._pending_results) == {0}
    assert dispatcher._active_task_ids == {0}


def test_workflow_executor_binds_callback_to_allocated_task_id(monkeypatch):
    executor = object.__new__(WorkflowExecutor)
    executor._task_id_generator = TaskIdGenerator()
    executor._dispatcher = Mock()
    monkeypatch.setattr(
        workflow_executor_module.perf_tracer, "register_task", lambda _: None
    )

    task_id = executor.submit({}, workflow=Mock(), callback_addr="http://callback")

    assert task_id == 0
    executor.dispatcher.register_callback.assert_called_once_with(0, "http://callback")
    submitted = executor.dispatcher.submit_task_input.call_args.args[0]
    assert submitted.task_id == 0


def test_workflow_executor_cancels_own_callback_when_submit_fails(monkeypatch):
    executor = object.__new__(WorkflowExecutor)
    executor._task_id_generator = TaskIdGenerator()
    executor._dispatcher = Mock()
    executor.dispatcher.submit_task_input.side_effect = ValueError("duplicate")
    monkeypatch.setattr(
        workflow_executor_module.perf_tracer, "register_task", lambda _: None
    )

    with pytest.raises(ValueError, match="duplicate"):
        executor.submit({}, workflow=Mock(), callback_addr="http://callback")

    executor.dispatcher.cancel_callback.assert_called_once_with(0, "http://callback")


def test_dynamic_batch_counts_rejections_as_attempts():
    dispatcher = object.__new__(BatchTaskDispatcher)
    dispatcher._input_cv = threading.Condition()
    dispatcher._pending_inputs = []
    dispatcher.staleness_manager = SimpleNamespace(get_pending_limit=lambda: 0)
    dispatcher.runner = SimpleNamespace(
        max_queue_size=4, get_input_queue_size=lambda: 4
    )
    dispatcher.enable_tracing = False
    dispatcher.wait_results = Mock(side_effect=[[None], ["accepted"]])

    results = dispatcher.active_submit_and_wait(iter(()), batch_size=2, dynamic_bs=True)

    assert results == ["accepted"]
    assert [
        call.kwargs["count"] for call in dispatcher.wait_results.call_args_list
    ] == [2, 1]


def test_fixed_batch_replaces_rejected_attempts_in_order():
    dispatcher = object.__new__(BatchTaskDispatcher)
    dispatcher._input_cv = threading.Condition()
    dispatcher._pending_inputs = []
    dispatcher.staleness_manager = SimpleNamespace(get_pending_limit=lambda: 0)
    dispatcher.runner = SimpleNamespace(
        max_queue_size=4, get_input_queue_size=lambda: 4
    )
    dispatcher.enable_tracing = False
    dispatcher.wait_results = Mock(side_effect=[[None, "first"], ["replacement"]])

    results = dispatcher.active_submit_and_wait(iter(()), batch_size=2)

    assert results == ["first", "replacement"]
    assert [
        call.kwargs["count"] for call in dispatcher.wait_results.call_args_list
    ] == [2, 1]


def test_task_id_generator_advances_past_explicit_id():
    generator = TaskIdGenerator()

    generator.reserve_at_least(7)

    assert generator.next() == 8


def test_workflow_executor_reserves_explicit_task_id(monkeypatch):
    executor = object.__new__(WorkflowExecutor)
    executor._task_id_generator = TaskIdGenerator()
    executor._dispatcher = Mock()
    monkeypatch.setattr(
        workflow_executor_module.perf_tracer, "register_task", lambda _: None
    )

    assert executor.submit({}, workflow=Mock(), task_id=7) == 7
    assert executor.submit({}, workflow=Mock()) == 8


def test_rollout_controller_reserves_explicit_task_id():
    controller = object.__new__(RolloutController)
    controller._task_id_generator = TaskIdGenerator()
    controller._dispatcher = Mock()
    controller._resolve_workflow_str = Mock(return_value="workflow.path")
    controller._resolve_should_accept_fn = Mock(return_value=None)

    assert controller.submit({}, workflow=Mock(), task_id=7) == 7
    assert controller.submit({}, workflow=Mock()) == 8


def test_remote_engine_delegates_callback_after_task_id_allocation():
    engine = object.__new__(RemoteInfEngine)
    engine.config = SimpleNamespace(agent=None)
    engine._resolve_workflow = Mock(return_value="resolved-workflow")
    engine._resolve_should_accept_fn = Mock(return_value="resolved-filter")
    engine.workflow_executor = Mock()
    engine.workflow_executor.submit.return_value = 9

    task_id = engine.submit(
        {},
        workflow=Mock(),
        callback_addr="http://callback",
    )

    assert task_id == 9
    engine.workflow_executor.dispatcher.register_callback.assert_not_called()
    engine.workflow_executor.submit.assert_called_once_with(
        {},
        workflow="resolved-workflow",
        should_accept_fn="resolved-filter",
        task_id=None,
        is_eval=False,
        callback_addr="http://callback",
    )
