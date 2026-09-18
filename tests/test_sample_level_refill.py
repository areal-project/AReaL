# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from areal.api import RolloutWorkflow
from areal.api.cli_args import InferenceEngineConfig
from areal.infra import workflow_context
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.infra.sample_capacity import SampleCapacity
from areal.infra.staleness_manager import StalenessManager
from areal.infra.workflow_executor import BatchTaskDispatcher, WorkflowExecutor
from areal.v2.inference_service.controller.workflow import InferenceServiceWorkflow


class _GatedWorkflow(RolloutWorkflow):
    def __init__(self, groups: int, size: int):
        self.size = size
        self.started = [threading.Event() for _ in range(groups)]
        self.gates: dict[tuple[int, int], asyncio.Event] = {}
        self.loop = None

    async def arun_episode(self, engine, data):
        self.loop = asyncio.get_running_loop()
        index = workflow_context.get().sample_idx
        group = data["group"]
        gate = asyncio.Event()
        self.gates[group, index] = gate
        if index == self.size - 1:
            self.started[group].set()
        await gate.wait()
        return {"input_ids": torch.tensor([[group, index]])}

    def release(self, group: int, index: int):
        self.loop.call_soon_threadsafe(self.gates[group, index].set)

    def release_all(self):
        if self.loop is not None:
            self.loop.call_soon_threadsafe(
                lambda: [gate.set() for gate in self.gates.values()]
            )


def _executor(*, group_budget=8, sample_limit=48):
    config = InferenceEngineConfig(
        backend="sglang:d1",
        consumer_batch_size=group_budget,
        max_concurrent_rollouts=4,
        max_concurrent_samples=sample_limit,
        check_trajectory_format=False,
    )
    engine = SimpleNamespace(get_version=lambda: 0)
    executor = WorkflowExecutor(config, engine)
    executor.initialize(train_data_parallel_size=1)
    return executor


def test_partial_groups_cannot_bypass_staleness_budget():
    executor = _executor(group_budget=4)
    inner = _GatedWorkflow(5, 12)
    workflow = GroupedRolloutWorkflow(inner, group_size=12, logger=MagicMock())
    try:
        for group in range(5):
            executor.submit({"group": group}, workflow, task_id=group)
        for group in range(4):
            assert inner.started[group].wait(5)
            for index in range(3):
                inner.release(group, index)
        with executor.dispatcher._input_cv:
            assert executor.dispatcher._input_cv.wait_for(
                lambda: executor.dispatcher.sample_capacity.running == 36, timeout=5
            )
            assert not executor.dispatcher._has_runner_capacity()
        assert not inner.started[4].is_set()
    finally:
        inner.release_all()
        executor.destroy()


def test_sample_notifications_deduplicate_and_isolate_attempts():
    capacity = SampleCapacity(4)
    attempt = capacity.reserve(7, 4)
    assert not capacity.complete(7, "old", 0)
    assert not capacity.complete(7, attempt, -1)
    assert not capacity.complete(7, attempt, 4)
    assert capacity.complete(7, attempt, 0)
    assert not capacity.complete(7, attempt, 0)
    assert capacity.running == 3
    assert capacity.finish(7, attempt)
    replacement = capacity.reserve(7, 2)
    assert not capacity.finish(7, attempt)
    assert not capacity.complete(7, attempt, 1)
    assert capacity.running == 2
    assert capacity.finish(7, replacement)
    assert capacity.running == 0


def test_admission_requires_whole_next_group_and_worker_uses_global_budget():
    manager = StalenessManager(SimpleNamespace(get_version=lambda: 0), 1, 1, 0)
    dispatcher = BatchTaskDispatcher(
        8, lambda _: None, manager, max_concurrent_samples=6
    )
    dispatcher.logger = MagicMock()
    first = SimpleNamespace(task_id=0, group_size=4, externally_admitted=True)
    second = SimpleNamespace(task_id=1, group_size=3, externally_admitted=True)
    dispatcher.submit_task_input(first)
    assert dispatcher._get_next_task_for_submission() is first
    dispatcher.submit_task_input(second)
    assert not dispatcher._has_runner_capacity()
    attempt = dispatcher.get_sample_attempt(0)
    dispatcher.sample_completed(0, attempt, 0)
    # Global controller already admitted this group; worker's local budget is full.
    assert manager.get_capacity(include_concurrency=False) == 0
    assert dispatcher._has_runner_capacity()
    assert dispatcher._get_next_task_for_submission() is second
    assert dispatcher.sample_capacity.running == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("serial", [False, True])
async def test_v2_reports_member_completion_before_group_export(serial):
    previous = workflow_context.get()
    reports = []
    workflow_context.set(
        workflow_context.WorkflowContext(
            task_id=1, sample_completion_callback=reports.append
        )
    )
    workflow = InferenceServiceWorkflow(
        SimpleNamespace(get_version=lambda: 0),
        agent=SimpleNamespace(run=AsyncMock(return_value=1.0)),
        group_size=2,
        serialize_group_samples=serial,
    )
    workflow._start_session = AsyncMock(
        return_value=("group", [("a", "key-a"), ("b", "key-b")])
    )
    workflow._set_last_reward = AsyncMock()

    async def export(*args, **kwargs):
        assert sorted(reports) == [0, 1]
        return None

    workflow._export_interactions = export
    try:
        assert await workflow._run_offline(MagicMock(), {}) is None
    finally:
        workflow_context.set(previous)


def test_queue_full_rolls_back_reservation_before_retry():
    from areal.infra.async_task_runner import TaskQueueFullError

    manager = StalenessManager(SimpleNamespace(get_version=lambda: 0), 1, 1, 0)
    dispatcher = BatchTaskDispatcher(
        1, lambda _: None, manager, max_concurrent_samples=4
    )
    dispatcher.logger = MagicMock()
    dispatcher.submit_task_input(SimpleNamespace(task_id=0, group_size=4))
    attempts = []

    def submit(*args, **kwargs):
        attempts.append(dispatcher.get_sample_attempt(0))
        assert manager.get_stats().running == 1
        assert dispatcher.sample_capacity.running == 4
        if len(attempts) == 1:
            raise TaskQueueFullError()
        dispatcher._shutdown_event.set()

    dispatcher.runner.submit = submit
    dispatcher._commit_loop()
    assert len(attempts) == 2
    assert attempts[0] != attempts[1]
    assert manager.get_stats().enqueued == 0
    assert manager.get_stats().running == 1


@pytest.mark.asyncio
async def test_remote_timeout_keeps_slots_until_late_confirmed_completion(monkeypatch):
    from areal.infra.controller import rollout_controller as module

    monkeypatch.setattr(module, "make_server", MagicMock())
    monkeypatch.setattr(module, "find_free_ports", lambda count: [1234])
    monkeypatch.setattr(module, "gethostip", lambda: "127.0.0.1")
    config = InferenceEngineConfig(backend="sglang:d1", request_timeout=0.01)
    scheduler = SimpleNamespace(async_call_engine=AsyncMock(return_value=0))
    controller = module.RolloutController(MagicMock(), config, scheduler)
    controller._worker_role = "rollout"
    controller.workers = [SimpleNamespace(id="worker")]
    manager = StalenessManager(controller, 1, 8, 0)
    controller._staleness_manager = manager
    dispatcher = BatchTaskDispatcher(
        8, controller._create_submit_callback, manager, max_concurrent_samples=4
    )
    dispatcher.logger = MagicMock()
    controller._dispatcher = dispatcher
    controller._start_callback_server()
    task = module._RemoteRolloutTaskInput(
        task_id=0,
        data={},
        workflow="test.Workflow",
        workflow_kwargs={},
        should_accept_fn=None,
        group_size=4,
    )
    try:
        dispatcher.submit_task_input(task)
        assert dispatcher._get_next_task_for_submission() is task
        attempt = dispatcher.get_sample_attempt(0)
        assert await controller._create_submit_callback(task)() is None
        assert manager.get_stats().rejected == 1
        assert dispatcher.sample_capacity.running == 4
        kwargs = scheduler.async_call_engine.call_args.kwargs
        assert kwargs["sample_attempt_id"] == attempt
        assert kwargs["max_retries"] == 1
        client = controller._callback_app.test_client()
        payload = {"task_id": 0, "attempt_id": attempt, "sample_idx": 0}
        for _ in range(2):
            assert (
                client.post("/callback/rollout_progress", json=payload).status_code
                == 200
            )
        assert dispatcher.sample_capacity.running == 3
        assert (
            client.post(
                "/callback/rollout_complete", json={"task_id": 0, "attempt_id": "old"}
            ).json["status"]
            == "ignored"
        )
        assert dispatcher.sample_capacity.running == 3
        assert (
            client.post("/callback/rollout_complete", json=payload).status_code == 200
        )
        assert dispatcher.sample_capacity.running == 0
        assert (
            client.post("/callback/rollout_complete", json=payload).json["status"]
            == "ignored"
        )
        assert manager.get_stats().rejected == 1
        assert not controller._pending_futures
    finally:
        controller._stop_callback_server()


@pytest.mark.asyncio
async def test_worker_progress_forwards_remote_attempt_and_terminal_identity():
    executor = _executor(sample_limit=4)
    dispatcher = executor.dispatcher
    posts = []
    dispatcher._post_callback = lambda addr, payload: posts.append((addr, payload))
    dispatcher.register_callback(5, "complete")
    dispatcher.register_progress_callback(5, "remote-attempt", "progress")

    class EmptyWorkflow(RolloutWorkflow):
        async def arun_episode(self, engine, data):
            return None

    try:
        executor.submit(
            {},
            GroupedRolloutWorkflow(EmptyWorkflow(), 4, MagicMock()),
            task_id=5,
            externally_admitted=True,
        )
        assert await asyncio.to_thread(executor.wait, 1, 5) == [None]
        progress = [payload for addr, payload in posts if addr == "progress"]
        complete = [payload for addr, payload in posts if addr == "complete"]
        assert sorted(payload["sample_idx"] for payload in progress) == [0, 1, 2, 3]
        assert all(payload["attempt_id"] == "remote-attempt" for payload in progress)
        assert complete == [{"task_id": 5, "attempt_id": "remote-attempt"}]
    finally:
        executor.destroy()


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_engine_adapter_forwards_attempt_and_confirms_submit_failure(backend):
    from areal.engine.sglang_remote import RemoteSGLangEngine
    from areal.engine.vllm_remote import RemotevLLMEngine
    from areal.infra.remote_inf_engine import RemoteInfEngine

    cls = RemoteSGLangEngine if backend == "sglang" else RemotevLLMEngine
    adapter = object.__new__(cls)
    remote = object.__new__(RemoteInfEngine)
    remote.config = InferenceEngineConfig(backend="sglang:d1")
    executor = MagicMock()
    remote._workflow_executor = executor
    remote._resolve_workflow = MagicMock(side_effect=ValueError("invalid workflow"))
    adapter._engine = remote
    with pytest.raises(ValueError, match="invalid workflow"):
        adapter.submit(
            {},
            "invalid.Workflow",
            task_id=9,
            callback_addr="complete",
            sample_progress_addr="progress",
            sample_attempt_id="attempt",
        )
    executor.dispatcher._post_callback.assert_called_once_with(
        "complete", {"task_id": 9, "attempt_id": "attempt"}
    )
    executor.submit.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_group_holds_slots_until_member_cleanup_finishes():
    from areal.infra.workflow_executor import _RolloutTaskInput

    entered = [asyncio.Event(), asyncio.Event()]
    cleaning = asyncio.Event()
    cleanup_allowed = asyncio.Event()

    class Workflow(RolloutWorkflow):
        async def arun_episode(self, engine, data):
            entered[workflow_context.get().sample_idx].set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await cleanup_allowed.wait()

    engine = SimpleNamespace(get_version=lambda: 0)
    config = InferenceEngineConfig(backend="sglang:d1")
    manager = StalenessManager(engine, 1, 1, 0)
    executor = WorkflowExecutor(config, engine, staleness_manager=manager)
    dispatcher = BatchTaskDispatcher(
        4, executor._create_workflow_task, manager, max_concurrent_samples=2
    )
    dispatcher.logger = MagicMock()
    executor._dispatcher = dispatcher
    executor.logger = MagicMock()
    pending = _RolloutTaskInput(
        1, {}, GroupedRolloutWorkflow(Workflow(), 2, MagicMock())
    )
    dispatcher.submit_task_input(pending)
    assert dispatcher._get_next_task_for_submission() is pending
    task = asyncio.create_task(executor._create_workflow_task(pending)())
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), 5)
    task.cancel()
    try:
        await asyncio.wait_for(cleaning.wait(), 5)
        assert dispatcher.sample_capacity.running == 2
        assert manager.get_stats().running == 1
    finally:
        cleanup_allowed.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert dispatcher.sample_capacity.running == 0
    assert manager.get_stats().running == 0
    assert manager.get_stats().rejected == 1


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "48"])
def test_sample_limit_invalid_config_is_rejected(limit):
    with pytest.raises(ValueError, match="max_concurrent_samples"):
        InferenceEngineConfig(backend="sglang:d1", max_concurrent_samples=limit)


def test_refill_collects_full_batch_while_initial_group_stragglers_remain():
    """Two later complete groups can feed training before either initial tail ends."""
    executor = _executor(group_budget=6, sample_limit=4)
    executor.staleness_manager.max_concurrent_rollouts = 2
    inner = _GatedWorkflow(4, 2)
    workflow = GroupedRolloutWorkflow(inner, group_size=2, logger=MagicMock())
    try:
        for group in range(4):
            executor.submit({"group": group}, workflow, task_id=group)
        assert inner.started[0].wait(5)
        assert inner.started[1].wait(5)
        # Half of each initial group completes; no initial group is trainable yet.
        inner.release(0, 0)
        inner.release(1, 0)
        assert inner.started[2].wait(5)
        inner.release(2, 0)
        inner.release(2, 1)
        assert inner.started[3].wait(5)
        inner.release(3, 0)
        inner.release(3, 1)
        batch = executor.wait(2, timeout=5)
        assert sorted(result["input_ids"][0, 0].item() for result in batch) == [2, 3]
        assert all(result["rollout_group"].row_counts == (1, 1) for result in batch)
        assert executor.staleness_manager.get_stats().running == 2
        assert executor.dispatcher.sample_capacity.running == 2
        # Switching to training must not require the two initial tails to finish.
        executor.pause()
        assert executor.is_paused()
        assert executor.dispatcher.sample_capacity.running == 2
        executor.resume()
        inner.release(0, 1)
        inner.release(1, 1)
        remaining = executor.wait(2, timeout=5)
        assert sorted(result["input_ids"][0, 0].item() for result in remaining) == [
            0,
            1,
        ]
        assert executor.dispatcher.sample_capacity.running == 0
    finally:
        inner.release_all()
        executor.destroy()


def test_group_admission_keeps_slots_until_initial_groups_finish():
    executor = _executor(group_budget=6, sample_limit=None)
    executor.staleness_manager.max_concurrent_rollouts = 2
    inner = _GatedWorkflow(4, 2)
    workflow = GroupedRolloutWorkflow(inner, group_size=2, logger=MagicMock())
    try:
        for group in range(4):
            executor.submit({"group": group}, workflow, task_id=group)
        assert inner.started[0].wait(5)
        assert inner.started[1].wait(5)
        inner.release(0, 0)
        inner.release(1, 0)
        # Independent of how soon those two short members return, group running
        # stays two and no complete training group can be delivered.
        with executor.dispatcher._input_cv:
            assert not executor.dispatcher._has_runner_capacity()
        assert not inner.started[2].is_set()
        assert executor.wait(2, timeout=0, raise_timeout=False) == []
        inner.release(0, 1)
        inner.release(1, 1)
        assert inner.started[2].wait(5)
        assert inner.started[3].wait(5)
        inner.release_all()
        batch = executor.wait(4, timeout=5)
        assert sorted(result["input_ids"][0, 0].item() for result in batch) == [
            0,
            1,
            2,
            3,
        ]
    finally:
        inner.release_all()
        executor.destroy()
