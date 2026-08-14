# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from areal.api import ModelRequest
from areal.api.cli_args import GenerationHyperparameters, SGLangConfig
from areal.engine.sglang_remote import SGLangBackend
from areal.experimental.openai.proxy import proxy_rollout_server
from areal.experimental.openai.proxy.proxy_rollout_server import (
    _deterministic_sampling_seed,
)
from areal.experimental.openai.proxy.server import SessionData
from areal.infra import workflow_context
from areal.infra import workflow_executor as workflow_executor_module
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.infra.workflow_executor import (
    BatchTaskDispatcher,
    TaskIdGenerator,
    WorkflowExecutor,
    _select_results,
)


def test_sampling_seed_is_stable_across_calls():
    assert _deterministic_sampling_seed("17:3", 0) == _deterministic_sampling_seed(
        "17:3", 0
    )


def test_sampling_seed_differs_per_request_and_per_sample():
    assert _deterministic_sampling_seed("17:3", 0) != _deterministic_sampling_seed(
        "17:3", 1
    )
    assert _deterministic_sampling_seed("17:3", 0) != _deterministic_sampling_seed(
        "17:4", 0
    )


def test_sampling_seed_identity_ignores_physical_session_suffix():
    sessions = [
        SessionData("17:3-0", sampling_seed_identity="17:3"),
        SessionData("17:3-1", sampling_seed_identity="17:3"),
    ]

    seeds = [
        _deterministic_sampling_seed(
            session.sampling_seed_identity,
            session.next_sampling_request_index(),
        )
        for session in sessions
    ]

    assert seeds[0] == seeds[1]


def test_sampling_request_indices_are_unique_under_concurrency():
    session = SessionData("17:3-0", sampling_seed_identity="17:3")

    with ThreadPoolExecutor(max_workers=8) as executor:
        indices = list(
            executor.map(lambda _: session.next_sampling_request_index(), range(32))
        )

    assert sorted(indices) == list(range(32))


@pytest.mark.asyncio
async def test_proxy_allocates_unique_seeds_before_concurrent_generation(monkeypatch):
    session = SessionData("17:3-0", sampling_seed_identity="17:3")
    monkeypatch.setattr(proxy_rollout_server, "_openai_client", object())
    monkeypatch.setattr(proxy_rollout_server, "_deterministic_sampling", True)
    monkeypatch.setitem(
        proxy_rollout_server._session_cache, session.session_id, session
    )

    async def create_fn(*, areal_cache, seed, temperature, top_p):
        await asyncio.sleep(0)
        return seed

    seeds = await asyncio.gather(
        *[
            proxy_rollout_server._call_client_create(
                create_fn,
                {"temperature": 1.0, "top_p": 1.0},
                session.session_id,
            )
            for _ in range(8)
        ]
    )

    expected = {
        _deterministic_sampling_seed(session.sampling_seed_identity, i)
        for i in range(8)
    }
    assert set(seeds) == expected


@pytest.mark.asyncio
async def test_proxy_explicit_seed_still_consumes_request_index(monkeypatch):
    session = SessionData("17:3-0", sampling_seed_identity="17:3")
    monkeypatch.setattr(proxy_rollout_server, "_openai_client", object())
    monkeypatch.setattr(proxy_rollout_server, "_deterministic_sampling", True)
    monkeypatch.setitem(
        proxy_rollout_server._session_cache, session.session_id, session
    )

    async def create_fn(*, areal_cache, seed, temperature, top_p):
        return seed

    explicit_seed = await proxy_rollout_server._call_client_create(
        create_fn,
        {"seed": 123, "temperature": 1.0, "top_p": 1.0},
        session.session_id,
    )
    derived_seed = await proxy_rollout_server._call_client_create(
        create_fn,
        {"temperature": 1.0, "top_p": 1.0},
        session.session_id,
    )

    assert explicit_seed == 123
    assert derived_seed == _deterministic_sampling_seed(
        session.sampling_seed_identity, 1
    )


@pytest.mark.asyncio
async def test_grouped_rollout_is_concurrent_and_sample_ordered():
    class _Workflow:
        active = 0
        max_active = 0

        async def arun_episode(self, engine, data):
            sample_idx = workflow_context.get().sample_idx
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01 * (3 - sample_idx))
            self.active -= 1
            return {"sample_idx": torch.tensor([[sample_idx]])}

    workflow = _Workflow()
    grouped = GroupedRolloutWorkflow(
        workflow=workflow,
        group_size=3,
        logger=Mock(),
    )
    engine = SimpleNamespace(config=SimpleNamespace(deterministic_sampling=True))

    result = await grouped.arun_episode(engine, {})

    assert workflow.max_active == 3
    assert result is not None
    assert result["sample_idx"].tolist() == [[0], [1], [2]]


def test_sglang_request_forwards_sampling_seed_when_set():
    req = ModelRequest(
        input_ids=[1, 2, 3],
        gconfig=GenerationHyperparameters(seed=12345),
    )

    request = SGLangBackend().build_generation_request(req, with_lora=False, version=0)

    assert request.payload["sampling_params"]["sampling_seed"] == 12345


def test_sglang_request_omits_sampling_seed_by_default():
    req = ModelRequest(input_ids=[1, 2, 3], gconfig=GenerationHyperparameters())

    request = SGLangBackend().build_generation_request(req, with_lora=False, version=0)

    assert "sampling_seed" not in request.payload["sampling_params"]


def test_sglang_server_args_enable_deterministic_inference(monkeypatch):
    monkeypatch.setattr(
        "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
        lambda *_: True,
    )
    args = SGLangConfig.build_args(
        SGLangConfig(
            model_path="test-model",
            enable_deterministic_inference=True,
        ),
        tp_size=1,
        base_gpu_id=0,
    )

    assert args["enable_deterministic_inference"] is True


@dataclass
class _FakeTimedResult:
    task_id: int
    create_time: float
    data: object | None = None


def test_select_results_is_task_ordered_when_deterministic():
    # Arrival order (create_time) deliberately disagrees with task id order.
    drained = [
        _FakeTimedResult(task_id=2, create_time=1.0),
        _FakeTimedResult(task_id=0, create_time=2.0),
        _FakeTimedResult(task_id=1, create_time=3.0),
    ]

    selected, pending = _select_results(drained, count=2, deterministic=True)

    assert [r.task_id for r in selected] == [0, 1]
    assert [r.task_id for r in pending] == [2]


def test_select_results_follows_explicit_submission_frontier():
    drained = [
        _FakeTimedResult(task_id=1, create_time=1.0),
        _FakeTimedResult(task_id=50, create_time=2.0),
        _FakeTimedResult(task_id=100, create_time=3.0),
    ]

    selected, pending = _select_results(
        drained,
        count=2,
        deterministic=True,
        task_frontier=(100, 1),
    )

    assert [r.task_id for r in selected] == [100, 1]
    assert [r.task_id for r in pending] == [50]


def test_select_results_is_arrival_ordered_by_default():
    drained = [
        _FakeTimedResult(task_id=2, create_time=1.0),
        _FakeTimedResult(task_id=0, create_time=2.0),
        _FakeTimedResult(task_id=1, create_time=3.0),
    ]

    selected, pending = _select_results(drained, count=2, deterministic=False)

    # Oldest-first selection is preserved; the selected order itself is
    # shuffled, so only membership is asserted here.
    assert {r.task_id for r in selected} == {2, 0}
    assert [r.task_id for r in pending] == [1]


def test_wait_results_does_not_skip_unfinished_task_frontier():
    dispatcher = object.__new__(BatchTaskDispatcher)
    dispatcher.deterministic_order = True
    dispatcher._result_cv = threading.Condition()
    dispatcher._active_task_ids = {0: None, 1: None, 2: None}
    dispatcher._shutdown_event = threading.Event()
    dispatcher._pending_results = {
        1: _FakeTimedResult(task_id=1, create_time=1.0, data="one"),
        2: _FakeTimedResult(task_id=2, create_time=2.0, data="two"),
    }
    dispatcher._check_thread_exception = lambda: None

    assert dispatcher.wait_results(2, timeout=0, raise_timeout=False) == []
    assert set(dispatcher._pending_results) == {1, 2}

    dispatcher._pending_results[0] = _FakeTimedResult(
        task_id=0, create_time=3.0, data="zero"
    )
    results = dispatcher.wait_results(2, timeout=0)

    assert results == ["zero", "one"]
    assert set(dispatcher._pending_results) == {2}
    assert dispatcher._active_task_ids == {2: None}


def test_wait_results_fails_fast_when_dispatcher_is_shutting_down():
    dispatcher = object.__new__(BatchTaskDispatcher)
    dispatcher.deterministic_order = True
    dispatcher._result_cv = threading.Condition()
    dispatcher._active_task_ids = {0: None}
    dispatcher._pending_results = {}
    dispatcher._shutdown_event = threading.Event()
    dispatcher._shutdown_event.set()
    dispatcher._check_thread_exception = lambda: None

    with pytest.raises(RuntimeError, match="shutting down"):
        dispatcher.wait_results(1, timeout=1)


def test_callback_registration_rejects_duplicate_task_id():
    dispatcher = object.__new__(BatchTaskDispatcher)
    dispatcher._result_cv = threading.Condition()
    dispatcher._task_callbacks = {}

    dispatcher.register_callback(7, "http://first")

    with pytest.raises(ValueError, match="already registered"):
        dispatcher.register_callback(7, "http://second")
    assert dispatcher._task_callbacks == {7: "http://first"}


def test_submit_task_input_rolls_back_when_enqueue_hook_fails():
    dispatcher = object.__new__(BatchTaskDispatcher)
    dispatcher._check_thread_exception = lambda: None
    dispatcher._result_cv = threading.Condition()
    dispatcher._input_cv = threading.Condition()
    dispatcher._active_task_ids = {}
    dispatcher._pending_inputs = deque()
    dispatcher.staleness_manager = SimpleNamespace(
        on_rollout_enqueued=Mock(side_effect=RuntimeError("hook failed"))
    )
    dispatcher.enable_tracing = False
    task_input = SimpleNamespace(task_id=7)

    with pytest.raises(RuntimeError, match="hook failed"):
        dispatcher.submit_task_input(task_input)

    assert dispatcher._active_task_ids == {}
    assert list(dispatcher._pending_inputs) == []


def test_wait_for_task_removes_middle_item_from_deterministic_frontier():
    dispatcher = object.__new__(BatchTaskDispatcher)
    dispatcher.deterministic_order = True
    dispatcher._result_cv = threading.Condition()
    dispatcher._active_task_ids = {0: None, 1: None, 2: None}
    dispatcher._shutdown_event = threading.Event()
    dispatcher._pending_results = {
        task_id: _FakeTimedResult(task_id, float(task_id), str(task_id))
        for task_id in range(3)
    }
    dispatcher._check_thread_exception = lambda: None

    assert dispatcher.wait_for_task(1, timeout=0) == "1"
    assert dispatcher.wait_results(2, timeout=0) == ["0", "2"]


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
    ] == [
        2,
        1,
    ]


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
    ] == [
        2,
        1,
    ]


def test_task_id_generator_advances_past_explicit_id():
    generator = TaskIdGenerator()

    generator.reserve_at_least(7)

    assert generator.next() == 8


def test_responses_and_completions_both_accept_seed():
    import inspect

    from areal.experimental.openai.client import (
        AsyncCompletionsWithReward,
        AsyncResponsesWithReward,
    )

    for cls in (AsyncCompletionsWithReward, AsyncResponsesWithReward):
        params = inspect.signature(cls.create).parameters
        assert "seed" in params, f"{cls.__name__}.create is missing a seed parameter"
