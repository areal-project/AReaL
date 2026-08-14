# SPDX-License-Identifier: Apache-2.0

import asyncio
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from areal.api import ModelRequest
from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    SGLangConfig,
)
from areal.engine.sglang_remote import SGLangBackend
from areal.experimental.openai.proxy import proxy_rollout_server
from areal.experimental.openai.proxy.proxy_rollout_server import (
    _deterministic_sampling_seed,
)
from areal.experimental.openai.proxy.server import SessionData
from areal.infra import workflow_context
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.infra.workflow_executor import _select_results


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


@pytest.mark.parametrize("attention_backend", [None, "flashinfer", "fa3", "triton"])
def test_sglang_deterministic_inference_accepts_supported_backends(
    monkeypatch, attention_backend
):
    monkeypatch.setattr(
        "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
        lambda *_: True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SGLangConfig.build_args(
            SGLangConfig(
                model_path="test-model",
                attention_backend=attention_backend,
                enable_deterministic_inference=True,
            ),
            tp_size=1,
            base_gpu_id=0,
        )


def test_sglang_deterministic_inference_warns_for_unsupported_backend(monkeypatch):
    monkeypatch.setattr(
        "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
        lambda *_: True,
    )

    with pytest.warns(UserWarning, match="only documented for attention backends"):
        SGLangConfig.build_args(
            SGLangConfig(
                model_path="test-model",
                attention_backend="torch_native",
                enable_deterministic_inference=True,
            ),
            tp_size=1,
            base_gpu_id=0,
        )


def test_deterministic_sampling_warns_for_async_staleness():
    with pytest.warns(UserWarning, match="max_head_offpolicyness=0"):
        InferenceEngineConfig(
            backend="sglang:d1",
            deterministic_sampling=True,
            max_head_offpolicyness=2,
        )


def test_deterministic_sampling_accepts_synchronous_rollouts():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        InferenceEngineConfig(
            backend="sglang:d1",
            deterministic_sampling=True,
            max_head_offpolicyness=0,
        )


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


def test_select_results_is_creation_ordered_by_default():
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


def test_responses_and_completions_both_accept_seed():
    import inspect

    from areal.experimental.openai.client import (
        AsyncCompletionsWithReward,
        AsyncResponsesWithReward,
    )

    for cls in (AsyncCompletionsWithReward, AsyncResponsesWithReward):
        params = inspect.signature(cls.create).parameters
        assert "seed" in params, f"{cls.__name__}.create is missing a seed parameter"
