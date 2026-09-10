# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from areal.api import ModelRequest, ModelResponse
from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    PPOConfig,
    SGLangConfig,
)
from areal.engine.sglang_remote import SGLangBackend
from areal.experimental.openai.client import ArealOpenAI
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
    _select_results,
)
from areal.v2.inference_service.data_proxy.session import SessionData as V2SessionData
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend


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


def test_v2_sampling_request_indices_are_unique_under_concurrency():
    session = V2SessionData("17:3-0", sampling_seed_identity="17:3")

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


def test_sglang_v2_request_forwards_sampling_seed_when_set():
    req = ModelRequest(
        input_ids=[1, 2, 3],
        gconfig=GenerationHyperparameters(seed=12345),
    )

    request = SGLangBridgeBackend().build_generation_request(
        req, with_lora=False, version=0
    )

    assert request.payload["sampling_params"]["sampling_seed"] == 12345


def test_sglang_v2_request_omits_sampling_seed_by_default():
    req = ModelRequest(input_ids=[1, 2, 3], gconfig=GenerationHyperparameters())

    request = SGLangBridgeBackend().build_generation_request(
        req, with_lora=False, version=0
    )

    assert "sampling_seed" not in request.payload["sampling_params"]


@pytest.mark.asyncio
async def test_areal_openai_forwards_seed_into_model_request(monkeypatch):
    monkeypatch.setattr(
        "areal.utils.hf_utils.pkg_version.is_version_greater_or_equal",
        lambda *_: False,
    )
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = [10, 11]
    tokenizer.decode.return_value = "ok"
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0

    class CapturingEngine:
        async def agenerate(self, req):
            self.request = req
            return ModelResponse(
                input_tokens=req.input_ids,
                output_tokens=[3],
                output_logprobs=[-0.1],
                output_versions=[0],
                stop_reason="length",
                tokenizer=tokenizer,
            )

    engine = CapturingEngine()
    client = ArealOpenAI(engine=engine, tokenizer=tokenizer, api_key="test")
    try:
        await client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=4,
            seed=12345,
        )
    finally:
        await client.close()

    assert engine.request.gconfig.seed == 12345


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


@pytest.mark.parametrize("attention_backend", ["flashinfer", "fa3", "triton", None])
def test_sglang_deterministic_inference_supported_backend_does_not_warn(
    monkeypatch, attention_backend
):
    monkeypatch.setattr(
        "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
        lambda *_: True,
    )
    mock_logger = Mock()
    monkeypatch.setattr("areal.api.cli_args.logger", mock_logger)

    SGLangConfig.build_args(
        SGLangConfig(
            model_path="test-model",
            attention_backend=attention_backend,
            enable_deterministic_inference=True,
        ),
        tp_size=1,
        base_gpu_id=0,
    )

    mock_logger.warning.assert_not_called()


def test_sglang_deterministic_inference_unsupported_backend_warns(monkeypatch):
    monkeypatch.setattr(
        "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
        lambda *_: True,
    )
    mock_logger = Mock()
    monkeypatch.setattr("areal.api.cli_args.logger", mock_logger)

    SGLangConfig.build_args(
        SGLangConfig(
            model_path="test-model",
            attention_backend="torch_native",
            enable_deterministic_inference=True,
        ),
        tp_size=1,
        base_gpu_id=0,
    )

    mock_logger.warning.assert_called_once()
    assert "torch_native" in mock_logger.warning.call_args.args


def _deterministic_ppo_config(
    *,
    gconfig: GenerationHyperparameters,
    eval_gconfig: GenerationHyperparameters | None = None,
) -> PPOConfig:
    return PPOConfig(
        rollout=InferenceEngineConfig(
            backend="sglang:d1",
            deterministic_sampling=True,
        ),
        gconfig=gconfig,
        eval_gconfig=eval_gconfig,
    )


def test_grouped_deterministic_training_rejects_shared_gconfig_seed():
    with pytest.raises(ValueError, match=r"shared gconfig\.seed"):
        _deterministic_ppo_config(
            gconfig=GenerationHyperparameters(n_samples=4, seed=42),
        )


def test_grouped_deterministic_evaluation_rejects_shared_gconfig_seed():
    with pytest.raises(ValueError, match=r"shared eval_gconfig\.seed"):
        _deterministic_ppo_config(
            gconfig=GenerationHyperparameters(n_samples=1),
            eval_gconfig=GenerationHyperparameters(n_samples=4, seed=42),
        )


def test_single_sample_deterministic_config_allows_explicit_seed():
    config = _deterministic_ppo_config(
        gconfig=GenerationHyperparameters(n_samples=1, seed=42),
    )

    assert config.gconfig.seed == 42


def test_grouped_deterministic_config_allows_derived_seeds():
    config = _deterministic_ppo_config(
        gconfig=GenerationHyperparameters(n_samples=4, seed=None),
    )

    assert config.gconfig.seed is None


def test_grouped_nondeterministic_config_preserves_explicit_seed():
    config = PPOConfig(
        rollout=InferenceEngineConfig(
            backend="sglang:d1",
            deterministic_sampling=False,
        ),
        gconfig=GenerationHyperparameters(n_samples=4, seed=42),
    )

    assert config.gconfig.seed == 42


@dataclass
class _FakeTimedResult:
    task_id: int
    create_time: float
    data: object | None = None


def test_select_results_is_task_ordered_without_shuffle_when_deterministic(
    monkeypatch,
):
    mock_shuffle = Mock()
    monkeypatch.setattr(workflow_executor_module.random, "shuffle", mock_shuffle)
    # Arrival order (create_time) deliberately disagrees with task id order.
    drained = [
        _FakeTimedResult(task_id=2, create_time=1.0),
        _FakeTimedResult(task_id=0, create_time=2.0),
        _FakeTimedResult(task_id=1, create_time=3.0),
    ]

    selected, pending = _select_results(drained, count=2, deterministic=True)

    assert [r.task_id for r in selected] == [0, 1]
    assert [r.task_id for r in pending] == [2]
    mock_shuffle.assert_not_called()


def test_select_results_is_arrival_ordered_and_shuffled_by_default(monkeypatch):
    mock_shuffle = Mock()
    monkeypatch.setattr(workflow_executor_module.random, "shuffle", mock_shuffle)
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
    mock_shuffle.assert_called_once_with(selected)


def test_wait_results_selects_completed_tasks_when_deterministic():
    dispatcher = object.__new__(BatchTaskDispatcher)
    dispatcher.deterministic_order = True
    dispatcher._result_cv = threading.Condition()
    dispatcher._active_task_ids = {0, 1, 2}
    dispatcher._pending_results = {
        1: _FakeTimedResult(task_id=1, create_time=1.0, data="one"),
        2: _FakeTimedResult(task_id=2, create_time=2.0, data="two"),
    }
    dispatcher._check_thread_exception = lambda: None

    results = dispatcher.wait_results(2, timeout=0)

    assert results == ["one", "two"]
    assert dispatcher._pending_results == {}
    assert dispatcher._active_task_ids == {0}


def test_wait_for_task_removes_result_before_deterministic_batch_selection():
    dispatcher = object.__new__(BatchTaskDispatcher)
    dispatcher.deterministic_order = True
    dispatcher._result_cv = threading.Condition()
    dispatcher._active_task_ids = {0, 1, 2}
    dispatcher._pending_results = {
        task_id: _FakeTimedResult(task_id, float(task_id), str(task_id))
        for task_id in range(3)
    }
    dispatcher._check_thread_exception = lambda: None

    assert dispatcher.wait_for_task(1, timeout=0) == "1"
    assert dispatcher.wait_results(2, timeout=0) == ["0", "2"]


def test_responses_and_completions_both_accept_seed():
    import inspect

    from areal.experimental.openai.client import (
        AsyncCompletionsWithReward,
        AsyncResponsesWithReward,
    )

    for cls in (AsyncCompletionsWithReward, AsyncResponsesWithReward):
        params = inspect.signature(cls.create).parameters
        assert "seed" in params, f"{cls.__name__}.create is missing a seed parameter"
