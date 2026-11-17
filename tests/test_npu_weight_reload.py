# SPDX-License-Identifier: Apache-2.0

import ast
import asyncio
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import torch
from torch import nn

from areal.engine.vllm_ext.npu_weight_reload import (
    NpuReloadModel,
    NpuWeightReload,
    weight_update_models,
)

VLLM_EXT = Path(__file__).parents[1] / "areal/engine/vllm_ext"


@pytest.fixture
def native_reload(monkeypatch):
    api = ModuleType("vllm.model_executor.model_loader.reload")
    meta = ModuleType("vllm.model_executor.model_loader.reload.meta")
    meta.SKIP_LOAD_TENSORS = set()
    meta.SKIP_TENSORS = set()
    api.initialize_layerwise_reload = MagicMock()
    api.finalize_layerwise_reload = MagicMock()
    monkeypatch.setitem(sys.modules, api.__name__, api)
    monkeypatch.setitem(sys.modules, meta.__name__, meta)
    return api, meta


@pytest.mark.parametrize("replace_storage", [False, True])
def test_reload_preserves_captured_storage(native_reload, replace_storage):
    model = nn.Linear(4, 2, bias=False)
    model.load_weights = MagicMock()
    reload = NpuWeightReload(model, None)
    reload.start()
    with torch.no_grad():
        if replace_storage:
            model.weight.data = model.weight.detach().clone()
        else:
            model.weight.fill_(3)
    if replace_storage:
        with pytest.raises(RuntimeError, match="captured tensor storage"):
            reload.finish()
    else:
        reload.finish()
        torch.testing.assert_close(
            reload.runtime_tensors["weight"], torch.full((2, 4), 3.0), rtol=0, atol=0
        )


def test_failed_finalization_is_not_retried(native_reload):
    api, _ = native_reload
    reload = NpuWeightReload(nn.Linear(4, 2, bias=False), None)
    reload.start()
    api.finalize_layerwise_reload.side_effect = ValueError("bad layout")
    with pytest.raises(ValueError, match="bad layout"):
        reload.finish()
    api.finalize_layerwise_reload.side_effect = None
    with pytest.raises(RuntimeError, match="cannot be retried"):
        reload.finish()
    api.finalize_layerwise_reload.assert_called_once()


def _runner(method="mtp", wrapped=False):
    target, draft = nn.Module(), nn.Module()
    target.load_weights = Mock(return_value=set())
    draft.load_weights = Mock(return_value=set())
    target_config = SimpleNamespace(model="target", dtype=torch.float32)
    draft_config = SimpleNamespace(model="draft", dtype=torch.float32)
    runner = SimpleNamespace(
        model=target,
        model_config=target_config,
        drafter=SimpleNamespace(
            model=SimpleNamespace(unwrap=lambda: draft) if wrapped else draft
        ),
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                method=method, draft_model_config=draft_config
            )
        ),
        device="cpu",
    )
    return runner, target, draft


@pytest.mark.parametrize("wrapped", [False, True])
def test_mtp_draft_loads_actor_weights(wrapped):
    runner, target, draft = _runner(wrapped=wrapped)
    draft_config = runner.vllm_config.speculative_config.draft_model_config
    assert weight_update_models(runner) == [
        (target, runner.model_config),
        (draft, draft_config),
    ]


@pytest.mark.parametrize("method", ["draft_model", "eagle", "ngram", None])
def test_independent_draft_keeps_own_checkpoint(method):
    runner, target, _ = _runner(method)
    if method is None:
        runner.vllm_config.speculative_config = None
    assert weight_update_models(runner) == [(target, runner.model_config)]


def test_mtp_rank_without_draft_uses_target_only():
    runner, target, _ = _runner()
    del runner.drafter
    assert weight_update_models(runner) == [(target, runner.model_config)]


@pytest.mark.parametrize("target_loads", [False, True])
@pytest.mark.parametrize("draft_fails", [False, True])
def test_shared_loader_skips_only_target_updates_and_restores_on_error(
    target_loads, draft_fails
):
    """Different checkpoint aliases can be owned exclusively by the draft loader."""
    shared = nn.Linear(2, 2, bias=False)
    loader = Mock()
    shared.weight.weight_loader = loader
    target, draft = nn.Module(), nn.Module()
    target.output = draft.input = shared
    target.load_weights = Mock(
        return_value={"output.weight"} if target_loads else set()
    )

    def load(weights):
        for _, tensor in weights:
            draft.input.weight.weight_loader(draft.input.weight, tensor)
        if draft_fails:
            raise ValueError("draft failure")

    draft.load_weights = load
    model = NpuReloadModel(target, draft)
    if draft_fails:
        with pytest.raises(ValueError, match="draft failure"):
            model.load_weights([("custom.alias", torch.ones(2, 2))])
    else:
        model.load_weights([("custom.alias", torch.ones(2, 2))])
    assert loader.call_count == (0 if target_loads else 1)
    assert shared.weight.weight_loader is loader


@pytest.fixture
def server_handlers():
    path = VLLM_EXT / "areal_vllm_server.py"
    # Isolate the handlers from optional serving imports; exercise their call order.
    selected = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in (
            "areal_pause_generation",
            "areal_continue_generation",
            "areal_update_weight_xccl",
        ):
            node.decorator_list = []
            selected.append(node)
    namespace = {
        "Request": object,
        "logger": MagicMock(),
        "_generation_run_event": asyncio.Event(),
        "build_response": lambda ret: ret,
        "to_json_response": lambda success, message: (success, message),
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


def _request(llm):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(engine_client=llm))
    )


@pytest.mark.asyncio
async def test_xccl_buckets_pause_once_until_finalization(server_handlers):
    """Repeated DP pauses must not run a dummy forward during partial reload."""
    event = server_handlers["_generation_run_event"]
    event.set()
    order = []
    partial_reload = False

    async def pause(**kwargs):
        assert not partial_reload, "DP pause would forward through unfinished weights"
        assert not event.is_set()
        order.append("pause")

    async def reset():
        order.append("reset")

    async def rpc(method):
        nonlocal partial_reload
        assert not event.is_set()
        partial_reload = method == "areal_update_weight_xccl"
        order.append(method)
        return [(True, "ok")]

    async def resume():
        assert not partial_reload
        order.append("resume")

    request = _request(
        SimpleNamespace(
            pause_generation=pause,
            reset_mm_cache=reset,
            collective_rpc=rpc,
            resume_generation=resume,
        )
    )
    for _ in range(2):
        order.clear()
        await server_handlers["areal_pause_generation"](request)
        for _ in range(3):
            assert await server_handlers["areal_update_weight_xccl"](request) == [
                (True, "ok")
            ]
            assert not event.is_set()
        assert order == ["pause", "reset"] + ["areal_update_weight_xccl"] * 3
        assert (await server_handlers["areal_continue_generation"](request))[0]
        assert order[-2:] == ["areal_finish_weight_update", "resume"]
        assert event.is_set()


@pytest.mark.asyncio
async def test_finalize_exception_does_not_resume_generation(server_handlers):
    llm = SimpleNamespace(
        collective_rpc=AsyncMock(side_effect=RuntimeError("reload failed")),
        resume_generation=AsyncMock(),
    )
    with pytest.raises(RuntimeError, match="reload failed"):
        await server_handlers["areal_continue_generation"](_request(llm))
    llm.resume_generation.assert_not_awaited()
    assert not server_handlers["_generation_run_event"].is_set()


@pytest.fixture
def worker(monkeypatch):
    path = VLLM_EXT / "vllm_worker_extension.py"
    cls = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
    )
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in (
            "areal_update_weight_xccl",
            "areal_finish_weight_update",
            "_begin_native_moe_reload",
        )
    ]
    namespace = {
        "torch": torch,
        "logger": MagicMock(),
        "current_platform": SimpleNamespace(device_type="npu", synchronize=MagicMock()),
        "set_current_vllm_config": lambda config: nullcontext(),
        "weight_update_models": weight_update_models,
        "process_weights_after_loading": MagicMock(),
        "NpuReloadModel": NpuReloadModel,
        "NpuWeightReload": MagicMock(),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    instance = namespace[cls.name]()
    instance.model_runner, target, draft = _runner()
    instance.sync = MagicMock()
    instance._uses_native_moe_reload = lambda: True
    instance.areal_weight_meta_names = ["first", "second"]
    instance.areal_weight_meta_shapes = [[2, 3], [3, 2]]
    instance.areal_weight_meta_dtypes = ["float32", "float32"]
    instance.areal_weight_meta_group_name = "test"
    instance.weight_update_groups = {"test": object()}
    broadcast = MagicMock()
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    monkeypatch.setattr(
        torch.distributed, "barrier", lambda: pytest.fail("Unexpected barrier")
    )
    return instance, broadcast, target, draft


def test_worker_fallback_xccl_reload_loads_target_and_draft(worker):
    instance, broadcast, target, draft = worker
    instance._uses_native_moe_reload = lambda: False
    assert instance.areal_update_weight_xccl()[0]
    assert broadcast.call_count == 2
    for model in (target, draft):
        assert [call.args[0][0][0] for call in model.load_weights.call_args_list] == [
            "first",
            "second",
        ]
    namespace = instance.areal_update_weight_xccl.__globals__
    assert namespace["process_weights_after_loading"].call_count == 2
    instance.sync.assert_called_once()


def test_worker_native_reload_uses_one_transaction_for_target_and_draft(worker):
    instance, _, target, draft = worker
    first = instance._begin_native_moe_reload()
    assert instance._begin_native_moe_reload() is first
    factory = instance._begin_native_moe_reload.__globals__["NpuWeightReload"]
    factory.assert_called_once()
    model, config = factory.call_args.args
    assert model.target is target and model.draft is draft
    assert config is instance.model_runner.model_config
    first.start.assert_called_once()


def test_worker_buckets_defer_finish_and_continue_is_idempotent(worker):
    instance, broadcast, _, _ = worker
    transaction = SimpleNamespace(load=MagicMock(), finish=MagicMock())
    instance._areal_weight_reload = transaction
    instance._begin_native_moe_reload = lambda: transaction
    assert instance.areal_update_weight_xccl()[0]
    instance.areal_weight_meta_group_name = "second_producer"
    instance.weight_update_groups["second_producer"] = object()
    assert instance.areal_update_weight_xccl()[0]
    assert broadcast.call_count == transaction.load.call_count == 4
    transaction.finish.assert_not_called()
    assert instance.areal_finish_weight_update()[0]
    assert instance.areal_finish_weight_update()[0]
    transaction.finish.assert_called_once()
