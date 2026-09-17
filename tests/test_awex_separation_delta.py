# SPDX-License-Identifier: Apache-2.0
"""Separation-specific distributed ordering tests for AWEX delta transfer."""

import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest

from tests import test_awex_delta_common as common

torch = common.torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("dte") is None or importlib.util.find_spec("awex") is None,
    reason="DTE/AWEX source path is not available",
)


class _FakeTransferPlanBuilder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def build_local_transfer_plan(self, *args, **kwargs):
        del args, kwargs
        return object()


def _capture_weight_update_groups(monkeypatch, mod):
    calls = []

    def _init_group(**kwargs):
        calls.append(kwargs)
        return kwargs.get("backend", "nccl")

    monkeypatch.setattr(mod, "init_weights_update_group", _init_group)
    monkeypatch.setattr(mod, "fetch_kv_metadata", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(mod, "TransferPlanBuilder", _FakeTransferPlanBuilder)
    return calls


def test_megatron_full_transfer_initializes_gloo_control_group(monkeypatch):
    """Separated Full sync must not fall back to a NCCL control barrier."""
    mod = common._load_megatron_adapter(monkeypatch)
    calls = _capture_weight_update_groups(monkeypatch, mod)
    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._dte_config = SimpleNamespace(enabled=False)

    adapter.init_weight_update_group(
        pair_name="pair",
        master_addr="127.0.0.1",
        master_port=23456,
        transfer_rank=8,
        world_size=16,
        kv_store_url="http://127.0.0.1:9999",
        infer_world_size=8,
        train_world_size=8,
        num_engines=2,
    )

    assert [call.get("backend", "nccl") for call in calls] == ["nccl", "gloo"]
    assert adapter._weights_update_group == "nccl"
    assert adapter._weights_update_group_gloo == "gloo"


def test_sglang_full_transfer_initializes_gloo_control_group(monkeypatch):
    """SGLang Full sync creates the same CPU control group as training."""
    mod = common._load_sglang_adapter(monkeypatch)
    calls = _capture_weight_update_groups(monkeypatch, mod)
    adapter = object.__new__(mod.AwexSGLangAdapter)
    adapter._dte_config = SimpleNamespace(enabled=False)
    adapter._get_model_context = lambda: {
        "tp_size": 4,
        "tp_rank": 0,
        "pp_size": 1,
        "pp_rank": 0,
    }

    adapter.init_weight_update_group(
        pair_name="pair",
        master_addr="127.0.0.1",
        master_port=23456,
        transfer_rank=0,
        world_size=16,
        kv_store_url="http://127.0.0.1:9999",
        infer_world_size=8,
        train_world_size=8,
        num_engines=2,
    )

    assert [call.get("backend", "nccl") for call in calls] == ["nccl", "gloo"]
    assert adapter._weights_update_group == "nccl"
    assert adapter._weights_update_group_gloo == "gloo"


def test_megatron_dte_init_caches_global_wire_dtypes_once(monkeypatch):
    mod = common._load_megatron_adapter(monkeypatch)
    _capture_weight_update_groups(monkeypatch, mod)
    calls = []
    expected = (torch.bfloat16, torch.float32)
    monkeypatch.setattr(
        mod,
        "synchronize_wire_dtypes",
        lambda plan, group: calls.append((plan, group)) or expected,
    )
    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._dte_config = SimpleNamespace(enabled=True)

    adapter.init_weight_update_group(
        pair_name="pair",
        master_addr="127.0.0.1",
        master_port=23456,
        transfer_rank=8,
        world_size=16,
        kv_store_url="http://127.0.0.1:9999",
        infer_world_size=8,
        train_world_size=8,
        num_engines=2,
    )

    assert calls == [(adapter._transfer_plan, "gloo")]
    assert adapter._separation_wire_dtypes == expected


def test_sglang_dte_init_caches_global_wire_dtypes_once(monkeypatch):
    mod = common._load_sglang_adapter(monkeypatch)
    _capture_weight_update_groups(monkeypatch, mod)
    calls = []
    expected = (torch.bfloat16, torch.float32)
    monkeypatch.setattr(
        mod,
        "synchronize_wire_dtypes",
        lambda plan, group: calls.append((plan, group)) or expected,
    )
    adapter = object.__new__(mod.AwexSGLangAdapter)
    adapter._dte_config = SimpleNamespace(enabled=True)
    adapter._get_model_context = lambda: {
        "tp_size": 4,
        "tp_rank": 0,
        "pp_size": 1,
        "pp_rank": 0,
    }

    adapter.init_weight_update_group(
        pair_name="pair",
        master_addr="127.0.0.1",
        master_port=23456,
        transfer_rank=0,
        world_size=16,
        kv_store_url="http://127.0.0.1:9999",
        infer_world_size=8,
        train_world_size=8,
        num_engines=2,
    )

    assert calls == [(adapter._transfer_plan, "gloo")]
    assert adapter._separation_wire_dtypes == expected


@pytest.mark.parametrize(
    ("loader_name", "adapter_name"),
    [
        ("_load_megatron_adapter", "AwexMegatronAdapter"),
        ("_load_sglang_adapter", "AwexSGLangAdapter"),
    ],
)
def test_dte_world_size_fails_before_group_or_metadata_init(
    monkeypatch, loader_name, adapter_name
):
    mod = getattr(common, loader_name)(monkeypatch)
    events = []

    def _reject(*args):
        events.append(("validate", args))
        raise ValueError("combined=5")

    monkeypatch.setattr(mod, "validate_dte_world_size", _reject)
    monkeypatch.setattr(
        mod,
        "fetch_kv_metadata",
        lambda *args: events.append(("metadata", args)),
    )
    monkeypatch.setattr(
        mod,
        "init_weights_update_group",
        lambda **kwargs: events.append(("group", kwargs)),
    )
    adapter = object.__new__(getattr(mod, adapter_name))
    adapter._dte_config = SimpleNamespace(enabled=True)

    with pytest.raises(ValueError, match="combined=5"):
        adapter.init_weight_update_group(
            pair_name="pair",
            master_addr="127.0.0.1",
            master_port=23456,
            transfer_rank=0,
            world_size=5,
            kv_store_url="http://127.0.0.1:9999",
            infer_world_size=3,
            train_world_size=2,
            num_engines=1,
        )

    assert events == [("validate", (5, 3, 2))]


def test_non_dte_megatron_keeps_odd_world_size_compatibility(monkeypatch):
    mod = common._load_megatron_adapter(monkeypatch)
    calls = _capture_weight_update_groups(monkeypatch, mod)
    monkeypatch.setattr(
        mod,
        "validate_dte_world_size",
        lambda *args: pytest.fail("dense AWEX must not use the DTE world-size gate"),
    )
    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._dte_config = SimpleNamespace(enabled=False)

    adapter.init_weight_update_group(
        pair_name="pair",
        master_addr="127.0.0.1",
        master_port=23456,
        transfer_rank=3,
        world_size=5,
        kv_store_url="http://127.0.0.1:9999",
        infer_world_size=3,
        train_world_size=2,
        num_engines=1,
    )

    assert [call.get("backend", "nccl") for call in calls] == ["nccl", "gloo"]


def test_megatron_enters_empty_local_dtype_round(monkeypatch):
    """A sender with only BF16 ops must still enter the global FP32 round."""
    mod = common._load_megatron_adapter(monkeypatch)
    from dte.core import colocate_protocol, delta_p2p

    op = SimpleNamespace(
        recv_shard_meta=SimpleNamespace(dtype=torch.bfloat16),
    )
    calls = []

    def _exchange(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(colocate_protocol, "two_round_delta_exchange", _exchange)
    monkeypatch.setattr(
        delta_p2p,
        "build_send_payloads_by_op",
        lambda ops, masks, params: {"operation_count": len(ops)},
    )
    monkeypatch.setattr(mod.torch.cuda, "current_device", lambda: 0)
    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._transfer_plan = SimpleNamespace(operations={0: [op]})
    adapter._weights_update_group = object()
    adapter._transfer_rank = 1
    adapter._world_size = 2
    adapter._separation_wire_dtypes = (torch.bfloat16, torch.float32)
    adapter._separation_delta_transport = SimpleNamespace(
        execute_recursive_partition_stream_transfer=lambda *args, **kwargs: None
    )

    adapter._execute_separation_delta_send({}, {}, version=7)

    assert [call["value_dtype"] for call in calls] == [
        torch.bfloat16,
        torch.float32,
    ]
    assert calls[0]["send_payloads_by_op"] == {"operation_count": 1}
    assert calls[1]["send_payloads_by_op"] == {"operation_count": 0}
    assert calls[1]["send_plan"].operations == {}


def test_sglang_enters_empty_local_dtype_round(monkeypatch):
    """A receiver with only FP32 ops must still enter the global BF16 round."""
    mod = common._load_sglang_adapter(monkeypatch)
    from dte.core import colocate_protocol

    op = SimpleNamespace(
        recv_shard_meta=SimpleNamespace(dtype=torch.float32),
    )
    calls = []
    monkeypatch.setattr(
        colocate_protocol,
        "two_round_delta_exchange",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(mod.torch.cuda, "current_device", lambda: 0)
    adapter = object.__new__(mod.AwexSGLangAdapter)
    adapter._transfer_plan = SimpleNamespace(operations={1: [op]})
    adapter._weights_update_group = object()
    adapter._transfer_rank = 0
    adapter._world_size = 2
    adapter._separation_wire_dtypes = (torch.bfloat16, torch.float32)
    adapter._separation_delta_transport = SimpleNamespace(
        execute_recursive_partition_stream_transfer=lambda *args, **kwargs: None
    )

    adapter._execute_separation_delta_recv({}, version=7)

    assert [call["value_dtype"] for call in calls] == [
        torch.bfloat16,
        torch.float32,
    ]
    assert calls[0]["recv_plan"].operations == {}
    assert calls[1]["recv_plan"].operations == {1: [op]}


def test_megatron_full_transfer_uses_gloo_completion_barrier(monkeypatch):
    """Full payload P2P completes through the sideband control group."""
    mod = common._load_megatron_adapter(monkeypatch)
    monkeypatch.setattr(
        mod, "nccl_build_send_ops", lambda *args, **kwargs: ([], [], [])
    )
    monkeypatch.setattr(mod, "batch_send_recv", lambda **kwargs: None)
    barriers = []
    monkeypatch.setattr(mod.dist, "barrier", lambda *, group: barriers.append(group))
    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._dte_config = SimpleNamespace(enabled=False)
    adapter._transfer_plan = object()
    adapter._weights_update_group = "nccl"
    adapter._weights_update_group_gloo = "gloo"
    adapter._transfer_rank = 8
    adapter._world_size = 16
    adapter.get_local_shard_parameters = lambda: {}

    adapter.execute_weight_update(version=1)

    assert barriers == ["gloo"]


def test_sglang_full_transfer_uses_gloo_completion_barrier(monkeypatch):
    """Receiver Full payload completion avoids the NCCL data group."""
    mod = common._load_sglang_adapter(monkeypatch)
    monkeypatch.setattr(
        mod, "nccl_build_recv_ops", lambda *args, **kwargs: ([], [], [])
    )
    monkeypatch.setattr(mod, "batch_send_recv", lambda **kwargs: None)
    barriers = []
    monkeypatch.setattr(mod.dist, "barrier", lambda *, group: barriers.append(group))
    adapter = object.__new__(mod.AwexSGLangAdapter)
    adapter._dte_config = SimpleNamespace(enabled=False)
    adapter._transfer_plan = object()
    adapter._weights_update_group = "nccl"
    adapter._weights_update_group_gloo = "gloo"
    adapter._transfer_rank = 0
    adapter.get_local_shard_parameters = lambda: {}

    adapter.execute_weight_update(version=1)

    assert barriers == ["gloo"]


def test_megatron_separation_delta_commits_tracker_after_receiver_barrier(
    monkeypatch,
):
    """A successful direct sparse transfer advances the periodic anchor."""
    mod = common._load_megatron_adapter(monkeypatch)
    events = []

    class _Tracker:
        def mark_delta_committed(self, version):
            events.append(("commit", version))

    monkeypatch.setattr(mod.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mod.dist,
        "barrier",
        lambda *, group: events.append(("barrier", group)),
    )
    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._transfer_plan = object()
    adapter._weights_update_group = "nccl"
    adapter._weights_update_group_gloo = "gloo"
    adapter._transfer_rank = 8
    adapter._world_size = 16
    adapter._delta_tracker = _Tracker()
    adapter.get_local_shard_parameters = lambda: {"w": torch.ones(1)}
    adapter._ensure_delta_components = lambda: None
    adapter._delta_detector = SimpleNamespace(
        capture_synced_state=lambda params: {},
        mark_synced=lambda version, state: events.append(("synced", version)),
    )
    adapter._delta_prepare_masks = lambda params, version: (
        {"w": torch.ones(1, dtype=torch.bool)},
        True,
    )
    adapter._execute_separation_delta_send = (
        lambda params, masks, version: events.append(("send", version))
    )

    adapter._execute_separation_weight_update(version=2)

    assert events == [
        ("send", 2),
        ("barrier", "gloo"),
        ("commit", 2),
        ("synced", 2),
    ]


def test_megatron_separation_failed_delta_does_not_advance_tracker(monkeypatch):
    """A failed sparse transfer must leave the anchor counter unchanged."""
    mod = common._load_megatron_adapter(monkeypatch)
    commits = []

    class _Tracker:
        def mark_delta_committed(self, version):
            commits.append(version)

    monkeypatch.setattr(mod.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mod.dist,
        "barrier",
        lambda **kwargs: pytest.fail("failed transfer must not reach barrier"),
    )
    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._transfer_plan = object()
    adapter._weights_update_group = "nccl"
    adapter._weights_update_group_gloo = "gloo"
    adapter._transfer_rank = 8
    adapter._world_size = 16
    adapter._delta_tracker = _Tracker()
    adapter.get_local_shard_parameters = lambda: {"w": torch.ones(1)}
    adapter._ensure_delta_components = lambda: None
    adapter._delta_detector = SimpleNamespace(
        capture_synced_state=lambda params: {}, mark_synced=lambda *args: None
    )
    adapter._delta_prepare_masks = lambda params, version: (
        {"w": torch.ones(1, dtype=torch.bool)},
        True,
    )
    adapter._execute_separation_delta_send = lambda *args: (_ for _ in ()).throw(
        RuntimeError("transfer failed")
    )

    with pytest.raises(RuntimeError, match="transfer failed"):
        adapter._execute_separation_weight_update(version=2)

    assert commits == []


def test_reconstructed_override_preserves_tensor_parallel_metadata(monkeypatch):
    """A plain theta_old tensor must gather like its live TP parameter."""
    mod = common._load_megatron_adapter(monkeypatch)
    param = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    param.tensor_model_parallel = True
    param.partition_dim = 0
    param.partition_stride = 1
    override = param.detach().clone()
    overrides = {id(param): override}
    captured = {}

    megatron_mod = types.ModuleType("areal.engine.megatron_utils.megatron")
    megatron_mod.get_named_parameters = lambda model, experts: [("qkv", param)]

    def _all_gather_param(name, tensor, **kwargs):
        del name, kwargs
        captured["uses_override"] = tensor is override
        captured["metadata"] = (
            tensor.tensor_model_parallel,
            tensor.partition_dim,
            tensor.partition_stride,
        )
        return tensor

    megatron_mod.all_gather_param = _all_gather_param
    megatron_mod.convert_to_hf = lambda config, model, name, tensor: [("w", tensor)]
    monkeypatch.setitem(
        sys.modules, "areal.engine.megatron_utils.megatron", megatron_mod
    )

    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._engine = SimpleNamespace(
        model=object(),
        tf_config=SimpleNamespace(num_moe_experts=None),
        hf_config=SimpleNamespace(model_type="qwen3", tie_word_embeddings=False),
        _duplicated_param_names=set(),
    )

    items = list(adapter._iter_hf_params(overrides, consume_overrides=True))

    assert len(items) == 1
    assert items[0][0] == "w"
    torch.testing.assert_close(items[0][1], override, rtol=0, atol=0)
    assert captured["uses_override"] is True
    assert captured["metadata"] == (True, 0, 1)
    assert overrides == {}


def test_streaming_generator_waits_async_batch_before_yield(monkeypatch):
    """HF conversion never starts with inversion all-reduces in flight."""
    monkeypatch.setenv("DTE_INVERSION_ALLREDUCE_WINDOW_MB", "0.000001")
    dd = common._load_delta_detect(monkeypatch)
    param = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    base_opt = torch.optim.AdamW([param], lr=1e-3)
    dp_group = object()
    in_flight = 0

    class _FakeWork:
        def wait(self):
            nonlocal in_flight
            in_flight -= 1

    def _fake_all_reduce(tensor, op=None, group=None, async_op=False):
        nonlocal in_flight
        del op
        assert group is dp_group
        if tensor.dtype == torch.int64:
            tensor[:] = torch.tensor([1, 0, param.numel()])
        if async_op:
            in_flight += 1
            return _FakeWork()
        return None

    monkeypatch.setattr(dd.torch.distributed, "all_reduce", _fake_all_reduce)

    class _FakeDistOpt:
        optimizer = base_opt
        shard_fp32_from_float16_groups = [[param]]
        model_float16_groups = [[param]]
        model_param_group_index_map = None
        data_parallel_group = dp_group

        def _get_model_param_range_map(self, model_param):
            assert model_param is param
            return {"param": SimpleNamespace(start=0, end=param.numel())}

    inv = dd.AdamWInversionDetector(SimpleNamespace(_offloaded_optimizer_states={}))
    common._bind_inversion_param_names(inv, ("w", param))
    producer = inv._iter_reconstruct_pre_step_mcore([_FakeDistOpt()])

    param_id, old = next(producer)

    assert param_id == id(param)
    assert old is not dd._NO_RECONSTRUCTION
    assert in_flight == 0


def test_missing_watermark_finishes_all_collectives_before_abort(monkeypatch):
    """A dense verdict cannot reduce the per-parameter collective count."""
    monkeypatch.setenv("DTE_INVERSION_ALLREDUCE_WINDOW_MB", "0.000001")
    dd = common._load_delta_detect(monkeypatch)
    params = [
        torch.nn.Parameter(torch.tensor([1.0, 2.0])),
        torch.nn.Parameter(torch.tensor([3.0, 4.0])),
    ]
    base_opt = torch.optim.AdamW(params, lr=1e-3)
    for param in params:
        param.grad = torch.ones_like(param)
    base_opt.step()
    dp_group = object()
    calls = 0

    class _FakeWork:
        def wait(self):
            return None

    def _fake_all_reduce(tensor, op=None, group=None, async_op=False):
        nonlocal calls
        del tensor, op
        assert group is dp_group
        calls += 1
        return _FakeWork() if async_op else None

    monkeypatch.setattr(dd.torch.distributed, "all_reduce", _fake_all_reduce)

    class _FakeDistOpt:
        optimizer = base_opt
        shard_fp32_from_float16_groups = [params]
        model_float16_groups = [params]
        model_param_group_index_map = None
        data_parallel_group = dp_group

        def _get_model_param_range_map(self, model_param):
            return {"param": SimpleNamespace(start=0, end=model_param.numel())}

    inv = dd.AdamWInversionDetector(SimpleNamespace(_offloaded_optimizer_states={}))
    common._bind_inversion_param_names(inv, ("w0", params[0]), ("w1", params[1]))
    lazy = dd._LazyReconstructDict(
        inv._iter_reconstruct_pre_step_mcore([_FakeDistOpt()])
    )

    lazy.finish()

    assert lazy.aborted is True
    assert calls == 2 * len(params)
