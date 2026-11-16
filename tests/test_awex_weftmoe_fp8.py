# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from areal.engine.awex.adapters.weftmoe_fp8 import convert_fp8


def _ipc_receiver(payload, queue):
    import gc

    from awex.util.tensor_util import (
        cuda_ipc_deserialize,
        reconstruct_tensors_from_groups,
    )

    torch.cuda.set_device(0)
    groups, metadata = cuda_ipc_deserialize(payload)
    views = reconstruct_tensors_from_groups(groups, metadata)
    # The receive buffers reinterpret the exact byte representation as E4M3.
    weight = torch.empty(views[0].shape, device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.empty_like(views[1])
    weight.view(torch.uint8).copy_(views[0])
    scale.copy_(views[1])
    result = (weight.view(torch.uint8).cpu().tolist(), scale.cpu().tolist())
    del views, groups
    gc.collect()
    torch.cuda.synchronize()
    queue.put(result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA IPC requires a GPU")
def test_awex_fp8_ipc_preserves_weight_bits_and_scales():
    import multiprocessing as mp

    from awex.util.cuda_ipc import cuda_ipc_allocation
    from awex.util.tensor_util import (
        cuda_ipc_serialize,
        group_tensors_by_shape_and_dtype,
    )

    data = (
        torch.arange(256, dtype=torch.int32, device="cuda")
        .to(torch.uint8)
        .reshape(16, 16)
    )
    scale = torch.tensor([[0.125, 0.25], [0.5, 1.0]], device="cuda")
    with cuda_ipc_allocation():
        groups, metadata = group_tensors_by_shape_and_dtype([data, scale])
    torch.cuda.synchronize()
    payload = cuda_ipc_serialize((groups, metadata))
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_ipc_receiver, args=(payload, queue))
    process.start()
    try:
        received_data, received_scale = queue.get(timeout=90)
        process.join(timeout=30)
        assert process.exitcode == 0
        assert received_data == data.cpu().tolist()
        assert received_scale == scale.cpu().tolist()
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        queue.close()


def test_qwen35_fp8_train_and_infer_byte_scale_mapping():
    from awex.models.registry import (
        get_infer_weights_converter,
        get_train_weights_converter,
    )
    from awex.sharding.mcore_sharding import get_mcore_sharding_strategy
    from awex.sharding.rank_info import RankInfo
    from awex.sharding.sglang_sharding import get_sglang_sharding_strategy

    config = SimpleNamespace(
        num_experts=16,
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_hidden_layers=2,
        moe_intermediate_size=128,
        tie_word_embeddings=False,
    )
    train_rank = RankInfo(
        tp_rank=0,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        dp_rank=3,
        dp_size=4,
        ep_rank=6,
        ep_size=8,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=2,
        attn_dp_rank=3,
        world_size=8,
        global_rank=6,
        local_rank=6,
        engine_rank=0,
        is_infer=False,
    )
    infer_rank = replace(
        train_rank,
        tp_rank=6,
        tp_size=8,
        attn_tp_size=1,
        attn_dp_rank=6,
        dp_size=8,
        dp_rank=6,
        is_infer=True,
    )
    args = SimpleNamespace(
        tp_size=8,
        ep_size=8,
        dp_size=8,
        enable_dp_attention=True,
        enable_dp_lm_head=False,
        moe_dense_tp_size=None,
    )
    architecture = "Qwen3_5MoeForConditionalGeneration"
    train = get_train_weights_converter(
        "mcore",
        architecture,
        config,
        train_rank,
        dict(infer_atten_tp_size=1),
        tf_config=config,
    )
    infer = get_infer_weights_converter(
        "sglang", architecture, config, infer_rank, args
    )
    train_sharding = get_mcore_sharding_strategy(architecture, train_rank)
    infer_sharding = get_sglang_sharding_strategy(architecture, args, infer_rank)
    for fc, sg, shape in (
        ("linear_fc1", "w13_weight", (2, 256, 256)),
        ("linear_fc2", "w2_weight", (2, 256, 128)),
    ):
        data = (
            torch.arange(torch.tensor(shape).prod().item()).to(torch.uint8).view(shape)
        )
        scales = (
            torch.arange(
                shape[0] * (shape[1] // 128) * (shape[2] // 128), dtype=torch.float32
            ).view(shape[0], shape[1] // 128, shape[2] // 128)
            + 1
        )
        source = {}
        for expert in range(2):
            source.update(
                convert_fp8(
                    train,
                    f"decoder.layers.0.mlp.experts.{fc}.weight{expert}",
                    data[expert],
                    scales[expert],
                    vp_stage=0,
                )
            )
        target = dict(
            convert_fp8(infer, f"model.layers.0.mlp.experts.{sg}", data, scales)
        )
        assert source.keys() == target.keys()
        assert any("experts.12." in name for name in target)
        assert any("experts.13." in name for name in target)
        for name, tensor in source.items():
            torch.testing.assert_close(tensor, target[name], rtol=0, atol=0)
            assert tensor.dtype == (
                torch.float32 if name.endswith("_scale_inv") else torch.uint8
            )
            meta = dict(
                shape=tuple(tensor.shape), dtype=tensor.dtype, numel=tensor.numel()
            )
            left = train_sharding.get_sharding_strategy(
                name, rank_info=train_rank, param_meta=meta
            )
            right = infer_sharding.get_sharding_strategy(
                name, rank_info=infer_rank, param_meta=meta
            )
            assert left == right


@pytest.mark.parametrize("transfer_fails", [False, True])
def test_fp8_reader_rebinds_after_offload_before_publishing(
    monkeypatch, transfer_fails
):
    import sys

    from areal.engine.awex.adapters.sglang_adapter import AwexSGLangAdapter

    monkeypatch.setenv("AREAL_USE_WEFTMOE", "1")
    monkeypatch.setenv("WEFTMOE_FP8", "1")
    events = []
    current = {"expert.weight": torch.zeros(4, dtype=torch.uint8)}

    def receive(*, step_id):
        assert reader.parameters["expert.weight"] is current["expert.weight"]
        if transfer_fails:
            raise RuntimeError("transfer failed")
        reader.parameters["expert.weight"].fill_(step_id)
        events.append(("receive", step_id))

    def commit(version):
        assert current["expert.weight"].tolist() == [version] * 4
        events.append(("commit", version))

    monkeypatch.setitem(
        sys.modules, "weftmoe.sglang", SimpleNamespace(commit_fp8_weights=commit)
    )
    reader = SimpleNamespace(
        parameters=dict(current),
        wire_parameter_provider=lambda: dict(current),
        update_weights=receive,
    )
    adapter = AwexSGLangAdapter.__new__(AwexSGLangAdapter)
    adapter._initialized = True
    adapter._legacy_reader = reader
    adapter._parameters = dict(current)
    adapter._qwen4_frozen_contract = None
    adapter._rebuild_derived_weights = lambda: events.append(("rebuild", None))
    stale = current["expert.weight"]
    adapter.release_fp8_bindings()
    assert not reader.parameters and not adapter._parameters
    current = {"expert.weight": torch.empty(4, dtype=torch.uint8)}
    if transfer_fails:
        with pytest.raises(RuntimeError, match="transfer failed"):
            adapter.update_weights(2)
        assert events == []
    else:
        adapter.update_weights(2)
        assert events == [("receive", 2), ("commit", 2), ("rebuild", None)]
        adapter.release_fp8_bindings()
        current = {"expert.weight": torch.empty(4, dtype=torch.uint8)}
        adapter.update_weights(3)
        assert events[-3:] == [("receive", 3), ("commit", 3), ("rebuild", None)]
    assert stale.tolist() == [0] * 4


def test_fp8_reader_initializes_canonical_views_without_reconverting(monkeypatch):
    from awex.reader.nccl_reader import NCCLWorkerWeightsReader

    from areal.engine.awex.adapters.sglang_adapter import _DeviceBoundWeightsReader

    original_model, original_converter = object(), object()
    weight = torch.arange(4, dtype=torch.uint8)
    reader = _DeviceBoundWeightsReader.__new__(_DeviceBoundWeightsReader)
    reader.model = original_model
    reader.weight_converter = original_converter
    reader.wire_parameter_provider = lambda: {"expert.weight": weight}

    def initialize(self):
        parameters = dict(self.model.named_parameters())
        assert parameters["expert.weight"] is weight
        assert self.weight_converter.convert_param("expert.weight", weight) == [
            ("expert.weight", weight)
        ]
        self.parameters = parameters

    monkeypatch.setattr(NCCLWorkerWeightsReader, "initialize", initialize)
    reader.initialize()
    assert reader.model is original_model
    assert reader.weight_converter is original_converter
    assert reader.parameters["expert.weight"] is weight


@pytest.mark.parametrize("colocated", [False, True])
def test_v2_fp8_is_rejected_before_transport_setup(monkeypatch, colocated):
    from areal.engine.awex.adapters.sglang_adapter import AwexSGLangAdapter

    monkeypatch.setenv("AREAL_USE_WEFTMOE", "1")
    monkeypatch.setenv("WEFTMOE_FP8", "1")
    adapter = AwexSGLangAdapter.__new__(AwexSGLangAdapter)
    kwargs = dict(
        pair_name="test",
        kv_store_url="unused",
        transfer_rank=0,
        infer_world_size=8,
        train_world_size=8,
        num_engines=1,
        master_port=1234,
    )
    with pytest.raises(NotImplementedError, match="v1 PPOTrainer colocated"):
        if colocated:
            adapter.init_colocate_weight_update(**kwargs)
        else:
            adapter.init_weight_update_group(
                **kwargs, master_addr="unused", world_size=16
            )
