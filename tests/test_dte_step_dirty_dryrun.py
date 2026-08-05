from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_dryrun_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "areal"
        / "v2"
        / "weight_update"
        / "awex"
        / "step_dirty_dryrun.py"
    )
    spec = importlib.util.spec_from_file_location("step_dirty_dryrun", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DRYRUN = _load_dryrun_module()
capture_bf16_optimizer_shards = _DRYRUN.capture_bf16_optimizer_shards
collect_copyback_dirty_bitsets = _DRYRUN.collect_copyback_dirty_bitsets
compare_bf16_optimizer_shards = _DRYRUN.compare_bf16_optimizer_shards
iter_optimizer_shard_refs = _DRYRUN.iter_optimizer_shard_refs


class _FakeOptimizer:
    def __init__(
        self, main_shards: list[torch.Tensor], model_params: list[torch.Tensor]
    ):
        self.shard_fp32_from_float16_groups = [main_shards]
        self.model_float16_groups = [model_params]


def test_iter_optimizer_shard_refs_uses_resolver_names() -> None:
    main0 = torch.tensor([1.0, 2.0], dtype=torch.float32)
    main1 = torch.tensor([3.0, 4.0], dtype=torch.float32)
    model0 = torch.empty(2)
    model1 = torch.empty(2)
    opt = _FakeOptimizer([main0, main1], [model0, model1])

    refs = iter_optimizer_shard_refs(
        [opt],
        name_resolver=lambda param: "param0" if param is model0 else "param1",
    )

    assert [ref.name for ref in refs] == ["param0", "param1"]
    assert refs[0].tensor is main0
    assert refs[1].tensor is main1


def test_compare_bf16_optimizer_shards_uses_bitwise_payload_semantics() -> None:
    main = torch.tensor([0.0, -0.0, 1.0, float("nan")], dtype=torch.float32)
    opt = _FakeOptimizer([main], [torch.empty(4)])
    refs = iter_optimizer_shard_refs([opt])
    capture = capture_bf16_optimizer_shards(refs, storage="cpu")

    main.copy_(torch.tensor([-0.0, -0.0, 1.0001, float("nan")], dtype=torch.float32))
    result = compare_bf16_optimizer_shards(capture)

    # +0.0 -> -0.0 changes the bf16 bit pattern. 1.0 -> 1.0001 does not cross
    # a bf16 boundary. NaN with the same casted payload bit pattern is unchanged.
    assert result.captured_params == 1
    assert result.captured_elements == 4
    assert result.changed_elements == 1
    assert result.bitset_bytes == 1
    assert result.indices_elements == 0
    assert result.indices_bytes == 0


def test_compare_bf16_optimizer_shards_can_materialize_indices(monkeypatch) -> None:
    def pack_bool_mask_to_uint8(mask: torch.Tensor) -> torch.Tensor:
        flat = mask.reshape(-1).to(torch.bool)
        padded = ((flat.numel() + 7) // 8) * 8
        if padded != flat.numel():
            flat = torch.cat(
                (flat, torch.zeros(padded - flat.numel(), dtype=torch.bool))
            )
        bits = flat.view(-1, 8).to(torch.uint8)
        return sum(bits[:, i] << i for i in range(8))

    def packed_bool_mask_to_indices(
        packed: torch.Tensor,
        numel: int,
        *,
        dtype: torch.dtype = torch.int64,
    ) -> torch.Tensor:
        values = []
        for i in range(numel):
            if int(packed[i // 8]) & (1 << (i % 8)):
                values.append(i)
        return torch.tensor(values, dtype=dtype)

    dte_module = ModuleType("dte")
    core_module = ModuleType("dte.core")
    core_module.pack_bool_mask_to_uint8 = pack_bool_mask_to_uint8
    core_module.packed_bool_mask_to_indices = packed_bool_mask_to_indices
    monkeypatch.setitem(sys.modules, "dte", dte_module)
    monkeypatch.setitem(sys.modules, "dte.core", core_module)

    main = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    opt = _FakeOptimizer([main], [torch.empty(4)])
    refs = iter_optimizer_shard_refs([opt])
    capture = capture_bf16_optimizer_shards(refs, storage="cpu")

    main.copy_(torch.tensor([-0.0, 1.0, 2.25, 3.0], dtype=torch.float32))
    result = compare_bf16_optimizer_shards(
        capture,
        pack_bitset=True,
        materialize_indices=True,
    )

    assert result.changed_elements == 2
    assert result.bitset_bytes == 1
    assert result.indices_elements == 2
    assert result.indices_bytes == 8
    assert result.pack_ms >= 0
    assert result.indices_ms >= 0


def test_capture_bf16_optimizer_shards_respects_byte_cap() -> None:
    main0 = torch.ones(4, dtype=torch.float32)
    main1 = torch.ones(4, dtype=torch.float32)
    opt = _FakeOptimizer([main0, main1], [torch.empty(4), torch.empty(4)])
    refs = iter_optimizer_shard_refs([opt])

    capture = capture_bf16_optimizer_shards(
        refs,
        max_snapshot_bytes=8,
        storage="cpu",
    )

    assert capture.total_params == 2
    assert capture.captured_params == 1
    assert capture.skipped_by_cap == 1
    assert capture.total_snapshot_bytes == 16
    assert capture.captured_snapshot_bytes == 8


def test_capture_bf16_optimizer_shards_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="storage"):
        capture_bf16_optimizer_shards([], storage="disk")
    with pytest.raises(ValueError, match="non-negative"):
        capture_bf16_optimizer_shards([], max_snapshot_bytes=-1)


def test_collect_copyback_dirty_bitsets_compares_before_copy(monkeypatch) -> None:
    def bitwise_changed_mask(current: torch.Tensor, baseline: torch.Tensor):
        return current.contiguous().view(torch.int16) != baseline.contiguous().view(
            torch.int16
        )

    def pack_bool_mask_to_uint8(mask: torch.Tensor) -> torch.Tensor:
        flat = mask.reshape(-1).to(torch.bool)
        padded = ((flat.numel() + 7) // 8) * 8
        if padded != flat.numel():
            flat = torch.cat(
                (flat, torch.zeros(padded - flat.numel(), dtype=torch.bool))
            )
        bits = flat.view(-1, 8).to(torch.uint8)
        return sum(bits[:, i] << i for i in range(8))

    dte_module = ModuleType("dte")
    core_module = ModuleType("dte.core")
    core_module.bitwise_changed_mask = bitwise_changed_mask
    core_module.pack_bool_mask_to_uint8 = pack_bool_mask_to_uint8
    monkeypatch.setitem(sys.modules, "dte", dte_module)
    monkeypatch.setitem(sys.modules, "dte.core", core_module)

    model_param = torch.empty(8, dtype=torch.bfloat16)
    main_shard = torch.tensor([1.0, 20.0, 3.0, 40.0], dtype=torch.float32)
    old_buffer = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)

    class _Optimizer:
        shard_fp32_from_float16_groups = [[main_shard]]
        model_float16_groups = [[model_param]]
        shard_fp32_groups = []
        model_fp32_groups = []
        model_param_gbuf_map = {model_param: (0, None, 0)}
        buffers = [SimpleNamespace(buckets=[SimpleNamespace(param_data=old_buffer)])]

        def _get_model_param_range_map(self, param):
            assert param is model_param
            return {
                "param": SimpleNamespace(start=2, end=6, size=4),
                "gbuf_world_in_bucket": SimpleNamespace(start=0, end=4, size=4),
            }

    result = collect_copyback_dirty_bitsets(
        _Optimizer(),
        name_resolver=lambda param: "module.module.decoder.layers.0.mlp.linear_fc2.weight",
    )

    assert result.complete is True
    assert len(result.records) == 1
    record = result.records[0]
    assert record["name"] == "module.module.decoder.layers.0.mlp.linear_fc2.weight"
    assert record["shape"] == (8,)
    assert record["shard_start"] == 2
    assert record["shard_numel"] == 4
    assert int(record["packed_bitset"][0]) == 0b0000_1010
