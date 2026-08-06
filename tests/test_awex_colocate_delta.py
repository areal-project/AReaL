# SPDX-License-Identifier: Apache-2.0
"""Colocate-specific CUDA-IPC payload tests for AWEX delta transfer."""

from tests import test_awex_delta_common as common

torch = common.torch


def test_bounded_full_ipc_payload_uses_independent_storage(monkeypatch):
    """Bounded full payload must use compact exporter-owned storage."""
    mod = common._load_megatron_adapter(monkeypatch)
    adapter = object.__new__(mod.AwexMegatronAdapter)

    live = torch.arange(4, dtype=torch.float32)
    owned = torch.arange(4, dtype=torch.float32) + 10
    adapter._live_module_storage_ptrs = lambda: {live.untyped_storage().data_ptr()}
    monkeypatch.setenv("DTE_COLOCATE_FULL_GROUP_MAX_BYTES", str(32))

    groups, metadata = adapter._full_tensors_for_ipc([live, owned])

    assert len(groups) == 1
    assert groups[0].data_ptr() not in {live.data_ptr(), owned.data_ptr()}
    assert metadata[0]["group_index"] == 0
    assert metadata[1]["group_index"] == 0
    rebuilt = []
    for meta in metadata:
        start = meta["offset"]
        end = start + meta["size"]
        rebuilt.append(
            groups[meta["group_index"]].view(-1)[start:end].view(meta["shape"])
        )
    assert torch.equal(rebuilt[0], live)
    assert torch.equal(rebuilt[1], owned)


def test_bounded_full_ipc_payload_respects_group_cap(monkeypatch):
    """Same-shape tensors are packed only up to the configured byte cap."""
    mod = common._load_megatron_adapter(monkeypatch)
    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._live_module_storage_ptrs = lambda: set()
    monkeypatch.setenv("DTE_COLOCATE_FULL_GROUP_MAX_BYTES", str(32))

    tensors = [torch.full((4,), i, dtype=torch.float32) for i in range(3)]
    groups, metadata = adapter._full_tensors_for_ipc(tensors)

    assert [g.numel() for g in groups] == [8, 4]
    assert [m["group_index"] for m in metadata] == [0, 0, 1]


def test_bounded_full_ipc_payload_handles_empty_tensors(monkeypatch):
    """CUDA IPC cannot export a zero-sized storage; use a dummy group."""
    mod = common._load_megatron_adapter(monkeypatch)
    adapter = object.__new__(mod.AwexMegatronAdapter)

    empty = torch.empty(0, 3)
    adapter._live_module_storage_ptrs = lambda: set()

    groups, metadata = adapter._full_tensors_for_ipc([empty])

    assert groups[0].numel() == 1
    assert metadata[0]["shape"] == empty.shape
    assert metadata[0]["size"] == 0


def test_sglang_decoded_delta_empty_detection(monkeypatch):
    """Empty live-apply deltas can skip receiver weight resume/apply."""
    mod = common._load_sglang_adapter(monkeypatch)

    decoded = type(
        "_Decoded",
        (),
        {
            "sparse": {"w": (torch.empty(0, dtype=torch.int32), torch.empty(0))},
            "dense": {},
        },
    )()
    assert mod.AwexSGLangAdapter._decoded_delta_is_empty(decoded)

    decoded.sparse = {"w": (torch.tensor([1], dtype=torch.int32), torch.ones(1))}
    assert not mod.AwexSGLangAdapter._decoded_delta_is_empty(decoded)

    decoded.sparse = {}
    decoded.dense = {"w": torch.ones(1)}
    assert not mod.AwexSGLangAdapter._decoded_delta_is_empty(decoded)
