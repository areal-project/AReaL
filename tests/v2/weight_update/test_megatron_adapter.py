# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from areal.v2.weight_update.awex.megatron_adapter import AwexMegatronAdapter


class _FakeDDP:
    pass


@pytest.mark.parametrize("chunks", [[object()], [_FakeDDP(), object()]])
def test_colocate_init_rejects_non_ddp_chunks_before_mutating_state(
    monkeypatch, chunks
):
    """Unsupported weight layouts fail during connect, before AWEX state exists."""
    monkeypatch.setattr("megatron.core.distributed.DistributedDataParallel", _FakeDDP)
    client_factory = MagicMock()
    monkeypatch.setattr(
        "areal.v2.weight_update.awex.megatron_adapter.httpx.Client", client_factory
    )
    adapter = AwexMegatronAdapter(SimpleNamespace(model=chunks))

    with pytest.raises(RuntimeError, match="megatron.wrap_with_ddp=true"):
        adapter.init_colocate_weight_update(
            pair_name="actor-rollout",
            kv_store_url="http://127.0.0.1:1234",
            transfer_rank=0,
            infer_world_size=1,
            train_world_size=1,
            num_engines=1,
            master_port=2345,
        )

    client_factory.assert_not_called()
    assert adapter._colocate_http_client is None
    assert not hasattr(adapter, "_colocate_pair_name")


def test_colocate_init_accepts_mcore_ddp_chunks(monkeypatch):
    """A fully wrapped MCore model reaches the colocated AWEX initialized state."""
    monkeypatch.setattr("megatron.core.distributed.DistributedDataParallel", _FakeDDP)
    client = object()
    monkeypatch.setattr(
        "areal.v2.weight_update.awex.megatron_adapter.httpx.Client", lambda: client
    )
    adapter = AwexMegatronAdapter(SimpleNamespace(model=[_FakeDDP(), _FakeDDP()]))

    adapter.init_colocate_weight_update(
        pair_name="actor-rollout",
        kv_store_url="http://127.0.0.1:1234",
        transfer_rank=3,
        infer_world_size=4,
        train_world_size=4,
        num_engines=1,
        master_port=2345,
    )

    assert adapter._colocate_pair_name == "actor-rollout"
    assert adapter._colocate_transfer_rank == 3
    assert adapter._colocate_http_client is client


@pytest.mark.parametrize("chunks", [[object()], [_FakeDDP(), object()]])
def test_colocate_preflight_rejects_non_ddp_without_mutating_state(monkeypatch, chunks):
    """Capability preflight performs only the MCore DDP layout check."""
    monkeypatch.setattr("megatron.core.distributed.DistributedDataParallel", _FakeDDP)
    adapter = AwexMegatronAdapter(SimpleNamespace(model=chunks))

    with pytest.raises(RuntimeError, match="megatron.wrap_with_ddp=true"):
        adapter.preflight_colocate_weight_update()

    assert adapter._colocate_http_client is None
    assert not hasattr(adapter, "_colocate_pair_name")


def test_colocate_preflight_accepts_mcore_ddp_without_mutating_state(monkeypatch):
    """Supported layouts pass without allocating a client or pair state."""
    monkeypatch.setattr("megatron.core.distributed.DistributedDataParallel", _FakeDDP)
    adapter = AwexMegatronAdapter(SimpleNamespace(model=[_FakeDDP()]))

    adapter.preflight_colocate_weight_update()

    assert adapter._colocate_http_client is None
    assert not hasattr(adapter, "_colocate_pair_name")


def test_fsdp_colocate_preflight_fails_with_actionable_error():
    """FSDP does not silently claim support for the Megatron-only path."""
    from areal.v2.weight_update.awex.fsdp_adapter import AwexFSDPAdapter

    adapter = AwexFSDPAdapter.__new__(AwexFSDPAdapter)

    with pytest.raises(RuntimeError, match="does not support FSDPEngine"):
        adapter.preflight_colocate_weight_update()
