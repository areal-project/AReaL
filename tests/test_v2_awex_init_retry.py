# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import mock

import pytest

from areal.v2.weight_update.awex.fsdp_adapter import AwexFSDPAdapter
from areal.v2.weight_update.awex.megatron_adapter import AwexMegatronAdapter
from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter

SEPARATION_ARGS = {
    "pair_name": "pair",
    "master_addr": "192.0.2.1",
    "master_port": 1234,
    "transfer_rank": 0,
    "world_size": 2,
    "kv_store_url": "http://gateway",
    "infer_world_size": 1,
    "train_world_size": 1,
    "num_engines": 1,
}

COLOCATE_ARGS = {
    "pair_name": "pair",
    "meta_server_addr": "192.0.2.1:1234",
    "transfer_rank": 1,
    "infer_world_size": 2,
    "train_world_size": 2,
    "num_engines": 1,
    "timeout_s": 30.0,
}


def _separation_fingerprint():
    return tuple(SEPARATION_ARGS.values())


def _colocate_fingerprint():
    return tuple(COLOCATE_ARGS.values())


def test_fsdp_separation_init_retry_is_idempotent():
    adapter = AwexFSDPAdapter(SimpleNamespace())
    adapter._init_fingerprint = _separation_fingerprint()

    adapter.init_weight_update_group(**SEPARATION_ARGS)


def test_sglang_separation_init_retry_is_idempotent():
    adapter = AwexSGLangAdapter(SimpleNamespace())
    adapter._separation_init_fingerprint = _separation_fingerprint()

    adapter.init_weight_update_group(**SEPARATION_ARGS)


def test_sglang_colocate_init_retry_is_idempotent():
    adapter = AwexSGLangAdapter(SimpleNamespace())
    adapter._colocate_init_fingerprint = _colocate_fingerprint()

    adapter.init_colocate_weight_update(**COLOCATE_ARGS)


def test_megatron_colocate_response_loss_retry_is_idempotent():
    engine = SimpleNamespace(cpu_group="cpu", _awex_adapter=None)
    adapter = AwexMegatronAdapter(engine)
    client = mock.Mock()

    with (
        mock.patch(
            "areal.v2.weight_update.awex.megatron_adapter.dist.get_world_size",
            return_value=2,
        ),
        mock.patch(
            "areal.v2.weight_update.awex.megatron_adapter.dist.get_rank",
            return_value=0,
        ),
        mock.patch(
            "awex.meta.meta_server.MetaServerClient", return_value=client
        ) as client_cls,
    ):
        adapter.init_colocate_weight_update(**COLOCATE_ARGS)
        adapter.init_colocate_weight_update(**COLOCATE_ARGS)

    client_cls.assert_called_once()
    client.put_object.assert_called_once()


def test_megatron_colocate_init_failure_rolls_back_for_retry():
    engine = SimpleNamespace(cpu_group="cpu", _awex_adapter=None)
    adapter = AwexMegatronAdapter(engine)
    backup = object()
    adapter._offloaded_weights["weight"] = backup
    adapter._released_tags.update({"weights", "optimizer"})
    broken_client = mock.Mock()
    broken_client.put_object.side_effect = RuntimeError("meta server unavailable")

    with (
        mock.patch(
            "areal.v2.weight_update.awex.megatron_adapter.dist.get_world_size",
            return_value=2,
        ),
        mock.patch(
            "areal.v2.weight_update.awex.megatron_adapter.dist.get_rank",
            return_value=0,
        ),
        mock.patch(
            "awex.meta.meta_server.MetaServerClient", return_value=broken_client
        ),
        pytest.raises(RuntimeError, match="meta server unavailable"),
    ):
        adapter.init_colocate_weight_update(**COLOCATE_ARGS)

    assert adapter._active_mode is None
    assert adapter._init_fingerprint is None
    assert adapter._meta_server_client is None
    assert engine._awex_adapter is None
    assert adapter._offloaded_weights == {"weight": backup}
    assert adapter._released_tags == {"weights", "optimizer"}

    healthy_client = mock.Mock()
    with (
        mock.patch(
            "areal.v2.weight_update.awex.megatron_adapter.dist.get_world_size",
            return_value=2,
        ),
        mock.patch(
            "areal.v2.weight_update.awex.megatron_adapter.dist.get_rank",
            return_value=0,
        ),
        mock.patch(
            "awex.meta.meta_server.MetaServerClient", return_value=healthy_client
        ),
    ):
        adapter.init_colocate_weight_update(**COLOCATE_ARGS)

    assert adapter._active_mode == "colocate"
    assert engine._awex_adapter is adapter
