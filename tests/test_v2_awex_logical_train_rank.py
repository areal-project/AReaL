# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

from areal.engine.weight_update.awex.colocate_megatron import (
    MegatronColocateBackend,
)


def test_lazy_initialize_uses_megatron_global_rank_for_logical_train_rank(
    monkeypatch,
):
    rank_info = SimpleNamespace(global_rank=7, world_size=32)
    converter = object()

    class MetaResolver:
        def __init__(self, *_args):
            pass

        def get_parameters_meta(self):
            return []

        def get_pp_stage_layer_id_map(self):
            return {}

    modules = {
        "awex": ModuleType("awex"),
        "awex.meta": ModuleType("awex.meta"),
        "awex.meta.train_meta_resolver": ModuleType("awex.meta.train_meta_resolver"),
        "awex.models": ModuleType("awex.models"),
        "awex.models.registry": ModuleType("awex.models.registry"),
        "awex.sharding": ModuleType("awex.sharding"),
        "awex.sharding.param_sharding": ModuleType("awex.sharding.param_sharding"),
        "awex.util": ModuleType("awex.util"),
        "awex.util.common": ModuleType("awex.util.common"),
    }
    modules["awex.meta.train_meta_resolver"].McoreParamMetaResolver = MetaResolver
    modules["awex.models.registry"].get_train_weights_converter = (
        lambda *_args, **_kwargs: converter
    )
    modules["awex.sharding.param_sharding"].get_rank_info_extractor = lambda _name: (
        lambda: rank_info
    )
    modules["awex.util.common"].get_ip_address = lambda: "192.0.2.10"
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    client = mock.Mock()
    client.get_object.side_effect = [
        {"infer_world_size": 16},
        4,
    ]
    engine = SimpleNamespace(
        model=[],
        hf_config=SimpleNamespace(architectures=["Model"]),
    )
    backend = MegatronColocateBackend(
        engine,
        physical_gpu_id_resolver=lambda: 3,
    )
    backend.configure(meta_server_client=client, timeout_s=30.0)

    with mock.patch(
        "areal.engine.weight_update.awex.colocate_megatron.dist.get_rank",
        return_value=1,
    ):
        backend.lazy_initialize()

    assert backend.logical_train_rank == 23
    client.add_object_to_set.assert_called_once_with(
        "training_device_rank_entries",
        ("192.0.2.10", 3, 23),
    )
