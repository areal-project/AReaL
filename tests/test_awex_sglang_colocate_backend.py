# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import mock

import pytest

from areal.engine.weight_update.awex import colocate_sglang, v1_sglang_adapter
from areal.engine.weight_update.awex.colocate_protocol import (
    ColocateKeyspace,
    ColocateTopology,
)
from areal.engine.weight_update.awex.colocate_sglang import SGLangColocateBackend
from areal.engine.weight_update.awex.v1_sglang_adapter import AwexColocateReader
from areal.v2.weight_update.awex import sglang_adapter
from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter


def _scheduler(*, tp_size=1, pp_size=1, tp_rank=0, pp_rank=0, model=None):
    model = model or SimpleNamespace(config=SimpleNamespace())
    return SimpleNamespace(
        server_args=SimpleNamespace(
            tp_size=tp_size,
            pp_size=pp_size,
            dp_size=1,
            ep_size=1,
        ),
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model=model, model_config=None)
        ),
    )


def test_v1_and_v2_facades_compose_shared_sglang_backend():
    scheduler = _scheduler()

    legacy = AwexColocateReader(scheduler)
    v2 = AwexSGLangAdapter(scheduler)

    assert isinstance(legacy._backend, SGLangColocateBackend)
    assert isinstance(v2._colocate_backend, SGLangColocateBackend)


def test_v1_and_v2_share_single_instance_meta_resolver():
    assert (
        v1_sglang_adapter.SingleInstanceMetaResolver
        is colocate_sglang.SingleInstanceMetaResolver
    )
    assert not hasattr(sglang_adapter, "SingleInstanceMetaResolver")


def test_v1_facade_preserves_legacy_metadata_publication_policy(monkeypatch):
    legacy = AwexColocateReader(_scheduler())
    legacy._backend.initialize = mock.Mock()
    monkeypatch.setattr(
        "areal.engine.weight_update.awex.v1_sglang_adapter.get_awex_infer_hf_config",
        lambda model, model_runner: ("v1_hf_config", model, model_runner),
    )
    monkeypatch.setattr(
        "areal.engine.weight_update.awex.v1_sglang_adapter.get_router_dtype",
        lambda config: ("v1_router_dtype", config),
    )

    legacy.initialize(
        meta_server_addr="192.0.2.1:1234",
        transfer_rank=0,
        infer_world_size=1,
        train_world_size=1,
        local_gpu_id=0,
    )

    assert (
        legacy._backend.initialize.call_args.kwargs["publish_infer_params_meta"]
        is False
    )
    model_runner = legacy._scheduler.tp_worker.model_runner
    assert legacy._backend.initialize.call_args.kwargs["infer_hf_config"] == (
        "v1_hf_config",
        model_runner.model,
        model_runner,
    )
    assert legacy._backend.initialize.call_args.kwargs["router_dtype"] == (
        "v1_router_dtype",
        model_runner.model.config,
    )


def test_v2_facade_resolves_server_base_rank_before_backend_init(monkeypatch):
    scheduler = _scheduler(tp_size=2, pp_size=2, tp_rank=1, pp_rank=1)
    adapter = AwexSGLangAdapter(scheduler)
    adapter._colocate_backend.initialize = mock.Mock()
    monkeypatch.setattr(
        "areal.v2.weight_update.awex.sglang_adapter.simple_hf_config",
        lambda config: ("v2_hf_config", config),
    )

    adapter.init_colocate_weight_update(
        pair_name="pair",
        meta_server_addr="192.0.2.1:1234",
        transfer_rank=4,
        infer_world_size=8,
        train_world_size=8,
        num_engines=2,
    )

    topology = adapter._colocate_backend.initialize.call_args.kwargs["topology"]
    assert topology == ColocateTopology(
        transfer_rank=7,
        infer_world_size=8,
        train_world_size=8,
        instance_world_size=4,
    )
    assert (
        adapter._colocate_backend.initialize.call_args.kwargs[
            "expected_num_infer_engines"
        ]
        == 2
    )
    assert (
        adapter._colocate_backend.initialize.call_args.kwargs[
            "publish_infer_params_meta"
        ]
        is True
    )
    assert adapter._colocate_backend.initialize.call_args.kwargs["infer_hf_config"] == (
        "v2_hf_config",
        adapter._get_model().config,
    )
    assert (
        adapter._colocate_backend.initialize.call_args.kwargs["router_dtype"] == "bf16"
    )


@pytest.mark.parametrize(
    ("publish_infer_params_meta", "expected_keys"),
    [
        (
            False,
            {
                ColocateKeyspace.INFER_CONF,
                ColocateKeyspace.NUM_INFER_ENGINES,
            },
        ),
        (
            True,
            {
                ColocateKeyspace.INFER_CONF,
                ColocateKeyspace.NUM_INFER_ENGINES,
                ColocateKeyspace.INFER_PARAMS_META,
            },
        ),
    ],
)
def test_backend_initialize_publishes_only_requested_metadata(
    publish_infer_params_meta, expected_keys
):
    backend = SGLangColocateBackend(_scheduler())
    client = mock.Mock()
    backend._build_instance_params_meta = mock.Mock(return_value=["metadata"])
    topology = ColocateTopology(
        transfer_rank=0,
        infer_world_size=1,
        train_world_size=1,
        instance_world_size=1,
    )

    with mock.patch("awex.meta.meta_server.MetaServerClient", return_value=client):
        backend.initialize(
            meta_server_addr="192.0.2.1:1234",
            topology=topology,
            infer_hf_config="hf_config",
            router_dtype="bf16",
            publish_infer_params_meta=publish_infer_params_meta,
        )

    assert {call.args[0] for call in client.put_object.call_args_list} == expected_keys


def test_backend_update_reuses_reader_without_extra_copy_collective_or_sync():
    events = []
    model = SimpleNamespace(
        config=SimpleNamespace(),
        post_load_weights=lambda: events.append("post_load"),
    )
    backend = SGLangColocateBackend(_scheduler(model=model))
    reader = SimpleNamespace(
        update_weights=lambda step_id: events.append(("transfer", step_id))
    )
    backend._reader = reader
    backend._initialized = True

    with mock.patch(
        "areal.engine.weight_update.awex.colocate_sglang.torch.cuda.synchronize",
        side_effect=lambda: events.append("synchronize"),
    ) as synchronize:
        backend.update_weights(3)
        backend.update_weights(4)

    assert events == [
        ("transfer", 3),
        "post_load",
        "synchronize",
        ("transfer", 4),
        "post_load",
        "synchronize",
    ]
    assert synchronize.call_count == 2


def test_backend_update_without_post_load_does_not_synchronize():
    backend = SGLangColocateBackend(_scheduler())
    backend._reader = mock.Mock()
    backend._initialized = True

    with mock.patch(
        "areal.engine.weight_update.awex.colocate_sglang.torch.cuda.synchronize"
    ) as synchronize:
        backend.update_weights(1)

    backend._reader.update_weights.assert_called_once_with(step_id=1)
    synchronize.assert_not_called()
