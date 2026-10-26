# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import mock
from unittest.mock import Mock

import flask
import pytest

from areal.engine.awex.adapters.megatron_adapter import AwexMegatronAdapter
from areal.engine.awex.adapters.sglang_adapter import AwexSGLangAdapter
from areal.v2.training_service.worker.awex import create_awex_blueprint

SEPARATION_ARGS = dict(
    pair_name="pair",
    master_addr="192.0.2.1",
    master_port=1234,
    transfer_rank=0,
    world_size=2,
    kv_store_url="http://gateway",
    infer_world_size=1,
    train_world_size=1,
    num_engines=1,
    timeout_s=30.0,
)


COLOCATE_ARGS = dict(
    pair_name="pair",
    meta_server_addr="192.0.2.1:1234",
    transfer_rank=2,
    infer_world_size=2,
    train_world_size=2,
    num_engines=1,
    timeout_s=30.0,
)


@pytest.mark.parametrize("adapter_type", [AwexMegatronAdapter, AwexSGLangAdapter])
def test_separation_init_retry_preserves_one_initialization(adapter_type):
    adapter = adapter_type(SimpleNamespace())
    adapter._initialize_separation_group = Mock()

    adapter.init_weight_update_group(**SEPARATION_ARGS)
    adapter.init_weight_update_group(**SEPARATION_ARGS)

    adapter._initialize_separation_group.assert_called_once()
    with pytest.raises(RuntimeError, match="differently"):
        adapter.init_weight_update_group(**{**SEPARATION_ARGS, "master_port": 1235})


def test_megatron_failed_init_keeps_residency_and_allows_retry(monkeypatch):
    state = SimpleNamespace(released_tags=frozenset({"weights", "optimizer"}))
    engine = SimpleNamespace(
        cpu_group="cpu", _awex_adapter=None, _ensure_weight_residency=lambda: state
    )
    engine.hf_config = SimpleNamespace(architectures=["Qwen3ForCausalLM"])
    adapter = AwexMegatronAdapter(engine)
    client = Mock()
    client.put_object.side_effect = RuntimeError("metadata unavailable")
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    client_factory = Mock(return_value=client)
    monkeypatch.setattr("awex.meta.meta_server.MetaServerClient", client_factory)

    with pytest.raises(RuntimeError, match="metadata unavailable"):
        adapter.init_colocate_weight_update(**COLOCATE_ARGS)
    assert adapter._colocate_init_fingerprint is None
    assert adapter.residency is state
    assert adapter._released_tags == {"weights", "optimizer"}

    client.put_object.side_effect = None
    adapter.init_colocate_weight_update(**COLOCATE_ARGS)
    adapter.init_colocate_weight_update(**COLOCATE_ARGS)
    assert client_factory.call_count == 2
    assert engine._awex_adapter is adapter
    with pytest.raises(RuntimeError, match="colocation"):
        adapter.init_weight_update_group(**SEPARATION_ARGS)


def test_sglang_colocate_init_retry_uses_instance_base_and_local_rank(monkeypatch):
    scheduler = SimpleNamespace(gpu_id=1)
    adapter = AwexSGLangAdapter(scheduler)
    adapter._get_model_context = lambda: dict(
        tp_size=2, pp_size=1, tp_rank=1, pp_rank=0
    )
    adapter.initialize = Mock()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")

    args = {**COLOCATE_ARGS, "transfer_rank": 0}
    adapter.init_colocate_weight_update(**args)
    adapter.init_colocate_weight_update(**args)

    adapter.initialize.assert_called_once()
    assert adapter.initialize.call_args.kwargs["transfer_rank"] == 1
    assert adapter.initialize.call_args.kwargs["local_gpu_id"] == 5
    with pytest.raises(RuntimeError, match="colocation"):
        adapter.init_weight_update_group(**SEPARATION_ARGS)


def test_sglang_failed_rank_zero_init_retries_and_publishes_metadata_once(monkeypatch):
    adapter = AwexSGLangAdapter(SimpleNamespace(gpu_id=0))
    adapter._get_model_context = lambda: dict(
        tp_size=2, pp_size=1, tp_rank=0, pp_rank=0
    )
    client = Mock()
    attempts = []

    def initialize(**kwargs):
        attempts.append(kwargs)
        adapter._legacy_meta_server_client = client
        adapter._legacy_infer_params_meta = ["native-meta"]
        if len(attempts) == 1:
            raise RuntimeError("temporary metadata failure")
        adapter._initialized = True

    adapter.initialize = initialize
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    args = {**COLOCATE_ARGS, "transfer_rank": 0}

    with pytest.raises(RuntimeError, match="temporary metadata failure"):
        adapter.init_colocate_weight_update(**args)
    assert adapter._legacy_meta_server_client is None
    assert adapter._legacy_infer_params_meta is None
    assert adapter._colocate_init_fingerprint is None
    assert not adapter._initialized

    adapter.init_colocate_weight_update(**args)
    adapter.init_colocate_weight_update(**args)

    assert len(attempts) == 2
    assert attempts[-1]["local_gpu_id"] == 6
    client.put_object.assert_called_once_with("infer_params_meta", ["native-meta"])


def test_pending_colocate_export_rejects_teardown_without_losing_state():
    adapter = AwexMegatronAdapter(SimpleNamespace())
    export = object()
    client = Mock()
    adapter._pending_ipc_export = export
    adapter._legacy_meta_server_client = client
    adapter._colocate_init_fingerprint = ("pair",)

    with pytest.raises(RuntimeError, match="awaits reader acknowledgement"):
        adapter.teardown_colocate_weight_update()

    assert adapter._pending_ipc_export is export
    assert adapter._legacy_meta_server_client is client
    assert adapter._colocate_init_fingerprint == ("pair",)


class _MetaServer:
    def __init__(self, events):
        self.events = events

    def wait_set_until_size(self, key, size, timeout):
        self.events.append(("wait", key, size, timeout))

    def delete_if_exists(self, key):
        self.events.append(("delete", key))


def _adapter(events):
    adapter = AwexMegatronAdapter.__new__(AwexMegatronAdapter)
    adapter._engine = SimpleNamespace(cpu_group="cpu")
    adapter._legacy_meta_server_client = _MetaServer(events)
    adapter._legacy_num_infer_engines = 4
    adapter._legacy_timeout_s = 120.0
    return adapter


def test_finish_colocate_update_cleans_reusable_keys_before_broadcast():
    events = []
    adapter = _adapter(events)

    with (
        mock.patch(
            "areal.engine.awex.adapters.megatron_adapter.dist.get_world_size",
            return_value=8,
        ),
        mock.patch(
            "areal.engine.awex.adapters.megatron_adapter.dist.get_rank",
            return_value=0,
        ),
        mock.patch(
            "areal.engine.awex.adapters.megatron_adapter.dist.barrier",
            side_effect=lambda group: events.append(("barrier", group)),
        ),
        mock.patch(
            "areal.engine.awex.adapters.megatron_adapter.dist.get_global_rank",
            return_value=2,
        ),
        mock.patch(
            "areal.engine.awex.adapters.megatron_adapter.dist.broadcast_object_list",
            side_effect=lambda status, src, group: events.append(
                ("broadcast", status[:], src, group)
            ),
        ),
    ):
        adapter.finish_colocate_weight_update(training_world_size=8)

    assert events == [
        ("barrier", "cpu"),
        ("wait", "finished_weights_update_engines", 4, 120.0),
        ("delete", "finished_weights_update_engines"),
        ("delete", "all_training_offloaded_weights"),
        ("broadcast", [None], 2, "cpu"),
    ]


def test_nonzero_rank_only_participates_in_finish_collectives():
    events = []
    adapter = _adapter(events)

    with (
        mock.patch(
            "areal.engine.awex.adapters.megatron_adapter.dist.get_world_size",
            return_value=8,
        ),
        mock.patch(
            "areal.engine.awex.adapters.megatron_adapter.dist.get_rank",
            return_value=3,
        ),
        mock.patch(
            "areal.engine.awex.adapters.megatron_adapter.dist.barrier",
            side_effect=lambda group: events.append(("barrier", group)),
        ),
        mock.patch(
            "areal.engine.awex.adapters.megatron_adapter.dist.get_global_rank",
            return_value=2,
        ),
        mock.patch(
            "areal.engine.awex.adapters.megatron_adapter.dist.broadcast_object_list",
            side_effect=lambda status, src, group: events.append(
                ("broadcast", status[:], src, group)
            ),
        ),
    ):
        adapter.finish_colocate_weight_update(training_world_size=8)

    assert events == [("barrier", "cpu"), ("broadcast", [None], 2, "cpu")]


def test_worker_execute_endpoint_runs_finish_after_transfer():
    events = []
    adapter = SimpleNamespace(
        parallelism_strategy={"world_size": 8},
        enable_colocate_memory_management=lambda: events.append("enable"),
        init_colocate_weight_update=lambda **_kwargs: events.append("init"),
        execute_colocate_weight_update=lambda version: events.append(
            ("execute", version)
        ),
        finish_colocate_weight_update=lambda training_world_size: events.append(
            ("finish", training_world_size)
        ),
        release_memory=lambda tags=None: None,
        resume_memory=lambda tags=None: None,
        teardown_colocate_weight_update=lambda: None,
    )

    def submit_to_engine_thread(_name, action):
        return action()

    def run_endpoint(_name, action, return_result=True):
        result = action()
        return flask.jsonify({"result": result} if return_result else {"status": "ok"})

    with mock.patch(
        "areal.v2.training_service.worker.awex._create_training_adapter",
        return_value=adapter,
    ):
        app = flask.Flask(__name__)
        app.register_blueprint(
            create_awex_blueprint(
                flask_module=flask,
                get_engine=lambda: object(),
                submit_to_engine_thread=submit_to_engine_thread,
                run_endpoint=run_endpoint,
            )
        )
        response = app.test_client().post(
            "/awex/execute_colocate_weight_update", json={"version": 3}
        )

    assert response.status_code == 200
    assert events == ["enable", ("execute", 3), ("finish", 8)]
