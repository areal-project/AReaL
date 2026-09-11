# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import MegatronEngineConfig, PPOActorConfig
from areal.engine.awex.colocate_reader import (
    _PhysicalDeviceMetaServerClient,
)
from areal.engine.awex.memory_saver import patch_tms_hook_mode
from areal.engine.awex.sglang_plugin import (
    AwexSchedulerPlugin,
    _load_sglang_plugins_if_available,
    _resolve_transfer_rank,
    _writer_version_key,
)


def test_load_sglang_plugins_accepts_runtime_without_registry(monkeypatch):
    import areal.engine.awex.sglang_plugin as plugin_module

    def _missing_registry(name):
        assert name == "sglang.srt.plugins"
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(plugin_module.importlib, "import_module", _missing_registry)

    assert _load_sglang_plugins_if_available() is False


def test_event_loop_patch_supports_current_metrics_api():
    class Scheduler:
        def __init__(self):
            self.forward_ct_decode = 7
            self.event_loop_overlap = lambda: None
            self.event_loop_normal = lambda: None
            self.calls = []

        def report_decode_stats(
            self, can_run_cuda_graph, running_batch=None, num_accepted_tokens=0
        ):
            self.calls.append((can_run_cuda_graph, running_batch, num_accepted_tokens))

    scheduler = Scheduler()
    AwexSchedulerPlugin(scheduler)._patch_event_loop()
    scheduler.report_decode_stats(True, running_batch=object(), num_accepted_tokens=3)

    assert scheduler._areal_awex_last_decode_stats_ct == 7
    assert scheduler.calls[0][0] is True
    assert scheduler.calls[0][2] == 3


def test_memory_transitions_are_idempotent():
    class Scheduler:
        def __init__(self):
            self.offload_tags = set()
            self.calls = []

        def release_memory_occupation(self, request):
            self.calls.append(("release", list(request.tags)))
            self.offload_tags.update(request.tags)

        def resume_memory_occupation(self, request):
            self.calls.append(("resume", list(request.tags)))
            self.offload_tags.difference_update(request.tags)

    scheduler = Scheduler()
    AwexSchedulerPlugin(scheduler)._patch_memory_transitions()
    request = SimpleNamespace(tags=["kv_cache"])

    scheduler.release_memory_occupation(request)
    scheduler.release_memory_occupation(request)
    scheduler.resume_memory_occupation(request)
    scheduler.resume_memory_occupation(request)

    assert scheduler.calls == [
        ("release", ["kv_cache"]),
        ("resume", ["kv_cache"]),
    ]


def test_tms_hook_mode_stays_preload_after_initialization(monkeypatch):
    import sys

    class Saver:
        def __init__(self):
            self._impl_ctor_kwargs = {}

        @property
        def hook_mode(self):
            raise AttributeError

        @hook_mode.setter
        def hook_mode(self, value):
            self._impl_ctor_kwargs["hook_mode"] = value

    saver = Saver()
    monkeypatch.setitem(
        sys.modules, "torch_memory_saver", SimpleNamespace(torch_memory_saver=saver)
    )
    monkeypatch.setenv("SGLANG_MEMORY_SAVER_CUDA_GRAPH", "1")

    patch_tms_hook_mode()
    saver.hook_mode = "torch"

    assert saver._impl_ctor_kwargs == {}


def test_awex_meta_client_uses_physical_device_for_colocate_identity():
    class Client:
        def __init__(self):
            self.calls = []

        def add_object_to_set(self, key, value):
            self.calls.append(("add", key, value))

        def get_object(self, key, *args, **kwargs):
            self.calls.append(("get", key, args, kwargs))

        def put_object(self, key, *args, **kwargs):
            self.calls.append(("put", key, args, kwargs))

        def get_object_then_delete(self, key, *args, **kwargs):
            self.calls.append(("delete", key, args, kwargs))

    client = Client()
    physical_client = _PhysicalDeviceMetaServerClient(client, physical_gpu_id=6)

    physical_client.add_object_to_set(
        "inference_device_rank_entries", ("10.0.0.1", 0, 6)
    )
    physical_client.get_object("training_serialized_weights_10.0.0.1_0_3")
    physical_client.put_object("weights_update_finished_10.0.0.1_0_3", True)
    physical_client.get_object_then_delete("write_finished_10.0.0.1_0_3")

    assert client.calls == [
        ("add", "inference_device_rank_entries", ("10.0.0.1", 6, 6)),
        ("get", "training_serialized_weights_10.0.0.1_6_3", (), {}),
        ("put", "weights_update_finished_10.0.0.1_6_3", (True,), {}),
        ("delete", "write_finished_10.0.0.1_6_3", (), {}),
    ]


def test_awex_weight_update_runs_without_grad_tracking():
    from areal.engine.awex.colocate_reader import AwexColocateReader

    grad_modes = []
    reader = SimpleNamespace(
        update_weights=lambda step_id: grad_modes.append(torch.is_grad_enabled())
    )
    instance = object.__new__(AwexColocateReader)
    instance._initialized = True
    instance._ensure_reader = lambda: reader
    instance._rebuild_derived_weights = lambda: None

    AwexColocateReader.update_weights(instance, 1)

    assert grad_modes == [False]


def test_transfer_rank_uses_global_rank_for_isolated_gpu(monkeypatch):
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("WORLD_SIZE", "8")

    assert (
        _resolve_transfer_rank(
            infer_world_size=8,
            gpu_id=0,
            node_id=0,
            nnodes=1,
            instance_world_size=1,
        )
        == 7
    )


@pytest.mark.parametrize(("tp_size", "pp_size"), [(4, 1), (1, 4)])
def test_scheduler_instance_world_size_includes_tp_and_pp(tp_size, pp_size):
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(tp_size=tp_size, pp_size=pp_size)
    )

    assert AwexSchedulerPlugin(scheduler)._instance_world_size() == 4


def test_transfer_rank_uses_scheduler_gpu_for_multi_gpu_server(monkeypatch):
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("WORLD_SIZE", "32")

    ranks = [
        _resolve_transfer_rank(
            infer_world_size=32,
            gpu_id=gpu_id,
            node_id=2,
            nnodes=4,
            instance_world_size=4,
        )
        for gpu_id in range(4)
    ]

    assert ranks == [16, 17, 18, 19]


def test_transfer_rank_falls_back_to_node_local_identity(monkeypatch):
    monkeypatch.delenv("AWEX_TRANSFER_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    assert (
        _resolve_transfer_rank(
            infer_world_size=16,
            gpu_id=3,
            node_id=1,
            nnodes=2,
            instance_world_size=1,
        )
        == 11
    )


def test_physical_gpu_id_uses_noncontiguous_visible_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5,6,7")
    scheduler = SimpleNamespace(gpu_id=1)
    gpu_id = AwexSchedulerPlugin(scheduler)._physical_gpu_id()

    assert gpu_id == 5
    assert _writer_version_key("10.0.0.1", gpu_id) == "awex_writer_version_10.0.0.1_5"


def test_physical_gpu_id_rejects_uuid_visible_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-deadbeef")

    with pytest.raises(ValueError, match="numeric CUDA_VISIBLE_DEVICES"):
        AwexSchedulerPlugin(SimpleNamespace(gpu_id=0))._physical_gpu_id()


def test_awex_rejects_megatron_without_ddp_flat_buffers():
    with pytest.raises(ValueError, match="requires megatron.wrap_with_ddp=true"):
        PPOActorConfig(
            backend="megatron:d1",
            weight_update_mode="awex",
            megatron=MegatronEngineConfig(wrap_with_ddp=False),
        )
