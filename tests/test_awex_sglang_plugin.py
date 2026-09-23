# SPDX-License-Identifier: Apache-2.0

import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import MegatronEngineConfig, PPOActorConfig
from areal.engine.awex.colocate_reader import (
    _get_awex_infer_hf_config,
    _get_router_dtype,
    _PhysicalDeviceMetaServerClient,
)
from areal.engine.awex.memory_saver import patch_tms_hook_mode
from areal.engine.awex.metadata import serialize_metadata_gc
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


def test_native_scheduler_hook_preserves_loops_and_runs_once(monkeypatch):
    class Scheduler:
        _engine_paused = False

        def _apply_war_barrier(self):
            pass

        def event_loop_overlap(self):
            pass

        def event_loop_normal(self):
            pass

        def process_input_requests(self, requests):
            self._engine_paused = requests == ["pause"]
            return "processed"

    scheduler = Scheduler()
    native_overlap = scheduler.event_loop_overlap
    native_normal = scheduler.event_loop_normal
    plugin = AwexSchedulerPlugin(scheduler)
    calls = []
    monkeypatch.setattr(plugin, "process_awex_queue", lambda: calls.append(True))
    plugin._patch_event_loop()
    plugin._patch_event_loop()
    assert scheduler.process_input_requests([]) == "processed"
    assert calls == []
    assert scheduler.process_input_requests(["pause"]) == "processed"
    assert calls == [True]
    assert scheduler.event_loop_overlap == native_overlap
    assert scheduler.event_loop_normal == native_normal


def test_awex_config_preserves_nested_router_and_vision_metadata(monkeypatch):
    import areal.engine.awex.colocate_reader as reader

    composite = object()

    def serialize(config):
        assert config is composite
        return {
            "text_config": {"router_dtype": "fp32"},
            "vision_config": {"hidden_size": 64},
        }

    monkeypatch.setattr(reader, "simple_hf_config", serialize)
    config = _get_awex_infer_hf_config(
        SimpleNamespace(config=object()),
        SimpleNamespace(model_config=SimpleNamespace(hf_config=composite)),
    )
    assert config.vision_config.hidden_size == 64
    assert _get_router_dtype(config) == "fp32"
    assert config.architectures == ["SimpleNamespace"]


def test_memory_transitions_are_idempotent(monkeypatch):
    import sys

    class ReleaseOutput:
        pass

    class ResumeOutput:
        pass

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.io_struct",
        SimpleNamespace(
            ReleaseMemoryOccupationReqOutput=ReleaseOutput,
            ResumeMemoryOccupationReqOutput=ResumeOutput,
        ),
    )

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
    assert isinstance(scheduler.release_memory_occupation(request), ReleaseOutput)
    scheduler.resume_memory_occupation(request)
    assert isinstance(scheduler.resume_memory_occupation(request), ResumeOutput)

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
    instance = AwexColocateReader(SimpleNamespace())
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


@pytest.mark.parametrize(
    "installed", ["0.5.9", "0.5.10.post1", "0.5.18.dev10+g85b539146"]
)
def test_supported_sglang_builds_are_accepted(monkeypatch, installed):
    import areal.engine.awex.sglang_plugin as plugin

    monkeypatch.setattr(plugin.pkg_version, "get_version", lambda name: installed)
    plugin.assert_supported_sglang_version()


def test_unverified_sglang_build_is_rejected(monkeypatch):
    import areal.engine.awex.sglang_plugin as plugin

    monkeypatch.setattr(plugin.pkg_version, "get_version", lambda name: "0.5.19.dev126")
    with pytest.raises(RuntimeError, match="Re-check Scheduler"):
        plugin.assert_supported_sglang_version()


@pytest.mark.parametrize(
    "visible,logical", [(None, 7), ("7", 0), ("4,5,6,7", 2), ("GPU-uuid", 0)]
)
def test_awex_reader_uses_model_device_independent_of_physical_ids(
    monkeypatch, visible, logical
):
    from areal.engine.awex.colocate_reader import _DeviceBoundWeightsReader

    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    reader = _DeviceBoundWeightsReader.__new__(_DeviceBoundWeightsReader)
    reader._model_device = torch.device("cuda", logical)
    reader.transfer_rank = 7
    devices = []
    monkeypatch.setattr(torch.cuda, "set_device", devices.append)
    monkeypatch.setattr(torch, "tensor", lambda value, **kwargs: kwargs["device"])

    reader._set_device()

    assert devices == [reader._model_device]
    assert reader.barrier_device == logical
    assert reader.backend == "nccl"
    assert reader.ready_tensor == reader._model_device


def test_awex_reader_rejects_model_weights_still_on_cpu():
    from areal.engine.awex.colocate_reader import _DeviceBoundWeightsReader

    with pytest.raises(RuntimeError, match="model weights resumed on CUDA"):
        _DeviceBoundWeightsReader(model=torch.nn.Linear(2, 2))


def test_receiver_initialization_failure_reaches_scheduler_loop():
    plugin = AwexSchedulerPlugin(SimpleNamespace())
    failure = SystemError("metadata tuple construction failed")
    plugin._initialization_error = failure
    with pytest.raises(
        RuntimeError, match="AWEX receiver initialization failed"
    ) as exc:
        plugin.process_awex_queue()
    assert exc.value.__cause__ is failure


def test_native_scheduler_surfaces_initialization_error_while_unpaused():
    calls = []
    scheduler = SimpleNamespace(
        _apply_war_barrier=lambda: None,
        _engine_paused=False,
        process_input_requests=lambda requests: calls.append(requests),
    )
    plugin = AwexSchedulerPlugin(scheduler)
    plugin._patch_event_loop()
    plugin._initialization_error = ValueError("invalid metadata")
    with pytest.raises(RuntimeError, match="AWEX receiver initialization failed"):
        scheduler.process_input_requests([])
    assert not calls


def test_scheduler_dispatcher_captures_gc_guard_and_registration_is_idempotent(
    monkeypatch,
):
    import sys

    import areal.engine.awex.sglang_plugin as plugin_module

    class Scheduler:
        def __init__(self):
            self.dispatch = {"freeze": self.handle_freeze_gc}

        def handle_freeze_gc(self, request):
            return request

    original_freeze = Scheduler.handle_freeze_gc
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.scheduler",
        SimpleNamespace(Scheduler=Scheduler),
    )
    monkeypatch.delenv("QWEN_AWEX_FROZEN_CONTRACT", raising=False)
    monkeypatch.setattr(plugin_module, "assert_supported_sglang_version", lambda: None)
    monkeypatch.setattr(AwexSchedulerPlugin, "bind", lambda self: None)
    monkeypatch.setattr(
        plugin_module, "_patch_execute_task_in_model_worker", lambda *a: None
    )
    plugin_module.register_awex_plugin()
    guarded = Scheduler.handle_freeze_gc
    assert guarded.__wrapped__ is original_freeze
    plugin_module.register_awex_plugin()
    assert Scheduler.handle_freeze_gc is guarded
    scheduler = Scheduler()
    assert scheduler.dispatch["freeze"].__func__ is guarded
    assert scheduler.dispatch["freeze"]("request") == "request"


def test_gc_scan_waits_until_metadata_tuple_is_complete():
    building = threading.Event()
    scan_requested = threading.Event()
    events = []
    marker = object()

    @serialize_metadata_gc
    def build_metadata():
        def dimensions():
            yield marker
            building.set()
            assert scan_requested.wait(5)
            yield 2

        result = tuple(dimensions())
        events.append("metadata complete")
        return result

    @serialize_metadata_gc
    def freeze_gc():
        # Holding references to a growing tuple here would cause SystemError
        # when tuple(dimensions()) resizes its allocation after iteration.
        retained = gc.get_referrers(marker)
        events.append("GC scan")
        return retained

    def request_scan():
        assert building.wait(5)
        scan_requested.set()
        return freeze_gc()

    with ThreadPoolExecutor(max_workers=2) as pool:
        metadata = pool.submit(build_metadata)
        scan = pool.submit(request_scan)
        assert metadata.result(timeout=10) == (marker, 2)
        scan.result(timeout=10)
    assert events == ["metadata complete", "GC scan"]


def test_metadata_gc_guard_releases_lock_after_failure():
    import pytest

    @serialize_metadata_gc
    def fail():
        raise ValueError("metadata failed")

    @serialize_metadata_gc
    def succeed():
        return "ready"

    with pytest.raises(ValueError, match="metadata failed"):
        fail()
    assert succeed() == "ready"


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("location", ["scheduler", "worker", "runner", "worker_direct"])
def test_legacy_model_worker_callback_preserves_native_tp_rank(rank, location):
    from areal.engine.awex.sglang_plugin import _patch_execute_task_in_model_worker

    runner = SimpleNamespace(model=object())
    worker = SimpleNamespace(model_runner=runner)
    scheduler = SimpleNamespace(
        tp_worker=worker, server_args=SimpleNamespace(tp_size=4)
    )
    owner = {
        "scheduler": scheduler,
        "worker": worker,
        "runner": runner,
        "worker_direct": worker,
    }[location]
    if location == "worker_direct":
        owner.tp_rank = rank
        owner.tp_size = 4
    else:
        owner.ps = SimpleNamespace(tp_rank=rank, tp_size=4)
    plugin = AwexSchedulerPlugin(scheduler)
    _patch_execute_task_in_model_worker(scheduler, plugin)
    task = SimpleNamespace(
        kwargs={}, task_func=lambda **kwargs: kwargs["model_context"]
    )

    context = scheduler.execute_task_in_model_worker(task)

    assert context["tp_rank"] == rank
    assert context["tp_size"] == 4


@pytest.mark.parametrize("rank", [None, -1, 4])
def test_legacy_model_worker_callback_rejects_unresolved_or_invalid_tp_rank(rank):
    from areal.engine.awex.sglang_plugin import _patch_execute_task_in_model_worker

    scheduler = SimpleNamespace(
        tp_rank=rank,
        server_args=SimpleNamespace(tp_size=4),
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(model=object())),
    )
    plugin = AwexSchedulerPlugin(scheduler)
    _patch_execute_task_in_model_worker(scheduler, plugin)
    task = SimpleNamespace(kwargs={}, task_func=lambda **kwargs: kwargs)

    with pytest.raises(RuntimeError, match="valid AWEX inference TP rank"):
        scheduler.execute_task_in_model_worker(task)


def test_legacy_model_worker_callback_allows_single_rank_without_parallel_state():
    from areal.engine.awex.sglang_plugin import _patch_execute_task_in_model_worker

    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(tp_size=1),
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(model=object())),
    )
    plugin = AwexSchedulerPlugin(scheduler)
    _patch_execute_task_in_model_worker(scheduler, plugin)
    task = SimpleNamespace(
        kwargs={}, task_func=lambda **kwargs: kwargs["model_context"]
    )
    assert scheduler.execute_task_in_model_worker(task)["tp_rank"] == 0
