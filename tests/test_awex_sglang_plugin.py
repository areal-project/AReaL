# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from areal.engine.awex.memory_saver import patch_tms_hook_mode
from areal.engine.awex.sglang_plugin import (
    AwexSchedulerPlugin,
    _load_sglang_plugins_if_available,
    _resolve_physical_gpu_id,
    _resolve_transfer_rank,
    _scheduler_instance_world_size,
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

    assert _scheduler_instance_world_size(scheduler) == 4


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


def test_physical_gpu_id_and_writer_key_are_node_local():
    gpu_id = _resolve_physical_gpu_id(
        transfer_rank=11,
        infer_world_size=16,
        nnodes=2,
    )

    assert gpu_id == 3
    assert _writer_version_key("10.0.0.1", gpu_id) == "awex_writer_version_10.0.0.1_3"
