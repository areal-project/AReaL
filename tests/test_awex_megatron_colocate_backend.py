# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from areal.engine.weight_update.awex import megatron
from areal.engine.weight_update.awex.megatron import MegatronColocateBackend


class _FakeStorage:
    def __init__(self, pointer):
        self._pointer = pointer

    def data_ptr(self):
        return self._pointer


class _FakeTensor:
    def __init__(self, name, pointer, events):
        self.name = name
        self._storage = _FakeStorage(pointer)
        self._events = events

    def untyped_storage(self):
        return self._storage

    def share_memory_(self):
        self._events.append(("share", self.name))
        return self


class _Chunk:
    def __init__(self, live_tensor):
        self._live_tensor = live_tensor

    def named_parameters(self):
        return [("live", self._live_tensor)]

    def named_buffers(self):
        return []


class _MetaServer:
    def __init__(self, events):
        self.events = events

    def put_object(self, key, value):
        del value
        self.events.append(("put", key))

    def add_object_to_set(self, key, value):
        del value
        self.events.append(("add", key))

    def get_object(self, key, timeout):
        del timeout
        self.events.append(("get", key))
        return True

    def delete_if_exists(self, key):
        self.events.append(("delete", key))


def _install_tensor_util(monkeypatch, events, grouped):
    awex = ModuleType("awex")
    util = ModuleType("awex.util")
    tensor_util = ModuleType("awex.util.tensor_util")

    def group_tensors(tensors):
        events.append(("group", tuple(t.name for t in tensors)))
        return [grouped], {"metadata": True}

    def serialize(payload):
        del payload
        events.append(("serialize",))
        return "serialized"

    def release(tensors):
        events.append(("release", tuple(t.name for t in tensors)))

    tensor_util.group_tensors_by_shape_and_dtype = group_tensors
    tensor_util.cuda_ipc_serialize = serialize
    tensor_util.release_tensors = release
    monkeypatch.setitem(sys.modules, "awex", awex)
    monkeypatch.setitem(sys.modules, "awex.util", util)
    monkeypatch.setitem(sys.modules, "awex.util.tensor_util", tensor_util)


def _ready_backend(monkeypatch, events):
    live = _FakeTensor("live", 1, events)
    converted = _FakeTensor("converted", 2, events)
    grouped = _FakeTensor("grouped", 3, events)
    engine = SimpleNamespace(model=[_Chunk(live)], optimizer=None)
    backend = MegatronColocateBackend(
        engine,
        physical_gpu_id_resolver=lambda: 0,
    )
    backend.configure(meta_server_client=_MetaServer(events), timeout_s=10.0)
    backend.initialized = True
    backend.rank_info = object()
    backend.ip_address = "192.0.2.1"
    backend.physical_gpu_id = 4
    backend.logical_train_rank = 9

    def release_memory(tags):
        events.append(("release_memory", tuple(tags)))
        backend.released_tags.update(tags)

    def resume_memory(tags):
        events.append(("resume_memory", tuple(tags)))
        backend.released_tags.difference_update(tags)

    monkeypatch.setattr(backend, "release_memory", release_memory)
    monkeypatch.setattr(backend, "resume_memory", resume_memory)
    monkeypatch.setattr(
        backend, "release_grad_memory", lambda: events.append(("release_grad",))
    )
    monkeypatch.setattr(backend, "lazy_initialize", lambda: events.append(("lazy",)))
    monkeypatch.setattr(
        backend,
        "convert_parameters",
        lambda: events.append(("convert",)) or {"weight": converted},
    )
    _install_tensor_util(monkeypatch, events, grouped)

    monkeypatch.setattr(
        megatron.torch.cuda,
        "synchronize",
        lambda: events.append(("synchronize",)),
    )
    monkeypatch.setattr(
        megatron.torch.cuda,
        "ipc_collect",
        lambda: events.append(("ipc_collect",)),
    )
    monkeypatch.setattr(
        megatron.torch.cuda,
        "empty_cache",
        lambda: events.append(("empty_cache",)),
    )
    monkeypatch.setattr(megatron.gc, "collect", lambda: events.append(("gc",)))
    barrier = mock.Mock()
    monkeypatch.setattr(megatron.dist, "barrier", barrier)
    return backend, barrier


@pytest.mark.parametrize(
    (
        "publish_before",
        "restore_state",
        "collect_after",
        "wrap_timeout",
        "expected_ipc_collects",
    ),
    [
        (True, False, False, False, 2),
        (False, True, True, True, 3),
    ],
)
def test_execute_preserves_facade_protocol_order_and_device_operation_counts(
    monkeypatch,
    publish_before,
    restore_state,
    collect_after,
    wrap_timeout,
    expected_ipc_collects,
):
    events = []
    backend, barrier = _ready_backend(monkeypatch, events)

    backend.execute_weight_update(
        5,
        publish_offloaded_before_payload=publish_before,
        restore_initial_weight_state=restore_state,
        collect_ipc_after_update=collect_after,
        wrap_reader_timeout=wrap_timeout,
    )

    offloaded = ("add", "all_training_offloaded_weights")
    payload = ("put", "training_serialized_weights_192.0.2.1_4_5")
    if publish_before:
        assert events.index(offloaded) < events.index(("share", "grouped"))
    else:
        assert events.index(payload) < events.index(offloaded)
    assert events.index(("put", "awex_writer_version_192.0.2.1_4")) < (
        events.index(payload)
    )
    assert events.index(
        ("delete", "training_serialized_weights_192.0.2.1_4_5")
    ) < events.index(("put", "write_finished_192.0.2.1_4_5"))
    assert events.count(("synchronize",)) == 3
    assert events.count(("ipc_collect",)) == expected_ipc_collects
    assert ("release", ("converted",)) in events
    assert ("release", ("live",)) not in events
    barrier.assert_not_called()


@pytest.mark.parametrize(
    ("restore_state", "collect_after", "weights_restored", "ipc_collects"),
    [
        (False, False, False, 1),
        (True, True, True, 2),
    ],
)
def test_execute_failure_preserves_each_facade_cleanup_policy(
    monkeypatch,
    restore_state,
    collect_after,
    weights_restored,
    ipc_collects,
):
    events = []
    backend, _ = _ready_backend(monkeypatch, events)
    backend.released_tags.add("weights")

    def fail_lazy_initialize():
        events.append(("lazy",))
        raise RuntimeError("resolver failed")

    monkeypatch.setattr(backend, "lazy_initialize", fail_lazy_initialize)

    with pytest.raises(RuntimeError, match="resolver failed"):
        backend.execute_weight_update(
            5,
            publish_offloaded_before_payload=False,
            restore_initial_weight_state=restore_state,
            collect_ipc_after_update=collect_after,
            wrap_reader_timeout=True,
        )

    assert ("weights" in backend.released_tags) is weights_restored
    assert events.count(("ipc_collect",)) == ipc_collects
