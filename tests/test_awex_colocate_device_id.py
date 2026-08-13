from types import SimpleNamespace

import areal.engine.awex.colocate_reader as colocate_reader
from areal.engine.awex.colocate_reader import _PhysicalDeviceNCCLWorkerWeightsReader
from areal.engine.awex.colocate_writer import resolve_physical_gpu_id
from areal.engine.awex.sglang_plugin import _resolve_colocate_transfer_rank


def test_physical_gpu_id_maps_through_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    assert resolve_physical_gpu_id(0) == 4
    assert resolve_physical_gpu_id(3) == 7


def test_physical_gpu_id_is_identity_without_visible_devices(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert resolve_physical_gpu_id(2) == 2


def test_physical_gpu_id_falls_back_on_non_integer_entries(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc,GPU-def")

    assert resolve_physical_gpu_id(1) == 1


def test_physical_gpu_id_falls_back_when_index_out_of_range(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")

    assert resolve_physical_gpu_id(2) == 2


def test_sglang_instances_get_unique_colocate_transfer_ranks(monkeypatch):
    """Four TP=2 SGLang instances must occupy distinct AWEX ranks 0 through 7."""
    transfer_ranks = []

    for visible_devices in ("0,1", "2,3", "4,5", "6,7"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
        transfer_ranks.extend(
            _resolve_colocate_transfer_rank(
                gpu_id=relative_gpu_id,
                node_id=0,
                n_gpus_per_node=8,
            )[0]
            for relative_gpu_id in range(2)
        )

    assert transfer_ranks == list(range(8))


def test_colocate_transfer_rank_includes_node_offset(monkeypatch):
    """The global transfer rank must include the physical node offset."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")

    transfer_rank, physical_gpu_id = _resolve_colocate_transfer_rank(
        gpu_id=1,
        node_id=2,
        n_gpus_per_node=8,
    )

    assert transfer_rank == 23
    assert physical_gpu_id == 7


class _FakeMetaServerClient:
    def __init__(self, inference_entries=(), training_entries=(), payload=None):
        self.inference_entries = set(inference_entries)
        self.training_entries = set(training_entries)
        self.payload = payload
        self.added = []
        self.requested_keys = []
        self.put_objects = []
        self.deleted_keys = []

    def add_object_to_set(self, key, value):
        self.added.append((key, value))
        if key == "inference_device_rank_entries":
            self.inference_entries.add(value)

    def wait_set_until_size(self, key, size, timeout):
        entries = (
            self.inference_entries
            if key == "inference_device_rank_entries"
            else self.training_entries
        )
        assert len(entries) == size

    def get_set(self, key):
        if key == "inference_device_rank_entries":
            return self.inference_entries
        return self.training_entries

    def get_object(self, key, timeout):
        self.requested_keys.append(key)
        return self.payload

    def put_object(self, key, value):
        self.put_objects.append((key, value))

    def get_object_then_delete(self, key):
        self.deleted_keys.append(key)


def test_awex_reader_registers_physical_device_for_colocate_mapping(monkeypatch):
    """Inference device metadata must match the writer's physical GPU keys."""

    class FakePlanBuilder:
        def __init__(self, *args):
            pass

        def build_local_transfer_plan(self, *args):
            return "plan"

    class FakeTransport:
        def __init__(self, rank, world_size):
            self.rank = rank
            self.world_size = world_size

    inference_entries = {("host", device_id, device_id) for device_id in range(8)}
    training_entries = {("host", device_id, device_id) for device_id in range(8)}
    client = _FakeMetaServerClient(inference_entries, training_entries)
    reader = _PhysicalDeviceNCCLWorkerWeightsReader.__new__(
        _PhysicalDeviceNCCLWorkerWeightsReader
    )
    reader.meta_server_client = client
    reader.physical_device_id = 7
    reader.transfer_rank = 7
    reader.infer_world_size = 8
    reader.training_world_size = 8
    reader.num_engines = 4
    reader.enable_debug_mode = False
    reader.timeout = 10
    reader.parameters_meta = []
    reader.training_params_meta = []

    monkeypatch.setattr(colocate_reader, "get_ip_address", lambda: "host")
    monkeypatch.setattr(colocate_reader, "TransferPlanBuilder", FakePlanBuilder)
    monkeypatch.setattr(
        "awex.transfer.nccl_stream_batch.NcclColocateStreamBatchTransport",
        FakeTransport,
    )

    reader._init_reader_in_colocate_mode()

    assert (
        "inference_device_rank_entries",
        ("host", 7, 7),
    ) in client.added
    assert reader.inference_device_mapping[("host", 7)] == 7
    assert reader.infer_to_train_device_mapping[7] == 7


def test_awex_reader_uses_physical_key_and_runtime_deserialize_device(monkeypatch):
    """IPC lookup uses physical id while CUDA import uses the visible id."""
    payload = (3, SimpleNamespace(), "serialized")
    client = _FakeMetaServerClient(payload=payload)
    reader = _PhysicalDeviceNCCLWorkerWeightsReader.__new__(
        _PhysicalDeviceNCCLWorkerWeightsReader
    )
    reader.meta_server_client = client
    reader.physical_device_id = 7
    reader.enable_colocate_mode = True
    reader.timeout = 10
    reader.rank_coordinate = "3-1-7"
    reader.ipc_backend = "cuda"

    reconstruct_calls = []
    monkeypatch.setattr(colocate_reader, "get_ip_address", lambda: "host")
    monkeypatch.setattr(colocate_reader.device_util, "current_device", lambda: 1)
    monkeypatch.setattr(colocate_reader, "get_gpu_status", lambda: "ok")
    monkeypatch.setattr(colocate_reader, "count_open_fds", lambda: 0)
    monkeypatch.setattr(
        colocate_reader,
        "reconstruct_ipc_weights",
        lambda payload, ipc_backend, device_id: (
            reconstruct_calls.append((payload, ipc_backend, device_id)) or {},
            1,
        ),
    )

    reader.collect_training_weights(step_id=5)

    assert client.requested_keys == ["training_serialized_weights_host_7_5"]
    assert reconstruct_calls == [("serialized", "cuda", 1)]


def test_awex_reader_signals_physical_key_but_barriers_on_runtime_device(
    monkeypatch,
):
    """Completion keys and NCCL barriers must use their respective identities."""

    class FakeTransport:
        def update_weights_in_colocate_mode(self, *args, **kwargs):
            pass

    client = _FakeMetaServerClient()
    reader = _PhysicalDeviceNCCLWorkerWeightsReader.__new__(
        _PhysicalDeviceNCCLWorkerWeightsReader
    )
    reader.meta_server_client = client
    reader.physical_device_id = 7
    reader.enable_colocate_mode = True
    reader.transfer_rank = 7
    reader.rank_coordinate = "3-1-7"
    reader.transfer_plan = SimpleNamespace(operations={})
    reader.send_ranks_sample = []
    reader.colocate_transport = FakeTransport()
    reader.train_to_infer_device_mapping = {}
    reader.infer_to_train_device_mapping = {}
    reader.infer_world_size = 8
    reader.send_transfer_plan = SimpleNamespace()
    reader.weights_update_group = SimpleNamespace()
    reader.deserialized_weights = {}
    reader.parameters = {}
    reader._history_update_weights_time = {}

    barrier_calls = []
    monkeypatch.setattr(
        reader, "collect_training_weights", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(colocate_reader, "get_ip_address", lambda: "host")
    monkeypatch.setattr(colocate_reader, "print_current_gpu_status", lambda *args: None)
    monkeypatch.setattr(colocate_reader, "compute_statistics", lambda *args: None)
    monkeypatch.setattr(colocate_reader.device_util, "synchronize", lambda: None)
    monkeypatch.setattr(colocate_reader.device_util, "current_device", lambda: 1)
    monkeypatch.setattr(colocate_reader.device_util, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(
        colocate_reader.dist,
        "barrier",
        lambda group, device_ids: barrier_calls.append((group, device_ids)),
    )

    reader._update_weights_in_colocate_mode(step_id=5)

    assert client.put_objects == [("weights_update_finished_host_7_5", True)]
    assert client.deleted_keys == ["write_finished_host_7_5"]
    assert barrier_calls == [(reader.weights_update_group, [1])]
