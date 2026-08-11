# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from areal.engine.awex import colocate_reader as awex_colocate_reader
from areal.engine.awex.colocate_reader import (
    AwexColocateReader,
    _BailingV3PhysicalKeyNCCLWorkerWeightsReader,
    _BoundedMemoryNcclColocateStreamBatchTransport,
    _patch_awex_qwen3_attention_names,
)
from areal.engine.awex.sglang_plugin import (
    AwexSchedulerPlugin,
    _load_sglang_plugins_if_available,
    _resolve_physical_gpu_id,
    _resolve_transfer_rank,
    _scheduler_instance_world_size,
    _writer_version_key,
)


def test_load_sglang_plugins_accepts_pinned_runtime_without_registry(monkeypatch):
    """SGLang 0.5.10 starts through launch_server without a plugin registry."""
    import areal.engine.awex.sglang_plugin as plugin_module

    def _missing_registry(name):
        assert name == "sglang.srt.plugins"
        raise ModuleNotFoundError(
            "No module named 'sglang.srt.plugins'",
            name="sglang.srt.plugins",
        )

    monkeypatch.setattr(plugin_module.importlib, "import_module", _missing_registry)

    assert _load_sglang_plugins_if_available() is False


class _SchedulerWithCurrentMetricsAPI:
    def __init__(self) -> None:
        self.forward_ct_decode = 7
        self.event_loop_overlap = lambda: None
        self.event_loop_normal = lambda: None
        self.calls = []

    def report_decode_stats(
        self,
        can_run_cuda_graph,
        running_batch=None,
        num_accepted_tokens=0,
    ) -> None:
        self.calls.append((can_run_cuda_graph, running_batch, num_accepted_tokens))


def test_patch_event_loop_supports_current_sglang_metrics_api():
    """AWEX accepts report_decode_stats without the removed legacy methods."""
    scheduler = _SchedulerWithCurrentMetricsAPI()

    AwexSchedulerPlugin(scheduler)._patch_event_loop()
    scheduler.report_decode_stats(
        True,
        running_batch=SimpleNamespace(),
        num_accepted_tokens=3,
    )

    assert scheduler._areal_awex_last_decode_stats_ct == 7
    assert scheduler.calls[0][0] is True
    assert scheduler.calls[0][2] == 3


def test_awex_memory_transitions_are_idempotent_across_retries():
    class _Scheduler:
        def __init__(self) -> None:
            self.offload_tags = set()
            self.calls = []

        def release_memory_occupation(self, request):
            self.calls.append(("release", list(request.tags)))
            self.offload_tags.update(request.tags)

        def resume_memory_occupation(self, request):
            self.calls.append(("resume", list(request.tags)))
            for tag in request.tags:
                self.offload_tags.remove(tag)

    scheduler = _Scheduler()
    plugin = AwexSchedulerPlugin(scheduler)
    plugin._patch_memory_transitions()
    request = SimpleNamespace(tags=["kv_cache"])

    scheduler.release_memory_occupation(request)
    scheduler.release_memory_occupation(request)
    scheduler.resume_memory_occupation(request)
    scheduler.resume_memory_occupation(request)

    assert scheduler.calls == [
        ("release", ["kv_cache"]),
        ("resume", ["kv_cache"]),
    ]
    assert scheduler.offload_tags == set()


def test_awex_converter_canonicalizes_qwen3_attention_names():
    """Qwen3 infer metadata matches AWEX's MCore canonical attention names."""
    from awex.converter.sglang_converter import SGlangToHFWeightConverter

    _patch_awex_qwen3_attention_names()
    converter = object.__new__(SGlangToHFWeightConverter)
    parameter = torch.ones(128, dtype=torch.bfloat16)

    expected = {
        "self_attn.q_norm.weight": "attention.query_layernorm.weight",
        "self_attn.k_norm.weight": "attention.key_layernorm.weight",
    }
    for name, canonical_name in expected.items():
        converted = converter._convert_layer_norm_param(name, parameter, "0")
        assert converted == [(canonical_name, parameter)]

    expected = {
        "self_attn.qkv_proj.weight": "attention.query_key_value_proj.weight",
        "self_attn.o_proj.weight": "attention.dense.weight",
    }
    for name, canonical_name in expected.items():
        converted = converter._convert_attention_param(name, parameter, "0")
        assert converted == [(canonical_name, parameter)]


@pytest.mark.parametrize(
    ("tie_word_embeddings", "include_lm_head", "expected_lm_head_count"),
    [
        (True, False, 1),
        (True, True, 1),
        (False, False, 0),
    ],
)
def test_colocate_reader_metadata_handles_tied_sglang_lm_head(
    monkeypatch,
    tie_word_embeddings,
    include_lm_head,
    expected_lm_head_count,
):
    """SGLang metadata aliases only a missing lm_head for tied Qwen3."""
    model = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=tie_word_embeddings)
    )
    reader = object.__new__(AwexColocateReader)
    reader._scheduler = SimpleNamespace(server_args=SimpleNamespace())
    reader._engine_rank = 0
    reader._get_model = lambda: model
    reader._build_model_context = lambda: {"pp_rank": 0, "pp_size": 1}
    params_meta = [
        {
            "name": "model.embed_tokens.weight",
            "numel": 32,
            "shape": (8, 4),
            "dtype": "bfloat16",
        },
        {
            "name": "model.layers.0.input_layernorm.weight",
            "numel": 4,
            "shape": (4,),
            "dtype": "bfloat16",
        },
    ]
    if include_lm_head:
        params_meta.append(
            {
                "name": "lm_head.weight",
                "numel": 64,
                "shape": (16, 4),
                "dtype": "float32",
            }
        )
    raw_meta = {"params_meta": params_meta}
    monkeypatch.setattr(
        awex_colocate_reader.InferParamMetaResolver,
        "_get_model_param_info",
        lambda *args, **kwargs: raw_meta,
    )

    result = reader._compute_local_raw_meta()

    lm_head_meta = [
        item for item in result["params_meta"] if item["name"] == "lm_head.weight"
    ]
    assert len(lm_head_meta) == expected_lm_head_count
    if tie_word_embeddings and not include_lm_head:
        assert lm_head_meta[0] == {
            "name": "lm_head.weight",
            "numel": 32,
            "shape": (8, 4),
            "dtype": "bfloat16",
        }
    elif include_lm_head:
        assert lm_head_meta[0]["numel"] == 64


def test_resolve_transfer_rank_uses_global_rank_for_isolated_gpu(monkeypatch):
    """One-GPU SGLang processes keep distinct AWEX NCCL identities."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("LOCAL_RANK", "0")
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
    """Both TP and PP schedulers make inherited server ranks unsafe."""
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(tp_size=tp_size, pp_size=pp_size)
    )

    assert _scheduler_instance_world_size(scheduler) == 4


def test_resolve_transfer_rank_uses_scheduler_gpu_for_multi_gpu_server(monkeypatch):
    """TP schedulers sharing one inherited rank keep distinct AWEX identities."""
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("WORLD_SIZE", "32")

    transfer_ranks = [
        _resolve_transfer_rank(
            infer_world_size=32,
            gpu_id=gpu_id,
            node_id=2,
            nnodes=4,
            instance_world_size=4,
        )
        for gpu_id in range(4)
    ]

    assert transfer_ranks == [16, 17, 18, 19]


def test_resolve_transfer_rank_falls_back_to_node_local_identity(monkeypatch):
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


def test_resolve_physical_gpu_id_ignores_isolated_cuda_device_zero():
    """Global rank 7 pairs with physical GPU 7 despite local CUDA device 0."""
    assert (
        _resolve_physical_gpu_id(
            transfer_rank=7,
            infer_world_size=8,
            nnodes=1,
        )
        == 7
    )


def test_resolve_physical_gpu_id_is_node_local_for_multinode():
    assert (
        _resolve_physical_gpu_id(
            transfer_rank=11,
            infer_world_size=16,
            nnodes=2,
        )
        == 3
    )


def test_writer_version_key_uses_physical_gpu_not_local_cuda_device():
    assert _writer_version_key("10.0.0.1", 7) == ("awex_writer_version_10.0.0.1_7")


def test_colocate_writer_wait_falls_back_for_legacy_awex(monkeypatch):
    consumed_keys = []
    client = SimpleNamespace(
        get_object_then_delete=lambda key: consumed_keys.append(key),
    )
    monkeypatch.setattr(
        awex_colocate_reader,
        "_awex_wait_colocate_write_finished",
        None,
    )

    awex_colocate_reader._wait_colocate_write_finished(
        client,
        "write_finished_host_7_3",
        weights_update_group=object(),
        transfer_rank=7,
    )

    assert consumed_keys == ["write_finished_host_7_3"]


def test_colocate_writer_wait_uses_new_awex_helper(monkeypatch):
    calls = []

    def helper(*args):
        calls.append(args)

    client = object()
    group = object()
    monkeypatch.setattr(
        awex_colocate_reader,
        "_awex_wait_colocate_write_finished",
        helper,
    )

    awex_colocate_reader._wait_colocate_write_finished(
        client,
        "write_finished_host_7_3",
        weights_update_group=group,
        transfer_rank=7,
    )

    assert calls == [(client, "write_finished_host_7_3", group, 7)]


class _FakeMetaServerClient:
    def __init__(self) -> None:
        self.keys = []

    def get_object(self, key, timeout):
        self.keys.append(key)
        return 8, SimpleNamespace(), ("serialized",)


def test_colocate_reader_uses_physical_gpu_for_ipc_key(monkeypatch):
    """An isolated CUDA device 0 reads the paired physical-GPU writer key."""
    reader = object.__new__(_BailingV3PhysicalKeyNCCLWorkerWeightsReader)
    reader._areal_physical_gpu_id = 7
    reader.enable_colocate_mode = True
    reader.meta_server_client = _FakeMetaServerClient()
    reader.timeout = 10
    reader.ipc_backend = "cuda"
    reader.rank_coordinate = "7-0-7"

    monkeypatch.setattr("awex.util.common.get_ip_address", lambda: "10.0.0.1")
    monkeypatch.setattr(
        "awex.util.tensor_util.cuda_ipc_deserialize",
        lambda serialized: ([torch.ones(1)], [], []),
    )
    monkeypatch.setattr(
        "awex.util.tensor_util.reconstruct_tensors_from_groups",
        lambda groups, metadata: [],
    )
    monkeypatch.setattr("awex.util.device.current_device", lambda: 0)
    monkeypatch.setattr("awex.util.device.synchronize", lambda device_id: None)
    monkeypatch.setattr("awex.util.gpu.get_gpu_status", lambda: "ok")
    monkeypatch.setattr("awex.util.system_util.count_open_fds", lambda: 0)

    reader.collect_training_weights(step_id=3)

    assert reader.meta_server_client.keys == [
        "training_serialized_weights_10.0.0.1_7_3"
    ]


def test_colocate_reader_constructs_physical_key_reader(monkeypatch):
    """The production adapter must preserve physical GPU ids for AWEX keys."""
    captured = {}

    class _FakePhysicalKeyReader:
        transfer_rank = 5

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def initialize(self):
            captured["initialized"] = True

    monkeypatch.setattr(
        awex_colocate_reader,
        "_BailingV3PhysicalKeyNCCLWorkerWeightsReader",
        _FakePhysicalKeyReader,
    )

    reader = object.__new__(AwexColocateReader)
    reader._reader = None
    reader._meta_server_client = SimpleNamespace(
        get_object=lambda key, timeout: {"weight": object()}
    )
    reader._build_model_context = lambda: object()
    reader._get_model = lambda: object()
    reader._infer_conf = {"infer_world_size": 8}
    reader._engine_rank = 5
    reader._num_infer_engines = 8
    reader._meta_server_addr = "node-a:1234"
    reader._infer_params_meta = {"weight": object()}
    reader._local_gpu_id = 5

    native_reader = reader._ensure_reader()

    assert isinstance(native_reader, _FakePhysicalKeyReader)
    assert captured["physical_gpu_id"] == 5
    assert captured["initialized"] is True


def test_colocate_reader_updates_weights_with_grad_disabled():
    """AWEX copies and derived-weight rebuilds run in the same no-grad mode."""
    grad_modes = []

    class _FakeNativeReader:
        def update_weights(self, step_id):
            grad_modes.append(("native", step_id, torch.is_grad_enabled()))
            self._areal_pre_ack_callback(step_id)
            list(self._areal_pre_ack_named_tensors_factory())

    reader = object.__new__(AwexColocateReader)
    reader._initialized = True
    reader._ensure_reader = lambda: _FakeNativeReader()
    reader._pre_process_model_weights = lambda: grad_modes.append(
        ("prepare", None, torch.is_grad_enabled())
    )
    reader._rebuild_derived_weights = lambda: grad_modes.append(
        ("derived", None, torch.is_grad_enabled())
    )
    reader._iter_model_parts = lambda: []

    with torch.enable_grad():
        reader.update_weights(version=3)
        caller_grad_enabled = torch.is_grad_enabled()

    assert grad_modes == [
        ("prepare", None, False),
        ("native", 3, False),
        ("derived", None, False),
    ]
    assert caller_grad_enabled is True


def test_colocate_reader_runs_model_weight_hooks(monkeypatch):
    calls = []

    class _Model:
        def pre_process_weights_if_quant(self):
            calls.append(("pre", None))

        def post_load_weights(self, is_nextn=False, weight_names="unexpected"):
            calls.append(("post_load", (is_nextn, weight_names)))

        def post_process_weights_if_quant(self):
            calls.append(("post", None))

    reader = object.__new__(AwexColocateReader)
    reader._get_model = lambda: [_Model(), _Model()]
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append(("sync", None)))

    reader._pre_process_model_weights()
    reader._rebuild_derived_weights()

    assert calls == [
        ("pre", None),
        ("pre", None),
        ("post_load", (False, None)),
        ("post", None),
        ("post_load", (True, None)),
        ("post", None),
        ("sync", None),
    ]


def test_native_reader_initializes_bounded_transport(monkeypatch):
    """The production reader wires the bounded-memory transport implementation."""
    from awex.transfer import transfer_plan

    class _MetaServerClient:
        def add_object_to_set(self, key, value):
            return None

        def wait_set_until_size(self, key, size, timeout):
            return None

        def get_set(self, key):
            return {("node-a", 0, 0)}

    class _PlanBuilder:
        def __init__(self, *args):
            pass

        def build_local_transfer_plan(self, *args):
            return SimpleNamespace(operations={})

    monkeypatch.setattr(transfer_plan, "TransferPlanBuilder", _PlanBuilder)
    monkeypatch.setattr(
        awex_colocate_reader.NcclColocateStreamBatchTransport,
        "__init__",
        lambda self, *args: None,
    )
    monkeypatch.setattr("awex.util.common.get_ip_address", lambda: "node-a")
    monkeypatch.setattr("awex.util.device.current_device", lambda: 0)

    reader = object.__new__(_BailingV3PhysicalKeyNCCLWorkerWeightsReader)
    reader._areal_physical_gpu_id = 0
    reader.transfer_rank = 0
    reader.infer_world_size = 1
    reader.training_world_size = 1
    reader.num_engines = 1
    reader.enable_debug_mode = False
    reader.parameters_meta = {}
    reader.training_params_meta = {}
    reader.meta_server_client = _MetaServerClient()
    reader.timeout = 1

    reader._init_reader_in_colocate_mode()

    assert isinstance(
        reader.colocate_transport,
        _BoundedMemoryNcclColocateStreamBatchTransport,
    )


def test_bounded_transport_defers_send_clones_until_execution(monkeypatch):
    """Building an AWEX transfer plan retains views instead of model-sized clones."""
    from awex.transfer import nccl_stream_batch
    from awex.util import device as device_util

    class _SourceTensor:
        def __init__(self) -> None:
            self.clone_calls = 0

        def clone(self):
            self.clone_calls += 1
            return self

    source = _SourceTensor()
    send_op = SimpleNamespace(
        send_shard_meta=SimpleNamespace(name="weight"),
        recv_rank=1,
    )
    send_plan = SimpleNamespace(operations={1: [send_op]})
    recv_plan = SimpleNamespace(operations={})
    transport = object.__new__(_BoundedMemoryNcclColocateStreamBatchTransport)

    def _inspect_plan(
        transfer_rank,
        world_size,
        all_send_p2p_ops,
        all_recv_p2p_ops,
        weights_update_group,
        rank_coordinate,
        step_id,
    ) -> None:
        del (
            transfer_rank,
            world_size,
            all_recv_p2p_ops,
            weights_update_group,
            rank_coordinate,
            step_id,
        )
        assert all_send_p2p_ops[1][0][1].tensor is source
        assert source.clone_calls == 0

    transport.execute_recursive_partition_stream_transfer = _inspect_plan
    monkeypatch.setattr(
        nccl_stream_batch,
        "hang_detector",
        SimpleNamespace(submit=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        "awex.transfer.nccl_comm.validate_rank_mappings", lambda *args: None
    )
    monkeypatch.setattr(
        "awex.transfer.transfer_plan.slice_tensor",
        lambda tensor, *args, **kwargs: tensor,
    )
    monkeypatch.setattr(device_util, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.distributed,
        "P2POp",
        lambda op, tensor, peer, group: SimpleNamespace(
            op=op, tensor=tensor, peer=peer, group=group
        ),
    )

    transport.update_weights_in_colocate_mode(
        train_to_infer_device_mapping={0: 0, 1: 1},
        infer_to_train_device_mapping={0: 0, 1: 1},
        transfer_rank=0,
        rank_coordinate="0-0-0",
        world_size=2,
        send_transfer_plan=send_plan,
        recv_transfer_plan=recv_plan,
        weights_update_group=object(),
        send_parameters={"weight": source},
        recv_parameters={},
        step_id=1,
    )

    assert source.clone_calls == 0


def test_bounded_transport_releases_each_send_clone_batch(monkeypatch):
    """Only one send tensor per active peer remains live during P2P execution."""
    from awex.util import device as device_util

    counters = {"live": 0, "max_live": 0, "clones": 0, "syncs": 0}

    class _Clone:
        def __init__(self) -> None:
            counters["live"] += 1
            counters["max_live"] = max(counters["max_live"], counters["live"])

        def __del__(self) -> None:
            counters["live"] -= 1

    class _SourceTensor:
        def clone(self):
            counters["clones"] += 1
            return _Clone()

    class _Work:
        def __init__(self, tensor) -> None:
            self.tensor = tensor

        def wait(self) -> None:
            self.tensor = None

    def _isend(tensor, peer, group):
        del peer, group
        return _Work(tensor)

    monkeypatch.setattr(torch.distributed, "isend", _isend)
    monkeypatch.setattr(device_util, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        device_util,
        "synchronize",
        lambda: counters.__setitem__("syncs", counters["syncs"] + 1),
    )

    transport = object.__new__(_BoundedMemoryNcclColocateStreamBatchTransport)
    transport._stream_pool = [object(), object()]
    ops = {
        peer: [
            (
                SimpleNamespace(recv_shard_meta=SimpleNamespace(dtype=None)),
                SimpleNamespace(
                    op=_isend,
                    tensor=_SourceTensor(),
                    peer=peer,
                    group=object(),
                ),
            )
            for _ in range(3)
        ]
        for peer in (1, 2)
    }

    count = transport._execute_ops_concurrent(ops, range(1, 3))

    assert count == 6
    assert counters == {
        "live": 0,
        "max_live": 2,
        "clones": 6,
        "syncs": 3,
    }


def test_bounded_transport_casts_send_to_receiver_dtype(monkeypatch):
    """P2P sends use the receiver dtype so NCCL wire sizes match."""
    from awex.util import device as device_util

    sent = []

    class _Work:
        def wait(self) -> None:
            return None

    def _isend(tensor, peer, group):
        del peer, group
        sent.append(tensor)
        return _Work()

    source = torch.ones(2, dtype=torch.bfloat16)
    plan_op = SimpleNamespace(
        recv_shard_meta=SimpleNamespace(dtype=torch.float32),
    )
    p2p_op = SimpleNamespace(
        op=_isend,
        tensor=source,
        peer=1,
        group=object(),
    )

    monkeypatch.setattr(torch.distributed, "isend", _isend)
    monkeypatch.setattr(device_util, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(device_util, "synchronize", lambda: None)

    transport = object.__new__(_BoundedMemoryNcclColocateStreamBatchTransport)
    transport._stream_pool = [object()]

    count = transport._execute_ops_concurrent(
        {1: [(plan_op, p2p_op)]},
        range(1, 2),
    )

    assert count == 1
    assert len(sent) == 1
    assert sent[0].dtype == torch.float32


def test_bounded_transport_prepares_send_on_transfer_stream(monkeypatch):
    """Send clones are ordered on the same stream as their NCCL operation."""
    from awex.util import device as device_util

    state = {"active_stream": None}
    transfer_stream = object()

    class _StreamContext:
        def __enter__(self):
            state["active_stream"] = transfer_stream

        def __exit__(self, exc_type, exc_value, traceback):
            state["active_stream"] = None

    class _SourceTensor:
        dtype = torch.bfloat16

        def clone(self):
            assert state["active_stream"] is transfer_stream
            return self

    class _Work:
        def wait(self) -> None:
            return None

    def _isend(tensor, peer, group):
        del tensor, peer, group
        assert state["active_stream"] is transfer_stream
        return _Work()

    monkeypatch.setattr(torch.distributed, "isend", _isend)
    monkeypatch.setattr(device_util, "stream", lambda stream: _StreamContext())
    monkeypatch.setattr(device_util, "synchronize", lambda: None)

    transport = object.__new__(_BoundedMemoryNcclColocateStreamBatchTransport)
    transport._stream_pool = [transfer_stream]
    plan_op = SimpleNamespace(
        recv_shard_meta=SimpleNamespace(dtype=torch.bfloat16),
    )
    p2p_op = SimpleNamespace(
        op=_isend,
        tensor=_SourceTensor(),
        peer=1,
        group=object(),
    )

    count = transport._execute_ops_concurrent(
        {1: [(plan_op, p2p_op)]},
        range(1, 2),
    )

    assert count == 1
    assert state["active_stream"] is None


def test_native_reader_reports_pre_ack_finite_failure_to_writer(monkeypatch):
    """A bad received tensor wakes the paired writer before IPC release."""
    events = []

    class _MetaServerClient:
        def put_object(self, key, value) -> None:
            events.append(("put", key, value))

    class _Transport:
        def update_weights_in_colocate_mode(self, *args, **kwargs) -> None:
            events.append(("transfer", kwargs["step_id"]))

    reader = object.__new__(_BailingV3PhysicalKeyNCCLWorkerWeightsReader)
    reader.enable_colocate_mode = True
    reader.collect_training_weights = lambda step_id, **kwargs: None
    reader.colocate_transport = _Transport()
    reader.train_to_infer_device_mapping = {}
    reader.infer_to_train_device_mapping = {}
    reader.transfer_rank = 6
    reader.rank_coordinate = "0-0-6"
    reader.infer_world_size = 8
    reader.send_transfer_plan = SimpleNamespace()
    reader.transfer_plan = SimpleNamespace(operations={})
    reader.send_ranks_sample = []
    reader.weights_update_group = object()
    reader.deserialized_weights = {}
    reader.parameters = {"model.layers.0.input_layernorm.weight": torch.ones(2)}
    reader._areal_physical_gpu_id = 6
    reader.meta_server_client = _MetaServerClient()

    monkeypatch.setattr(
        "awex.util.common.get_ip_address",
        lambda: "node-a",
    )

    def _reject(*args, **kwargs):
        events.append(("finite_check", kwargs["stage"]))
        if kwargs["stage"] == "awex_reader_pre_ack":
            raise FloatingPointError("non-finite layernorm")

    monkeypatch.setattr(awex_colocate_reader, "check_named_tensors_finite", _reject)

    with pytest.raises(FloatingPointError, match="non-finite layernorm"):
        reader._update_weights_in_colocate_mode(step_id=5)

    assert events[:3] == [
        ("finite_check", "awex_reader_ipc_imported"),
        ("transfer", 5),
        ("finite_check", "awex_reader_pre_ack"),
    ]
    assert events[3] == (
        "put",
        "weights_update_finished_node-a_6_5",
        {
            "ok": False,
            "error": "FloatingPointError: non-finite layernorm",
        },
    )


def test_native_reader_reports_derived_rebuild_failure_before_success_ack(monkeypatch):
    """A derived-weight failure is returned to the writer before IPC release."""
    events = []

    reader = object.__new__(_BailingV3PhysicalKeyNCCLWorkerWeightsReader)
    reader.enable_colocate_mode = True
    reader.collect_training_weights = lambda step_id, **kwargs: None
    reader.colocate_transport = SimpleNamespace(
        update_weights_in_colocate_mode=lambda *args, **kwargs: events.append(
            ("transfer", kwargs["step_id"])
        )
    )
    reader.train_to_infer_device_mapping = {}
    reader.infer_to_train_device_mapping = {}
    reader.transfer_rank = 0
    reader.rank_coordinate = "0-0-0"
    reader.infer_world_size = 1
    reader.send_transfer_plan = SimpleNamespace()
    reader.transfer_plan = SimpleNamespace(operations={})
    reader.send_ranks_sample = []
    reader.weights_update_group = object()
    reader.deserialized_weights = {}
    reader.parameters = {"weight": torch.ones(1)}
    reader._areal_physical_gpu_id = 0
    reader._areal_pre_ack_callback = lambda step_id: (_ for _ in ()).throw(
        RuntimeError("derived rebuild failed")
    )
    reader.meta_server_client = SimpleNamespace(
        put_object=lambda key, value: events.append(("put", key, value))
    )

    monkeypatch.setattr("awex.util.common.get_ip_address", lambda: "node-a")
    monkeypatch.setattr(
        awex_colocate_reader,
        "check_named_tensors_finite",
        lambda *args, **kwargs: events.append(("finite", kwargs["stage"])),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="derived rebuild failed"):
        reader._update_weights_in_colocate_mode(step_id=7)

    completions = [event for event in events if event[0] == "put"]
    assert len(completions) == 1
    assert completions[0][2]["ok"] is False
    assert "derived rebuild failed" in completions[0][2]["error"]


def test_pre_ack_callback_rejects_remote_rank_failure(monkeypatch):
    """Every reader rank rejects a version when one rank fails rebuild."""
    reader = object.__new__(_BailingV3PhysicalKeyNCCLWorkerWeightsReader)
    reader.weights_update_group = object()
    reader._areal_pre_ack_callback = lambda step_id: None

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "gloo")

    def _remote_failure(tensor, op, group):
        tensor.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", _remote_failure)

    with pytest.raises(RuntimeError, match="another rank"):
        reader._run_pre_ack_callback(step_id=3)
