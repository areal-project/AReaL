# SPDX-License-Identifier: Apache-2.0

import threading
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from areal.v2.weight_update import nccl_group
from areal.v2.weight_update.awex import fsdp_adapter, megatron_adapter, sglang_adapter
from areal.v2.weight_update.awex.deadline import TransferDeadline
from areal.v2.weight_update.awex.fsdp_adapter import AwexFSDPAdapter


def test_setup_batch_isend_irecv_uses_sidecar_for_final_barrier(monkeypatch):
    """The liveness payload uses NCCL while its final barrier uses the sidecar."""
    process_group = MagicMock(name="nccl_group")
    barrier_group = MagicMock(name="gloo_group")
    barrier = MagicMock()

    monkeypatch.setattr(nccl_group.current_platform, "current_device", lambda: 0)
    monkeypatch.setattr(nccl_group.current_platform, "device_type", "cpu")
    monkeypatch.setattr(nccl_group.current_platform, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "full", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(nccl_group.dist, "barrier", barrier)

    nccl_group.setup_batch_isend_irecv(
        process_group,
        rank=0,
        world_size=1,
        barrier_group=barrier_group,
    )

    barrier.assert_called_once_with(group=barrier_group)


def test_setup_batch_isend_irecv_defaults_to_payload_group(monkeypatch):
    """Callers without a sidecar retain the existing payload-group barrier."""
    process_group = MagicMock(name="nccl_group")
    barrier = MagicMock()

    monkeypatch.setattr(nccl_group.current_platform, "current_device", lambda: 0)
    monkeypatch.setattr(nccl_group.current_platform, "device_type", "cpu")
    monkeypatch.setattr(nccl_group.current_platform, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "full", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(nccl_group.dist, "barrier", barrier)

    nccl_group.setup_batch_isend_irecv(
        process_group,
        rank=0,
        world_size=1,
    )

    barrier.assert_called_once_with(group=process_group, device_ids=[0])


def test_megatron_adapter_initializes_nccl_and_gloo_groups(monkeypatch):
    """The training adapter creates matching payload and sidecar groups."""
    adapter = megatron_adapter.AwexMegatronAdapter(MagicMock())
    builder = MagicMock()
    builder.build_local_transfer_plan.return_value = MagicMock()
    initializer = MagicMock(side_effect=[MagicMock(), MagicMock()])

    monkeypatch.setattr(
        megatron_adapter, "fetch_kv_metadata", lambda *args: (MagicMock(), MagicMock())
    )
    monkeypatch.setattr(
        megatron_adapter, "TransferPlanBuilder", lambda **kwargs: builder
    )
    monkeypatch.setattr(megatron_adapter, "init_weights_update_group", initializer)

    adapter.init_weight_update_group(
        pair_name="actor-rollout",
        master_addr="127.0.0.1",
        master_port=29500,
        transfer_rank=2,
        world_size=4,
        kv_store_url="http://kv-store",
        infer_world_size=2,
        train_world_size=2,
        num_engines=1,
    )

    assert initializer.call_args_list == [
        call(
            master_address="127.0.0.1",
            master_port=29500,
            rank=2,
            world_size=4,
            group_name="awex_actor-rollout",
            role="training",
        ),
        call(
            master_address="127.0.0.1",
            master_port=29500,
            rank=2,
            world_size=4,
            group_name="awex_actor-rollout_gloo",
            backend="gloo",
            role="training",
        ),
    ]


def test_fsdp_adapter_initializes_nccl_and_gloo_groups(monkeypatch):
    """The FSDP training adapter creates matching payload and sidecar groups."""
    adapter = fsdp_adapter.AwexFSDPAdapter(MagicMock())
    builder = MagicMock()
    builder.build_local_transfer_plan.return_value = MagicMock()
    initializer = MagicMock(side_effect=[MagicMock(), MagicMock()])

    monkeypatch.setattr(
        fsdp_adapter, "fetch_kv_metadata", lambda *args: (MagicMock(), MagicMock())
    )
    monkeypatch.setattr(fsdp_adapter, "TransferPlanBuilder", lambda **kwargs: builder)
    monkeypatch.setattr(fsdp_adapter, "init_weights_update_group", initializer)

    adapter.init_weight_update_group(
        pair_name="actor-rollout",
        master_addr="127.0.0.1",
        master_port=29500,
        transfer_rank=2,
        world_size=4,
        kv_store_url="http://kv-store",
        infer_world_size=2,
        train_world_size=2,
        num_engines=1,
    )

    assert initializer.call_args_list == [
        call(
            master_address="127.0.0.1",
            master_port=29500,
            rank=2,
            world_size=4,
            group_name="awex_actor-rollout",
            role="training",
        ),
        call(
            master_address="127.0.0.1",
            master_port=29500,
            rank=2,
            world_size=4,
            group_name="awex_actor-rollout_gloo",
            backend="gloo",
            role="training",
        ),
    ]


def test_sglang_adapter_initializes_nccl_and_gloo_groups(monkeypatch):
    """The inference adapter creates matching payload and sidecar groups."""
    adapter = sglang_adapter.AwexSGLangAdapter(MagicMock())
    monkeypatch.setattr(
        sglang_adapter, "current_platform", SimpleNamespace(device_type="cpu")
    )
    builder = MagicMock()
    builder.build_local_transfer_plan.return_value = MagicMock()
    initializer = MagicMock(side_effect=[MagicMock(), MagicMock()])

    monkeypatch.setattr(
        adapter,
        "_get_model_context",
        lambda: {"tp_size": 1, "tp_rank": 0, "pp_size": 1, "pp_rank": 0},
    )
    monkeypatch.setattr(
        sglang_adapter, "fetch_kv_metadata", lambda *args: (MagicMock(), MagicMock())
    )
    monkeypatch.setattr(sglang_adapter, "TransferPlanBuilder", lambda **kwargs: builder)
    monkeypatch.setattr(sglang_adapter, "init_weights_update_group", initializer)

    adapter.init_weight_update_group(
        pair_name="actor-rollout",
        master_addr="127.0.0.1",
        master_port=29500,
        transfer_rank=1,
        world_size=4,
        kv_store_url="http://kv-store",
        infer_world_size=2,
        train_world_size=2,
        num_engines=2,
    )

    assert initializer.call_args_list == [
        call(
            master_address="127.0.0.1",
            master_port=29500,
            rank=1,
            world_size=4,
            group_name="awex_actor-rollout",
            role="inference",
            timeout=timedelta(seconds=150),
        ),
        call(
            master_address="127.0.0.1",
            master_port=29500,
            rank=1,
            world_size=4,
            group_name="awex_actor-rollout_gloo",
            backend="gloo",
            role="inference",
            timeout=timedelta(seconds=150),
        ),
    ]


@pytest.mark.parametrize(
    ("adapter_cls", "module"),
    [
        (fsdp_adapter.AwexFSDPAdapter, fsdp_adapter),
        (megatron_adapter.AwexMegatronAdapter, megatron_adapter),
        (sglang_adapter.AwexSGLangAdapter, sglang_adapter),
    ],
)
def test_awex_adapters_use_sidecar_for_setup_barrier(adapter_cls, module, monkeypatch):
    """Both AWEX adapters retain NCCL payloads and select the Gloo barrier."""
    adapter = adapter_cls(MagicMock())
    payload_group = MagicMock(name="nccl_group")
    sidecar_group = MagicMock(name="gloo_group")
    adapter._weights_update_group = payload_group
    adapter._weights_update_group_gloo = sidecar_group
    adapter._transfer_rank = 3
    setup = MagicMock()
    monkeypatch.setattr(module, "setup_batch_isend_irecv", setup)

    adapter.batch_isend_irecv(world_size=4)

    setup.assert_called_once_with(
        payload_group,
        3,
        4,
        barrier_group=sidecar_group,
    )


@pytest.mark.parametrize(
    ("adapter_cls", "module", "build_ops_name"),
    [
        (
            fsdp_adapter.AwexFSDPAdapter,
            fsdp_adapter,
            "nccl_build_send_ops",
        ),
        (
            megatron_adapter.AwexMegatronAdapter,
            megatron_adapter,
            "nccl_build_send_ops",
        ),
        (
            sglang_adapter.AwexSGLangAdapter,
            sglang_adapter,
            "nccl_build_recv_ops",
        ),
    ],
)
def test_awex_adapters_use_sidecar_for_completion_barrier(
    adapter_cls, module, build_ops_name, monkeypatch
):
    """Payload ops stay on NCCL while the completion barrier uses Gloo."""
    adapter = adapter_cls(MagicMock())
    payload_group = MagicMock(name="nccl_group")
    sidecar_group = MagicMock(name="gloo_group")
    adapter._transfer_plan = MagicMock()
    adapter._weights_update_group = payload_group
    adapter._weights_update_group_gloo = sidecar_group
    adapter._transfer_rank = 0
    adapter.get_local_shard_parameters = MagicMock(return_value={})
    monkeypatch.setattr(module, build_ops_name, lambda *args, **kwargs: ([], [], None))
    monkeypatch.setattr(module, "batch_send_recv", MagicMock())
    barrier = MagicMock()
    distributed = getattr(module, "dist", module.torch.distributed)
    monkeypatch.setattr(distributed, "barrier", barrier)

    adapter.execute_weight_update(version=1)

    if module is sglang_adapter and module.current_platform.device_type == "cuda":
        barrier.assert_called_once_with(group=sidecar_group, async_op=True)
        assert barrier.return_value.wait.call_args.kwargs["timeout"].total_seconds() > 0
    else:
        barrier.assert_called_once_with(group=sidecar_group)


def test_awex_transfer_deadline_never_aborts_the_model_default_group():
    with pytest.raises(ValueError, match="dedicated AWEX"):
        TransferDeadline(dist.group.WORLD, 1.0)


def test_awex_transfer_success_cancels_and_joins_old_watchdog(monkeypatch):
    abort = MagicMock()
    monkeypatch.setattr(
        torch.distributed.distributed_c10d, "_abort_process_group", abort
    )
    first, second = object(), object()
    for group in (first, second):
        deadline = TransferDeadline(group, 1.0)
        with deadline:
            deadline.check()
        assert deadline.stopped and not deadline.aborted
    abort.assert_not_called()


def test_awex_abort_requires_failure_even_if_transfer_returns_normally(monkeypatch):
    aborted = threading.Event()
    payload, sidecar = object(), object()
    abort = MagicMock(side_effect=lambda group: aborted.set())
    monkeypatch.setattr(
        torch.distributed.distributed_c10d, "_abort_process_group", abort
    )
    adapter = sglang_adapter.AwexSGLangAdapter(MagicMock())
    adapter._weights_update_group = payload
    adapter._weights_update_group_gloo = sidecar
    adapter._transfer_plan = object()
    adapter._transfer_timeout_s = 0.01
    adapter.get_local_shard_parameters = lambda: {}
    original, contiguous = MagicMock(), MagicMock()
    monkeypatch.setattr(
        sglang_adapter,
        "current_platform",
        SimpleNamespace(device_type="cuda", synchronize=MagicMock()),
    )
    monkeypatch.setattr(
        sglang_adapter,
        "nccl_build_recv_ops",
        lambda *args: ([], [(original, contiguous)], None),
    )

    def transfer(**kwargs):
        assert aborted.wait(5), "Deadline watchdog did not run"

    monkeypatch.setattr(sglang_adapter, "batch_send_recv", transfer)
    barrier = MagicMock()
    monkeypatch.setattr(sglang_adapter.dist, "barrier", barrier)
    with pytest.raises(TimeoutError, match="deadline exceeded"):
        adapter.execute_weight_update(3)
    abort.assert_called_once_with(payload)
    original.copy_.assert_not_called()
    barrier.assert_not_called()
    assert adapter._weights_update_group is None
    assert adapter._weights_update_group_gloo is sidecar
    assert adapter._transfer_deadline is None


def test_awex_failed_abort_retains_group_and_blocks_teardown(monkeypatch):
    payload = object()
    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_abort_process_group",
        MagicMock(side_effect=RuntimeError("abort failed")),
    )
    monkeypatch.setattr(
        sglang_adapter, "current_platform", SimpleNamespace(device_type="cuda")
    )
    adapter = sglang_adapter.AwexSGLangAdapter(MagicMock())
    adapter._weights_update_group = payload
    with pytest.raises(RuntimeError, match="abort failed"):
        with adapter._bound_transfer():
            raise RuntimeError("peer died")
    assert adapter._weights_update_group is payload
    assert adapter._transfer_deadline is not None
    destroy = MagicMock()
    monkeypatch.setattr(dist, "destroy_process_group", destroy)
    with pytest.raises(RuntimeError, match="abort failed"):
        adapter.teardown_weight_update_group()
    destroy.assert_not_called()


def test_awex_context_exit_rejects_success_after_abort_has_started(monkeypatch):
    started, finish = threading.Event(), threading.Event()

    def abort(group):
        started.set()
        assert finish.wait(5)

    monkeypatch.setattr(
        torch.distributed.distributed_c10d, "_abort_process_group", abort
    )
    deadline = TransferDeadline(object(), 0.01)
    with pytest.raises(TimeoutError, match="deadline exceeded"):
        with deadline:
            assert started.wait(5)
            finish.set()
            # No transfer exception: final exit must still reject this operation.
    assert deadline.aborted and deadline.stopped


def test_awex_unfinished_gloo_confirmation_aborts_only_payload_group(monkeypatch):
    abort, destroy, barrier = MagicMock(), MagicMock(), MagicMock()
    payload, sidecar = object(), object()
    barrier.return_value.wait.return_value = False
    monkeypatch.setattr(
        torch.distributed.distributed_c10d, "_abort_process_group", abort
    )
    monkeypatch.setattr(
        sglang_adapter, "current_platform", SimpleNamespace(device_type="cuda")
    )
    monkeypatch.setattr(dist, "barrier", barrier)
    monkeypatch.setattr(dist, "destroy_process_group", destroy)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    adapter = sglang_adapter.AwexSGLangAdapter(MagicMock())
    adapter._weights_update_group = payload
    adapter._weights_update_group_gloo = sidecar
    with pytest.raises(RuntimeError, match="barrier did not finish"):
        with adapter._bound_transfer():
            adapter._complete_transfer()
    abort.assert_called_once_with(payload)
    assert adapter._weights_update_group is None
    adapter.teardown_weight_update_group()
    destroy.assert_called_once_with(sidecar)


def test_awex_live_abort_thread_prevents_teardown_and_new_group(monkeypatch):
    started, finish = threading.Event(), threading.Event()
    payload = object()

    def abort(group):
        started.set()
        assert finish.wait(5)

    monkeypatch.setattr(
        torch.distributed.distributed_c10d, "_abort_process_group", abort
    )
    monkeypatch.setattr(
        sglang_adapter, "current_platform", SimpleNamespace(device_type="cuda")
    )
    adapter = sglang_adapter.AwexSGLangAdapter(MagicMock())
    adapter._weights_update_group = payload
    real_join = None
    try:
        with pytest.raises(RuntimeError, match="still running"):
            with adapter._bound_transfer():
                thread = adapter._transfer_deadline._thread
                real_join = thread.join
                # Exercise the join-expired path without waiting 150 test seconds.
                monkeypatch.setattr(thread, "join", lambda timeout: real_join(0.01))
                raise RuntimeError("sender died")
        assert started.wait(5)
        assert adapter._weights_update_group is payload
        with pytest.raises(RuntimeError, match="still running"):
            adapter.teardown_weight_update_group()
        with pytest.raises(RuntimeError, match="still running"):
            adapter.init_weight_update_group(
                "new-pair", "127.0.0.1", 12345, 0, 2, "http://kv", 1, 1, 1
            )
        assert adapter._weights_update_group is payload
    finally:
        finish.set()
        if real_join is not None:
            real_join(5)


def test_awex_pair_thread_isolates_torch_state_and_retires_after_teardown(monkeypatch):
    platform = SimpleNamespace(
        device_type="cuda", current_device=lambda: 1, synchronize=MagicMock()
    )
    monkeypatch.setattr(sglang_adapter, "current_platform", platform)
    device = MagicMock(side_effect=lambda index: nullcontext())
    monkeypatch.setattr(torch.cuda, "device", device)
    local = threading.local()

    class Adapter:
        def __init__(self):
            self._transfer_executor = None
            self._transfer_thread_id = None
            self._transfer_device = None
            self._transfer_deadline = None
            self._weights_update_group = None
            self._weights_update_group_gloo = None

        def _finish_transfer_deadline(self):
            pass

        @sglang_adapter._on_transfer_thread(start=True)
        def initialize(self):
            assert not hasattr(local, "old_pair")
            assert torch.is_inference_mode_enabled()
            local.old_pair = True
            return threading.get_ident()

        @sglang_adapter._on_transfer_thread(stop=True)
        def teardown(self):
            assert local.old_pair is True
            return threading.get_ident()

    adapter = Adapter()
    main = threading.get_ident()
    previous_mode = torch.is_inference_mode_enabled()
    for _ in range(2):
        worker = adapter.initialize()
        assert worker != main
        assert adapter.teardown() == worker
        assert adapter._transfer_executor is None
        assert adapter._transfer_thread_id is None
    assert torch.is_inference_mode_enabled() == previous_mode
    assert not hasattr(local, "old_pair")
    assert all(args == (1,) for args, kwargs in device.call_args_list)


def test_sglang_synchronizes_weight_copies_before_gloo_barrier(monkeypatch):
    """The success barrier runs only after inference weights reach the device."""
    adapter = sglang_adapter.AwexSGLangAdapter(MagicMock())
    adapter._transfer_plan = MagicMock()
    adapter._weights_update_group = MagicMock(name="nccl_group")
    adapter._weights_update_group_gloo = MagicMock(name="gloo_group")
    adapter._transfer_rank = 0
    adapter.get_local_shard_parameters = MagicMock(return_value={})

    events = []
    original = MagicMock()
    contiguous = MagicMock()
    original.copy_.side_effect = lambda value: events.append("copy")
    monkeypatch.setattr(
        sglang_adapter,
        "nccl_build_recv_ops",
        lambda *args, **kwargs: ([], [(original, contiguous)], None),
    )
    monkeypatch.setattr(sglang_adapter, "batch_send_recv", MagicMock())
    platform = MagicMock()
    platform.synchronize.side_effect = lambda: events.append("synchronize")
    monkeypatch.setattr(sglang_adapter, "current_platform", platform, raising=False)
    monkeypatch.setattr(
        sglang_adapter.dist,
        "barrier",
        lambda **kwargs: events.append("barrier"),
    )

    adapter.execute_weight_update(version=1)

    original.copy_.assert_called_once_with(contiguous)
    assert events == ["copy", "synchronize", "barrier"]


@pytest.mark.parametrize(
    ("adapter_cls", "module"),
    [
        (fsdp_adapter.AwexFSDPAdapter, fsdp_adapter),
        (megatron_adapter.AwexMegatronAdapter, megatron_adapter),
        (sglang_adapter.AwexSGLangAdapter, sglang_adapter),
    ],
)
def test_awex_adapters_destroy_payload_and_sidecar_groups(
    adapter_cls, module, monkeypatch
):
    """Adapter teardown destroys both process groups and clears their handles."""
    adapter = adapter_cls(MagicMock())
    payload_group = MagicMock(name="nccl_group")
    sidecar_group = MagicMock(name="gloo_group")
    adapter._weights_update_group = payload_group
    adapter._weights_update_group_gloo = sidecar_group
    destroy = MagicMock()
    distributed = getattr(module, "dist", module.torch.distributed)
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "destroy_process_group", destroy)

    adapter.teardown_weight_update_group()

    assert destroy.call_args_list == [call(payload_group), call(sidecar_group)]
    assert adapter._weights_update_group is None
    assert adapter._weights_update_group_gloo is None
    if adapter_cls in (
        megatron_adapter.AwexMegatronAdapter,
        sglang_adapter.AwexSGLangAdapter,
    ):
        assert adapter._separation_wire_dtypes is None


def test_teardown_attempts_sidecar_after_payload_failure_and_can_retry(monkeypatch):
    from types import SimpleNamespace

    from areal.v2.weight_update.awex import teardown_weight_update_groups

    adapter = SimpleNamespace(
        _weights_update_group="payload", _weights_update_group_gloo="sidecar"
    )
    destroy = MagicMock(side_effect=[RuntimeError("payload cleanup unknown"), None])
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy)

    with pytest.raises(ExceptionGroup, match="cleanup incomplete"):
        teardown_weight_update_groups(adapter)
    assert destroy.call_args_list == [call("payload"), call("sidecar")]
    assert adapter._weights_update_group == "payload"
    assert adapter._weights_update_group_gloo is None

    destroy.side_effect = None
    teardown_weight_update_groups(adapter)
    assert destroy.call_args_list[-1] == call("payload")
    assert adapter._weights_update_group is None


@pytest.mark.parametrize("sharded", [False, True])
def test_fsdp_export_matches_wire_metadata_and_preserves_master(sharded):
    """FP32 storage exports BF16 shards without changing master or Adam state."""
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
    try:
        mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("dp_shard",))
        master = torch.nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(2, 4))
        optimizer = torch.optim.AdamW([master], lr=0.01)
        master.sum().backward()
        optimizer.step()
        original = master.detach().clone()
        moments = {key: value.clone() for key, value in optimizer.state[master].items()}
        parameter = master
        if sharded:
            parameter = torch.nn.Parameter(
                DTensor.from_local(
                    master.detach(),
                    mesh,
                    [Shard(0)],
                    run_check=False,
                    shape=torch.Size([4, 4]),
                    stride=(4, 1),
                )
            )
        cast = MagicMock(side_effect=lambda tensor: tensor.to(torch.bfloat16))
        engine = SimpleNamespace(
            model=SimpleNamespace(
                named_parameters=lambda: [("layer.weight", parameter)]
            ),
            model_config=SimpleNamespace(tie_word_embeddings=False),
            world_mesh=mesh,
            rank=0,
            world_size=2,
            data_parallel_world_size=2,
            dp_rank=0,
            is_vision_model=False,
            _compute_dtype=lambda: torch.bfloat16,
            _cast_to_compute_dtype=cast,
        )
        adapter = AwexFSDPAdapter(engine)

        metadata = adapter.get_weight_metadata()[0]
        cast.assert_not_called()  # Metadata discovery cannot allocate export buffers.
        exported = adapter.get_local_shard_parameters()["layer.weight"]

        assert metadata.dtype == exported.dtype == torch.bfloat16
        assert metadata.shards[0].dtype == torch.bfloat16
        assert metadata.replicas[0].shards[0].dtype == torch.bfloat16
        assert metadata.shards[0].shape == tuple(master.shape)
        assert metadata.global_shape == ((4, 4) if sharded else (2, 4))
        assert metadata.shards[0].global_offset == (0, 0)
        torch.testing.assert_close(
            exported, original.to(torch.bfloat16), rtol=0, atol=0
        )
        assert parameter.dtype == master.dtype == torch.float32
        torch.testing.assert_close(master, original, rtol=0, atol=0)
        for key, value in moments.items():
            torch.testing.assert_close(
                optimizer.state[master][key], value, rtol=0, atol=0
            )
    finally:
        dist.destroy_process_group()


def test_fsdp_wire_dtype_preserves_nonfloating_parameters():
    """Integer parameters retain their storage type in metadata."""
    adapter = AwexFSDPAdapter(SimpleNamespace(_compute_dtype=lambda: torch.bfloat16))
    assert adapter._wire_dtype(torch.arange(4, dtype=torch.int64)) == torch.int64
