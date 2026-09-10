# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

from areal.v2.weight_update import nccl_group
from areal.v2.weight_update.awex import fsdp_adapter, megatron_adapter, sglang_adapter
from areal.v2.weight_update.awex.state import AwexPairState

PAIR_NAME = "test-pair"


def _install_pair_state(adapter, payload_group, sidecar_group, transfer_rank):
    if isinstance(adapter, fsdp_adapter.AwexFSDPAdapter):
        adapter._pair_states[PAIR_NAME] = AwexPairState(
            payload_group,
            sidecar_group,
            MagicMock(),
            transfer_rank,
        )
        return

    adapter._active_pair_name = PAIR_NAME
    adapter._transfer_plan = MagicMock()
    adapter._weights_update_group = payload_group
    adapter._weights_update_group_gloo = sidecar_group
    adapter._transfer_rank = transfer_rank
    adapter._dte_config = SimpleNamespace(enabled=False)


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


def test_process_group_timeout_is_forwarded_as_timedelta(monkeypatch):
    group = MagicMock()
    init_group = MagicMock(return_value=group)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(nccl_group, "init_custom_process_group", init_group)

    result = nccl_group.init_weights_update_group(
        "127.0.0.1",
        29500,
        rank=0,
        world_size=1,
        group_name="timeout-test",
        timeout_s=2.5,
    )

    assert result is group
    assert init_group.call_args.kwargs["timeout"] == timedelta(seconds=2.5)


def test_adapter_uses_decreasing_setup_budget(monkeypatch):
    adapter = fsdp_adapter.AwexFSDPAdapter(MagicMock())
    builder = MagicMock()
    builder.build_local_transfer_plan.return_value = MagicMock()
    metadata = MagicMock(return_value=(MagicMock(), MagicMock()))
    initializer = MagicMock(side_effect=[MagicMock(), MagicMock()])
    clock = iter([100.0, 101.0, 102.0, 103.0])

    monkeypatch.setattr(fsdp_adapter.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(fsdp_adapter, "fetch_kv_metadata", metadata)
    monkeypatch.setattr(fsdp_adapter, "TransferPlanBuilder", lambda **kwargs: builder)
    monkeypatch.setattr(fsdp_adapter, "init_weights_update_group", initializer)

    adapter.init_weight_update_group(
        pair_name=PAIR_NAME,
        master_addr="127.0.0.1",
        master_port=29500,
        transfer_rank=0,
        world_size=2,
        kv_store_url="http://kv-store",
        infer_world_size=1,
        train_world_size=1,
        num_engines=1,
        process_group_timeout_s=10.0,
    )

    metadata.assert_called_once_with("http://kv-store", PAIR_NAME, timeout_s=9.0)
    assert [item.kwargs["timeout_s"] for item in initializer.call_args_list] == [
        8.0,
        7.0,
    ]


def test_megatron_adapter_initializes_nccl_and_gloo_groups(monkeypatch):
    """The training adapter creates matching payload and sidecar groups."""
    adapter = megatron_adapter.AwexMegatronAdapter(MagicMock())
    builder = MagicMock()
    builder.build_local_transfer_plan.return_value = MagicMock()
    initializer = MagicMock(side_effect=[MagicMock(), MagicMock()])

    monkeypatch.setattr(
        megatron_adapter,
        "fetch_kv_metadata",
        lambda *args, **kwargs: (MagicMock(), MagicMock()),
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
            timeout_s=None,
        ),
        call(
            master_address="127.0.0.1",
            master_port=29500,
            rank=2,
            world_size=4,
            group_name="awex_actor-rollout_gloo",
            backend="gloo",
            role="training",
            timeout_s=None,
        ),
    ]


def test_fsdp_adapter_initializes_nccl_and_gloo_groups(monkeypatch):
    """The FSDP training adapter creates matching payload and sidecar groups."""
    adapter = fsdp_adapter.AwexFSDPAdapter(MagicMock())
    builder = MagicMock()
    builder.build_local_transfer_plan.return_value = MagicMock()
    initializer = MagicMock(side_effect=[MagicMock(), MagicMock()])

    monkeypatch.setattr(
        fsdp_adapter,
        "fetch_kv_metadata",
        lambda *args, **kwargs: (MagicMock(), MagicMock()),
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
            timeout_s=None,
        ),
        call(
            master_address="127.0.0.1",
            master_port=29500,
            rank=2,
            world_size=4,
            group_name="awex_actor-rollout_gloo",
            backend="gloo",
            role="training",
            timeout_s=None,
        ),
    ]


def test_sglang_adapter_initializes_nccl_and_gloo_groups(monkeypatch):
    """The inference adapter creates matching payload and sidecar groups."""
    adapter = sglang_adapter.AwexSGLangAdapter(MagicMock())
    builder = MagicMock()
    builder.build_local_transfer_plan.return_value = MagicMock()
    initializer = MagicMock(side_effect=[MagicMock(), MagicMock()])

    monkeypatch.setattr(
        adapter,
        "_get_model_context",
        lambda: {"tp_size": 1, "tp_rank": 0, "pp_size": 1, "pp_rank": 0},
    )
    monkeypatch.setattr(
        sglang_adapter,
        "fetch_kv_metadata",
        lambda *args, **kwargs: (MagicMock(), MagicMock()),
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
            timeout_s=None,
        ),
        call(
            master_address="127.0.0.1",
            master_port=29500,
            rank=1,
            world_size=4,
            group_name="awex_actor-rollout_gloo",
            backend="gloo",
            role="inference",
            timeout_s=None,
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
    _install_pair_state(adapter, payload_group, sidecar_group, 3)
    setup = MagicMock()
    monkeypatch.setattr(module, "setup_batch_isend_irecv", setup)

    adapter.batch_isend_irecv(PAIR_NAME, world_size=4)

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
    _install_pair_state(adapter, payload_group, sidecar_group, 0)
    adapter.get_local_shard_parameters = MagicMock(return_value={})
    monkeypatch.setattr(module, build_ops_name, lambda *args, **kwargs: ([], [], None))
    monkeypatch.setattr(module, "batch_send_recv", MagicMock())
    barrier = MagicMock()
    distributed = getattr(module, "dist", module.torch.distributed)
    monkeypatch.setattr(distributed, "barrier", barrier)

    adapter.execute_weight_update(PAIR_NAME, version=1)

    barrier.assert_called_once_with(group=sidecar_group)


def test_sglang_synchronizes_weight_copies_before_gloo_barrier(monkeypatch):
    """The success barrier runs only after inference weights reach the device."""
    adapter = sglang_adapter.AwexSGLangAdapter(MagicMock())
    _install_pair_state(
        adapter,
        MagicMock(name="nccl_group"),
        MagicMock(name="gloo_group"),
        0,
    )
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

    adapter.execute_weight_update(PAIR_NAME, version=1)

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
    _install_pair_state(adapter, payload_group, sidecar_group, 0)
    destroy = MagicMock()
    distributed = getattr(module, "dist", module.torch.distributed)
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "destroy_process_group", destroy)

    adapter.teardown_weight_update_group(PAIR_NAME)
    adapter.teardown_weight_update_group(PAIR_NAME)

    assert destroy.call_args_list == [call(payload_group), call(sidecar_group)]
    if adapter_cls is fsdp_adapter.AwexFSDPAdapter:
        assert PAIR_NAME not in adapter._pair_states
    else:
        assert adapter._weights_update_group is None
        assert adapter._weights_update_group_gloo is None
    if adapter_cls in (
        megatron_adapter.AwexMegatronAdapter,
        sglang_adapter.AwexSGLangAdapter,
    ):
        assert adapter._separation_wire_dtypes is None


@pytest.mark.parametrize(
    "adapter_cls",
    [megatron_adapter.AwexMegatronAdapter, sglang_adapter.AwexSGLangAdapter],
)
def test_parked_pair_can_be_reactivated_without_crossing_state(adapter_cls):
    adapter = adapter_cls(MagicMock())
    a_payload, a_control = MagicMock(), MagicMock()
    b_payload, b_control = MagicMock(), MagicMock()
    adapter._active_pair_name = "pair-a"
    adapter._weights_update_group = a_payload
    adapter._weights_update_group_gloo = a_control
    adapter._transfer_plan = MagicMock(name="plan-a")
    adapter._transfer_rank = 1
    adapter._pair_states["pair-b"] = AwexPairState(
        b_payload, b_control, MagicMock(name="plan-b"), 2
    )

    adapter._activate_pair("pair-b")
    assert (adapter._weights_update_group, adapter._transfer_rank) == (b_payload, 2)
    adapter._activate_pair("pair-a")
    assert (adapter._weights_update_group, adapter._transfer_rank) == (a_payload, 1)
    assert adapter._pair_states["pair-b"].control_group is b_control
