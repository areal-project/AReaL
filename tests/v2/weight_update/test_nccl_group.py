# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, call

import pytest
import torch

from areal.v2.weight_update import nccl_group
from areal.v2.weight_update.awex import fsdp_adapter, megatron_adapter, sglang_adapter
from areal.v2.weight_update.awex.state import AwexPairState

PAIR_NAME = "actor-rollout"


def _install_pair_state(adapter, payload_group, sidecar_group, rank=0):
    adapter._pair_states[PAIR_NAME] = AwexPairState(
        weights_update_group=payload_group,
        control_group=sidecar_group,
        transfer_plan=MagicMock(),
        transfer_rank=rank,
    )


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
    payload_group = MagicMock(name="nccl_group")
    sidecar_group = MagicMock(name="gloo_group")
    initializer = MagicMock(side_effect=[payload_group, sidecar_group])

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
    state = adapter._pair_states[PAIR_NAME]
    assert state.weights_update_group is payload_group
    assert state.control_group is sidecar_group


def test_fsdp_init_preserves_original_error_and_retains_failed_cleanup(
    monkeypatch,
):
    """A sidecar init failure remains primary and leaked payload is retryable."""
    adapter = fsdp_adapter.AwexFSDPAdapter(MagicMock())
    builder = MagicMock()
    builder.build_local_transfer_plan.return_value = MagicMock()
    payload_group = MagicMock(name="nccl_group")
    initializer = MagicMock(
        side_effect=[payload_group, RuntimeError("gloo init failed")]
    )
    destroy = MagicMock(side_effect=RuntimeError("payload cleanup failed"))

    monkeypatch.setattr(
        fsdp_adapter, "fetch_kv_metadata", lambda *args: (MagicMock(), MagicMock())
    )
    monkeypatch.setattr(fsdp_adapter, "TransferPlanBuilder", lambda **kwargs: builder)
    monkeypatch.setattr(fsdp_adapter, "init_weights_update_group", initializer)
    monkeypatch.setattr(fsdp_adapter.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        fsdp_adapter.torch.distributed, "destroy_process_group", destroy
    )

    with pytest.raises(RuntimeError, match="gloo init failed"):
        adapter.init_weight_update_group(
            pair_name=PAIR_NAME,
            master_addr="127.0.0.1",
            master_port=29500,
            transfer_rank=2,
            world_size=4,
            kv_store_url="http://kv-store",
            infer_world_size=2,
            train_world_size=2,
            num_engines=1,
        )

    destroy.assert_called_once_with(payload_group)
    state = adapter._pair_states[PAIR_NAME]
    assert state.weights_update_group is payload_group
    assert state.control_group is None


def test_fsdp_partial_pair_must_be_torn_down_before_reinitialization():
    adapter = fsdp_adapter.AwexFSDPAdapter(MagicMock())
    adapter._pair_states[PAIR_NAME] = AwexPairState(
        weights_update_group=MagicMock(),
        control_group=None,
        transfer_plan=MagicMock(),
        transfer_rank=0,
    )

    with pytest.raises(RuntimeError, match="partially initialized"):
        adapter.init_weight_update_group(
            pair_name=PAIR_NAME,
            master_addr="127.0.0.1",
            master_port=29500,
            transfer_rank=2,
            world_size=4,
            kv_store_url="http://kv-store",
            infer_world_size=2,
            train_world_size=2,
            num_engines=1,
        )


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
        ),
        call(
            master_address="127.0.0.1",
            master_port=29500,
            rank=1,
            world_size=4,
            group_name="awex_actor-rollout_gloo",
            backend="gloo",
            role="inference",
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
    _install_pair_state(adapter, payload_group, sidecar_group, rank=3)
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
    _install_pair_state(adapter, payload_group, sidecar_group)
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
    _install_pair_state(adapter, payload_group, sidecar_group)
    destroy = MagicMock()
    distributed = getattr(module, "dist", module.torch.distributed)
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "destroy_process_group", destroy)

    adapter.teardown_weight_update_group(PAIR_NAME)

    assert destroy.call_args_list == [call(payload_group), call(sidecar_group)]
    assert PAIR_NAME not in adapter._pair_states


@pytest.mark.parametrize(
    ("adapter_cls", "module"),
    [
        (fsdp_adapter.AwexFSDPAdapter, fsdp_adapter),
        (megatron_adapter.AwexMegatronAdapter, megatron_adapter),
        (sglang_adapter.AwexSGLangAdapter, sglang_adapter),
    ],
)
def test_awex_teardown_retains_only_failed_group_for_retry(
    adapter_cls, module, monkeypatch
):
    adapter = adapter_cls(MagicMock())
    payload_group = MagicMock(name="nccl_group")
    sidecar_group = MagicMock(name="gloo_group")
    _install_pair_state(adapter, payload_group, sidecar_group)

    def fail_payload(group):
        if group is payload_group:
            raise RuntimeError("payload teardown failed")

    destroy = MagicMock(side_effect=fail_payload)
    distributed = getattr(module, "dist", module.torch.distributed)
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "destroy_process_group", destroy)

    with pytest.raises(RuntimeError, match="Failed to teardown AWEX pair"):
        adapter.teardown_weight_update_group(PAIR_NAME)

    assert destroy.call_args_list == [call(payload_group), call(sidecar_group)]
    state = adapter._pair_states[PAIR_NAME]
    assert state.weights_update_group is payload_group
    assert state.control_group is None

    destroy.reset_mock()
    destroy.side_effect = None
    adapter.teardown_weight_update_group(PAIR_NAME)

    destroy.assert_called_once_with(payload_group)
    assert PAIR_NAME not in adapter._pair_states
