# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.engine.awex.adapters.megatron_adapter import AwexMegatronAdapter
from areal.engine.awex.utils import resolve_physical_gpu_id
from areal.v2.inference_service.sglang.scheduler import AwexSchedulerBridge


def test_numeric_cuda_visibility_maps_local_to_physical(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    assert resolve_physical_gpu_id(2) == 6


def test_strict_colocate_mapping_rejects_gpu_uuid(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-deadbeef")

    with pytest.raises(ValueError, match="numeric CUDA_VISIBLE_DEVICES"):
        resolve_physical_gpu_id(0)


def test_negative_device_index_is_rejected(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")

    with pytest.raises(ValueError, match="nonnegative"):
        resolve_physical_gpu_id(-1)


def _scheduler_bridge_result(include_device):
    result = []
    bridge = object.__new__(AwexSchedulerBridge)
    bridge._adapter = SimpleNamespace(parallelism_strategy={"world_size": 4})
    bridge._push_result = result.append

    bridge.awex_report_parallelism(include_device=include_device)

    return result[0]


def test_separation_parallelism_report_does_not_require_physical_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-deadbeef")

    assert _scheduler_bridge_result(include_device=False) == {"world_size": 4}


def test_colocate_parallelism_report_includes_physical_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setattr("areal.utils.network.gethostip", lambda: "192.0.2.1")
    monkeypatch.setattr("torch.cuda.current_device", lambda: 2)

    assert _scheduler_bridge_result(include_device=True) == {
        "world_size": 4,
        "ip": "192.0.2.1",
        "device_id": 6,
    }


def test_native_converter_rank_is_not_replaced_by_physical_pair_rank(monkeypatch):
    adapter = AwexMegatronAdapter(SimpleNamespace())
    adapter._legacy_transfer_rank = 14
    adapter._parameters_meta = []
    adapter._rank_info = SimpleNamespace(global_rank=3, world_size=8)
    adapter.configure_model_converter = Mock()
    client = Mock()
    client.get_object.side_effect = [{"infer_world_size": 8}, 2]
    adapter._legacy_meta_server_client = client
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 3)
    monkeypatch.setattr("awex.util.common.get_ip_address", lambda: "192.0.2.1")

    adapter._lazy_initialize_legacy_colocate()

    assert adapter._legacy_logical_train_rank == 11
    client.add_object_to_set.assert_called_once_with(
        "training_device_rank_entries", ("192.0.2.1", 4, 11)
    )


def test_physical_base_from_cvd_maps_guard_mask_to_physical_base():
    from areal.v2.inference_service.sglang.launch_server import physical_base_from_cvd

    assert physical_base_from_cvd("4,5,6,7") == 4
    assert physical_base_from_cvd("0,1,2,3") == 0
    assert physical_base_from_cvd("0,1,2,3,4,5,6,7") == 0


@pytest.mark.parametrize("cvd", ["", "0,2,3,4", "5,4", "0,1,1", "-1,0", "GPU-uuid"])
def test_physical_base_from_cvd_rejects_invalid_device_order(cvd):
    from areal.v2.inference_service.sglang.launch_server import physical_base_from_cvd

    with pytest.raises(ValueError):
        physical_base_from_cvd(cvd)
