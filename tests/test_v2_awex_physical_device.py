# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from areal.v2.inference_service.sglang.scheduler import AwexSchedulerBridge
from areal.v2.weight_update.awex import resolve_physical_gpu_id


def test_numeric_cuda_visibility_maps_local_to_physical(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    assert resolve_physical_gpu_id(2, strict=True) == 6


def test_strict_colocate_mapping_rejects_gpu_uuid(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-deadbeef")

    with pytest.raises(RuntimeError, match="numeric physical GPU"):
        resolve_physical_gpu_id(0, strict=True)


def test_nonstrict_mapping_preserves_legacy_fallback(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-deadbeef")

    assert resolve_physical_gpu_id(0) == 0


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
