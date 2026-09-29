# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.infra.platforms.cuda import CudaPlatform


@pytest.mark.parametrize(
    "visible,local_rank,uuid",
    [("4", 0, "four"), ("7,4", 1, "four"), ("GPU-four", 0, "GPU-four")],
)
def test_numa_affinity_uses_cuda_device_uuid(monkeypatch, visible, local_rank, uuid):
    import areal.infra.platforms.cuda as cuda

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    nvml = SimpleNamespace(
        nvmlInit=Mock(),
        nvmlShutdown=Mock(),
        nvmlDeviceGetHandleByIndex=Mock(
            side_effect=AssertionError("physical index lookup")
        ),
        nvmlDeviceGetHandleByUUID=Mock(return_value="actual-device"),
        nvmlDeviceSetCpuAffinity=Mock(),
    )
    properties = Mock(return_value=SimpleNamespace(uuid=uuid))
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(cuda.torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(cuda.os, "sched_getaffinity", lambda _: {2, 3})

    CudaPlatform.set_numa_affinity(local_rank)

    properties.assert_called_once_with(local_rank)
    nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with(
        uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
    )
    nvml.nvmlDeviceSetCpuAffinity.assert_called_once_with("actual-device")
    nvml.nvmlDeviceGetHandleByIndex.assert_not_called()
    nvml.nvmlShutdown.assert_called_once()


def test_numa_affinity_uuid_unavailable_skips_binding(monkeypatch):
    import areal.infra.platforms.cuda as cuda

    nvml = SimpleNamespace(
        nvmlInit=Mock(),
        nvmlShutdown=Mock(),
        nvmlDeviceGetHandleByUUID=Mock(),
        nvmlDeviceSetCpuAffinity=Mock(),
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(
        cuda.torch.cuda, "get_device_properties", lambda _: SimpleNamespace()
    )

    CudaPlatform.set_numa_affinity(0)

    nvml.nvmlDeviceSetCpuAffinity.assert_not_called()
    nvml.nvmlShutdown.assert_called_once()
