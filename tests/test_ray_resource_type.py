# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.infra.utils import ray as ray_utils


@pytest.fixture
def cpu_only_driver(monkeypatch):
    import torch

    monkeypatch.setattr(ray_utils.ray, "is_initialized", lambda: True)
    monkeypatch.setattr("areal.infra.platforms.is_npu_available", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_local_gpu_driver_reports_gpu(monkeypatch):
    import torch

    monkeypatch.setattr("areal.infra.platforms.is_npu_available", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert ray_utils.ray_resource_type() == "GPU"


def test_cpu_only_driver_reports_cluster_gpu(cpu_only_driver, monkeypatch):
    monkeypatch.setattr(
        ray_utils.ray, "cluster_resources", lambda: {"CPU": 32.0, "GPU": 64.0}
    )

    assert ray_utils.ray_resource_type() == "GPU"


def test_cpu_only_driver_reports_cluster_npu(cpu_only_driver, monkeypatch):
    monkeypatch.setattr(
        ray_utils.ray, "cluster_resources", lambda: {"CPU": 32.0, "NPU": 16.0}
    )

    assert ray_utils.ray_resource_type() == "NPU"


def test_cpu_only_driver_without_cluster_accelerators_reports_cpu(
    cpu_only_driver, monkeypatch
):
    monkeypatch.setattr(ray_utils.ray, "cluster_resources", lambda: {"CPU": 32.0})

    assert ray_utils.ray_resource_type() == "CPU"


def test_disconnected_cpu_driver_does_not_query_cluster(monkeypatch):
    import torch

    monkeypatch.setattr("areal.infra.platforms.is_npu_available", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ray_utils.ray, "is_initialized", lambda: False)

    def fail():
        raise AssertionError("cluster_resources must not be queried when disconnected")

    monkeypatch.setattr(ray_utils.ray, "cluster_resources", fail)

    assert ray_utils.ray_resource_type() == "CPU"
