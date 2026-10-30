# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.infra import platforms
from areal.infra.platforms.cpu import CpuPlatform
from areal.infra.platforms.cuda import CudaPlatform
from areal.infra.platforms.npu import NPUPlatform
from areal.infra.platforms.rocm import ROCmPlatform
from areal.infra.platforms.xpu import XpuPlatform


def test_init_platform_selects_xpu_when_available(monkeypatch):
    import torch

    monkeypatch.setattr(platforms, "is_npu_available", False)
    monkeypatch.setattr(platforms, "is_xpu_available", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert isinstance(platforms._init_platform(), XpuPlatform)


def test_init_platform_prefers_npu_over_xpu(monkeypatch):
    """NPU is checked before XPU, so enabling both must still select NPU."""
    monkeypatch.setattr(platforms, "is_npu_available", True)
    monkeypatch.setattr(platforms, "is_xpu_available", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert isinstance(platforms._init_platform(), NPUPlatform)


def test_init_platform_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(platforms, "is_npu_available", False)
    monkeypatch.setattr(platforms, "is_xpu_available", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert isinstance(platforms._init_platform(), CpuPlatform)


def test_xpu_visible_devices_use_level_zero_affinity_mask():
    env_vars = {}

    XpuPlatform.update_env_vars_for_visible_devices(env_vars, [0, 2])

    assert env_vars == {"ZE_AFFINITY_MASK": "0,2"}


def test_xpu_visible_devices_do_not_set_empty_ray_noset_key():
    """``ray_experimental_noset`` is empty for XPU; it must not leak as a key."""
    env_vars = {}

    XpuPlatform.update_env_vars_for_visible_devices(env_vars, [1])

    assert "" not in env_vars


@pytest.mark.parametrize(
    "platform_cls,local_rank,expected",
    [
        (XpuPlatform, 0, "xpu:0"),
        (XpuPlatform, 3, "xpu:3"),
        (CudaPlatform, 0, "cuda:0"),
        (CudaPlatform, 2, "cuda:2"),
        (ROCmPlatform, 1, "cuda:1"),  # ROCm reports device_type "cuda" in torch
    ],
)
def test_device_of_qualifies_device_type(platform_cls, local_rank, expected):
    """``device_of`` must spell out the device type, never rely on a bare index."""
    assert platform_cls.device_of(local_rank) == torch.device(expected)


def test_device_of_builds_npu_string():
    """NPU's torch backend is registered by ``torch_npu``, which may be absent.

    Assert on the string ``device_of`` builds so this covers the NPU branch on a
    machine without the Ascend plugin installed.
    """
    try:
        expected = torch.device("npu:1")
    except RuntimeError:
        pytest.skip("torch_npu not installed; 'npu' device type unregistered")
    assert NPUPlatform.device_of(1) == expected


def test_device_of_ignores_local_rank_on_cpu():
    assert CpuPlatform.device_of(5) == torch.device("cpu")


def test_device_of_differs_from_bare_index_on_xpu():
    """Regression guard: ``torch.device(0)`` resolves to CUDA on a CUDA build.

    This is the bug ``device_of`` exists to prevent, so assert the two forms
    genuinely disagree rather than trusting them to match. On an XPU-only wheel
    CUDA is not compiled in and the bare index already resolves to ``xpu:0``,
    so there is no disagreement to observe -- only the device type is asserted.
    """
    assert XpuPlatform.device_of(0).type == "xpu"
    if torch.backends.cuda.is_built():
        assert XpuPlatform.device_of(0) != torch.device(0)


def test_xpu_platform_metadata():
    assert XpuPlatform.device_type == "xpu"
    assert XpuPlatform.communication_backend == "xccl"
    assert XpuPlatform.ray_device_key == "XPU"
    assert XpuPlatform.device_control_env_var == "ZE_AFFINITY_MASK"


def test_set_device_delegates_to_torch_xpu():
    """``Platform.__getattr__`` must route ``set_device`` to ``torch.xpu``.

    Engine setup calls ``current_platform.set_device(local_rank)`` with no XPU
    code of its own, so the delegation in ``Platform.__getattr__`` is the only
    thing making it land on ``torch.xpu``. Assert on identity rather than
    calling it, so an upstream rename or removal fails here instead of
    misrouting tensors to CUDA at runtime.
    """
    assert XpuPlatform().set_device is torch.xpu.set_device
    assert XpuPlatform().current_device is torch.xpu.current_device


def test_set_device_delegation_is_instance_only():
    """Delegation is an *instance* hook, so class-level access must fail loudly.

    ``__getattr__`` is only consulted on instances. Every call site reaches
    ``set_device`` through the ``current_platform`` proxy instance, and this
    pins that contract: a future caller writing ``XpuPlatform.set_device``
    gets an immediate AttributeError rather than a silent miss.
    """
    with pytest.raises(AttributeError, match="set_device"):
        XpuPlatform.set_device  # noqa: B018 - attribute access is the assertion


def test_unknown_attribute_raises_rather_than_falling_back():
    """A name absent from ``torch.xpu`` must raise, never resolve elsewhere."""
    with pytest.raises(AttributeError, match="definitely_not_a_torch_api"):
        XpuPlatform().definitely_not_a_torch_api  # noqa: B018 - access is the assertion


def test_set_device_delegates_per_platform():
    """Each platform delegates to its own ``torch`` submodule, not a shared one."""
    assert CudaPlatform().set_device is torch.cuda.set_device
    assert XpuPlatform().set_device is not CudaPlatform().set_device
