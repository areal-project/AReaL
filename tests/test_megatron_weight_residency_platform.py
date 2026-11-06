# SPDX-License-Identifier: Apache-2.0

"""Weight-residency memory calls must go through the device platform abstraction.

Regression test for the Ascend NPU path: the residency adapter used to call
``torch.cuda.synchronize()`` / ``torch.cuda.empty_cache()`` directly, which
raises on non-CUDA accelerators (``torch.cuda.synchronize()``) or silently does
nothing (``torch.cuda.empty_cache()`` leaves the NPU allocator cache in place).
"""

from unittest import mock

from areal.engine.megatron_utils import weight_residency
from areal.engine.megatron_utils.weight_residency import MegatronWeightResidency


class _StubEngine:
    """Engine stand-in: residency offload paths return early on empty members."""

    model = None
    optimizer = None


class _PlatformSpy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synchronize(self) -> None:
        self.calls.append("synchronize")

    def clear_memory(self) -> None:
        self.calls.append("clear_memory")


def test_release_memory_delegates_to_platform():
    spy = _PlatformSpy()
    with mock.patch.object(weight_residency, "current_platform", spy):
        MegatronWeightResidency(_StubEngine()).release_memory(["weights"])

    assert spy.calls == ["clear_memory"]


def test_release_memory_syncs_through_platform_for_all_tags():
    """The full release path must sync via the platform too, never ``torch.cuda``."""
    spy = _PlatformSpy()
    with mock.patch.object(weight_residency, "current_platform", spy):
        MegatronWeightResidency(_StubEngine()).release_memory()

    assert "clear_memory" in spy.calls
    assert set(spy.calls) <= {"synchronize", "clear_memory"}


def test_resume_memory_delegates_to_platform():
    spy = _PlatformSpy()
    residency = MegatronWeightResidency(_StubEngine())
    residency._released_tags.update({"weights", "optimizer"})
    with mock.patch.object(weight_residency, "current_platform", spy):
        residency.resume_memory()

    assert spy.calls == ["synchronize"]
