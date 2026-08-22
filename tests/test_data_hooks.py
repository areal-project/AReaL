# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from areal.trainer.ppo.actor import PPOActor
from areal.utils.data_hook import DataHookContext, DataHookManager, DataHookOutput

_EVENTS: list[str] = []


class _OrderHook:
    def __init__(self, name: str) -> None:
        self.name = name

    def setup(self, context: DataHookContext) -> None:
        _EVENTS.append(f"setup:{self.name}:{context.role}")

    def test_pre(self, data: list[str]) -> DataHookOutput:
        _EVENTS.append(f"pre:{self.name}")
        return DataHookOutput(data=[*data, self.name], state=f"state:{self.name}")

    def test_post(self, data: list[str], state: str) -> list[str]:
        _EVENTS.append(f"post:{self.name}:{state}")
        return [*data, f"post:{self.name}"]

    def close(self) -> None:
        _EVENTS.append(f"close:{self.name}")


class _FailingSetupHook:
    def setup(self, context: DataHookContext) -> None:
        _EVENTS.append(f"setup:bad:{context.role}")
        raise RuntimeError("setup failed")

    def close(self) -> None:
        _EVENTS.append("close:bad")


class _InterruptingCloseHook:
    def close(self) -> None:
        raise KeyboardInterrupt


@dataclass
class _Spec:
    class_path: str
    kwargs: dict[str, Any] = field(default_factory=dict)


def test_data_hook_manager_orders_setup_pre_post_and_close():
    """Middleware enters in YAML order and unwinds in reverse order."""
    _EVENTS.clear()
    specs = [
        _Spec(f"{__name__}._OrderHook", {"name": "a"}),
        _Spec(f"{__name__}._OrderHook", {"name": "b"}),
    ]
    manager = DataHookManager(specs, role="teacher")

    data, state = manager.run_pre("test_pre", ["start"])
    result = manager.run_post("test_post", data, state)
    manager.close()
    manager.close()

    assert result == ["start", "a", "b", "post:b", "post:a"]
    assert _EVENTS == [
        "setup:a:teacher",
        "setup:b:teacher",
        "pre:a",
        "pre:b",
        "post:b:state:b",
        "post:a:state:a",
        "close:b",
        "close:a",
    ]


def test_data_hook_manager_rolls_back_partial_setup():
    """A later setup failure closes the failing hook and earlier hooks."""
    _EVENTS.clear()
    specs = [
        _Spec(f"{__name__}._OrderHook", {"name": "a"}),
        _Spec(f"{__name__}._FailingSetupHook"),
    ]

    with pytest.raises(RuntimeError, match="setup failed"):
        DataHookManager(specs, role="teacher")

    assert _EVENTS == [
        "setup:a:teacher",
        "setup:bad:teacher",
        "close:bad",
        "close:a",
    ]


def test_data_hook_manager_preserves_base_exceptions_during_close():
    """Cleanup reports process-control exceptions without replacing their type."""
    manager = DataHookManager(
        [_Spec(f"{__name__}._InterruptingCloseHook")], role="teacher"
    )

    with pytest.raises(BaseExceptionGroup) as exc_info:
        manager.close()

    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], KeyboardInterrupt)


def test_ppo_actor_data_hooks_require_one_explicit_role():
    """A configured worker cannot silently change from teacher to actor hooks."""
    _EVENTS.clear()
    actor = object.__new__(PPOActor)
    actor._data_hook_specs = (_Spec(f"{__name__}._OrderHook", {"name": "role"}),)
    actor.data_hook_role = None
    actor.data_hooks = None
    actor._data_hooks_closed = False

    actor.setup_data_hooks("teacher")
    actor.setup_data_hooks("teacher")
    with pytest.raises(RuntimeError, match="already set up"):
        actor.setup_data_hooks("actor")
    actor.close_data_hooks()
    actor.close_data_hooks()

    assert _EVENTS == ["setup:role:teacher", "close:role"]
