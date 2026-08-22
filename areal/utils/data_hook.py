# SPDX-License-Identifier: Apache-2.0

"""Worker-local data lifecycle hooks.

Hooks are intentionally small middleware objects.  They are imported from a
configured class path, set up once in each worker, and may transform data before
and after a lifecycle event.  Pre hooks run in configuration order; post hooks
run in reverse order so nested transformations unwind correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from areal.utils.dynamic_import import import_from_string


@dataclass(frozen=True)
class DataHookContext:
    """Immutable context supplied once when a worker hook is initialized."""

    role: str


@dataclass
class DataHookOutput:
    """Result of a pre hook, including optional call-local post-hook state."""

    data: Any
    state: Any = None


@dataclass
class DataHookSpec:
    """Import path and constructor arguments for one worker-local hook."""

    class_path: str
    kwargs: dict[str, Any] = field(default_factory=dict)


class DataHookSpecLike(Protocol):
    """Structural type accepted by :class:`DataHookManager`."""

    class_path: str
    kwargs: dict[str, Any]


@dataclass
class _HookInvocation:
    hook: Any
    state: Any


class DataHookManager:
    """Load and execute ordered, fail-fast worker-local data hooks."""

    def __init__(
        self,
        specs: list[DataHookSpecLike] | tuple[DataHookSpecLike, ...] | None,
        *,
        role: str,
    ) -> None:
        self.context = DataHookContext(role=role)
        self._hooks: list[Any] = []
        self._closed = False

        try:
            for spec in specs or ():
                class_path = (
                    spec.get("class_path")
                    if isinstance(spec, dict)
                    else spec.class_path
                )
                kwargs = (
                    spec.get("kwargs", {}) if isinstance(spec, dict) else spec.kwargs
                )
                hook_cls = import_from_string(class_path)
                hook = hook_cls(**kwargs)
                self._hooks.append(hook)
                setup = getattr(hook, "setup", None)
                if setup is not None:
                    setup(self.context)
        except BaseException as setup_error:
            try:
                self.close()
            except BaseException as close_error:
                raise BaseExceptionGroup(
                    "Data hook setup and rollback failed",
                    [setup_error, close_error],
                ) from setup_error
            raise

    @property
    def enabled(self) -> bool:
        return bool(self._hooks)

    def run_pre(self, method_name: str, data: Any) -> tuple[Any, list[_HookInvocation]]:
        """Run a pre-hook chain and retain state for its matching post chain."""
        invocations: list[_HookInvocation] = []
        current = data
        for hook in self._hooks:
            method = getattr(hook, method_name, None)
            if method is None:
                continue
            output = method(current)
            if isinstance(output, DataHookOutput):
                current = output.data
                state = output.state
            else:
                current = output
                state = None
            invocations.append(_HookInvocation(hook=hook, state=state))
            if current is None:
                break
        return current, invocations

    def run_post(
        self,
        method_name: str,
        data: Any,
        invocations: list[_HookInvocation],
    ) -> Any:
        """Unwind a post-hook chain in reverse pre-hook order."""
        current = data
        for invocation in reversed(invocations):
            method = getattr(invocation.hook, method_name, None)
            if method is not None:
                current = method(current, invocation.state)
        return current

    def close(self) -> None:
        """Close initialized hooks once, in reverse setup order."""
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for hook in reversed(self._hooks):
            close = getattr(hook, "close", None)
            if close is None:
                continue
            try:
                close()
            except BaseException as exc:
                errors.append(exc)
        self._hooks.clear()
        if errors:
            raise BaseExceptionGroup("Data hook cleanup failed", errors)
