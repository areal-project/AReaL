# SPDX-License-Identifier: Apache-2.0
"""Context-local scalar capture; callers own selection and publication policy."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

Metrics = dict[str, list[float]]
_CAPTURE: ContextVar[tuple[Metrics, Callable[[str], bool]] | None] = ContextVar(
    "scalar_capture", default=None
)


def capture_scalar(key: str, value: float) -> bool:
    """Return whether the active context accepted this fully qualified scalar."""
    capture = _CAPTURE.get()
    if capture is None:
        return False
    metrics, select = capture
    if not select(key):
        return False
    metrics.setdefault(key, []).append(float(value))
    return True


@contextmanager
def capture_scalars(*, select: Callable[[str], bool]) -> Iterator[Metrics]:
    """Capture selected scalars instead of publishing them to their tracker.

    Nested contexts restore the parent on exit, including on exceptions. Async
    children inherit their parent's buffer; concurrent contexts stay isolated.
    Scalars with an explicit reduction group bypass capture in the tracker.
    The selector must be a pure key predicate: it runs under the tracker lock.
    Callers must await child work before transferring the captured buffer.
    """
    metrics: Metrics = {}
    token = _CAPTURE.set((metrics, select))
    try:
        yield metrics
    finally:
        _CAPTURE.reset(token)


@contextmanager
def suspend_scalar_capture() -> Iterator[None]:
    """Publish already captured observations without capturing them again."""
    token = _CAPTURE.set(None)
    try:
        yield
    finally:
        _CAPTURE.reset(token)
