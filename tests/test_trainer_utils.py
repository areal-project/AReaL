# SPDX-License-Identifier: Apache-2.0

import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.utils import cleanup
from areal.utils.cleanup import run_batch_cleanups


def test_reclaim_cpu_memory_collects_cycles_before_trimming(monkeypatch):
    """Unreachable owners must be destroyed before malloc_trim sees free pages."""

    class Owner:
        pass

    owner = Owner()
    owner.cycle = owner
    released = weakref.ref(owner)
    del owner

    def trim(pad):
        assert pad == 0
        assert released() is None
        return 1

    malloc_trim = Mock(side_effect=trim)
    monkeypatch.setattr(
        cleanup.ctypes, "CDLL", lambda _: SimpleNamespace(malloc_trim=malloc_trim)
    )

    assert cleanup.reclaim_cpu_memory()
    malloc_trim.assert_called_once_with(0)
    assert malloc_trim.argtypes == [cleanup.ctypes.c_size_t]
    assert malloc_trim.restype is cleanup.ctypes.c_int


@pytest.mark.parametrize("library_error", [False, True])
def test_reclaim_cpu_memory_without_malloc_trim_still_collects(
    monkeypatch, library_error
):
    """Non-glibc platforms keep Python cleanup without requiring a new library."""
    collect = Mock()
    monkeypatch.setattr(cleanup.gc, "collect", collect)

    def load_library(_):
        if library_error:
            raise OSError("unavailable")
        return SimpleNamespace()

    monkeypatch.setattr(cleanup.ctypes, "CDLL", load_library)

    assert not cleanup.reclaim_cpu_memory()
    collect.assert_called_once_with()


def test_run_batch_cleanups_continues_after_one_role_fails():
    calls = []
    actor_error = RuntimeError("actor cleanup failed")

    def fail_actor():
        calls.append("actor")
        raise actor_error

    def record(role):
        return lambda: calls.append(role)

    with pytest.raises(RuntimeError, match="actor cleanup failed") as exc_info:
        run_batch_cleanups(
            [
                ("actor", fail_actor),
                ("critic", record("critic")),
                ("ref", record("ref")),
                ("data", record("data")),
            ]
        )

    assert exc_info.value is actor_error
    assert calls == ["actor", "critic", "ref", "data"]


def test_run_batch_cleanups_preserves_first_of_multiple_failures():
    calls = []

    def fail(role):
        def cleanup():
            calls.append(role)
            raise RuntimeError(f"{role} cleanup failed")

        return cleanup

    with pytest.raises(RuntimeError, match="actor cleanup failed") as exc_info:
        run_batch_cleanups(
            [
                ("actor", fail("actor")),
                ("critic", lambda: calls.append("critic")),
                ("data", fail("data")),
            ]
        )

    assert calls == ["actor", "critic", "data"]
    assert exc_info.value.__notes__ == [
        "clear_batches failed first for role: actor",
        "Additional clear_batches failure for data: RuntimeError: data cleanup failed",
    ]
