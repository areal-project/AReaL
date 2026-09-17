# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.utils.cleanup import run_batch_cleanups


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
