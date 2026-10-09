# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable


def run_batch_cleanups(
    cleanups: Iterable[tuple[str, Callable[[], None]]],
) -> None:
    """Run every batch cleanup before propagating any ordinary failures."""
    failures: list[tuple[str, Exception]] = []
    for role, cleanup in cleanups:
        try:
            cleanup()
        except Exception as error:
            failures.append((role, error))

    if failures:
        first_role, first_error = failures[0]
        first_error.add_note(f"clear_batches failed first for role: {first_role}")
        for role, error in failures[1:]:
            first_error.add_note(
                "Additional clear_batches failure for "
                f"{role}: {type(error).__name__}: {error}"
            )
        raise first_error
