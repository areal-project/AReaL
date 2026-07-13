# SPDX-License-Identifier: Apache-2.0

"""Small rollback helper for process-local multi-index publication."""

from __future__ import annotations

_MISSING = object()


def _atomic_publish(
    *,
    mapping_writes: tuple[tuple[dict, object, object], ...],
    sequence_appends: tuple[tuple[dict, object, object], ...] = (),
) -> None:
    """Publish mapping writes and indexed list appends as one local unit.

    The helper deliberately catches ``BaseException`` because cancellation,
    ``KeyboardInterrupt``, and allocation failures must not leave only some
    indexes visible.  Store indexes are private built-in dictionaries and
    lists.  Rollback calls the built-in mutation methods directly so a fault-
    injection subclass cannot interrupt both publication and its own undo.
    """

    undo: list[tuple[str, object, object, object]] = []
    try:
        for mapping, key, value in mapping_writes:
            previous = mapping.get(key, _MISSING)
            undo.append(("mapping", mapping, key, previous))
            mapping[key] = value
        for mapping, key, value in sequence_appends:
            sequence = mapping.get(key, _MISSING)
            if sequence is _MISSING:
                undo.append(("mapping", mapping, key, _MISSING))
                mapping[key] = [value]
                continue
            if type(sequence) is not list:
                raise TypeError("indexed publication sequence must be a list")
            undo.append(("sequence", sequence, len(sequence), _MISSING))
            sequence.append(value)
    except BaseException:
        for kind, target, key_or_length, previous in reversed(undo):
            try:
                if kind == "sequence":
                    list.__delitem__(target, slice(key_or_length, None))
                elif previous is _MISSING:
                    dict.pop(target, key_or_length, None)
                else:
                    dict.__setitem__(target, key_or_length, previous)
            except BaseException:
                # A second process-level interruption can still make rollback
                # impossible.  Continue undoing the other independent indexes
                # and preserve the original publication exception.
                pass
        raise
