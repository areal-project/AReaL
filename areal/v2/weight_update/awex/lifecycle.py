# SPDX-License-Identifier: Apache-2.0

"""Fence replaceable communication groups inside one long-lived worker."""

from __future__ import annotations


class WeightUpdateLifecycle:
    """Called on the execution thread; retired pair names cannot be revived.

    A pair name identifies one incarnation. State survives adapter replacement.
    Unfenced direct callers retain the legacy endpoint behavior.
    """

    def __init__(self) -> None:
        self.active_pair: str | None = None
        self.ready = False
        self.fenced = False
        self.retired: set[str] = set()

    def begin(self, pair_name: str, *, fenced: bool = False) -> bool:
        if pair_name in self.retired:
            raise RuntimeError("Weight update pair has already been retired")
        if self.active_pair is not None:
            if self.active_pair == pair_name and self.ready:
                return False
            raise RuntimeError("Existing weight update pair requires cleanup")
        self.active_pair = pair_name
        self.fenced = fenced
        self.ready = False
        return True

    def connected(self) -> None:
        self.ready = True

    def require(self, pair_name: str | None) -> None:
        if pair_name is None and not self.fenced:
            return
        if not self.ready or pair_name != self.active_pair:
            raise RuntimeError("Weight update request belongs to an inactive pair")

    def begin_teardown(self, pair_name: str | None) -> bool:
        target = pair_name or self.active_pair
        if target is None or target in self.retired:
            return False
        if self.fenced and pair_name is None:
            raise RuntimeError("Weight update teardown requires its pair identity")
        if self.active_pair is not None and target != self.active_pair:
            self.retired.add(target)
            raise RuntimeError("Cannot tear down a different weight update pair")
        if self.active_pair is None:
            self.retired.add(target)
            return False
        self.ready = False
        return True

    def closed(self) -> None:
        if self.active_pair is not None:
            self.retired.add(self.active_pair)
        self.active_pair = None
        self.ready = False
        self.fenced = False
