# SPDX-License-Identifier: Apache-2.0

"""Deterministic constraint merge shared by in-memory and disk-backed ranking."""

from collections import Counter
from collections.abc import Callable, Iterable

from .types import InsufficientDataError, RankedCandidate, SelectionPolicy, positive_int


def select_ranked_candidates(
    policy: SelectionPolicy,
    target_count: int,
    ordered: Callable[[str | None], Iterable[RankedCandidate]],
) -> tuple[RankedCandidate, ...]:
    """Merge globally ordered rows, applying group minima before global fill.

    ordered(None) supplies all eligible candidates and ordered(group) that group.
    Providers must order by (priority, tie_breaker, task_id). This is an explicit
    greedy policy, not a solver for arbitrary overlapping constraints.
    """
    positive_int(target_count, "target_count")
    if sum(policy.minimum_per_group.values()) > target_count:
        raise ValueError("Selection group minima exceed target_count")
    selected: list[RankedCandidate] = []
    ids: set[str] = set()
    buckets: Counter[str] = Counter()

    def admit(candidate: RankedCandidate) -> bool:
        if candidate.task_id in ids:
            return False
        cap = policy.maximum_per_bucket.get(candidate.bucket)
        if cap is not None and buckets[candidate.bucket] >= cap:
            return False
        selected.append(candidate)
        ids.add(candidate.task_id)
        buckets[candidate.bucket] += 1
        return True

    for group, minimum in policy.minimum_per_group.items():
        admitted = 0
        if minimum:
            for candidate in ordered(group):
                admitted += admit(candidate)
                if admitted == minimum:
                    break
        if admitted < minimum:
            raise InsufficientDataError(f"Too few eligible {group} cases")
    if len(selected) < target_count:
        for candidate in ordered(None):
            admit(candidate)
            if len(selected) == target_count:
                break
    if len(selected) != target_count:
        raise InsufficientDataError("Too few diverse cases under the bucket caps")
    return tuple(selected)
