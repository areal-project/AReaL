# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from areal.infra.rpc.rtensor import RTensor
from areal.utils.seqpack import balanced_greedy_partition, ffd_allocate, kk_allocate


@dataclass(slots=True)
class AllocationInput:
    """DP allocation request for trajectory-like items.

    ``group_size`` binds adjacent items into an atomic allocation unit before
    cost-based algorithms run. ``capacity`` is used by capacity-style algorithms
    (``ffd`` and ``kk``); controller dispatch uses ``ffd_equal`` to preserve
    equal item counts across DP groups.
    """

    items: list[dict[str, Any]]
    n_groups: int
    algorithm: str
    group_size: int = 1
    capacity: int = int(1e12)


@dataclass(slots=True)
class AllocationOutput:
    """DP allocation result for trajectory-like items.

    ``group_indices`` always index ``items`` on this output object. DTA may
    normalize grouped trajectories into sequence-level items before allocating.
    """

    items: list[dict[str, Any]]
    group_indices: list[list[int]]
    metrics: Any | None = None


@dataclass(slots=True)
class _AtomicUnit:
    indices: list[int]
    cost: int


def _item_weight(item: dict[str, Any]) -> int:
    attn_mask = item.get("attention_mask")
    if isinstance(attn_mask, torch.Tensor):
        return int(attn_mask.sum().item())
    if isinstance(attn_mask, RTensor):
        return attn_mask.data.numel()
    for value in item.values():
        if isinstance(value, RTensor):
            return value.data.numel()
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            return value.numel()
    return 1


def _contains_rtensor(obj: Any) -> bool:
    if isinstance(obj, RTensor):
        return True
    if isinstance(obj, dict):
        return any(_contains_rtensor(value) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_rtensor(item) for item in obj)
    return False


def _make_atomic_units(
    items: list[dict[str, Any]], group_size: int
) -> list[_AtomicUnit]:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    if len(items) % group_size != 0:
        raise ValueError(
            f"item count ({len(items)}) must be divisible by group_size ({group_size})"
        )

    units: list[_AtomicUnit] = []
    for group_start in range(0, len(items), group_size):
        indices = list(range(group_start, group_start + group_size))
        cost = sum(_item_weight(items[idx]) for idx in indices)
        units.append(_AtomicUnit(indices=indices, cost=cost))
    return units


def _ffd_allocate(req: AllocationInput) -> AllocationOutput:
    units = _make_atomic_units(req.items, req.group_size)
    costs = [unit.cost for unit in units]
    unit_groups = ffd_allocate(costs, capacity=req.capacity, min_groups=req.n_groups)
    group_indices = [
        [idx for unit_idx in unit_group for idx in units[unit_idx].indices]
        for unit_group in unit_groups
    ]
    return AllocationOutput(items=req.items, group_indices=group_indices)


def _kk_allocate(req: AllocationInput) -> AllocationOutput:
    units = _make_atomic_units(req.items, req.group_size)
    costs = [unit.cost for unit in units]
    unit_groups = kk_allocate(costs, capacity=req.capacity, min_groups=req.n_groups)
    group_indices = [
        [idx for unit_idx in unit_group for idx in units[unit_idx].indices]
        for unit_group in unit_groups
    ]
    return AllocationOutput(items=req.items, group_indices=group_indices)


def _ffd_equal_allocate(req: AllocationInput) -> AllocationOutput:
    units = _make_atomic_units(req.items, req.group_size)
    costs = [unit.cost for unit in units]
    unit_groups = balanced_greedy_partition(costs, K=req.n_groups)
    group_indices = [
        [idx for unit_idx in unit_group for idx in units[unit_idx].indices]
        for unit_group in unit_groups
    ]
    return AllocationOutput(items=req.items, group_indices=group_indices)


def _dta_allocate(req: AllocationInput) -> AllocationOutput:
    if req.group_size != 1:
        raise ValueError(
            "packing_algorithm='dta' is incompatible with group_size > 1. "
            "DTA requires sequence-level independence."
        )
    from areal.experimental.dta.allocation import allocate_dta_trajectories

    items = req.items
    if _contains_rtensor(items):
        # TODO(agent): This controller-side localization can become a bottleneck.
        items = RTensor.localize(items)

    dta_allocation = allocate_dta_trajectories(items, n_groups=req.n_groups)
    return AllocationOutput(
        items=dta_allocation.items,
        group_indices=dta_allocation.group_indices,
        metrics=dta_allocation.metrics,
    )


class _TrajectoryAllocateFn(Protocol):
    def __call__(self, req: AllocationInput) -> AllocationOutput: ...


_TRAJECTORY_ALLOCATE_FNS: dict[str, _TrajectoryAllocateFn] = {
    "ffd": _ffd_allocate,
    "kk": _kk_allocate,
    "ffd_equal": _ffd_equal_allocate,
    "dta": _dta_allocate,
}


def get_dp_allocate_fn(algorithm: str) -> _TrajectoryAllocateFn:
    """Return the DP allocation adapter for a rollout packing algorithm."""
    try:
        return _TRAJECTORY_ALLOCATE_FNS[algorithm]
    except KeyError as err:
        raise ValueError(
            f"Unknown trajectory packing algorithm '{algorithm}'. "
            f"Supported algorithms: {sorted(_TRAJECTORY_ALLOCATE_FNS)}"
        ) from err


def allocate_trajectories(req: AllocationInput) -> AllocationOutput:
    """Allocate trajectory-like items across data-parallel groups.

    ``group_indices`` in the returned object always index ``AllocationOutput.items``.
    """
    return get_dp_allocate_fn(req.algorithm)(req)
