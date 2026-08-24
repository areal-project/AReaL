# SPDX-License-Identifier: Apache-2.0

"""Utilities for traversing Megatron optimizer residency owners."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from inspect import getattr_static
from typing import Any, Protocol

_MISSING = object()


class _WarningLogger(Protocol):
    def warning(self, message: str, *args: Any) -> Any: ...


def _getattr_descriptor_once(optimizer: Any, attribute: str) -> Any:
    """Read a dynamic descriptor once without hiding descriptor AttributeError."""
    if getattr_static(optimizer, attribute, _MISSING) is _MISSING:
        return _MISSING
    return getattr(optimizer, attribute)


def _chain_children(optimizer: Any) -> tuple[Any, ...] | None:
    """Return a chain node's children, or ``None`` for a lifecycle leaf."""
    attribute = "chained_optimizers"
    children = _getattr_descriptor_once(optimizer, attribute)
    if children is _MISSING:
        attribute = "optimizers"
        children = _getattr_descriptor_once(optimizer, attribute)
    if children is _MISSING:
        return None
    try:
        return tuple(children)
    except TypeError as exc:
        raise TypeError(
            f"optimizer chain attribute {attribute!r} on "
            f"{type(optimizer).__name__} must be iterable"
        ) from exc


def iter_megatron_optimizer_leaves(
    optimizer: Any | None,
    *,
    logger: _WarningLogger | None = None,
) -> Iterator[Any]:
    """Yield unique lifecycle leaves in stable depth-first order."""
    if optimizer is None:
        return

    active_chains: set[int] = set()
    expanded_chains: set[int] = set()
    yielded_leaves: set[int] = set()
    stack: list[tuple[bool, Any]] = [(False, optimizer)]
    while stack:
        exiting, node = stack.pop()
        node_id = id(node)
        if exiting:
            active_chains.remove(node_id)
            expanded_chains.add(node_id)
            continue
        if node is None:
            continue

        children = _chain_children(node)
        if children is None:
            if node_id not in yielded_leaves:
                yielded_leaves.add(node_id)
                yield node
            continue
        if node_id in active_chains:
            node_type = type(node).__name__
            if logger is not None:
                logger.warning(
                    "Detected optimizer chain cycle at %s; skipping repeated node",
                    node_type,
                )
            continue
        if node_id in expanded_chains:
            continue

        active_chains.add(node_id)
        stack.append((True, node))
        stack.extend((False, child) for child in reversed(children))


def get_managed_base_optimizer(optimizer: Any) -> Any | None:
    """Return a leaf's base optimizer when it owns CPU state residency."""
    base_optimizer = getattr(optimizer, "optimizer", optimizer)
    if base_optimizer is not None and getattr(
        base_optimizer, "manages_cpu_residency", False
    ):
        return base_optimizer
    return None


@dataclass(frozen=True)
class OptimizerResidencyEntry:
    """One stable optimizer leaf and its optional managed residency owner."""

    leaf: Any
    base_optimizer: Any | None
    managed_optimizer: Any | None


@dataclass(frozen=True)
class OptimizerResidencyPlan:
    """Stable leaves used by one successful AWEX release/resume cycle."""

    entries: tuple[OptimizerResidencyEntry, ...]

    @property
    def has_ordinary_optimizers(self) -> bool:
        return any(entry.managed_optimizer is None for entry in self.entries)


def build_optimizer_residency_plan(
    optimizer: Any | None,
    *,
    logger: _WarningLogger | None = None,
) -> OptimizerResidencyPlan:
    """Classify leaves as managed staged state or ordinary GPU state."""
    entries: list[OptimizerResidencyEntry] = []
    for leaf in iter_megatron_optimizer_leaves(optimizer, logger=logger):
        base_optimizer = getattr(leaf, "optimizer", leaf)
        entries.append(
            OptimizerResidencyEntry(
                leaf=leaf,
                base_optimizer=base_optimizer,
                managed_optimizer=get_managed_base_optimizer(leaf),
            )
        )
    return OptimizerResidencyPlan(tuple(entries))


@contextmanager
def checkpoint_awex_residency(
    adapter: Any,
    optimizer: Any | None,
    *,
    with_model: bool,
    with_optimizer: bool,
) -> Iterator[None]:
    """Temporarily restore model weights required by one checkpoint operation."""
    released_tags = getattr(adapter, "_released_tags", None)
    if not isinstance(released_tags, set):
        raise TypeError("AWEX checkpoint residency requires a released-tag set")
    plan = build_optimizer_residency_plan(optimizer)
    leased_tags: list[str] = []
    if with_model and "weights" in released_tags:
        leased_tags.append("weights")
    if with_optimizer and "optimizer" in released_tags and plan.has_ordinary_optimizers:
        # Managed slabs are already the authoritative checkpoint source.
        # Ordinary optimizer state must return to its normal GPU layout before
        # MCore constructs and loads its sharded checkpoint state.
        leased_tags.append("optimizer")
    if leased_tags:
        adapter.resume_memory(tags=leased_tags)
    yield
    if leased_tags:
        adapter.release_memory(tags=leased_tags)
