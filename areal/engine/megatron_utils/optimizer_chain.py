# SPDX-License-Identifier: Apache-2.0

"""Utilities for traversing nested Megatron optimizer chains."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
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
        # Older/custom ChainedOptimizer variants use this spelling.
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
    _cycle_node_types: list[str] | None = None,
) -> Iterator[Any]:
    """Yield unique lifecycle leaves in stable depth-first, left-to-right order.

    Both chain nodes and leaves are deduplicated by object identity. Cycles are
    detected using the active DFS path, logged when a logger is supplied, and
    skipped without recursively entering the repeated node.
    """
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
            if _cycle_node_types is not None:
                _cycle_node_types.append(node_type)
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


def get_megatron_optimizer_chain_children(
    optimizer: Any,
) -> tuple[Any, ...] | None:
    """Return one chain descriptor read using the shared single-read rules."""
    return _chain_children(optimizer)


def iter_megatron_optimizer_leaf_paths(
    optimizer: Any | None,
) -> Iterator[tuple[tuple[int, ...], Any]]:
    """Yield canonical tree paths without flattening nested optimizer chains.

    Checkpoint schemas require a tree, not lifecycle's identity-deduplicated
    traversal. Shared nodes and cycles are ambiguous and therefore rejected.
    """
    if optimizer is None:
        return

    active: set[int] = set()
    seen: dict[int, tuple[int, ...]] = {}

    def visit(
        node: Any, path: tuple[int, ...]
    ) -> Iterator[tuple[tuple[int, ...], Any]]:
        if node is None:
            raise ValueError(f"optimizer checkpoint tree contains None at path {path}")
        node_identity = id(node)
        if node_identity in active:
            raise ValueError(
                f"optimizer checkpoint tree contains a cycle at path {path}"
            )
        if node_identity in seen:
            raise ValueError(
                "optimizer checkpoint tree contains a shared node at path "
                f"{path}; first seen at {seen[node_identity]}"
            )
        seen[node_identity] = path
        children = _chain_children(node)
        if children is None:
            yield path, node
            return
        active.add(node_identity)
        try:
            for child_index, child in enumerate(children):
                yield from visit(child, (*path, child_index))
        finally:
            active.remove(node_identity)

    yield from visit(optimizer, ())


def get_managed_base_optimizer(optimizer: Any) -> Any | None:
    """Return a leaf's base optimizer when it owns CPU state residency."""
    base_optimizer = getattr(optimizer, "optimizer", optimizer)
    if base_optimizer is not None and getattr(
        base_optimizer, "manages_cpu_residency", False
    ):
        return base_optimizer
    return None


@contextmanager
def checkpoint_awex_residency(
    adapter: Any,
    optimizer: Any | None,
    *,
    with_model: bool,
    with_optimizer: bool,
) -> Iterator[None]:
    """Lease only AWEX resources required by one checkpoint operation.

    The original released-tag set is restored on exit. A chain containing only
    managed leaves never resumes the optimizer tag because its CPU slab is
    already the authoritative checkpoint source.
    """
    recovery = getattr(adapter, "_optimizer_rollback_recovery", None)
    if recovery is not None:
        raise RuntimeError(
            "AWEX optimizer lifecycle has unresolved rollback recovery; "
            "checkpoint is unsafe"
        )
    released_tags = getattr(adapter, "_released_tags", None)
    if not isinstance(released_tags, set):
        raise TypeError("AWEX checkpoint residency requires a released-tag set")
    original_tags = frozenset(released_tags)
    leaves = tuple(iter_megatron_optimizer_leaves(optimizer))
    all_managed = bool(leaves) and all(
        get_managed_base_optimizer(leaf) is not None for leaf in leaves
    )
    leased_tags: list[str] = []
    if with_model and "weights" in original_tags:
        leased_tags.append("weights")
    if with_optimizer and "optimizer" in original_tags and not all_managed:
        leased_tags.append("optimizer")

    if leased_tags:
        adapter.resume_memory(tags=leased_tags)
    try:
        yield
    except BaseException as original:
        if leased_tags:
            try:
                adapter.release_memory(tags=leased_tags)
            except BaseException as restore_error:
                original.add_note(
                    f"AWEX checkpoint residency restore failed: {restore_error!r}"
                )
        raise
    else:
        if leased_tags:
            adapter.release_memory(tags=leased_tags)
    if frozenset(released_tags) != original_tags:
        raise RuntimeError(
            "AWEX checkpoint residency did not restore the original released tags"
        )


class OptimizerLeafKind(Enum):
    """Lifecycle strategy fixed for one AWEX release/resume cycle."""

    MANAGED = auto()
    HDO = auto()
    ORDINARY = auto()


@dataclass(frozen=True)
class OptimizerLeafLifecycle:
    """Strong-reference snapshot of a leaf and its release-time capabilities."""

    leaf: Any
    kind: OptimizerLeafKind
    base_optimizer: Any | None = None
    managed_optimizer: Any | None = None
    offload_to_cpu: Callable[[], Any] | None = None
    restore_from_cpu: Callable[[], Any] | None = None


@dataclass
class OptimizerUndoAction:
    """One idempotent, fine-grained inverse of a completed release mutation."""

    restore: Callable[[], Any]
    description: str


@dataclass
class OptimizerEntryJournal:
    """Per-entry release undo data and retryable restore progress."""

    actions: list[OptimizerUndoAction]
    release_completed: bool = False
    restore_completed: bool = False
    hdo_release_attempted: bool = False


@dataclass
class OptimizerLifecyclePlan:
    """Release-time snapshot plus mutable rollback/resume progress journals."""

    entries: tuple[OptimizerLeafLifecycle, ...]
    cycle_node_types: tuple[str, ...]
    journals: list[OptimizerEntryJournal]
    resume_index: int = 0

    def has_pending_recovery(self) -> bool:
        """Whether rollback still owes any ordinary or HDO restore."""
        return any(
            journal.actions or journal.hdo_release_attempted
            for journal in self.journals
        )

    def has_pending_ordinary_undo(self) -> bool:
        """Compatibility predicate used by v2 entry-journal pruning."""
        return any(journal.actions for journal in self.journals)


def classify_optimizer_leaves(
    optimizer: Any | None,
    *,
    use_hdo_lifecycle: bool,
    logger: _WarningLogger | None = None,
) -> OptimizerLifecyclePlan:
    """Materialize and classify all leaves before lifecycle mutation begins."""
    cycle_node_types: list[str] = []
    leaves = tuple(
        iter_megatron_optimizer_leaves(
            optimizer,
            logger=logger,
            _cycle_node_types=cycle_node_types,
        )
    )
    classified: list[OptimizerLeafLifecycle] = []
    for leaf in leaves:
        base_optimizer = getattr(leaf, "optimizer", leaf)
        managed_optimizer = None
        if base_optimizer is not None and getattr(
            base_optimizer, "manages_cpu_residency", False
        ):
            managed_optimizer = base_optimizer
        if managed_optimizer is not None:
            classified.append(
                OptimizerLeafLifecycle(
                    leaf=leaf,
                    kind=OptimizerLeafKind.MANAGED,
                    base_optimizer=base_optimizer,
                    managed_optimizer=managed_optimizer,
                )
            )
            continue

        offload_to_cpu = None
        restore_from_cpu = None
        if use_hdo_lifecycle:
            offload_to_cpu = getattr(leaf, "offload_to_cpu", None)
            restore_from_cpu = getattr(leaf, "restore_from_cpu", None)
        if callable(offload_to_cpu) and callable(restore_from_cpu):
            classified.append(
                OptimizerLeafLifecycle(
                    leaf=leaf,
                    kind=OptimizerLeafKind.HDO,
                    base_optimizer=base_optimizer,
                    offload_to_cpu=offload_to_cpu,
                    restore_from_cpu=restore_from_cpu,
                )
            )
        else:
            classified.append(
                OptimizerLeafLifecycle(
                    leaf=leaf,
                    kind=OptimizerLeafKind.ORDINARY,
                    base_optimizer=base_optimizer,
                )
            )
    return OptimizerLifecyclePlan(
        entries=tuple(classified),
        cycle_node_types=tuple(cycle_node_types),
        journals=[OptimizerEntryJournal(actions=[]) for _ in classified],
    )


def _report_rollback_errors(
    original: BaseException,
    rollback_errors: list[tuple[str, BaseException]],
    logger: Any | None,
) -> None:
    for description, error in rollback_errors:
        note = f"AWEX rollback failed for {description}: {error!r}"
        original.add_note(note)
        if logger is not None:
            try:
                logger.error(
                    note,
                    exc_info=(type(error), error, error.__traceback__),
                )
            except BaseException as logging_error:
                original.add_note(
                    f"AWEX failed to log rollback error: {logging_error!r}"
                )


def _restore_ordinary_journal(
    journal: OptimizerEntryJournal,
) -> list[tuple[str, BaseException]]:
    errors: list[tuple[str, BaseException]] = []
    for action in tuple(reversed(journal.actions)):
        try:
            action.restore()
        except BaseException as error:
            errors.append((action.description, error))
        else:
            journal.actions.remove(action)
    return errors


def rollback_optimizer_lifecycle(
    plan: OptimizerLifecyclePlan,
    original: BaseException,
    *,
    logger: Any | None = None,
) -> None:
    """Best-effort rollback that never lets one undo block earlier entries."""
    rollback_errors: list[tuple[str, BaseException]] = []
    last_attempted = max(
        (
            index
            for index, journal in enumerate(plan.journals)
            if journal.release_completed
            or journal.hdo_release_attempted
            or journal.actions
        ),
        default=-1,
    )
    for index in range(last_attempted, -1, -1):
        entry = plan.entries[index]
        journal = plan.journals[index]
        if entry.kind is OptimizerLeafKind.HDO and journal.hdo_release_attempted:
            try:
                entry.restore_from_cpu()
            except BaseException as error:
                rollback_errors.append((f"HDO leaf {index}", error))
            else:
                journal.hdo_release_attempted = False
        elif entry.kind is OptimizerLeafKind.ORDINARY:
            rollback_errors.extend(_restore_ordinary_journal(journal))
        journal.release_completed = False
        journal.restore_completed = (
            not journal.actions and not journal.hdo_release_attempted
        )
    _report_rollback_errors(original, rollback_errors, logger)


def retry_optimizer_recovery(
    plan: OptimizerLifecyclePlan,
    *,
    logger: Any | None = None,
) -> None:
    """Retry only rollback work that remains pending, committing each success."""
    recovery_errors: list[tuple[str, BaseException]] = []
    for index in range(len(plan.entries) - 1, -1, -1):
        entry = plan.entries[index]
        journal = plan.journals[index]
        if entry.kind is OptimizerLeafKind.HDO and journal.hdo_release_attempted:
            try:
                entry.restore_from_cpu()
            except BaseException as error:
                recovery_errors.append((f"HDO leaf {index}", error))
            else:
                journal.hdo_release_attempted = False
        elif entry.kind is OptimizerLeafKind.ORDINARY and journal.actions:
            recovery_errors.extend(_restore_ordinary_journal(journal))
        journal.restore_completed = (
            not journal.actions and not journal.hdo_release_attempted
        )

    if recovery_errors:
        description, primary = recovery_errors[0]
        primary.add_note(f"AWEX recovery retry failed for {description}")
        _report_rollback_errors(primary, recovery_errors[1:], logger)
        raise primary


def release_optimizer_lifecycle(
    plan: OptimizerLifecyclePlan,
    release_ordinary: Callable[
        [int, OptimizerLeafLifecycle, OptimizerEntryJournal], None
    ],
    *,
    logger: Any | None = None,
) -> None:
    """Release all entries, rolling back granular journals on any failure."""
    for index, (entry, journal) in enumerate(zip(plan.entries, plan.journals)):
        try:
            if entry.kind is OptimizerLeafKind.MANAGED:
                entry.managed_optimizer.drain()
            elif entry.kind is OptimizerLeafKind.HDO:
                journal.hdo_release_attempted = True
                entry.offload_to_cpu()
            else:
                release_ordinary(index, entry, journal)
        except BaseException as original:
            rollback_optimizer_lifecycle(plan, original, logger=logger)
            raise
        journal.release_completed = True


def resume_optimizer_lifecycle(
    plan: OptimizerLifecyclePlan,
    *,
    logger: Any | None = None,
) -> None:
    """Restore only the unfinished suffix, committing each successful entry."""
    del logger
    while plan.resume_index < len(plan.entries):
        index = plan.resume_index
        entry = plan.entries[index]
        journal = plan.journals[index]
        if entry.kind is OptimizerLeafKind.MANAGED:
            pass
        elif entry.kind is OptimizerLeafKind.HDO:
            entry.restore_from_cpu()
            journal.hdo_release_attempted = False
        else:
            errors = _restore_ordinary_journal(journal)
            if errors:
                description, error = errors[0]
                for extra_description, extra_error in errors[1:]:
                    error.add_note(
                        f"Additional AWEX restore failure for "
                        f"{extra_description}: {extra_error!r}"
                    )
                error.add_note(f"AWEX ordinary restore failed for {description}")
                raise error
        journal.restore_completed = True
        plan.resume_index += 1
