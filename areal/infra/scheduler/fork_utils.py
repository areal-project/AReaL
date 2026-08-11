# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from areal.infra.scheduler.exceptions import WorkerCleanupError, WorkerCreationError


class ForkRoleState(Enum):
    """Explicit lifecycle state for a scheduler-owned fork role."""

    RESERVING = "reserving"
    ACTIVE = "active"
    CLEANUP_PENDING = "cleanup_pending"


@dataclass
class ForkOwnership:
    """Owner and reservation keys retained until formal cleanup succeeds."""

    owner_role: str
    reservation_indices: set[int] = field(default_factory=set)
    state: ForkRoleState = ForkRoleState.RESERVING


def ensure_fork_role_available(
    workers_by_role: dict[str, list[Any]],
    colocated_roles: dict[str, str],
    ownership_by_role: dict[str, ForkOwnership],
    role: str,
) -> None:
    """Reject creation when a role already exists or still owns resources."""
    ownership = ownership_by_role.get(role)
    if ownership is not None:
        raise WorkerCreationError(
            role,
            "Fork role still owns resources",
            f"State is {ownership.state.value}; call delete_workers({role!r}) first",
        )
    if role in workers_by_role or role in colocated_roles:
        raise WorkerCreationError(
            role,
            "Worker group already exists",
            f"Use delete_workers({role!r}) first to remove existing workers",
        )


def ensure_fork_role_queryable(
    workers_by_role: dict[str, list[Any]],
    ownership_by_role: dict[str, ForkOwnership],
    role: str,
) -> None:
    """Prevent pending fork ownership from masquerading as colocation."""
    ownership = ownership_by_role.get(role)
    if ownership is None:
        return
    if ownership.state is ForkRoleState.ACTIVE and role in workers_by_role:
        return
    raise WorkerCleanupError(
        role,
        [
            f"fork role is {ownership.state.value}; "
            f"call delete_workers({role!r}) before querying or recreating it"
        ],
    )


def ensure_fork_target_queryable(
    workers_by_role: dict[str, list[Any]],
    colocated_roles: dict[str, str],
    ownership_by_role: dict[str, ForkOwnership],
    role: str,
) -> None:
    """Reject a pending role anywhere in a colocation owner chain."""
    current = role
    seen: set[str] = set()
    while True:
        ensure_fork_role_queryable(workers_by_role, ownership_by_role, current)
        if current in workers_by_role:
            return
        if current in seen:
            return
        seen.add(current)
        parent = colocated_roles.get(current)
        if parent is None:
            return
        current = parent


def fork_reservation_indices(
    ownership_by_role: dict[str, ForkOwnership], role: str
) -> set[int]:
    """Return a snapshot of reservation keys owned by one role."""
    ownership = ownership_by_role.get(role)
    return set() if ownership is None else set(ownership.reservation_indices)


def mark_fork_role_state(
    ownership_by_role: dict[str, ForkOwnership],
    role: str,
    state: ForkRoleState,
) -> None:
    """Transition an existing fork ownership record."""
    ownership = ownership_by_role.get(role)
    if ownership is not None:
        ownership.state = state


async def reserve_fork_ports(
    session: Any,
    guard_url: str,
    role: str,
    worker_index: int,
    count: int,
    *,
    attempts: int = 2,
) -> tuple[str, list[int]]:
    """Reserve an idempotent fixed port group, retrying a lost response.

    The Guard keys reservations by ``(role, worker_index)``. Reissuing the
    same request therefore recovers the original reservation when the first
    response was lost after commit.
    """
    error: BaseException | None = None
    for _ in range(attempts):
        try:
            async with session.post(
                f"{guard_url}/reserve_worker_ports",
                json={
                    "role": role,
                    "worker_index": worker_index,
                    "count": count,
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {error_text}")
                result = await response.json()

            host = result.get("host")
            ports = result.get("ports")
            if not isinstance(host, str) or not host:
                raise RuntimeError(f"invalid reservation host {host!r}")
            if (
                not isinstance(ports, list)
                or len(ports) != count
                or any(not isinstance(port, int) for port in ports)
            ):
                raise RuntimeError(f"invalid reservation ports {ports!r}")
            return host, ports
        except Exception as exc:  # noqa: BLE001
            error = exc

    assert error is not None
    raise error


def retain_fork_reservation(
    ownership_by_role: dict[str, ForkOwnership],
    colocated_roles: dict[str, str],
    fork_parent_roles: dict[str, str],
    role: str,
    target_role: str,
    worker_index: int,
) -> None:
    """Record reservation ownership before contacting the owner Guard."""
    ownership = ownership_by_role.get(role)
    if ownership is None:
        ownership = ForkOwnership(owner_role=target_role)
        ownership_by_role[role] = ownership
    elif ownership.owner_role != target_role:
        raise WorkerCreationError(
            role,
            "Fork ownership conflict",
            f"Role is owned by {ownership.owner_role!r}, not {target_role!r}",
        )
    ownership.reservation_indices.add(worker_index)
    colocated_roles[role] = target_role
    fork_parent_roles[role] = target_role


def discard_fork_reservation(
    ownership_by_role: dict[str, ForkOwnership],
    role: str,
    worker_index: int,
) -> None:
    """Forget one reservation only after Guard cleanup is confirmed."""
    ownership = ownership_by_role.get(role)
    if ownership is None:
        return
    ownership.reservation_indices.discard(worker_index)
    if not ownership.reservation_indices:
        ownership_by_role.pop(role, None)


async def release_fork_reservation(
    session: Any,
    guard_url: str,
    role: str,
    worker_index: int,
) -> None:
    """Idempotently remove a forked child and its fixed port reservation."""
    async with session.post(
        f"{guard_url}/kill_forked_worker",
        json={
            "role": role,
            "worker_index": worker_index,
            "release_ports": True,
        },
    ) as response:
        if response.status != 200:
            error_text = await response.text()
            raise RuntimeError(f"HTTP {response.status}: {error_text}")
        result = await response.json()
        if result.get("status") != "success":
            raise RuntimeError(result.get("error", "Unknown cleanup error"))


async def reconcile_fork_response(
    session: Any,
    guard_url: str,
    role: str,
    worker_index: int,
    allocated_ports: list[int],
) -> int | None:
    """Resolve an uncertain ``/fork`` outcome against the owner Guard.

    A live child with the exact reserved ports is accepted. If the Guard
    confirms that no child is alive, the fixed reservation is released and
    ``None`` is returned. Any transport or protocol error is propagated so the
    scheduler keeps its provisional ownership metadata for a later retry.
    """
    async with session.post(
        f"{guard_url}/forked_worker_status",
        json={"role": role, "worker_index": worker_index},
    ) as response:
        if response.status != 200:
            error_text = await response.text()
            raise RuntimeError(f"HTTP {response.status}: {error_text}")
        result = await response.json()
        if result.get("status") != "success":
            raise RuntimeError(result.get("error", "Unknown fork status error"))

    if result.get("alive"):
        actual_ports = result.get("ports")
        if actual_ports != allocated_ports:
            raise RuntimeError(
                f"forked worker owns unexpected ports {actual_ports}; "
                f"expected {allocated_ports}"
            )
        pid = result.get("pid")
        if not isinstance(pid, int):
            raise RuntimeError(f"forked worker returned invalid pid {pid!r}")
        return pid

    await release_fork_reservation(session, guard_url, role, worker_index)
    return None


def discard_provisional_worker(
    workers_by_role: dict[str, list[Any]],
    ownership_by_role: dict[str, ForkOwnership],
    colocated_roles: dict[str, str],
    fork_parent_roles: dict[str, str],
    role: str,
    worker_id: str,
) -> None:
    """Forget one provisional key only after Guard cleanup is confirmed."""
    retained = [
        worker
        for worker in workers_by_role.get(role, [])
        if worker.worker.id != worker_id
    ]
    if retained:
        workers_by_role[role] = retained
        return
    workers_by_role.pop(role, None)
    if role not in ownership_by_role:
        colocated_roles.pop(role, None)
        fork_parent_roles.pop(role, None)
