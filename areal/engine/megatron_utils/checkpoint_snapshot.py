# SPDX-License-Identifier: Apache-2.0

"""Bounded-memory, ownership-safe disk snapshots for optimizer rollback.

This private format targets synchronous managed-optimizer loads.  Snapshot
roots must already exist and every path component must be a real directory.
All mutating operations use retained directory file descriptors and relative
names; display paths are never used as deletion authority.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import stat
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import torch

_SCHEMA = "areal.gpu_staged_adamw.rollback"
_VERSION = 2
_MARKER = "owner.json"
_DIR_PREFIX = "areal-managed-rollback-"
_DEFAULT_PARENT = Path("/tmp")
MIN_SNAPSHOT_CHUNK_BYTES = 1 * 1024 * 1024
MAX_SNAPSHOT_CHUNK_BYTES = 512 * 1024 * 1024
MAX_SNAPSHOT_CHUNKS_PER_SLAB = 131_072
MAX_SNAPSHOT_HEADER_BYTES = 16 * 1024 * 1024
_HEADER_BYTES_PER_CHUNK = 128
_HEADER_FIXED_BYTES = 64 * 1024
_CAPACITY_MARGIN_BYTES = 64 * 1024 * 1024
_CAPACITY_MARGIN_RATIO = 0.05


@dataclass(frozen=True)
class SnapshotRequirement:
    """Required bytes for one trusted rollback snapshot filesystem."""

    parent: Path
    required_bytes: int


@dataclass(frozen=True)
class SnapshotCapacityReport:
    """Serializable local capacity observation for a trusted filesystem."""

    filesystem_id: int
    root_path: str
    root_device: int
    root_inode: int
    required_bytes: int
    free_bytes: int


@dataclass(frozen=True)
class _FileIdentity:
    name: str
    device: int
    inode: int


class _MoveStage(Enum):
    PLANNED = auto()
    MOVED = auto()
    CLEANED = auto()


@dataclass
class _ArtifactMove:
    """Pre-registered rename authority for one snapshot artifact."""

    partial_name: str
    final_name: str
    device: int
    inode: int
    file_type: int
    stage: _MoveStage = _MoveStage.PLANNED


class _FDState(Enum):
    OPEN = auto()
    CLOSE_PENDING = auto()
    CLOSED = auto()
    OWNERSHIP_LOST = auto()


class _RestoreState(Enum):
    COPY_PENDING = auto()
    DATA_RESTORED = auto()
    RESTORE_COMPLETE = auto()


@dataclass
class _OwnedFD:
    """Own an FD until it is detached immediately before the close syscall."""

    fd: int
    device: int
    inode: int
    file_type: int
    state: _FDState = _FDState.OPEN
    close_diagnostic: str | None = None
    ownership_diagnostic: str | None = None

    @classmethod
    def capture(cls, fd: int) -> _OwnedFD:
        opened = os.fstat(fd)
        return cls(
            fd=fd,
            device=opened.st_dev,
            inode=opened.st_ino,
            file_type=stat.S_IFMT(opened.st_mode),
        )

    @property
    def expected_signature(self) -> tuple[int, int, int]:
        return (self.device, self.inode, self.file_type)

    @staticmethod
    def _signature(opened: os.stat_result) -> tuple[int, int, int]:
        return (
            opened.st_dev,
            opened.st_ino,
            stat.S_IFMT(opened.st_mode),
        )

    def _lose_ownership(
        self, *, actual: tuple[int, int, int] | str, original: BaseException | None
    ) -> None:
        self.ownership_diagnostic = (
            f"original_preclose={original!r}, expected={self.expected_signature!r}, "
            f"replacement={actual!r}"
        )
        self.fd = -1
        self.state = _FDState.OWNERSHIP_LOST

    def _revalidate(self, *, original: BaseException | None = None) -> None:
        try:
            opened = os.fstat(self.fd)
        except OSError as error:
            if error.errno == errno.EBADF:
                self._lose_ownership(actual="EBADF", original=original)
                raise RuntimeError(
                    "rollback snapshot FD ownership changed: "
                    f"{self.ownership_diagnostic}"
                ) from error
            raise
        actual = self._signature(opened)
        if actual != self.expected_signature:
            self._lose_ownership(actual=actual, original=original)
            raise RuntimeError(
                f"rollback snapshot FD ownership changed: {self.ownership_diagnostic}"
            )

    def close(self) -> None:
        if self.state is _FDState.OWNERSHIP_LOST:
            raise RuntimeError(
                "rollback snapshot FD ownership changed; replacement will not "
                f"be closed: {self.ownership_diagnostic}"
            )
        if self.state is _FDState.CLOSED:
            if self.close_diagnostic is not None:
                raise RuntimeError(
                    "rollback snapshot FD close result is indeterminate; the "
                    "consumed descriptor will not be retried: "
                    f"{self.close_diagnostic}"
                )
            return
        self._revalidate()
        self.state = _FDState.CLOSE_PENDING
        # This hook is the only retryable failure boundary: the owner still
        # holds the descriptor because no close call has been entered.
        try:
            _prepare_fd_close(self)
        except BaseException as original:
            try:
                self._revalidate(original=original)
            except RuntimeError as ownership_error:
                original.add_note(str(ownership_error))
            raise
        fd = self.fd
        self.fd = -1
        self.state = _FDState.CLOSED
        try:
            _close_fd(fd)
        except BaseException as error:
            # Linux close errors are ambiguous.  The integer was detached
            # before entering close and is permanently consumed: never fstat
            # or retry it, even if the kernel has already reused the number.
            self.close_diagnostic = repr(error)
            raise


@dataclass
class _RestoreFDJournal:
    """Close-only journal shared with cleanup; it has no restore authority."""

    owner: _OwnedFD | None = None
    state: _RestoreState = _RestoreState.COPY_PENDING
    diagnostic: str | None = None

    @property
    def restore_complete(self) -> bool:
        return self.state is _RestoreState.RESTORE_COMPLETE

    def finalize(self) -> None:
        owner = self.owner
        if owner is None:
            return
        try:
            owner.close()
        except BaseException as error:
            self.diagnostic = repr(error)
            if owner.state is _FDState.CLOSED:
                # The integer was detached before close and is consumed even
                # though the result was indeterminate.  Never retry it.
                self.owner = None
                if self.state is _RestoreState.DATA_RESTORED:
                    self.state = _RestoreState.RESTORE_COMPLETE
            raise
        self.owner = None
        self.diagnostic = None
        if self.state is _RestoreState.DATA_RESTORED:
            self.state = _RestoreState.RESTORE_COMPLETE


@dataclass
class _TrustedRoot:
    path: Path
    fd: int
    device: int
    inode: int
    filesystem_id: int
    _fd_owner: _OwnedFD | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.fd >= 0 and self._fd_owner is None:
            self._fd_owner = _OwnedFD.capture(self.fd)

    def validate(self) -> None:
        current = os.fstat(self.fd)
        if not stat.S_ISDIR(current.st_mode) or (
            current.st_dev,
            current.st_ino,
        ) != (self.device, self.inode):
            raise RuntimeError("rollback snapshot root ownership changed")
        reopened = _open_trusted_root(self.path)
        try:
            if (reopened.device, reopened.inode) != (self.device, self.inode):
                raise RuntimeError("rollback snapshot root ownership changed")
        finally:
            reopened.close()

    def close(self) -> None:
        if self.fd < 0:
            if self._fd_owner is not None:
                self._fd_owner.close()
            return
        assert self._fd_owner is not None
        try:
            self._fd_owner.close()
        finally:
            if self._fd_owner.state in (
                _FDState.CLOSED,
                _FDState.OWNERSHIP_LOST,
            ):
                self.fd = -1
        if self._fd_owner.close_diagnostic is None:
            self._fd_owner = None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _write_all(fd: int, payload: bytes | memoryview) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(fd, view[written:])
        if count <= 0:
            raise OSError("short write while creating optimizer rollback snapshot")
        written += count


def _read_exact(fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        payload = os.read(fd, remaining)
        if not payload:
            raise OSError(
                "short read while validating optimizer rollback snapshot: "
                f"expected {size} bytes"
            )
        chunks.append(payload)
        remaining -= len(payload)
    return b"".join(chunks)


def _read_exact_buffer(fd: int, size: int) -> bytearray:
    payload = bytearray(size)
    view = memoryview(payload)
    offset = 0
    while offset < size:
        count = os.readv(fd, (view[offset:],))
        if count <= 0:
            raise OSError(
                "short read while reading optimizer rollback snapshot: "
                f"expected {size} bytes"
            )
        offset += count
    return payload


def _close_fd(fd: int) -> None:
    os.close(fd)


def _prepare_fd_close(owner: _OwnedFD) -> None:
    """Injectable pre-close boundary; failures here retain retry authority."""

    del owner


def _directory_flags() -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _file_flags(flags: int) -> int:
    flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _lexical_absolute_path(path: str | os.PathLike[str] | None) -> Path:
    configured = os.fspath(_DEFAULT_PARENT if path is None else path)
    return Path(os.path.abspath(configured))


def _open_trusted_root(path: str | os.PathLike[str] | None) -> _TrustedRoot:
    """Open an existing directory tree without following any symlink.

    The deliberately strict contract rejects symlinks in every parent
    component, not just the configured root.  This avoids silently changing
    deletion authority through ``resolve()``.
    """

    absolute = _lexical_absolute_path(path)
    current_fd = os.open(os.sep, _directory_flags())
    try:
        for component in absolute.parts[1:]:
            entry = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            if stat.S_ISLNK(entry.st_mode):
                raise RuntimeError(
                    f"rollback snapshot root contains a symlink component: {absolute}"
                )
            if not stat.S_ISDIR(entry.st_mode):
                raise NotADirectoryError(
                    f"rollback snapshot root is not a directory: {absolute}"
                )
            next_fd = os.open(component, _directory_flags(), dir_fd=current_fd)
            opened = os.fstat(next_fd)
            if (opened.st_dev, opened.st_ino) != (entry.st_dev, entry.st_ino):
                _close_fd(next_fd)
                raise RuntimeError(
                    "rollback snapshot root changed while it was being opened"
                )
            _close_fd(current_fd)
            current_fd = next_fd
        root_state = os.fstat(current_fd)
        filesystem_state = os.fstatvfs(current_fd)
        missing = object()
        raw_filesystem_id = getattr(filesystem_state, "f_fsid", missing)
        if type(raw_filesystem_id) is not int or raw_filesystem_id == 0:
            raise OSError(
                "rollback snapshot filesystem has no stable nonzero integer "
                f"f_fsid (actual={raw_filesystem_id!r}); shared capacity "
                "cannot be verified"
            )
        filesystem_id = raw_filesystem_id
        return _TrustedRoot(
            path=absolute,
            fd=current_fd,
            device=root_state.st_dev,
            inode=root_state.st_ino,
            filesystem_id=filesystem_id,
        )
    except BaseException:
        _close_fd(current_fd)
        raise


def _regular_identity(fd: int, name: str) -> _FileIdentity:
    state = os.fstat(fd)
    if not stat.S_ISREG(state.st_mode):
        raise OSError(f"rollback snapshot file is not regular: {name}")
    return _FileIdentity(name=name, device=state.st_dev, inode=state.st_ino)


def _open_owned_regular_readonly(directory_fd: int, expected: _FileIdentity) -> int:
    entry = os.stat(expected.name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(entry.st_mode) or (
        entry.st_dev,
        entry.st_ino,
    ) != (expected.device, expected.inode):
        raise RuntimeError(f"rollback snapshot file ownership changed: {expected.name}")
    fd = os.open(
        expected.name,
        _file_flags(os.O_RDONLY),
        dir_fd=directory_fd,
    )
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (expected.device, expected.inode):
            raise RuntimeError(
                f"rollback snapshot file ownership changed: {expected.name}"
            )
    except BaseException:
        _close_fd(fd)
        raise
    return fd


def _read_bounded_json_at(
    directory_fd: int, expected: _FileIdentity, maximum_bytes: int
) -> Any:
    fd = _open_owned_regular_readonly(directory_fd, expected)
    try:
        size = os.fstat(fd).st_size
        if size > maximum_bytes:
            raise OSError(
                f"rollback snapshot metadata is too large: {expected.name} has "
                f"{size} bytes"
            )
        payload = _read_exact(fd, size)
        if os.read(fd, 1):
            raise OSError(
                f"rollback snapshot metadata grew while reading: {expected.name}"
            )
    finally:
        _close_fd(fd)
    return json.loads(payload)


def _validate_chunk_layout(numel: int, element_size: int, chunk_bytes: int) -> int:
    payload = numel * element_size
    chunks = max(1, math.ceil(payload / chunk_bytes))
    header_estimate = _HEADER_FIXED_BYTES + chunks * _HEADER_BYTES_PER_CHUNK
    if chunk_bytes % element_size:
        raise ValueError(
            "rollback snapshot chunk size must align to FP32 elements: "
            f"chunk_bytes={chunk_bytes}, alignment={element_size}"
        )
    if not MIN_SNAPSHOT_CHUNK_BYTES <= chunk_bytes <= MAX_SNAPSHOT_CHUNK_BYTES:
        raise ValueError(
            "rollback snapshot chunk size is outside the supported range: "
            f"chunk_bytes={chunk_bytes}, allowed=[{MIN_SNAPSHOT_CHUNK_BYTES}, "
            f"{MAX_SNAPSHOT_CHUNK_BYTES}], chunk_count={chunks}"
        )
    if (
        chunks > MAX_SNAPSHOT_CHUNKS_PER_SLAB
        or header_estimate > MAX_SNAPSHOT_HEADER_BYTES
    ):
        raise ValueError(
            "rollback snapshot chunk metadata exceeds the bounded budget: "
            f"chunk_bytes={chunk_bytes}, chunk_count={chunks}, "
            f"max_chunks={MAX_SNAPSHOT_CHUNKS_PER_SLAB}, "
            f"estimated_header_bytes={header_estimate}, "
            f"max_header_bytes={MAX_SNAPSHOT_HEADER_BYTES}"
        )
    return chunks


def validate_snapshot_chunk_bytes(chunk_bytes: int) -> None:
    """Validate configuration before any snapshot root or file is touched."""

    _validate_chunk_layout(
        1, torch.tensor([], dtype=torch.float32).element_size(), chunk_bytes
    )


def _probe_filesystem(root: _TrustedRoot) -> None:
    probe_name = f".areal-rollback-probe-{uuid.uuid4().hex}"
    fd = os.open(
        probe_name,
        _file_flags(os.O_CREAT | os.O_EXCL | os.O_RDWR),
        0o600,
        dir_fd=root.fd,
    )
    attempted_unlink = False
    try:
        if not hasattr(os, "posix_fallocate"):
            raise OSError("rollback snapshot filesystem reservation is unsupported")
        payload = b"areal-rollback-probe"
        os.posix_fallocate(fd, 0, len(payload))
        _write_all(fd, payload)
        os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        if _read_exact(fd, len(payload)) != payload:
            raise OSError("rollback snapshot filesystem failed read-after-write probe")
        _regular_identity(fd, probe_name)
    finally:
        try:
            _close_fd(fd)
        finally:
            try:
                attempted_unlink = True
                os.unlink(probe_name, dir_fd=root.fd)
            except FileNotFoundError:
                if not attempted_unlink:
                    raise


def _filesystem_free_bytes(root: _TrustedRoot) -> int:
    state = os.fstatvfs(root.fd)
    return int(state.f_bavail) * int(state.f_frsize)


def _required_with_margin(required: int) -> int:
    return required + max(
        _CAPACITY_MARGIN_BYTES, math.ceil(required * _CAPACITY_MARGIN_RATIO)
    )


def preflight_snapshot_requirements(
    requirements: tuple[SnapshotRequirement, ...],
) -> tuple[SnapshotCapacityReport, ...]:
    """Validate roots/filesystems locally without creating snapshot files."""

    by_root: dict[tuple[str, int, int], SnapshotCapacityReport] = {}
    probed_roots: set[tuple[int, int]] = set()
    for requirement in requirements:
        if requirement.required_bytes < 0:
            raise ValueError("rollback snapshot required bytes must be non-negative")
        root = _open_trusted_root(requirement.parent)
        try:
            root_key = (root.device, root.inode)
            if root_key not in probed_roots:
                _probe_filesystem(root)
                probed_roots.add(root_key)
            free = _filesystem_free_bytes(root)
            root_identity = (os.fspath(root.path), root.device, root.inode)
            previous = by_root.get(root_identity)
            if previous is None:
                report = SnapshotCapacityReport(
                    filesystem_id=root.filesystem_id,
                    root_path=os.fspath(root.path),
                    root_device=root.device,
                    root_inode=root.inode,
                    required_bytes=requirement.required_bytes,
                    free_bytes=free,
                )
            else:
                report = SnapshotCapacityReport(
                    filesystem_id=previous.filesystem_id,
                    root_path=previous.root_path,
                    root_device=previous.root_device,
                    root_inode=previous.root_inode,
                    required_bytes=(
                        previous.required_bytes + requirement.required_bytes
                    ),
                    free_bytes=min(previous.free_bytes, free),
                )
            by_root[root_identity] = report
        finally:
            root.close()
    reports = tuple(by_root.values())
    local_filesystems: dict[int, list[SnapshotCapacityReport]] = {}
    for report in reports:
        local_filesystems.setdefault(report.filesystem_id, []).append(report)
    for filesystem_id, filesystem_reports in local_filesystems.items():
        payload = sum(report.required_bytes for report in filesystem_reports)
        free = min(report.free_bytes for report in filesystem_reports)
        required = _required_with_margin(payload)
        if free < required:
            raise OSError(
                "insufficient rollback snapshot capacity on "
                f"filesystem_id={filesystem_id}: required={payload}, "
                f"safety_required={required}, available={free}, "
                f"roots={sorted(report.root_path for report in filesystem_reports)}"
            )
    return reports


def validate_shared_snapshot_capacity(
    local_reports: tuple[SnapshotCapacityReport, ...], process_group: Any
) -> None:
    """Aggregate capacity for every confirmed shared filesystem on WORLD.

    Linux ``f_fsid`` is the filesystem sharing identity.  A zero/absent value
    is rejected during local preflight rather than pretending that per-rank
    capacity observations are independent.
    """

    import torch.distributed as dist

    # Structural unit-test leaves may implement their own non-disk journal and
    # therefore produce no reports outside an initialized distributed job.
    # A real disk-backed leaf must never bypass the explicit WORLD vote.
    if not dist.is_initialized():
        if local_reports:
            raise RuntimeError(
                "shared rollback snapshot capacity requires an initialized "
                "distributed process group"
            )
        return
    world_size = dist.get_world_size(process_group)
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, local_reports, group=process_group)
    by_filesystem: dict[int, list[tuple[int, SnapshotCapacityReport]]] = {}
    by_root_identity: dict[
        tuple[str, int, int], list[tuple[int, SnapshotCapacityReport]]
    ] = {}
    for rank, rank_reports in enumerate(gathered):
        if not isinstance(rank_reports, tuple):
            raise TypeError("invalid rollback snapshot capacity report from rank")
        for report in rank_reports:
            if not isinstance(report, SnapshotCapacityReport):
                raise TypeError("invalid rollback snapshot capacity report from rank")
            if type(report.filesystem_id) is not int or report.filesystem_id == 0:
                raise ValueError(
                    "invalid rollback snapshot filesystem f_fsid report: "
                    f"rank={rank}, root={report.root_path!r}, "
                    f"dev={report.root_device}, inode={report.root_inode}, "
                    f"f_fsid={report.filesystem_id!r}"
                )
            if (
                type(report.root_device) is not int
                or type(report.root_inode) is not int
                or type(report.required_bytes) is not int
                or type(report.free_bytes) is not int
                or report.required_bytes < 0
                or report.free_bytes < 0
            ):
                raise TypeError(
                    "invalid rollback snapshot filesystem capacity report: "
                    f"rank={rank}, report={report!r}"
                )
            ranked = (rank, report)
            by_filesystem.setdefault(report.filesystem_id, []).append(ranked)
            root_identity = (
                os.path.abspath(report.root_path),
                report.root_device,
                report.root_inode,
            )
            by_root_identity.setdefault(root_identity, []).append(ranked)
    for root_identity, ranked_reports in by_root_identity.items():
        filesystem_ids = {report.filesystem_id for _, report in ranked_reports}
        if len(filesystem_ids) != 1:
            details = ", ".join(
                f"rank={rank} root={report.root_path!r} "
                f"dev={report.root_device} inode={report.root_inode} "
                f"f_fsid={report.filesystem_id}"
                for rank, report in ranked_reports
            )
            raise RuntimeError(
                "rollback snapshot filesystem identity conflict for canonical "
                f"root={root_identity[0]!r}: {details}"
            )
    for filesystem_id, ranked_reports in by_filesystem.items():
        reports = [report for _, report in ranked_reports]
        required = sum(report.required_bytes for report in reports)
        free = min(report.free_bytes for report in reports)
        safety_required = _required_with_margin(required)
        if free < safety_required:
            raise OSError(
                "insufficient shared rollback snapshot capacity: "
                f"filesystem_id={filesystem_id}, required={required}, "
                f"safety_required={safety_required}, min_available={free}, "
                f"roots={sorted({report.root_path for report in reports})}"
            )


def discover_orphaned_snapshot_directories(
    parent: str | os.PathLike[str] | None = None,
) -> tuple[Path, ...]:
    """Read-only discovery; ownership is never inferred strongly enough to delete."""

    try:
        root = _open_trusted_root(parent)
    except FileNotFoundError:
        return ()
    try:
        discovered: list[Path] = []
        for name in os.listdir(root.fd):
            if not name.startswith(_DIR_PREFIX):
                continue
            entry = os.stat(name, dir_fd=root.fd, follow_symlinks=False)
            if not stat.S_ISDIR(entry.st_mode):
                continue
            candidate_fd = os.open(name, _directory_flags(), dir_fd=root.fd)
            try:
                try:
                    marker = os.stat(
                        _MARKER, dir_fd=candidate_fd, follow_symlinks=False
                    )
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(marker.st_mode):
                    discovered.append(root.path / name)
            finally:
                _close_fd(candidate_fd)
        return tuple(sorted(discovered))
    finally:
        root.close()


def _create_owned_directory(
    root: _TrustedRoot, *, rank: int, identity_digest: str
) -> tuple[str, int, os.stat_result]:
    for _ in range(128):
        name = f"{_DIR_PREFIX}r{rank}-l{identity_digest}-{uuid.uuid4().hex}"
        try:
            os.mkdir(name, 0o700, dir_fd=root.fd)
        except FileExistsError:
            continue
        directory_fd = os.open(name, _directory_flags(), dir_fd=root.fd)
        state = os.fstat(directory_fd)
        linked = os.stat(name, dir_fd=root.fd, follow_symlinks=False)
        if (state.st_dev, state.st_ino) != (linked.st_dev, linked.st_ino):
            _close_fd(directory_fd)
            raise RuntimeError("rollback snapshot directory changed during creation")
        return name, directory_fd, state
    raise FileExistsError("could not allocate a unique rollback snapshot directory")


@dataclass
class DiskSnapshotBuildCleanup:
    """Retryable cleanup for a snapshot that never gained rollback authority."""

    directory: Path
    parent: Path
    directory_device: int
    directory_inode: int
    pending_paths: list[Path]
    pending_fd_owners: list[_OwnedFD] = field(default_factory=list)
    directory_pending: bool = True
    _root: _TrustedRoot | None = field(default=None, repr=False)
    _directory_fd: int = field(default=-1, repr=False)
    _file_identities: dict[str, _FileIdentity] = field(default_factory=dict, repr=False)
    _unlink_attempted: set[str] = field(default_factory=set, repr=False)
    _rmdir_attempted: bool = field(default=False, repr=False)
    _cleanup_token: str = field(default_factory=lambda: uuid.uuid4().hex, repr=False)
    _quarantine_names: dict[str, str] = field(default_factory=dict, repr=False)
    _directory_quarantine_name: str | None = field(default=None, repr=False)
    _moves: dict[str, _ArtifactMove] = field(default_factory=dict, repr=False)
    _directory_fd_owner: _OwnedFD | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._directory_fd >= 0 and self._directory_fd_owner is None:
            self._directory_fd_owner = _OwnedFD.capture(self._directory_fd)

    def register_move(
        self,
        artifact: str,
        *,
        partial_name: str,
        final_name: str,
        identity: _FileIdentity,
    ) -> None:
        """Journal rename authority before the first namespace mutation."""

        if artifact in self._moves:
            raise RuntimeError(f"snapshot artifact move already planned: {artifact}")
        opened = os.stat(
            partial_name,
            dir_fd=self._directory_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (identity.device, identity.inode):
            raise RuntimeError(
                f"snapshot artifact ownership changed before rename: {artifact}"
            )
        self._moves[artifact] = _ArtifactMove(
            partial_name=partial_name,
            final_name=final_name,
            device=identity.device,
            inode=identity.inode,
            file_type=stat.S_IFMT(opened.st_mode),
        )

    def mark_move_moved(self, artifact: str) -> None:
        move = self._moves[artifact]
        move.stage = _MoveStage.MOVED

    def _remove_pending_name(self, name: str) -> None:
        self.pending_paths[:] = [
            path for path in self.pending_paths if path.name != name
        ]
        self._file_identities.pop(name, None)

    def _commit_artifact_cleanup(self, name: str) -> None:
        pending_names = {path.name for path in self.pending_paths}
        for move in self._moves.values():
            if name not in (move.partial_name, move.final_name):
                continue
            if not pending_names.intersection((move.partial_name, move.final_name)):
                move.stage = _MoveStage.CLEANED

    def _reconcile_moves(self) -> None:
        """Find each pre-registered inode at either rename candidate.

        The stage is only an optimization/diagnostic.  Namespace identity is
        authoritative so a rename that completed before raising is recoverable.
        """

        for move in self._moves.values():
            if move.stage is _MoveStage.CLEANED:
                continue
            matches: list[str] = []
            for name in (move.partial_name, move.final_name):
                try:
                    entry = os.stat(
                        name,
                        dir_fd=self._directory_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    quarantine = self._quarantine_names.get(name)
                    if quarantine is None:
                        continue
                    try:
                        entry = os.stat(
                            quarantine,
                            dir_fd=self._directory_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        continue
                if stat.S_IFMT(entry.st_mode) != move.file_type or (
                    entry.st_dev,
                    entry.st_ino,
                ) != (move.device, move.inode):
                    raise RuntimeError(
                        "partial rollback snapshot move candidate ownership "
                        f"changed: {name}"
                    )
                matches.append(name)
                self._file_identities[name] = _FileIdentity(
                    name=name,
                    device=move.device,
                    inode=move.inode,
                )
            if not matches:
                attempted = any(
                    self._quarantine_names.get(name) in self._unlink_attempted
                    for name in (move.partial_name, move.final_name)
                )
                if attempted:
                    move.stage = _MoveStage.CLEANED
                    self._remove_pending_name(move.partial_name)
                    self._remove_pending_name(move.final_name)
                    continue
                raise RuntimeError(
                    "partial rollback snapshot artifact inode disappeared: "
                    f"partial={move.partial_name}, final={move.final_name}"
                )
            for name in (move.partial_name, move.final_name):
                if name not in matches:
                    self._remove_pending_name(name)
            if move.final_name in matches:
                move.stage = _MoveStage.MOVED

    def _finalize_authority(self, errors: list[BaseException]) -> None:
        if self._directory_fd_owner is not None:
            try:
                self._directory_fd_owner.close()
            except BaseException as error:
                errors.append(error)
            finally:
                if self._directory_fd_owner.state in (
                    _FDState.CLOSED,
                    _FDState.OWNERSHIP_LOST,
                ):
                    self._directory_fd = -1
            if self._directory_fd_owner.close_diagnostic is None:
                self._directory_fd = -1
                self._directory_fd_owner = None
        if self._directory_fd_owner is None and self._root is not None:
            try:
                self._root.close()
            except BaseException as error:
                errors.append(error)
            if self._root.fd < 0:
                self._root = None

    def _validate_directory(self) -> None:
        if self._root is None:
            raise RuntimeError("partial rollback snapshot cleanup lost root authority")
        self._root.validate()
        opened = os.fstat(self._directory_fd)
        if (opened.st_dev, opened.st_ino) != (
            self.directory_device,
            self.directory_inode,
        ):
            raise RuntimeError("partial rollback snapshot directory ownership changed")
        linked_name = self.directory.name
        if self._directory_quarantine_name is not None:
            try:
                os.stat(
                    self._directory_quarantine_name,
                    dir_fd=self._root.fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                linked_name = self._directory_quarantine_name
        try:
            linked = os.stat(
                linked_name,
                dir_fd=self._root.fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            if self._rmdir_attempted:
                self.directory_pending = False
                return
            raise RuntimeError(
                "partial rollback snapshot directory ownership changed"
            ) from None
        if not stat.S_ISDIR(linked.st_mode) or (
            linked.st_dev,
            linked.st_ino,
        ) != (self.directory_device, self.directory_inode):
            raise RuntimeError("partial rollback snapshot directory ownership changed")

    def cleanup(self) -> None:
        errors: list[BaseException] = []
        for owner in tuple(self.pending_fd_owners):
            try:
                owner.close()
            except BaseException as error:
                errors.append(error)
                continue
            self.pending_fd_owners.remove(owner)
        if not self.directory_pending and not self.pending_paths:
            self._finalize_authority(errors)
            if errors:
                primary = errors[0]
                for error in errors[1:]:
                    primary.add_note(
                        f"additional rollback snapshot build cleanup failure: {error!r}"
                    )
                raise primary
            return
        self._validate_directory()
        self._reconcile_moves()
        for path in tuple(self.pending_paths):
            name = path.name
            expected = self._file_identities.get(name)
            try:
                entry = os.stat(name, dir_fd=self._directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                quarantine = self._quarantine_names.get(name)
                if expected is None or quarantine is None:
                    if expected is not None:
                        raise RuntimeError(
                            f"partial rollback snapshot file ownership changed: {name}"
                        ) from None
                    self.pending_paths.remove(path)
                    self._commit_artifact_cleanup(name)
                    continue
                try:
                    quarantined = os.stat(
                        quarantine,
                        dir_fd=self._directory_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    if quarantine not in self._unlink_attempted:
                        raise RuntimeError(
                            f"partial rollback snapshot file ownership changed: {name}"
                        ) from None
                    self.pending_paths.remove(path)
                    continue
                if not stat.S_ISREG(quarantined.st_mode) or (
                    quarantined.st_dev,
                    quarantined.st_ino,
                ) != (expected.device, expected.inode):
                    raise RuntimeError(
                        f"partial rollback snapshot file ownership changed: {name}"
                    )
                try:
                    self._unlink_attempted.add(quarantine)
                    os.unlink(quarantine, dir_fd=self._directory_fd)
                except BaseException as error:
                    errors.append(error)
                    continue
                self.pending_paths.remove(path)
                self._commit_artifact_cleanup(name)
                continue
            if expected is None:
                raise RuntimeError(
                    f"partial rollback snapshot contains an unowned file: {name}"
                )
            if expected is not None and (
                not stat.S_ISREG(entry.st_mode)
                or (entry.st_dev, entry.st_ino) != (expected.device, expected.inode)
            ):
                raise RuntimeError(
                    f"partial rollback snapshot file ownership changed: {name}"
                )
            quarantine = self._quarantine_names.setdefault(
                name, f".delete-{self._cleanup_token}-{name}"
            )
            try:
                quarantined = os.stat(
                    quarantine,
                    dir_fd=self._directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                try:
                    os.rename(
                        name,
                        quarantine,
                        src_dir_fd=self._directory_fd,
                        dst_dir_fd=self._directory_fd,
                    )
                except BaseException as error:
                    errors.append(error)
                    continue
                quarantined = os.stat(
                    quarantine,
                    dir_fd=self._directory_fd,
                    follow_symlinks=False,
                )
            if not stat.S_ISREG(quarantined.st_mode) or (
                quarantined.st_dev,
                quarantined.st_ino,
            ) != (expected.device, expected.inode):
                raise RuntimeError(
                    f"partial rollback snapshot file ownership changed: {name}"
                )
            try:
                self._unlink_attempted.add(quarantine)
                os.unlink(quarantine, dir_fd=self._directory_fd)
            except BaseException as error:
                errors.append(error)
                continue
            self.pending_paths.remove(path)
            self._commit_artifact_cleanup(name)
        unexpected = set(os.listdir(self._directory_fd)) - {
            path.name for path in self.pending_paths
        }
        if unexpected:
            raise RuntimeError(
                "partial rollback snapshot directory contains unexpected files: "
                f"{sorted(unexpected)}"
            )
        if self.directory_pending and not self.pending_paths:
            if any(
                move.stage is not _MoveStage.CLEANED for move in self._moves.values()
            ):
                raise RuntimeError(
                    "partial rollback snapshot cleanup has unfinished artifact moves"
                )
            self._validate_directory()
            try:
                if self._directory_quarantine_name is None:
                    self._directory_quarantine_name = (
                        f".delete-{self._cleanup_token}-{self.directory.name}"
                    )
                try:
                    os.stat(
                        self._directory_quarantine_name,
                        dir_fd=self._root.fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    os.rename(
                        self.directory.name,
                        self._directory_quarantine_name,
                        src_dir_fd=self._root.fd,
                        dst_dir_fd=self._root.fd,
                    )
                linked = os.stat(
                    self._directory_quarantine_name,
                    dir_fd=self._root.fd,
                    follow_symlinks=False,
                )
                if not stat.S_ISDIR(linked.st_mode) or (
                    linked.st_dev,
                    linked.st_ino,
                ) != (self.directory_device, self.directory_inode):
                    raise RuntimeError(
                        "partial rollback snapshot directory ownership changed"
                    )
                self._rmdir_attempted = True
                assert self._root is not None
                os.rmdir(self._directory_quarantine_name, dir_fd=self._root.fd)
            except BaseException as error:
                errors.append(error)
            else:
                self.directory_pending = False
        if not self.directory_pending:
            self._finalize_authority(errors)
        if errors:
            primary = errors[0]
            for error in errors[1:]:
                primary.add_note(
                    f"additional rollback snapshot build cleanup failure: {error!r}"
                )
            raise primary


@dataclass
class DiskTensorRollbackSnapshot:
    """Chunked disk snapshot and retry journal for one FP32 CPU slab."""

    directory: Path
    parent: Path
    owner_token: str
    leaf_identity: Any
    slab_key: str
    data_path: Path
    header_path: Path
    chunk_bytes: int
    numel: int
    byte_length: int
    chunk_checksums: tuple[str, ...]
    checksum: str
    next_chunk: int = 0
    _restore_fd_journal: _RestoreFDJournal = field(
        default_factory=_RestoreFDJournal, repr=False
    )
    _root: _TrustedRoot | None = field(default=None, repr=False)
    _directory_fd: int = field(default=-1, repr=False)
    _directory_device: int = field(default=-1, repr=False)
    _directory_inode: int = field(default=-1, repr=False)
    _marker_state: dict[str, Any] = field(default_factory=dict, repr=False)
    _marker_identity: _FileIdentity | None = field(default=None, repr=False)
    _data_identity: _FileIdentity | None = field(default=None, repr=False)
    _header_identity: _FileIdentity | None = field(default=None, repr=False)
    _cleanup: DiskSnapshotCleanup | None = field(default=None, repr=False)

    @classmethod
    def required_bytes(cls, tensor: torch.Tensor, chunk_bytes: int) -> int:
        chunks = _validate_chunk_layout(
            tensor.numel(), tensor.element_size(), chunk_bytes
        )
        return (
            tensor.numel() * tensor.element_size()
            + _HEADER_FIXED_BYTES
            + chunks * _HEADER_BYTES_PER_CHUNK
        )

    @property
    def _restore_fd_owner(self) -> _OwnedFD | None:
        return self._restore_fd_journal.owner

    @property
    def restore_complete(self) -> bool:
        return self._restore_fd_journal.restore_complete

    @classmethod
    def create(
        cls,
        tensor: torch.Tensor,
        *,
        parent: str | os.PathLike[str] | None,
        leaf_identity: Any,
        slab_key: str,
        chunk_bytes: int,
        rank: int,
    ) -> DiskTensorRollbackSnapshot:
        if tensor.device.type != "cpu" or tensor.dtype is not torch.float32:
            raise TypeError("rollback snapshot source must be a CPU FP32 tensor")
        if not tensor.is_contiguous():
            raise ValueError("rollback snapshot source slab must be contiguous")
        _validate_chunk_layout(tensor.numel(), tensor.element_size(), chunk_bytes)
        root = _open_trusted_root(parent)
        identity_payload = _canonical_json(leaf_identity)
        stable_identity = json.loads(identity_payload)
        identity_digest = hashlib.sha256(identity_payload).hexdigest()[:12]
        directory_name = ""
        directory_fd = -1
        build_cleanup: DiskSnapshotBuildCleanup | None = None
        open_fds: dict[int, _OwnedFD] = {}
        try:
            directory_name, directory_fd, directory_state = _create_owned_directory(
                root, rank=rank, identity_digest=identity_digest
            )
            directory = root.path / directory_name
            owner_token = uuid.uuid4().hex
            marker_state = {
                "schema": _SCHEMA,
                "version": _VERSION,
                "owner_token": owner_token,
                "nonce": uuid.uuid4().hex,
                "pid": os.getpid(),
                "rank": rank,
                "created_ns": time.time_ns(),
                "root_device": root.device,
                "root_inode": root.inode,
                "directory_device": directory_state.st_dev,
                "directory_inode": directory_state.st_ino,
                "leaf_identity": stable_identity,
            }
            names = {
                "marker": _MARKER,
                "data": f"{slab_key}.data",
                "header": f"{slab_key}.header.json",
                "partial_data": f".{slab_key}.data.partial",
                "partial_header": f".{slab_key}.header.partial",
            }
            build_cleanup = DiskSnapshotBuildCleanup(
                directory=directory,
                parent=root.path,
                directory_device=directory_state.st_dev,
                directory_inode=directory_state.st_ino,
                pending_paths=[directory / name for name in names.values()],
                _root=root,
                _directory_fd=directory_fd,
            )
            create_flags = _file_flags(os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            marker_fd = os.open(
                names["marker"], create_flags, 0o600, dir_fd=directory_fd
            )
            open_fds[marker_fd] = _OwnedFD.capture(marker_fd)
            marker_identity = _regular_identity(marker_fd, names["marker"])
            build_cleanup._file_identities[names["marker"]] = marker_identity
            _write_all(marker_fd, _canonical_json(marker_state))
            os.fsync(marker_fd)
            open_fds[marker_fd].close()
            del open_fds[marker_fd]
            os.fsync(directory_fd)

            byte_length = tensor.numel() * tensor.element_size()
            data_fd = os.open(
                names["partial_data"], create_flags, 0o600, dir_fd=directory_fd
            )
            open_fds[data_fd] = _OwnedFD.capture(data_fd)
            partial_data_identity = _regular_identity(data_fd, names["partial_data"])
            build_cleanup._file_identities[names["partial_data"]] = (
                partial_data_identity
            )
            if byte_length:
                if not hasattr(os, "posix_fallocate"):
                    raise OSError(
                        "rollback snapshot filesystem reservation is unsupported"
                    )
                os.posix_fallocate(data_fd, 0, byte_length)
            checksums: list[str] = []
            full_checksum = hashlib.sha256()
            chunk_numel = chunk_bytes // tensor.element_size()
            for offset in range(0, tensor.numel(), chunk_numel):
                view = memoryview(tensor[offset : offset + chunk_numel].numpy()).cast(
                    "B"
                )
                checksums.append(hashlib.sha256(view).hexdigest())
                full_checksum.update(view)
                _write_all(data_fd, view)
            os.fsync(data_fd)
            open_fds[data_fd].close()
            del open_fds[data_fd]

            header = {
                "schema": _SCHEMA,
                "version": _VERSION,
                "owner_token": owner_token,
                "leaf_identity": stable_identity,
                "slab_key": slab_key,
                "dtype": "torch.float32",
                "numel": tensor.numel(),
                "byte_length": byte_length,
                "chunk_bytes": chunk_bytes,
                "chunk_checksums": checksums,
                "checksum": full_checksum.hexdigest(),
            }
            header_payload = _canonical_json(header)
            if len(header_payload) > MAX_SNAPSHOT_HEADER_BYTES:
                raise ValueError(
                    "rollback snapshot materialized header exceeds budget: "
                    f"bytes={len(header_payload)}, max={MAX_SNAPSHOT_HEADER_BYTES}"
                )
            header_fd = os.open(
                names["partial_header"], create_flags, 0o600, dir_fd=directory_fd
            )
            open_fds[header_fd] = _OwnedFD.capture(header_fd)
            partial_header_identity = _regular_identity(
                header_fd, names["partial_header"]
            )
            build_cleanup._file_identities[names["partial_header"]] = (
                partial_header_identity
            )
            _write_all(header_fd, header_payload)
            os.fsync(header_fd)
            open_fds[header_fd].close()
            del open_fds[header_fd]
            build_cleanup.register_move(
                "data",
                partial_name=names["partial_data"],
                final_name=names["data"],
                identity=partial_data_identity,
            )
            os.replace(
                names["partial_data"],
                names["data"],
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            build_cleanup.mark_move_moved("data")
            build_cleanup.register_move(
                "header",
                partial_name=names["partial_header"],
                final_name=names["header"],
                identity=partial_header_identity,
            )
            os.replace(
                names["partial_header"],
                names["header"],
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            build_cleanup.mark_move_moved("header")
            build_cleanup._file_identities.pop(names["partial_data"])
            build_cleanup._file_identities.pop(names["partial_header"])
            data_identity = _FileIdentity(
                names["data"],
                partial_data_identity.device,
                partial_data_identity.inode,
            )
            header_identity = _FileIdentity(
                names["header"],
                partial_header_identity.device,
                partial_header_identity.inode,
            )
            build_cleanup._file_identities[names["data"]] = data_identity
            build_cleanup._file_identities[names["header"]] = header_identity
            os.fsync(directory_fd)
            snapshot = cls(
                directory=directory,
                parent=root.path,
                owner_token=owner_token,
                leaf_identity=stable_identity,
                slab_key=slab_key,
                data_path=directory / names["data"],
                header_path=directory / names["header"],
                chunk_bytes=chunk_bytes,
                numel=tensor.numel(),
                byte_length=byte_length,
                chunk_checksums=tuple(checksums),
                checksum=full_checksum.hexdigest(),
                _root=root,
                _directory_fd=directory_fd,
                _directory_device=directory_state.st_dev,
                _directory_inode=directory_state.st_ino,
                _marker_state=marker_state,
                _marker_identity=marker_identity,
                _data_identity=data_identity,
                _header_identity=header_identity,
            )
            snapshot.verify()
            return snapshot
        except BaseException as original:
            if build_cleanup is not None:
                build_cleanup.pending_fd_owners.extend(open_fds.values())
                # Keep the exact journal on the original failure even when the
                # first best-effort pass succeeds.  Callers and diagnostics can
                # then verify/retry the same inode-scoped cleanup idempotently.
                setattr(original, "_areal_snapshot_build_cleanup", build_cleanup)
                try:
                    build_cleanup.cleanup()
                except BaseException as cleanup_error:
                    original.add_note(
                        "rollback snapshot partial-create cleanup is pending: "
                        f"{cleanup_error!r}"
                    )
            else:
                for owner in tuple(open_fds.values()):
                    try:
                        owner.close()
                    except BaseException:
                        pass
                if directory_fd >= 0:
                    _close_fd(directory_fd)
                root.close()
            raise

    def _validate_directory(self) -> None:
        if self._root is None or self._directory_fd < 0:
            raise RuntimeError("rollback snapshot cleanup already completed")
        self._root.validate()
        opened = os.fstat(self._directory_fd)
        if (opened.st_dev, opened.st_ino) != (
            self._directory_device,
            self._directory_inode,
        ):
            raise RuntimeError("rollback snapshot directory ownership changed")
        try:
            linked = os.stat(
                self.directory.name,
                dir_fd=self._root.fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "rollback snapshot directory ownership changed"
            ) from None
        if not stat.S_ISDIR(linked.st_mode) or (
            linked.st_dev,
            linked.st_ino,
        ) != (self._directory_device, self._directory_inode):
            raise RuntimeError("rollback snapshot directory ownership changed")
        assert self._marker_identity is not None
        marker = _read_bounded_json_at(
            self._directory_fd, self._marker_identity, 1024 * 1024
        )
        if marker != self._marker_state:
            raise RuntimeError("rollback snapshot ownership marker mismatch")

    def _read_header(self) -> dict[str, Any]:
        self._validate_directory()
        expected = {
            "schema": _SCHEMA,
            "version": _VERSION,
            "owner_token": self.owner_token,
            "leaf_identity": self.leaf_identity,
            "slab_key": self.slab_key,
            "dtype": "torch.float32",
            "numel": self.numel,
            "byte_length": self.byte_length,
            "chunk_bytes": self.chunk_bytes,
            "chunk_checksums": list(self.chunk_checksums),
            "checksum": self.checksum,
        }
        assert self._header_identity is not None
        try:
            header = _read_bounded_json_at(
                self._directory_fd,
                self._header_identity,
                MAX_SNAPSHOT_HEADER_BYTES,
            )
        except BaseException as error:
            raise RuntimeError(
                f"rollback snapshot header is unreadable for {self.slab_key}"
            ) from error
        if header != expected:
            raise RuntimeError(
                f"rollback snapshot header mismatch for {self.slab_key}: "
                f"expected={expected!r}, actual={header!r}"
            )
        return header

    def verify(self) -> None:
        self._read_header()
        assert self._data_identity is not None
        fd = _open_owned_regular_readonly(self._directory_fd, self._data_identity)
        full_checksum = hashlib.sha256()
        bytes_read = 0
        try:
            for expected_checksum in self.chunk_checksums:
                remaining = self.byte_length - bytes_read
                payload = _read_exact_buffer(fd, min(self.chunk_bytes, remaining))
                if hashlib.sha256(payload).hexdigest() != expected_checksum:
                    raise RuntimeError(
                        f"rollback snapshot checksum mismatch for {self.slab_key}"
                    )
                full_checksum.update(payload)
                bytes_read += len(payload)
            if os.read(fd, 1):
                raise RuntimeError(
                    f"rollback snapshot contains extra bytes for {self.slab_key}"
                )
            if full_checksum.hexdigest() != self.checksum:
                raise RuntimeError(
                    f"rollback snapshot checksum mismatch for {self.slab_key}"
                )
        finally:
            _close_fd(fd)

    def restore_into(self, target: torch.Tensor) -> None:
        if target.device.type != "cpu" or target.dtype is not torch.float32:
            raise TypeError("rollback restore target must be a CPU FP32 tensor")
        if target.numel() != self.numel or not target.is_contiguous():
            raise ValueError(
                f"rollback restore target mismatch for {self.slab_key}: "
                f"expected numel={self.numel}, got={target.numel()}"
            )
        if self.restore_complete:
            self.cleanup()
            return
        # A retryable pre-close failure retains the original descriptor owner.
        # Finalize it before opening another descriptor or touching the files.
        if self._restore_fd_owner is not None:
            self._finalize_restore_fd()
            if self.restore_complete:
                self.cleanup()
                return
        self._read_header()
        assert self._data_identity is not None
        owner = _OwnedFD.capture(
            _open_owned_regular_readonly(self._directory_fd, self._data_identity)
        )
        # Publish close authority before the first fallible read.  A retry can
        # therefore finish this exact owner instead of leaking it and opening
        # another descriptor.
        self._restore_fd_journal.owner = owner
        fd = owner.fd
        try:
            for chunk_index in range(self.next_chunk, len(self.chunk_checksums)):
                offset = chunk_index * self.chunk_bytes
                os.lseek(fd, offset, os.SEEK_SET)
                size = min(self.chunk_bytes, self.byte_length - offset)
                payload = _read_exact_buffer(fd, size)
                if (
                    hashlib.sha256(payload).hexdigest()
                    != self.chunk_checksums[chunk_index]
                ):
                    raise RuntimeError(
                        "rollback snapshot checksum mismatch for "
                        f"{self.slab_key} chunk {chunk_index}"
                    )
                values = torch.frombuffer(payload, dtype=torch.float32)
                element_offset = offset // target.element_size()
                target[element_offset : element_offset + values.numel()].copy_(values)
                self.next_chunk = chunk_index + 1
            if os.fstat(fd).st_size != self.byte_length:
                raise RuntimeError(
                    f"rollback snapshot byte length mismatch for {self.slab_key}"
                )
            self._restore_fd_journal.state = _RestoreState.DATA_RESTORED
        except BaseException as original:
            try:
                self._finalize_restore_fd()
            except BaseException as close_error:
                original.add_note(
                    f"rollback snapshot restore FD finalization failed: {close_error!r}"
                )
            raise
        self._finalize_restore_fd()
        self.cleanup()

    def _finalize_restore_fd(self) -> None:
        owner = self._restore_fd_owner
        if owner is None:
            if self._restore_fd_journal.state is _RestoreState.DATA_RESTORED:
                self._restore_fd_journal.state = _RestoreState.RESTORE_COMPLETE
            return
        try:
            self._restore_fd_journal.finalize()
        except BaseException:
            # Once the close syscall has been entered the integer is consumed.
            # It is safe to complete the data action, but retain a bounded
            # diagnostic for the caller that observed the close failure.  A
            # retry must never operate on that integer.
            if owner.state is _FDState.CLOSED:
                assert self._restore_fd_owner is None
            # CLOSE_PENDING remains retryable and keeps the sole owner.
            # OWNERSHIP_LOST is terminal and deliberately remains in the
            # journal so cleanup cannot touch a replacement descriptor or
            # remove the recovery files.
            raise

    def cleanup(self) -> None:
        if self._restore_fd_owner is not None:
            self._finalize_restore_fd()
        self.cleanup_artifact().cleanup()

    def cleanup_artifact(self) -> DiskSnapshotCleanup:
        if self._cleanup is None:
            assert self._root is not None
            assert self._marker_identity is not None
            assert self._data_identity is not None
            assert self._header_identity is not None
            self._cleanup = DiskSnapshotCleanup(
                directory=self.directory,
                parent=self.parent,
                owner_token=self.owner_token,
                data_path=self.data_path,
                header_path=self.header_path,
                _root=self._root,
                _directory_fd=self._directory_fd,
                _directory_device=self._directory_device,
                _directory_inode=self._directory_inode,
                _marker_state=self._marker_state,
                _marker_identity=self._marker_identity,
                _data_identity=self._data_identity,
                _header_identity=self._header_identity,
                _restore_fd_journal=self._restore_fd_journal,
            )
        return self._cleanup


@dataclass
class DiskSnapshotCleanup:
    """Cleanup-only authority; it contains no restore target or restore API."""

    directory: Path
    parent: Path
    owner_token: str
    data_path: Path
    header_path: Path
    data_pending: bool = True
    header_pending: bool = True
    marker_pending: bool = True
    directory_pending: bool = True
    _root: _TrustedRoot | None = field(default=None, repr=False)
    _directory_fd: int = field(default=-1, repr=False)
    _directory_device: int = field(default=-1, repr=False)
    _directory_inode: int = field(default=-1, repr=False)
    _marker_state: dict[str, Any] = field(default_factory=dict, repr=False)
    _marker_identity: _FileIdentity | None = field(default=None, repr=False)
    _data_identity: _FileIdentity | None = field(default=None, repr=False)
    _header_identity: _FileIdentity | None = field(default=None, repr=False)
    _unlink_attempted: set[str] = field(default_factory=set, repr=False)
    _rmdir_attempted: bool = field(default=False, repr=False)
    _quarantine_names: dict[str, str] = field(default_factory=dict, repr=False)
    _directory_quarantine_name: str | None = field(default=None, repr=False)
    _directory_fd_owner: _OwnedFD | None = field(default=None, repr=False)
    _restore_fd_journal: _RestoreFDJournal | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._directory_fd >= 0 and self._directory_fd_owner is None:
            self._directory_fd_owner = _OwnedFD.capture(self._directory_fd)

    def _finalize_fds(self) -> None:
        # Directory authority is always finalized before its parent root.
        if self._directory_fd_owner is not None:
            try:
                self._directory_fd_owner.close()
            finally:
                if self._directory_fd_owner.state in (
                    _FDState.CLOSED,
                    _FDState.OWNERSHIP_LOST,
                ):
                    self._directory_fd = -1
            if self._directory_fd_owner.close_diagnostic is None:
                self._directory_fd = -1
                self._directory_fd_owner = None
        if self._directory_fd_owner is None and self._root is not None:
            self._root.close()
            if self._root.fd < 0:
                self._root = None
        if self._directory_fd < 0 and self._root is None:
            self._marker_state.clear()

    def _release_restore_fd_journal(self) -> None:
        journal = self._restore_fd_journal
        if journal is None:
            return
        if journal.owner is not None:
            raise RuntimeError(
                "rollback snapshot restore FD finalization is still pending"
            )
        journal.diagnostic = None
        self._restore_fd_journal = None

    def _validate_directory(self, *, marker_required: bool = True) -> None:
        if self._root is None:
            if not self.directory_pending:
                return
            raise RuntimeError("rollback cleanup lost trusted root authority")
        self._root.validate()
        opened = os.fstat(self._directory_fd)
        if (opened.st_dev, opened.st_ino) != (
            self._directory_device,
            self._directory_inode,
        ):
            raise RuntimeError("rollback snapshot directory ownership changed")
        linked_name = self.directory.name
        if self._directory_quarantine_name is not None:
            try:
                os.stat(
                    self._directory_quarantine_name,
                    dir_fd=self._root.fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                linked_name = self._directory_quarantine_name
        try:
            linked = os.stat(
                linked_name,
                dir_fd=self._root.fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            if self._rmdir_attempted:
                self.directory_pending = False
                return
            raise RuntimeError(
                "rollback snapshot directory ownership changed"
            ) from None
        if not stat.S_ISDIR(linked.st_mode) or (
            linked.st_dev,
            linked.st_ino,
        ) != (self._directory_device, self._directory_inode):
            raise RuntimeError("rollback snapshot directory ownership changed")
        if marker_required:
            assert self._marker_identity is not None
            marker = _read_bounded_json_at(
                self._directory_fd, self._marker_identity, 1024 * 1024
            )
            if marker != self._marker_state:
                raise RuntimeError("rollback snapshot ownership marker mismatch")

    def _unlink_file(self, pending_field: str, expected: _FileIdentity | None) -> None:
        if not getattr(self, pending_field):
            return
        assert expected is not None
        self._validate_directory(marker_required=self.marker_pending)
        quarantine = self._quarantine_names.setdefault(
            expected.name,
            f".delete-{self.owner_token}-{expected.name}",
        )
        try:
            quarantined = os.stat(
                quarantine, dir_fd=self._directory_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            try:
                entry = os.stat(
                    expected.name,
                    dir_fd=self._directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if quarantine not in self._unlink_attempted:
                    raise RuntimeError(
                        f"rollback snapshot file ownership changed: {expected.name}"
                    ) from None
                setattr(self, pending_field, False)
                return
            if not stat.S_ISREG(entry.st_mode) or (
                entry.st_dev,
                entry.st_ino,
            ) != (expected.device, expected.inode):
                raise RuntimeError(
                    f"rollback snapshot file ownership changed: {expected.name}"
                )
            os.rename(
                expected.name,
                quarantine,
                src_dir_fd=self._directory_fd,
                dst_dir_fd=self._directory_fd,
            )
            quarantined = os.stat(
                quarantine, dir_fd=self._directory_fd, follow_symlinks=False
            )
        if not stat.S_ISREG(quarantined.st_mode) or (
            quarantined.st_dev,
            quarantined.st_ino,
        ) != (expected.device, expected.inode):
            raise RuntimeError(
                f"rollback snapshot file ownership changed: {expected.name}"
            )
        self._unlink_attempted.add(quarantine)
        os.unlink(quarantine, dir_fd=self._directory_fd)
        setattr(self, pending_field, False)

    def cleanup(self) -> None:
        if self._restore_fd_journal is not None:
            self._restore_fd_journal.finalize()
        if not self.directory_pending:
            self._finalize_fds()
            self._release_restore_fd_journal()
            return
        self._unlink_file("data_pending", self._data_identity)
        self._unlink_file("header_pending", self._header_identity)
        if self.marker_pending:
            self._validate_directory()
            remaining = set(os.listdir(self._directory_fd))
            if remaining - {_MARKER}:
                raise RuntimeError(
                    "rollback snapshot directory contains unexpected files: "
                    f"{sorted(remaining - {_MARKER})}"
                )
            assert self._marker_identity is not None
            self._unlink_file("marker_pending", self._marker_identity)
        if self.directory_pending:
            self._validate_directory(marker_required=False)
            if os.listdir(self._directory_fd):
                raise RuntimeError(
                    "rollback snapshot directory contains unexpected files"
                )
            assert self._root is not None
            if self._directory_quarantine_name is None:
                self._directory_quarantine_name = (
                    f".delete-{self.owner_token}-{self.directory.name}"
                )
            try:
                os.stat(
                    self._directory_quarantine_name,
                    dir_fd=self._root.fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                os.rename(
                    self.directory.name,
                    self._directory_quarantine_name,
                    src_dir_fd=self._root.fd,
                    dst_dir_fd=self._root.fd,
                )
            linked = os.stat(
                self._directory_quarantine_name,
                dir_fd=self._root.fd,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(linked.st_mode) or (
                linked.st_dev,
                linked.st_ino,
            ) != (self._directory_device, self._directory_inode):
                raise RuntimeError("rollback snapshot directory ownership changed")
            self._rmdir_attempted = True
            os.rmdir(self._directory_quarantine_name, dir_fd=self._root.fd)
            self.directory_pending = False
            os.fsync(self._root.fd)
        if not self.directory_pending:
            self._finalize_fds()
            self._release_restore_fd_journal()


def snapshot_parent(parent: str | os.PathLike[str] | None) -> Path:
    """Validate and return the configured trusted snapshot root."""

    root = _open_trusted_root(parent)
    try:
        return root.path
    finally:
        root.close()
