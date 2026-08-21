# SPDX-License-Identifier: Apache-2.0

"""Ownership-safe visibility markers for managed asynchronous checkpoints.

This is a deliberately narrow Linux implementation.  Marker operations are
relative to a retained checkpoint-directory descriptor and never follow a
marker symlink.  The marker is an integrity/ownership guard against stale or
accidentally replaced files; it is not a security boundary against a process
which can arbitrarily modify the checkpoint directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from areal.engine.megatron_utils.checkpoint_snapshot import _OwnedFD

MANAGED_ASYNC_INCOMPLETE = ".areal-managed-async-incomplete.json"
MANAGED_ASYNC_COMPLETE = ".areal-managed-async-complete.json"
MANAGED_ASYNC_TEMP_PREFIX = ".areal-managed-async-complete.tmp."

_SCHEMA = "areal.managed-async-checkpoint-marker"
_VERSION = 2
_MAX_MARKER_BYTES = 1024 * 1024
_MAX_METADATA_BYTES = 64 * 1024 * 1024
_MARKER_MODE = 0o600
_BACKEND = "torch_dist"

_COMMON_FIELDS = {
    "schema",
    "version",
    "state",
    "checkpoint_path",
    "checkpoint_directory",
    "checkpoint_id",
    "logical_call_id",
    "mcore_async_call_index",
    "participant_world_size",
    "participant_ranks",
    "control_group",
    "managed_leaves",
    "managed_leaves_digest",
    "request_digest",
    "created_time_ns",
    "mcore_version",
    "backend",
    "metadata",
}


@dataclass(frozen=True)
class MarkerFileIdentity:
    name: str
    device: int
    inode: int
    file_type: int
    size: int


@dataclass
class DirectoryAuthority:
    """Retained openat chain from ``/`` to the checkpoint directory."""

    path: str
    component_names: tuple[str, ...]
    owners: list[_OwnedFD]

    @property
    def final_owner(self) -> _OwnedFD:
        if not self.owners:
            raise RuntimeError("managed async checkpoint directory authority is closed")
        return self.owners[-1]

    @property
    def dir_fd(self) -> int:
        fd = self.final_owner.fd
        if fd < 0:
            raise RuntimeError("managed async checkpoint directory authority is closed")
        return fd

    def validate(self) -> None:
        if len(self.owners) != len(self.component_names) + 1:
            raise RuntimeError(
                "managed async checkpoint directory authority is incomplete"
            )
        for index, owner in enumerate(self.owners):
            opened = os.fstat(owner.fd)
            actual = (opened.st_dev, opened.st_ino, stat.S_IFMT(opened.st_mode))
            if actual != owner.expected_signature or actual[2] != stat.S_IFDIR:
                raise RuntimeError(
                    "managed async checkpoint directory FD ownership changed: "
                    f"expected={owner.expected_signature}, actual={actual}"
                )
            if index == 0:
                continue
            parent = self.owners[index - 1]
            name = self.component_names[index - 1]
            named = os.stat(name, dir_fd=parent.fd, follow_symlinks=False)
            named_signature = (
                named.st_dev,
                named.st_ino,
                stat.S_IFMT(named.st_mode),
            )
            if named_signature != owner.expected_signature:
                raise RuntimeError(
                    "managed async checkpoint parent traversal was replaced: "
                    f"component={name!r}, expected={owner.expected_signature}, "
                    f"actual={named_signature}"
                )

    def close(self) -> None:
        errors: list[BaseException] = []
        for owner in reversed(tuple(self.owners)):
            try:
                owner.close()
            except BaseException as error:
                errors.append(error)
            if owner.fd < 0:
                self.owners.remove(owner)
        if errors:
            primary = errors[0]
            for error in errors[1:]:
                primary.add_note(f"another directory FD close failed: {error!r}")
            raise primary


@dataclass
class ManagedAsyncMarkerAuthority:
    """Authority and post-commit cleanup journal for one checkpoint."""

    path: str
    directory_device: int
    directory_inode: int
    payload_base: dict[str, Any]
    directory: DirectoryAuthority
    incomplete: MarkerFileIdentity | None = None
    temporary: MarkerFileIdentity | None = None
    complete: MarkerFileIdentity | None = None
    short_fd_owners: list[_OwnedFD] = field(default_factory=list, repr=False)
    prepared: bool = False
    committed: bool = False
    cleanup_pending: bool = False
    cleanup_validation_pending: bool = False
    directory_fsync_pending: bool = False
    cleanup_diagnostic: str | None = None
    diagnostics: list[str] = field(default_factory=list)

    @property
    def dir_fd(self) -> int:
        return self.directory.dir_fd

    def validate_directory(self) -> None:
        self.directory.validate()
        opened = os.fstat(self.dir_fd)
        if (opened.st_dev, opened.st_ino) != (
            self.directory_device,
            self.directory_inode,
        ):
            raise RuntimeError("managed async checkpoint final directory changed")

    def finalize_pending_fds(self) -> None:
        errors: list[BaseException] = []
        for owner in tuple(self.short_fd_owners):
            try:
                owner.close()
            except BaseException as error:
                errors.append(error)
            if owner.fd < 0:
                self.short_fd_owners.remove(owner)
        if errors:
            primary = errors[0]
            for error in errors[1:]:
                primary.add_note(f"another marker FD close failed: {error!r}")
            raise primary

    def close(self) -> None:
        self.finalize_pending_fds()
        self.directory.close()


@dataclass(frozen=True)
class MarkerPublishOutcome:
    committed: bool
    cleanup_pending: bool
    cleanup_diagnostic: str | None


def canonical_checkpoint_path(path: str) -> str:
    """Canonicalize spelling without resolving any symlink."""
    return os.path.abspath(os.path.normpath(path))


def new_checkpoint_id() -> str:
    return uuid.uuid4().hex


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def canonical_leaf_identities(
    identities: dict[tuple[int, ...], dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    leaves = [
        {"tree_path": list(path), "identity": identity}
        for path, identity in sorted(identities.items())
    ]
    return leaves, _digest(leaves)


def canonical_ranked_leaf_identities(
    ranked_leaves: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Validate and digest the all-rank ordered leaf manifest."""
    for expected_rank, entry in enumerate(ranked_leaves):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"rank", "leaves"}
            or entry["rank"] != expected_rank
            or not isinstance(entry["leaves"], list)
        ):
            raise RuntimeError(
                "managed async marker rank/leaf manifest is malformed at "
                f"rank {expected_rank}: {entry!r}"
            )
    return ranked_leaves, _digest(ranked_leaves)


def build_marker_payload(
    *,
    state: str,
    checkpoint_path: str,
    checkpoint_directory: tuple[int, int],
    checkpoint_id: str,
    logical_call_id: int,
    mcore_async_call_index: int,
    participant_ranks: tuple[int, ...],
    control_group_backend: str,
    managed_leaves: list[dict[str, Any]],
    managed_leaves_digest: str,
    mcore_version: str,
    created_time_ns: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": _SCHEMA,
        "version": _VERSION,
        "state": state,
        "checkpoint_path": canonical_checkpoint_path(checkpoint_path),
        "checkpoint_directory": {
            "device": checkpoint_directory[0],
            "inode": checkpoint_directory[1],
        },
        "checkpoint_id": checkpoint_id,
        "logical_call_id": logical_call_id,
        "mcore_async_call_index": mcore_async_call_index,
        "participant_world_size": len(participant_ranks),
        "participant_ranks": list(participant_ranks),
        "control_group": {
            "backend": control_group_backend,
            "ranks": list(participant_ranks),
            "requires_world_membership": True,
        },
        "managed_leaves": managed_leaves,
        "managed_leaves_digest": managed_leaves_digest,
        "request_digest": "",
        "created_time_ns": time.time_ns()
        if created_time_ns is None
        else created_time_ns,
        "mcore_version": mcore_version,
        "backend": _BACKEND,
        "metadata": metadata,
    }
    payload["request_digest"] = _digest(_request_binding(payload))
    return payload


def _request_binding(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "schema",
            "version",
            "checkpoint_path",
            "checkpoint_directory",
            "checkpoint_id",
            "logical_call_id",
            "mcore_async_call_index",
            "participant_world_size",
            "participant_ranks",
            "control_group",
            "managed_leaves",
            "managed_leaves_digest",
            "created_time_ns",
            "mcore_version",
            "backend",
        )
    }


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("short write while creating managed async marker")
        view = view[count:]


def _identity(name: str, info: os.stat_result) -> MarkerFileIdentity:
    return MarkerFileIdentity(
        name=name,
        device=info.st_dev,
        inode=info.st_ino,
        file_type=stat.S_IFMT(info.st_mode),
        size=info.st_size,
    )


def _validate_regular_marker_stat(
    name: str,
    info: os.stat_result,
    *,
    expected: MarkerFileIdentity | None = None,
) -> MarkerFileIdentity:
    kind = (
        "incomplete managed async save marker"
        if name == MANAGED_ASYNC_INCOMPLETE
        else "managed async marker"
    )
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{kind} {name!r} is not a regular file")
    if stat.S_IMODE(info.st_mode) != _MARKER_MODE:
        raise RuntimeError(
            f"{kind} {name!r} has unsafe permissions "
            f"{oct(stat.S_IMODE(info.st_mode))}; expected 0o600"
        )
    if info.st_uid != os.geteuid():
        raise RuntimeError(
            f"{kind} {name!r} is owned by uid={info.st_uid}; "
            f"expected uid={os.geteuid()}"
        )
    if info.st_size <= 1 or info.st_size > _MAX_MARKER_BYTES:
        raise RuntimeError(f"{kind} {name!r} has invalid size {info.st_size}")
    actual = _identity(name, info)
    if expected is not None and actual != expected:
        raise RuntimeError(
            f"{kind} {name!r} inode/type/size changed: "
            f"expected={expected}, actual={actual}"
        )
    return actual


def _lstat_at(dir_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _finalize_short_fd(authority: ManagedAsyncMarkerAuthority, owner: _OwnedFD) -> None:
    try:
        owner.close()
    finally:
        if owner.fd < 0 and owner in authority.short_fd_owners:
            authority.short_fd_owners.remove(owner)


def _create_file_at(
    authority: ManagedAsyncMarkerAuthority, name: str, payload: dict[str, Any]
) -> MarkerFileIdentity:
    authority.finalize_pending_fds()
    dir_fd = authority.dir_fd
    encoded = _canonical_json(payload)
    if len(encoded) > _MAX_MARKER_BYTES:
        raise RuntimeError(
            f"managed async marker payload is too large: {len(encoded)} bytes"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(name, flags, _MARKER_MODE, dir_fd=dir_fd)
    owner = _OwnedFD.capture(fd)
    authority.short_fd_owners.append(owner)
    try:
        _write_all(fd, encoded)
        os.fsync(fd)
        info = os.fstat(fd)
        identity = _validate_regular_marker_stat(name, info)
        named = _lstat_at(dir_fd, name)
        if named is None or (named.st_dev, named.st_ino) != (
            identity.device,
            identity.inode,
        ):
            raise RuntimeError(
                f"managed async marker {name!r} was replaced during creation"
            )
        return identity
    finally:
        _finalize_short_fd(authority, owner)


def _read_file_at(
    authority: ManagedAsyncMarkerAuthority,
    name: str,
    *,
    expected: MarkerFileIdentity | None = None,
) -> tuple[dict[str, Any], MarkerFileIdentity]:
    authority.finalize_pending_fds()
    dir_fd = authority.dir_fd
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(name, flags, dir_fd=dir_fd)
    owner = _OwnedFD.capture(fd)
    authority.short_fd_owners.append(owner)
    try:
        before = os.fstat(fd)
        identity = _validate_regular_marker_stat(name, before, expected=expected)
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, remaining)
            if not chunk:
                raise OSError(f"short read while reading managed async marker {name!r}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise RuntimeError(f"managed async marker {name!r} grew while reading")
        after = os.fstat(fd)
        _validate_regular_marker_stat(name, after, expected=identity)
        named = _lstat_at(dir_fd, name)
        if named is None or (named.st_dev, named.st_ino) != (
            identity.device,
            identity.inode,
        ):
            raise RuntimeError(
                f"managed async marker {name!r} was replaced while reading"
            )
        try:
            payload = json.loads(b"".join(chunks))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"managed async marker {name!r} is invalid JSON"
            ) from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"managed async marker {name!r} must contain an object")
        return payload, identity
    finally:
        _finalize_short_fd(authority, owner)


def _open_directory_chain(path: str, *, create_final: bool) -> DirectoryAuthority:
    canonical = canonical_checkpoint_path(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    root = _OwnedFD.capture(os.open("/", flags))
    owners = [root]
    components = tuple(part for part in canonical.split(os.sep) if part)
    try:
        for index, component in enumerate(components):
            parent = owners[-1]
            is_final = index == len(components) - 1
            named = _lstat_at(parent.fd, component)
            if is_final and create_final:
                if named is not None:
                    raise FileExistsError(
                        "managed async checkpoint requires a fresh directory: "
                        f"{canonical}"
                    )
                os.mkdir(component, 0o700, dir_fd=parent.fd)
                named = _lstat_at(parent.fd, component)
            if named is None:
                raise FileNotFoundError(
                    f"managed async checkpoint path component is missing: {component!r}"
                )
            if stat.S_ISLNK(named.st_mode) or not stat.S_ISDIR(named.st_mode):
                raise RuntimeError(
                    "managed async checkpoint path component must be a real "
                    f"directory; rejected {component!r} in {canonical!r}"
                )
            fd = os.open(component, flags, dir_fd=parent.fd)
            owner = _OwnedFD.capture(fd)
            owners.append(owner)
            opened = os.fstat(fd)
            if (
                opened.st_dev,
                opened.st_ino,
                stat.S_IFMT(opened.st_mode),
            ) != (named.st_dev, named.st_ino, stat.S_IFDIR):
                raise RuntimeError(
                    "managed async checkpoint path component changed while opening: "
                    f"{component!r}"
                )
    except BaseException as original:
        for owner in reversed(owners):
            try:
                owner.close()
            except BaseException as close_error:
                original.add_note(f"path authority close failed: {close_error!r}")
        raise
    return DirectoryAuthority(
        path=canonical,
        component_names=components,
        owners=owners,
    )


def _open_directory(path: str) -> ManagedAsyncMarkerAuthority:
    directory = _open_directory_chain(path, create_final=False)
    opened = os.fstat(directory.dir_fd)
    return ManagedAsyncMarkerAuthority(
        path=directory.path,
        directory_device=opened.st_dev,
        directory_inode=opened.st_ino,
        payload_base={},
        directory=directory,
    )


def create_incomplete_marker(
    *,
    path: str,
    checkpoint_id: str,
    logical_call_id: int,
    mcore_async_call_index: int,
    participant_ranks: tuple[int, ...],
    control_group_backend: str,
    managed_leaves: list[dict[str, Any]],
    managed_leaves_digest: str,
    mcore_version: str,
) -> ManagedAsyncMarkerAuthority:
    canonical = canonical_checkpoint_path(path)
    directory = _open_directory_chain(canonical, create_final=True)
    opened = os.fstat(directory.dir_fd)
    authority = ManagedAsyncMarkerAuthority(
        path=canonical,
        directory_device=opened.st_dev,
        directory_inode=opened.st_ino,
        payload_base={},
        directory=directory,
    )
    payload = build_marker_payload(
        state="incomplete",
        checkpoint_path=canonical,
        checkpoint_directory=(authority.directory_device, authority.directory_inode),
        checkpoint_id=checkpoint_id,
        logical_call_id=logical_call_id,
        mcore_async_call_index=mcore_async_call_index,
        participant_ranks=participant_ranks,
        control_group_backend=control_group_backend,
        managed_leaves=managed_leaves,
        managed_leaves_digest=managed_leaves_digest,
        mcore_version=mcore_version,
    )
    authority.payload_base = payload
    try:
        authority.incomplete = _create_file_at(
            authority, MANAGED_ASYNC_INCOMPLETE, payload
        )
        os.fsync(authority.dir_fd)
    except BaseException as original:
        # Publish the journal to the manager even though construction did not
        # return normally. A retryable pre-close failure must not strand a raw
        # descriptor in a local finally block.
        setattr(original, "marker_authority", authority)
        try:
            authority.close()
        except BaseException as close_error:
            original.add_note(
                f"managed async marker directory close also failed: {close_error!r}"
            )
        raise
    return authority


def _metadata_identity(authority: ManagedAsyncMarkerAuthority) -> dict[str, Any]:
    authority.validate_directory()
    authority.finalize_pending_fds()
    name = "metadata.json"
    fd = os.open(
        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=authority.dir_fd
    )
    owner = _OwnedFD.capture(fd)
    authority.short_fd_owners.append(owner)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("MCore metadata.json is not a regular file")
        if before.st_size <= 0 or before.st_size > _MAX_METADATA_BYTES:
            raise RuntimeError(f"MCore metadata.json has invalid size {before.st_size}")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise OSError("short read while hashing MCore metadata.json")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise RuntimeError("MCore metadata.json changed while hashing")
        named = _lstat_at(authority.dir_fd, name)
        if named is None or (named.st_dev, named.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise RuntimeError("MCore metadata.json was replaced while hashing")
        return {
            "name": name,
            "device": before.st_dev,
            "inode": before.st_ino,
            "size": before.st_size,
            "sha256": digest.hexdigest(),
        }
    finally:
        _finalize_short_fd(authority, owner)


def _validate_payload_shape(payload: dict[str, Any], *, state: str) -> None:
    fields = set(payload)
    if fields != _COMMON_FIELDS:
        raise RuntimeError(
            "managed async marker fields mismatch: "
            f"missing={sorted(_COMMON_FIELDS - fields)}, "
            f"extra={sorted(fields - _COMMON_FIELDS)}"
        )
    if payload["schema"] != _SCHEMA or payload["version"] != _VERSION:
        raise RuntimeError("unsupported managed async checkpoint marker schema")
    if payload["state"] != state:
        raise RuntimeError(
            f"managed async marker state mismatch: expected={state!r}, "
            f"actual={payload['state']!r}"
        )
    integer_fields = (
        "logical_call_id",
        "mcore_async_call_index",
        "participant_world_size",
        "created_time_ns",
    )
    for field_name in integer_fields:
        value = payload[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(
                f"managed async marker field {field_name!r} must be a non-negative int"
            )
    try:
        checkpoint_uuid = uuid.UUID(hex=payload["checkpoint_id"])
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("managed async marker checkpoint_id is invalid") from error
    if checkpoint_uuid.hex != payload["checkpoint_id"]:
        raise RuntimeError("managed async marker checkpoint_id is invalid")
    if not isinstance(payload["checkpoint_path"], str):
        raise RuntimeError("managed async marker checkpoint_path must be a string")
    directory = payload["checkpoint_directory"]
    if not isinstance(directory, dict) or set(directory) != {"device", "inode"}:
        raise RuntimeError("managed async marker directory identity is invalid")
    if any(
        isinstance(directory[key], bool) or not isinstance(directory[key], int)
        for key in directory
    ) or any(directory[key] < 0 for key in directory):
        raise RuntimeError(
            "managed async marker directory identity must contain integers"
        )
    ranks = payload["participant_ranks"]
    if (
        payload["participant_world_size"] <= 0
        or not isinstance(ranks, list)
        or any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks)
        or ranks != list(range(payload["participant_world_size"]))
    ):
        raise RuntimeError("managed async marker participant ranks are invalid")
    control = payload["control_group"]
    if not isinstance(control, dict) or set(control) != {
        "backend",
        "ranks",
        "requires_world_membership",
    }:
        raise RuntimeError("managed async marker control-group identity is invalid")
    if (
        control["backend"] != "gloo"
        or control["ranks"] != ranks
        or control["requires_world_membership"] is not True
    ):
        raise RuntimeError("managed async marker control-group contract mismatch")
    leaves = payload["managed_leaves"]
    if (
        not isinstance(leaves, list)
        or _digest(leaves) != payload["managed_leaves_digest"]
    ):
        raise RuntimeError("managed async marker leaf identity digest mismatch")
    if payload["request_digest"] != _digest(_request_binding(payload)):
        raise RuntimeError("managed async marker request payload digest mismatch")
    if payload["mcore_version"] != "0.17.0" or payload["backend"] != _BACKEND:
        raise RuntimeError("managed async marker MCore/backend contract mismatch")
    metadata = payload["metadata"]
    if state == "incomplete":
        if metadata is not None:
            raise RuntimeError("incomplete managed async marker must not bind metadata")
    elif not isinstance(metadata, dict) or set(metadata) != {
        "name",
        "device",
        "inode",
        "size",
        "sha256",
    }:
        raise RuntimeError("complete managed async marker metadata identity is invalid")


def _validate_expected_payload(
    payload: dict[str, Any],
    *,
    state: str,
    path: str,
    directory_identity: tuple[int, int],
    participant_ranks: tuple[int, ...],
    managed_leaves: list[dict[str, Any]] | None,
    managed_leaves_digest: str | None,
) -> None:
    _validate_payload_shape(payload, state=state)
    if payload["checkpoint_path"] != canonical_checkpoint_path(path):
        raise RuntimeError("managed async marker checkpoint path mismatch")
    directory = payload["checkpoint_directory"]
    if (directory["device"], directory["inode"]) != directory_identity:
        raise RuntimeError(
            "managed async marker checkpoint directory identity mismatch"
        )
    if payload["participant_ranks"] != list(participant_ranks):
        raise RuntimeError("managed async marker participant rank mismatch")
    if managed_leaves is None or managed_leaves_digest is None:
        raise RuntimeError(
            "managed async checkpoint marker requires a managed optimizer identity"
        )
    if (
        payload["managed_leaves"] != managed_leaves
        or payload["managed_leaves_digest"] != managed_leaves_digest
    ):
        raise RuntimeError("managed async marker leaf identity mismatch")


def _complete_payload(authority: ManagedAsyncMarkerAuthority) -> dict[str, Any]:
    payload = dict(authority.payload_base)
    payload["state"] = "complete"
    payload["metadata"] = _metadata_identity(authority)
    return payload


def _validate_owned_complete(
    authority: ManagedAsyncMarkerAuthority, complete_payload: dict[str, Any]
) -> None:
    final_info = _lstat_at(authority.dir_fd, MANAGED_ASYNC_COMPLETE)
    if final_info is None:
        raise RuntimeError("managed async complete marker disappeared")
    identity = _validate_regular_marker_stat(MANAGED_ASYNC_COMPLETE, final_info)
    owned_inode = (
        (authority.complete.device, authority.complete.inode)
        if authority.complete is not None
        else (
            (authority.temporary.device, authority.temporary.inode)
            if authority.temporary is not None
            else None
        )
    )
    if owned_inode is None or (identity.device, identity.inode) != owned_inode:
        raise RuntimeError("managed async complete marker ownership changed")
    authority.complete = identity
    payload, _ = _read_file_at(
        authority, MANAGED_ASYNC_COMPLETE, expected=authority.complete
    )
    if payload != complete_payload:
        raise RuntimeError("managed async complete marker payload changed")


def retry_post_commit_cleanup(authority: ManagedAsyncMarkerAuthority) -> bool:
    """Best-effort cleanup after the visibility commit point.

    Failure here never revokes ``authority.committed``.  The caller may retry
    while retained directory/short-FD authority remains.
    """
    if not authority.committed:
        raise RuntimeError("managed async marker cleanup requested before commit")
    errors: list[BaseException] = []
    try:
        authority.finalize_pending_fds()
    except BaseException as error:
        errors.append(error)

    if not errors and authority.cleanup_validation_pending:
        try:
            authority.validate_directory()
            _validate_owned_complete(authority, _complete_payload(authority))
        except BaseException as error:
            errors.append(error)
        else:
            authority.cleanup_validation_pending = False

    temp_name = f"{MANAGED_ASYNC_TEMP_PREFIX}{authority.payload_base['checkpoint_id']}"
    if not errors and authority.temporary is not None:
        try:
            authority.validate_directory()
            current_temp = _lstat_at(authority.dir_fd, temp_name)
            if current_temp is None:
                authority.temporary = None
            else:
                if authority.temporary is None or (
                    current_temp.st_dev,
                    current_temp.st_ino,
                ) != (authority.temporary.device, authority.temporary.inode):
                    raise RuntimeError(
                        "managed async temporary marker ownership changed"
                    )
                try:
                    os.unlink(temp_name, dir_fd=authority.dir_fd)
                except BaseException as unlink_error:
                    after = _lstat_at(authority.dir_fd, temp_name)
                    if after is None:
                        authority.temporary = None
                        authority.diagnostics.append(
                            "temporary marker unlink committed before error: "
                            f"{unlink_error!r}"
                        )
                    else:
                        raise
                else:
                    authority.temporary = None
        except BaseException as error:
            errors.append(error)

    if not errors and authority.directory_fsync_pending:
        try:
            authority.validate_directory()
            os.fsync(authority.dir_fd)
        except BaseException as error:
            errors.append(error)
        else:
            authority.directory_fsync_pending = False

    if (
        not errors
        and authority.temporary is None
        and not authority.short_fd_owners
        and not authority.cleanup_validation_pending
        and not authority.directory_fsync_pending
    ):
        try:
            authority.directory.close()
        except BaseException as error:
            errors.append(error)

    authority.cleanup_pending = bool(
        authority.temporary
        or authority.short_fd_owners
        or authority.directory.owners
        or authority.cleanup_validation_pending
        or authority.directory_fsync_pending
        or errors
    )
    if errors:
        authority.cleanup_diagnostic = repr(errors[-1])
        authority.diagnostics.append(authority.cleanup_diagnostic)
    elif not authority.cleanup_pending:
        authority.cleanup_diagnostic = None
    return not authority.cleanup_pending


def prepare_complete_marker(authority: ManagedAsyncMarkerAuthority) -> None:
    """Durably publish and validate ``complete`` while keeping load fenced.

    This phase is reversible from the loader's perspective because the owned
    ``incomplete`` marker remains present.  It is therefore safe to run before
    the all-rank commit decision.
    """
    if authority.committed:
        raise RuntimeError("managed async marker is already committed")
    if authority.prepared:
        authority.validate_directory()
        _validate_owned_complete(authority, _complete_payload(authority))
        return
    authority.validate_directory()
    authority.finalize_pending_fds()
    incomplete_info = _lstat_at(authority.dir_fd, MANAGED_ASYNC_INCOMPLETE)
    final_info = _lstat_at(authority.dir_fd, MANAGED_ASYNC_COMPLETE)
    if incomplete_info is not None:
        incomplete_payload, incomplete_identity = _read_file_at(
            authority,
            MANAGED_ASYNC_INCOMPLETE,
            expected=authority.incomplete,
        )
        authority.incomplete = incomplete_identity
    elif final_info is not None and authority.incomplete is not None:
        # Reconcile an unlink which took effect before its caller observed an
        # exception.  A final marker must already prove forward progress.
        incomplete_payload = dict(authority.payload_base)
        authority.incomplete = None
    elif authority.incomplete is None and final_info is not None:
        incomplete_payload = dict(authority.payload_base)
    else:
        raise RuntimeError(
            "managed async incomplete marker disappeared before complete publish"
        )
    _validate_expected_payload(
        incomplete_payload,
        state="incomplete",
        path=authority.path,
        directory_identity=(authority.directory_device, authority.directory_inode),
        participant_ranks=tuple(incomplete_payload["participant_ranks"]),
        managed_leaves=incomplete_payload["managed_leaves"],
        managed_leaves_digest=incomplete_payload["managed_leaves_digest"],
    )
    complete_payload = _complete_payload(authority)

    temp_name = f"{MANAGED_ASYNC_TEMP_PREFIX}{incomplete_payload['checkpoint_id']}"
    temp_info = _lstat_at(authority.dir_fd, temp_name)
    if authority.temporary is not None and temp_info is None and final_info is not None:
        # Same post-effect reconciliation for temporary unlink.
        authority.temporary = None
    if authority.temporary is None and final_info is None:
        authority.temporary = _create_file_at(authority, temp_name, complete_payload)
        temp_info = _lstat_at(authority.dir_fd, temp_name)
    elif authority.temporary is not None:
        _read_file_at(authority, temp_name, expected=authority.temporary)

    if final_info is None:
        assert authority.temporary is not None
        try:
            os.link(
                temp_name,
                MANAGED_ASYNC_COMPLETE,
                src_dir_fd=authority.dir_fd,
                dst_dir_fd=authority.dir_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            # A racing publisher or external file won.  Reconcile by inode;
            # never overwrite it even when its payload happens to match.
            final_info = _lstat_at(authority.dir_fd, MANAGED_ASYNC_COMPLETE)
            if final_info is None:
                raise
        else:
            final_info = _lstat_at(authority.dir_fd, MANAGED_ASYNC_COMPLETE)
    assert final_info is not None
    final_identity = _validate_regular_marker_stat(MANAGED_ASYNC_COMPLETE, final_info)
    owned_final_inode = (
        (authority.temporary.device, authority.temporary.inode)
        if authority.temporary is not None
        else (
            (authority.complete.device, authority.complete.inode)
            if authority.complete is not None
            else None
        )
    )
    if (
        owned_final_inode is None
        or (
            final_identity.device,
            final_identity.inode,
        )
        != owned_final_inode
    ):
        raise FileExistsError(
            "managed async complete marker already exists and is not owned by "
            "this request"
        )
    authority.complete = final_identity
    authority.validate_directory()
    os.fsync(authority.dir_fd)

    _validate_owned_complete(authority, complete_payload)
    authority.prepared = True


def commit_prepared_marker(
    authority: ManagedAsyncMarkerAuthority,
) -> MarkerPublishOutcome:
    """Cross the visibility commit point after the all-rank decision.

    Once the owned incomplete marker is absent, every subsequent operation is
    cleanup-only and cannot revoke the committed checkpoint.
    """
    if authority.committed:
        retry_post_commit_cleanup(authority)
        return MarkerPublishOutcome(
            committed=True,
            cleanup_pending=authority.cleanup_pending,
            cleanup_diagnostic=authority.cleanup_diagnostic,
        )
    if not authority.prepared:
        raise RuntimeError("managed async complete marker was not prepared")

    authority.validate_directory()
    current_incomplete = _lstat_at(authority.dir_fd, MANAGED_ASYNC_INCOMPLETE)
    if current_incomplete is not None:
        if authority.incomplete is None or (
            current_incomplete.st_dev,
            current_incomplete.st_ino,
        ) != (authority.incomplete.device, authority.incomplete.inode):
            raise RuntimeError("managed async incomplete marker ownership changed")
        try:
            os.unlink(MANAGED_ASYNC_INCOMPLETE, dir_fd=authority.dir_fd)
        except BaseException as unlink_error:
            after = _lstat_at(authority.dir_fd, MANAGED_ASYNC_INCOMPLETE)
            if after is not None:
                if (after.st_dev, after.st_ino) != (
                    authority.incomplete.device,
                    authority.incomplete.inode,
                ):
                    raise RuntimeError(
                        "managed async incomplete marker was replaced during commit"
                    ) from unlink_error
                raise
            authority.incomplete = None
            authority.committed = True
            authority.diagnostics.append(
                f"incomplete marker unlink committed before error: {unlink_error!r}"
            )
        else:
            after = _lstat_at(authority.dir_fd, MANAGED_ASYNC_INCOMPLETE)
            if after is not None:
                raise RuntimeError(
                    "managed async incomplete marker still exists after unlink"
                )
            authority.incomplete = None
            authority.committed = True

    # The absence of the owned incomplete marker is the irreversible
    # visibility commit point. Everything below is cleanup-only.
    if not authority.committed:
        authority.committed = True
    authority.cleanup_validation_pending = True
    authority.directory_fsync_pending = True
    retry_post_commit_cleanup(authority)
    return MarkerPublishOutcome(
        committed=True,
        cleanup_pending=authority.cleanup_pending,
        cleanup_diagnostic=authority.cleanup_diagnostic,
    )


def publish_complete_marker(
    authority: ManagedAsyncMarkerAuthority,
) -> MarkerPublishOutcome:
    """Single-rank convenience wrapper used by marker-level tests."""
    if not authority.committed:
        prepare_complete_marker(authority)
    return commit_prepared_marker(authority)


def _validate_metadata_from_payload(
    authority: ManagedAsyncMarkerAuthority, payload: dict[str, Any]
) -> None:
    expected = payload["metadata"]
    actual = _metadata_identity(authority)
    if actual != expected:
        raise RuntimeError(
            "managed async marker metadata.json identity/digest mismatch: "
            f"expected={expected}, actual={actual}"
        )


def has_managed_async_marker(path: str) -> bool:
    """Probe marker presence without requiring a distributed manifest."""
    authority = _open_directory(path)
    try:
        authority.validate_directory()
        return any(
            name in (MANAGED_ASYNC_INCOMPLETE, MANAGED_ASYNC_COMPLETE)
            or name.startswith(MANAGED_ASYNC_TEMP_PREFIX)
            for name in os.listdir(authority.dir_fd)
        )
    finally:
        authority.close()


def validate_load_marker(
    *,
    path: str,
    participant_ranks: tuple[int, ...] | None,
    managed_leaves: list[dict[str, Any]] | None,
    managed_leaves_digest: str | None,
) -> dict[str, Any] | None:
    authority = _open_directory(path)
    try:
        authority.validate_directory()
        names = os.listdir(authority.dir_fd)
        related = [
            name
            for name in names
            if name in (MANAGED_ASYNC_INCOMPLETE, MANAGED_ASYNC_COMPLETE)
            or name.startswith(MANAGED_ASYNC_TEMP_PREFIX)
        ]
        if not related:
            return None
        if participant_ranks is None:
            raise RuntimeError(
                "incomplete managed async save or complete managed marker requires "
                "an explicit checkpoint participant manifest"
            )
        for name in related:
            info = _lstat_at(authority.dir_fd, name)
            assert info is not None
            _validate_regular_marker_stat(name, info)
        incomplete = MANAGED_ASYNC_INCOMPLETE in related
        temporary = any(name.startswith(MANAGED_ASYNC_TEMP_PREFIX) for name in related)
        complete = MANAGED_ASYNC_COMPLETE in related
        incomplete_payload = None
        if incomplete:
            incomplete_payload, _ = _read_file_at(authority, MANAGED_ASYNC_INCOMPLETE)
            _validate_expected_payload(
                incomplete_payload,
                state="incomplete",
                path=path,
                directory_identity=(
                    authority.directory_device,
                    authority.directory_inode,
                ),
                participant_ranks=participant_ranks,
                managed_leaves=managed_leaves,
                managed_leaves_digest=managed_leaves_digest,
            )
        complete_payload = None
        if complete:
            complete_payload, _ = _read_file_at(authority, MANAGED_ASYNC_COMPLETE)
            _validate_expected_payload(
                complete_payload,
                state="complete",
                path=path,
                directory_identity=(
                    authority.directory_device,
                    authority.directory_inode,
                ),
                participant_ranks=participant_ranks,
                managed_leaves=managed_leaves,
                managed_leaves_digest=managed_leaves_digest,
            )
            _validate_metadata_from_payload(authority, complete_payload)
        if incomplete_payload is not None and complete_payload is not None:
            if (
                incomplete_payload["request_digest"]
                != complete_payload["request_digest"]
            ):
                raise RuntimeError(
                    "incomplete and complete managed async markers belong to "
                    "different requests"
                )
        if incomplete:
            raise RuntimeError(
                f"checkpoint is an incomplete managed async save: {path}"
            )
        if not complete:
            raise RuntimeError("managed async marker set is incomplete")
        assert complete_payload is not None
        if temporary:
            temp_names = sorted(
                name for name in related if name.startswith(MANAGED_ASYNC_TEMP_PREFIX)
            )
            expected_temp = (
                f"{MANAGED_ASYNC_TEMP_PREFIX}{complete_payload['checkpoint_id']}"
            )
            if temp_names != [expected_temp]:
                raise RuntimeError(
                    "checkpoint has an unowned temporary managed async marker"
                )
            temp_payload, temp_identity = _read_file_at(authority, expected_temp)
            complete_info = _lstat_at(authority.dir_fd, MANAGED_ASYNC_COMPLETE)
            assert complete_info is not None
            if (
                temp_identity.device,
                temp_identity.inode,
            ) != (complete_info.st_dev, complete_info.st_ino):
                raise RuntimeError(
                    "temporary managed async marker is not linked to complete"
                )
            if temp_payload != complete_payload:
                raise RuntimeError(
                    "temporary managed async marker payload does not match complete"
                )
        return complete_payload
    finally:
        authority.close()
