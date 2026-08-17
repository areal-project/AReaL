# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno
import gc
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from areal.engine.megatron_utils import checkpoint_snapshot as snapshot_module
from areal.engine.megatron_utils import checkpointer as checkpointer_module
from areal.engine.megatron_utils.checkpoint_snapshot import (
    MAX_SNAPSHOT_CHUNK_BYTES,
    MAX_SNAPSHOT_CHUNKS_PER_SLAB,
    MIN_SNAPSHOT_CHUNK_BYTES,
    DiskSnapshotCleanup,
    DiskTensorRollbackSnapshot,
    SnapshotCapacityReport,
    SnapshotRequirement,
    discover_orphaned_snapshot_directories,
    preflight_snapshot_requirements,
    validate_shared_snapshot_capacity,
)
from areal.engine.megatron_utils.checkpointer import MegatronCheckpointManager
from areal.engine.megatron_utils.gpu_staged_optimizer import (
    GPUStagedAdamW,
    GPUStagedAdamWConfig,
)
from areal.engine.megatron_utils.gpu_staged_optimizer_checkpoint import (
    ManagedCheckpointPhaseError,
    ManagedCheckpointTransactionPhase,
    abort_managed_checkpoint_load,
    begin_managed_checkpoint_load,
    commit_managed_checkpoint_load,
    create_managed_checkpoint_load_transaction,
    decide_managed_checkpoint_commit,
    is_managed_optimizer_tensor_checkpoint_key,
    merge_managed_optimizer_tensor_manifests,
    poison_managed_checkpoint_transaction,
    prepare_managed_checkpoint_commit,
    prepare_managed_checkpoint_load,
    prepare_managed_checkpoint_recovery,
    reset_managed_optimizer_from_model,
    retry_managed_checkpoint_cleanup,
    validate_managed_optimizer_outer_state,
    validate_managed_optimizer_source_tensor_metadata,
    vote_managed_checkpoint_phase,
)
from areal.engine.megatron_utils.optimizer_chain import checkpoint_awex_residency
from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


def _direct_checkpoint_optimizer(
    *, snapshot_root: str | None = None, snapshot_chunk_mb: float = 64.0
) -> tuple[torch.nn.Parameter, GPUStagedAdamW]:
    param = torch.nn.Parameter(
        torch.linspace(-1, 1, 5, device="cuda", dtype=torch.bfloat16)
    )
    optimizer = GPUStagedAdamW(
        [param],
        staged_config=GPUStagedAdamWConfig(
            buffer_count=1,
            bucket_size_mb=4 / (1024 * 1024),
            checkpoint_snapshot_root=snapshot_root,
            checkpoint_snapshot_chunk_mb=snapshot_chunk_mb,
        ),
    )
    optimizer.bind_owned_params(optimizer.param_groups)
    return param, optimizer


def _direct_checkpoint_optimizer_with_groups(
    group_count: int,
) -> GPUStagedAdamW:
    groups = [
        {
            "params": [
                torch.nn.Parameter(
                    torch.linspace(
                        group_index,
                        group_index + 1,
                        3,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                )
            ],
            "lr": 0.01 * (group_index + 1),
        }
        for group_index in range(group_count)
    ]
    optimizer = GPUStagedAdamW(
        groups,
        staged_config=GPUStagedAdamWConfig(
            buffer_count=1,
            bucket_size_mb=4 / (1024 * 1024),
        ),
    )
    optimizer.bind_owned_params(optimizer.param_groups)
    return optimizer


def _detached_state_dict(optimizer: GPUStagedAdamW) -> dict:
    state_dict = optimizer.state_dict()
    return {
        "state": {
            index: {key: value.clone() for key, value in state.items()}
            for index, state in state_dict["state"].items()
        },
        "param_groups": [dict(group) for group in state_dict["param_groups"]],
    }


def _instrument_slab_rollback_actions(
    optimizer: GPUStagedAdamW, callback
) -> dict[str, object]:
    """Inject failures around disk-backed slab restores without replacing slabs."""
    assert optimizer._checkpoint_rollback is not None
    originals: dict[str, object] = {}
    for action in optimizer._checkpoint_rollback.actions:
        if not action.name.startswith("slab."):
            continue
        slab_name = action.name.removeprefix("slab.")
        original_restore = action.restore
        originals[slab_name] = original_restore

        def wrapped_restore(
            target, snapshot, *, name=slab_name, restore=original_restore
        ):
            callback(name)
            restore(target, snapshot)

        action.restore = wrapped_restore
    return originals


def _create_disk_snapshot(
    tmp_path: Path,
    source: torch.Tensor,
    *,
    chunk_bytes: int = 1024 * 1024,
    slab_key: str = "master",
) -> DiskTensorRollbackSnapshot:
    return DiskTensorRollbackSnapshot.create(
        source,
        parent=tmp_path,
        leaf_identity={"version": 1, "tree_path": [0], "signature": "test"},
        slab_key=slab_key,
        chunk_bytes=chunk_bytes,
        rank=0,
    )


def test_disk_rollback_snapshot_header_and_chunked_restore(tmp_path: Path) -> None:
    source = torch.arange((9 * 1024 * 1024) // 4 + 1, dtype=torch.float32)
    expected = source.clone()
    snapshot = _create_disk_snapshot(tmp_path, source)
    header = json.loads(snapshot.header_path.read_text())

    assert snapshot.directory.name.startswith("areal-managed-rollback-r0-l")
    assert header["schema"] == "areal.gpu_staged_adamw.rollback"
    assert header["version"] == 2
    assert header["dtype"] == "torch.float32"
    assert header["numel"] == source.numel()
    assert header["byte_length"] == source.numel() * source.element_size()
    assert len(header["chunk_checksums"]) == 10
    assert snapshot.data_path.stat().st_size == header["byte_length"]
    assert snapshot.directory.stat().st_mode & 0o777 == 0o700
    assert snapshot.data_path.stat().st_mode & 0o777 == 0o600
    assert snapshot.header_path.stat().st_mode & 0o777 == 0o600

    source.fill_(-1)
    snapshot.restore_into(source)

    torch.testing.assert_close(source, expected, rtol=0.0, atol=0.0)
    assert snapshot.restore_complete
    assert snapshot.next_chunk == len(snapshot.chunk_checksums)
    assert not snapshot.directory.exists()


def test_disk_rollback_snapshot_checksum_failure_is_retryable(
    tmp_path: Path,
) -> None:
    source = torch.arange(13, dtype=torch.float32)
    expected = source.clone()
    snapshot = _create_disk_snapshot(tmp_path, source)
    original_payload = snapshot.data_path.read_bytes()
    corrupted = bytearray(original_payload)
    corrupted[0] ^= 0xFF
    snapshot.data_path.write_bytes(corrupted)
    source.zero_()

    with pytest.raises(RuntimeError, match="checksum mismatch.*chunk 0"):
        snapshot.restore_into(source)

    assert snapshot.next_chunk == 0
    assert snapshot.data_path.exists()
    snapshot.data_path.write_bytes(original_payload)
    snapshot.restore_into(source)
    torch.testing.assert_close(source, expected, rtol=0.0, atol=0.0)


def test_disk_rollback_snapshot_truncation_retains_pending_tail(
    tmp_path: Path,
) -> None:
    source = torch.arange((2 * 1024 * 1024) // 4 + 13, dtype=torch.float32)
    expected = source.clone()
    snapshot = _create_disk_snapshot(tmp_path, source)
    original_payload = snapshot.data_path.read_bytes()
    snapshot.data_path.write_bytes(original_payload[:-3])
    source.zero_()

    with pytest.raises(OSError, match="short read"):
        snapshot.restore_into(source)

    assert 0 < snapshot.next_chunk < len(snapshot.chunk_checksums)
    snapshot.data_path.write_bytes(original_payload)
    snapshot.restore_into(source)
    torch.testing.assert_close(source, expected, rtol=0.0, atol=0.0)


def test_disk_rollback_snapshot_retry_starts_at_first_pending_chunk(
    tmp_path: Path, monkeypatch
) -> None:
    source = torch.arange((2 * 1024 * 1024) // 4 + 13, dtype=torch.float32)
    expected = source.clone()
    snapshot = _create_disk_snapshot(tmp_path, source)
    source.fill_(-1)
    original_read = snapshot_module._read_exact_buffer
    reads = 0

    def fail_second_read(fd: int, size: int) -> bytearray:
        nonlocal reads
        reads += 1
        if reads == 2:
            raise OSError("injected chunk read failure")
        return original_read(fd, size)

    monkeypatch.setattr(snapshot_module, "_read_exact_buffer", fail_second_read)
    with pytest.raises(OSError, match="chunk read failure"):
        snapshot.restore_into(source)
    assert snapshot.next_chunk == 1
    first_chunk = snapshot.chunk_bytes // source.element_size()
    torch.testing.assert_close(
        source[:first_chunk], expected[:first_chunk], rtol=0.0, atol=0.0
    )
    source_version = source._version

    monkeypatch.setattr(snapshot_module, "_read_exact_buffer", original_read)
    snapshot.restore_into(source)
    assert source._version - source_version == len(snapshot.chunk_checksums) - 1
    torch.testing.assert_close(source, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("close_before_error", [False, True])
def test_disk_rollback_snapshot_close_failure_does_not_replay_chunks(
    tmp_path: Path, monkeypatch, close_before_error: bool
) -> None:
    source = torch.arange(13, dtype=torch.float32)
    expected = source.clone()
    snapshot = _create_disk_snapshot(tmp_path, source)
    source.zero_()
    original_close = snapshot_module._close_fd
    close_attempts = 0

    def fail_close_once(fd: int) -> None:
        nonlocal close_attempts
        try:
            fd_path = os.readlink(f"/proc/self/fd/{fd}")
        except FileNotFoundError:
            fd_path = ""
        if fd_path.endswith("master.data") and close_attempts == 0:
            close_attempts += 1
            if close_before_error:
                original_close(fd)
            raise OSError("injected restore close failure")
        original_close(fd)

    monkeypatch.setattr(snapshot_module, "_close_fd", fail_close_once)
    with pytest.raises(OSError, match="restore close failure"):
        snapshot.restore_into(source)
    assert snapshot.restore_complete
    assert snapshot.next_chunk == len(snapshot.chunk_checksums)
    restored_version = source._version

    snapshot.restore_into(source)
    assert source._version == restored_version
    torch.testing.assert_close(source, expected, rtol=0.0, atol=0.0)
    assert not snapshot.directory.exists()


def test_disk_rollback_restore_rejects_reused_pending_fd(
    tmp_path: Path, monkeypatch
) -> None:
    source = torch.arange(13, dtype=torch.float32)
    snapshot = _create_disk_snapshot(tmp_path, source)
    source.zero_()
    original_close = snapshot_module._close_fd
    replacement_source_fd = os.open("/dev/null", os.O_RDONLY)
    replacement_fd = -1
    injected = False

    def close_reuse_then_fail(fd: int) -> None:
        nonlocal injected, replacement_fd, replacement_source_fd
        try:
            fd_path = os.readlink(f"/proc/self/fd/{fd}")
        except FileNotFoundError:
            fd_path = ""
        if fd_path.endswith("master.data") and not injected:
            injected = True
            original_close(fd)
            os.dup2(replacement_source_fd, fd)
            original_close(replacement_source_fd)
            replacement_source_fd = -1
            replacement_fd = fd
            raise OSError("injected restore close failure after FD reuse")
        original_close(fd)

    monkeypatch.setattr(snapshot_module, "_close_fd", close_reuse_then_fail)
    try:
        with pytest.raises(OSError, match="restore close failure after FD reuse"):
            snapshot.restore_into(source)
        assert not hasattr(snapshot, "_pending_fd")
        snapshot.restore_into(source)
        os.fstat(replacement_fd)
    finally:
        monkeypatch.setattr(snapshot_module, "_close_fd", original_close)
        if replacement_source_fd >= 0:
            original_close(replacement_source_fd)
        if replacement_fd >= 0:
            try:
                original_close(replacement_fd)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
        snapshot.cleanup()


@pytest.mark.parametrize("slab_key", ["master", "exp_avg", "exp_avg_sq"])
def test_disk_rollback_restore_retries_preclose_failure_without_fd_leak(
    tmp_path: Path, monkeypatch, slab_key: str
) -> None:
    source = torch.arange(13, dtype=torch.float32)
    expected = source.clone()
    snapshot = _create_disk_snapshot(tmp_path, source, slab_key=slab_key)
    source.zero_()
    original_prepare = snapshot_module._prepare_fd_close
    original_close = snapshot_module._close_fd
    failed_fd = -1
    failures_remaining = 3
    close_attempts = 0

    def fail_restore_preclose_once(owner) -> None:
        nonlocal close_attempts, failed_fd, failures_remaining
        fd_path = os.readlink(f"/proc/self/fd/{owner.fd}")
        if fd_path.endswith(f"{slab_key}.data") and failures_remaining:
            close_attempts += 1
            failures_remaining -= 1
            failed_fd = owner.fd
            raise OSError("injected restore pre-close failure")
        original_prepare(owner)

    monkeypatch.setattr(
        snapshot_module, "_prepare_fd_close", fail_restore_preclose_once
    )
    try:
        owner = None
        restored_version = -1
        fd_count = len(os.listdir("/proc/self/fd"))
        for attempt in range(3):
            with pytest.raises(OSError, match="restore pre-close failure"):
                snapshot.restore_into(source)
            assert not snapshot.restore_complete
            assert snapshot._restore_fd_owner is not None
            if owner is None:
                owner = snapshot._restore_fd_owner
                restored_version = source._version
            else:
                assert snapshot._restore_fd_owner is owner
                assert source._version == restored_version
            assert snapshot._restore_fd_owner.fd == failed_fd
            os.fstat(failed_fd)
            assert len(os.listdir("/proc/self/fd")) == fd_count + 1
            assert close_attempts == attempt + 1

        monkeypatch.setattr(snapshot_module, "_prepare_fd_close", original_prepare)
        snapshot.restore_into(source)
        assert snapshot.restore_complete
        assert snapshot._restore_fd_owner is None
        assert source._version == restored_version
        with pytest.raises(OSError) as exc_info:
            os.fstat(failed_fd)
        assert exc_info.value.errno == errno.EBADF
        torch.testing.assert_close(source, expected, rtol=0.0, atol=0.0)
    finally:
        monkeypatch.setattr(snapshot_module, "_prepare_fd_close", original_prepare)
        if failed_fd >= 0:
            try:
                original_close(failed_fd)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
        snapshot.cleanup()


def test_disk_rollback_cleanup_finalizes_pending_restore_fd_before_unlink(
    tmp_path: Path, monkeypatch
) -> None:
    source = torch.arange(13, dtype=torch.float32)
    snapshot = _create_disk_snapshot(tmp_path, source)
    cleanup = snapshot.cleanup_artifact()
    source.zero_()
    original_prepare = snapshot_module._prepare_fd_close
    failures_remaining = 2

    def fail_restore_preclose(owner) -> None:
        nonlocal failures_remaining
        fd_path = os.readlink(f"/proc/self/fd/{owner.fd}")
        if fd_path.endswith("master.data") and failures_remaining:
            failures_remaining -= 1
            raise OSError("injected restore pre-close failure")
        original_prepare(owner)

    monkeypatch.setattr(snapshot_module, "_prepare_fd_close", fail_restore_preclose)
    with pytest.raises(OSError, match="restore pre-close failure"):
        snapshot.restore_into(source)
    owner = snapshot._restore_fd_owner
    assert owner is not None

    with pytest.raises(OSError, match="restore pre-close failure"):
        cleanup.cleanup()
    assert snapshot._restore_fd_owner is owner
    assert snapshot.data_path.exists()
    assert snapshot.header_path.exists()

    monkeypatch.setattr(snapshot_module, "_prepare_fd_close", original_prepare)
    cleanup.cleanup()
    assert snapshot._restore_fd_owner is None
    assert not snapshot.directory.exists()


def test_disk_rollback_snapshot_fsync_failure_removes_partial_files(
    tmp_path: Path, monkeypatch
) -> None:
    source = torch.arange(17, dtype=torch.float32)
    original_fsync = snapshot_module.os.fsync
    failed = False

    def fail_data_fsync(fd: int) -> None:
        nonlocal failed
        fd_path = os.readlink(f"/proc/self/fd/{fd}")
        if fd_path.endswith(".master.data.partial") and not failed:
            failed = True
            raise OSError("injected snapshot fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(snapshot_module.os, "fsync", fail_data_fsync)
    with pytest.raises(OSError, match="snapshot fsync failure"):
        _create_disk_snapshot(tmp_path, source)
    assert discover_orphaned_snapshot_directories(tmp_path) == ()


def test_disk_rollback_snapshot_preflight_capacity_is_side_effect_free(
    tmp_path: Path, monkeypatch
) -> None:
    source = torch.arange(17, dtype=torch.float32)
    before = source.clone()
    pointer = source.untyped_storage().data_ptr()
    version = source._version
    monkeypatch.setattr(snapshot_module, "_filesystem_free_bytes", lambda root: 0)

    with pytest.raises(OSError, match="insufficient rollback snapshot capacity"):
        preflight_snapshot_requirements((SnapshotRequirement(tmp_path, 4096),))

    torch.testing.assert_close(source, before, rtol=0.0, atol=0.0)
    assert source._version == version
    assert source.untyped_storage().data_ptr() == pointer
    assert discover_orphaned_snapshot_directories(tmp_path) == ()


def test_disk_rollback_snapshot_rejects_symlink_root(tmp_path: Path) -> None:
    """A configured snapshot root must itself be a real directory."""
    real_root = tmp_path / "real"
    real_root.mkdir(mode=0o700)
    symlink_root = tmp_path / "linked"
    symlink_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises((OSError, RuntimeError), match="symlink"):
        preflight_snapshot_requirements((SnapshotRequirement(symlink_root, 4096),))

    assert tuple(real_root.iterdir()) == ()


def test_disk_rollback_snapshot_rejects_symlink_parent_component(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir(mode=0o700)
    snapshot_root = real_parent / "snapshots"
    snapshot_root.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises((OSError, RuntimeError), match="symlink"):
        preflight_snapshot_requirements(
            (SnapshotRequirement(linked_parent / "snapshots", 4096),)
        )

    assert tuple(snapshot_root.iterdir()) == ()


@pytest.mark.parametrize("filesystem_id", [0, True, "7", None])
def test_disk_snapshot_root_rejects_unstable_local_fsid(
    tmp_path: Path, monkeypatch, filesystem_id
) -> None:
    class FilesystemState:
        f_fsid = filesystem_id

    monkeypatch.setattr(snapshot_module.os, "fstatvfs", lambda fd: FilesystemState())
    with pytest.raises(OSError, match="nonzero integer f_fsid"):
        preflight_snapshot_requirements((SnapshotRequirement(tmp_path, 4096),))
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    "chunk_bytes",
    [4, MIN_SNAPSHOT_CHUNK_BYTES + 2, MAX_SNAPSHOT_CHUNK_BYTES + 4],
)
def test_disk_rollback_snapshot_rejects_unsafe_chunk_before_root_access(
    tmp_path: Path, chunk_bytes: int
) -> None:
    missing_root = tmp_path / "must-not-be-created"

    with pytest.raises(ValueError, match="chunk size"):
        DiskTensorRollbackSnapshot.create(
            torch.ones(1, dtype=torch.float32),
            parent=missing_root,
            leaf_identity={"tree_path": [0]},
            slab_key="master",
            chunk_bytes=chunk_bytes,
            rank=0,
        )

    assert not missing_root.exists()


def test_disk_rollback_snapshot_rejects_unbounded_chunk_metadata() -> None:
    chunk_numel = MIN_SNAPSHOT_CHUNK_BYTES // 4
    tensor = torch.empty(
        (MAX_SNAPSHOT_CHUNKS_PER_SLAB + 1) * chunk_numel,
        dtype=torch.float32,
        device="meta",
    )

    with pytest.raises(ValueError, match="chunk metadata exceeds"):
        DiskTensorRollbackSnapshot.required_bytes(tensor, MIN_SNAPSHOT_CHUNK_BYTES)


def test_shared_snapshot_capacity_sums_ranks_on_same_filesystem(
    monkeypatch,
) -> None:
    mib = 1024 * 1024
    local = (SnapshotCapacityReport(7, "/snapshot", 1, 10, 80 * mib, 160 * mib),)
    remote = (SnapshotCapacityReport(7, "/snapshot", 1, 10, 80 * mib, 160 * mib),)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def gather(output, value, *, group) -> None:
        output[:] = [value, remote]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(OSError, match="insufficient shared.*required=167772160"):
        validate_shared_snapshot_capacity(local, object())


def test_shared_snapshot_capacity_groups_distinct_filesystems(monkeypatch) -> None:
    mib = 1024 * 1024
    local = (SnapshotCapacityReport(7, "/snapshot-a", 1, 10, 80 * mib, 160 * mib),)
    remote = (SnapshotCapacityReport(8, "/snapshot-b", 2, 20, 80 * mib, 160 * mib),)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def gather(output, value, *, group) -> None:
        output[:] = [value, remote]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    validate_shared_snapshot_capacity(local, object())


def test_shared_snapshot_capacity_sums_same_fsid_across_distinct_roots(
    monkeypatch,
) -> None:
    mib = 1024 * 1024
    local = (SnapshotCapacityReport(7, "/snapshot-a", 1, 10, 80 * mib, 160 * mib),)
    remote = (SnapshotCapacityReport(7, "/snapshot-b", 1, 20, 80 * mib, 160 * mib),)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def gather(output, value, *, group) -> None:
        output[:] = [value, remote]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(OSError, match="insufficient shared.*required=167772160"):
        validate_shared_snapshot_capacity(local, object())


@pytest.mark.parametrize(
    ("local", "remote"),
    [
        (
            SnapshotCapacityReport(0, "/snapshot", 1, 10, 1, 1 << 30),
            SnapshotCapacityReport(0, "/snapshot", 1, 10, 1, 1 << 30),
        ),
        (
            SnapshotCapacityReport(7, "/snapshot", 1, 10, 1, 1 << 30),
            SnapshotCapacityReport(8, "/snapshot", 1, 10, 1, 1 << 30),
        ),
    ],
)
def test_shared_snapshot_capacity_rejects_unstable_filesystem_identity(
    monkeypatch,
    local: SnapshotCapacityReport,
    remote: SnapshotCapacityReport,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def gather(output, value, *, group) -> None:
        output[:] = [value, (remote,)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises((OSError, RuntimeError, ValueError), match="filesystem|f_fsid"):
        validate_shared_snapshot_capacity((local,), object())


@pytest.mark.parametrize("error_type", [RuntimeError, ImportError])
def test_disk_rollback_snapshot_partial_write_fails_without_authority(
    tmp_path: Path, monkeypatch, error_type: type[BaseException]
) -> None:
    source = torch.arange(17, dtype=torch.float32)
    original_write = snapshot_module._write_all

    def fail_first_payload(fd: int, payload) -> None:
        fd_path = os.readlink(f"/proc/self/fd/{fd}")
        if fd_path.endswith(".master.data.partial"):
            os.write(fd, memoryview(payload)[:4])
            raise error_type("injected partial write")
        original_write(fd, payload)

    monkeypatch.setattr(snapshot_module, "_write_all", fail_first_payload)
    with pytest.raises(error_type, match="injected partial write"):
        _create_disk_snapshot(tmp_path, source)

    assert discover_orphaned_snapshot_directories(tmp_path) == ()


def test_disk_rollback_partial_create_cleanup_is_retained_and_retryable(
    tmp_path: Path, monkeypatch
) -> None:
    source = torch.arange(17, dtype=torch.float32)
    original_write = snapshot_module._write_all
    original_unlink = snapshot_module.os.unlink
    cleanup_failed = False

    def fail_payload(fd: int, payload) -> None:
        fd_path = os.readlink(f"/proc/self/fd/{fd}")
        if fd_path.endswith(".master.data.partial"):
            raise OSError("injected snapshot write failure")
        original_write(fd, payload)

    def fail_partial_cleanup_once(path, *args, **kwargs) -> None:
        nonlocal cleanup_failed
        if os.fspath(path).endswith(".master.data.partial") and not cleanup_failed:
            cleanup_failed = True
            raise OSError("injected partial cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(snapshot_module, "_write_all", fail_payload)
    monkeypatch.setattr(snapshot_module.os, "unlink", fail_partial_cleanup_once)
    with pytest.raises(OSError, match="snapshot write failure") as exc_info:
        _create_disk_snapshot(tmp_path, source)

    cleanup = getattr(exc_info.value, "_areal_snapshot_build_cleanup")
    assert cleanup.pending_paths
    assert cleanup.directory.exists()
    assert any(
        "partial-create cleanup is pending" in note for note in exc_info.value.__notes__
    )

    monkeypatch.setattr(snapshot_module.os, "unlink", original_unlink)
    cleanup.cleanup()
    assert not cleanup.directory.exists()


def test_disk_rollback_partial_create_cleanup_reconciles_partial_rename(
    tmp_path: Path, monkeypatch
) -> None:
    original_replace = snapshot_module.os.replace
    replace_count = 0

    def fail_second_replace(src, dst, *args, **kwargs) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("injected second snapshot rename failure")
        original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(snapshot_module.os, "replace", fail_second_replace)
    with pytest.raises(
        OSError, match="second snapshot rename failure"
    ) as exception_info:
        _create_disk_snapshot(tmp_path, torch.arange(17, dtype=torch.float32))

    cleanup = getattr(exception_info.value, "_areal_snapshot_build_cleanup", None)
    assert cleanup is not None
    monkeypatch.setattr(snapshot_module.os, "replace", original_replace)
    cleanup.cleanup()
    assert discover_orphaned_snapshot_directories(tmp_path) == ()


def test_disk_rollback_partial_create_moves_finish_cleaned(
    tmp_path: Path, monkeypatch
) -> None:
    original_replace = snapshot_module.os.replace
    replace_count = 0

    def fail_second_replace(src, dst, *args, **kwargs) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("injected header rename failure")
        original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(snapshot_module.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="header rename failure") as exception_info:
        _create_disk_snapshot(tmp_path, torch.arange(17, dtype=torch.float32))

    cleanup = getattr(exception_info.value, "_areal_snapshot_build_cleanup")
    monkeypatch.setattr(snapshot_module.os, "replace", original_replace)
    cleanup.cleanup()
    assert cleanup._moves
    assert {move.stage for move in cleanup._moves.values()} == {
        snapshot_module._MoveStage.CLEANED
    }
    cleanup.cleanup()
    assert discover_orphaned_snapshot_directories(tmp_path) == ()


def test_disk_rollback_partial_create_reconciles_rename_then_error(
    tmp_path: Path, monkeypatch
) -> None:
    original_replace = snapshot_module.os.replace
    failed = False

    def rename_then_fail(src, dst, *args, **kwargs) -> None:
        nonlocal failed
        original_replace(src, dst, *args, **kwargs)
        if not failed:
            failed = True
            raise OSError("injected post-rename failure")

    monkeypatch.setattr(snapshot_module.os, "replace", rename_then_fail)
    with pytest.raises(OSError, match="post-rename failure") as exception_info:
        _create_disk_snapshot(tmp_path, torch.arange(17, dtype=torch.float32))

    cleanup = getattr(exception_info.value, "_areal_snapshot_build_cleanup")
    monkeypatch.setattr(snapshot_module.os, "replace", original_replace)
    cleanup.cleanup()
    assert discover_orphaned_snapshot_directories(tmp_path) == ()


def test_disk_rollback_partial_create_reconciles_move_stage_update_error(
    tmp_path: Path, monkeypatch
) -> None:
    original_mark = snapshot_module.DiskSnapshotBuildCleanup.mark_move_moved
    failed = False

    def fail_mark_once(self, artifact: str) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected move journal update failure")
        original_mark(self, artifact)

    monkeypatch.setattr(
        snapshot_module.DiskSnapshotBuildCleanup,
        "mark_move_moved",
        fail_mark_once,
    )
    with pytest.raises(OSError, match="move journal update failure") as exc_info:
        _create_disk_snapshot(tmp_path, torch.arange(17, dtype=torch.float32))

    cleanup = getattr(exc_info.value, "_areal_snapshot_build_cleanup")
    monkeypatch.setattr(
        snapshot_module.DiskSnapshotBuildCleanup,
        "mark_move_moved",
        original_mark,
    )
    cleanup.cleanup()
    assert discover_orphaned_snapshot_directories(tmp_path) == ()


def test_disk_rollback_partial_cleanup_does_not_delete_unknown_file(
    tmp_path: Path, monkeypatch
) -> None:
    original_replace = snapshot_module.os.replace
    replace_count = 0

    def add_unknown_then_fail(src, dst, *args, **kwargs) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            directory_fd = kwargs["dst_dir_fd"]
            unknown_fd = os.open(
                "unknown",
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
                dir_fd=directory_fd,
            )
            os.close(unknown_fd)
            raise OSError("injected rename failure with unknown file")
        original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(snapshot_module.os, "replace", add_unknown_then_fail)
    with pytest.raises(OSError, match="rename failure with unknown") as exc_info:
        _create_disk_snapshot(tmp_path, torch.arange(17, dtype=torch.float32))

    cleanup = getattr(exc_info.value, "_areal_snapshot_build_cleanup")
    assert (cleanup.directory / "unknown").exists()
    with pytest.raises(RuntimeError, match="unexpected files"):
        cleanup.cleanup()
    assert (cleanup.directory / "unknown").exists()
    os.unlink("unknown", dir_fd=cleanup._directory_fd)
    cleanup.cleanup()
    assert not cleanup.directory.exists()


def test_disk_rollback_snapshot_cleanup_failure_retries_exact_paths(
    tmp_path: Path, monkeypatch
) -> None:
    source = torch.arange(17, dtype=torch.float32)
    snapshot = _create_disk_snapshot(tmp_path, source)
    cleanup = snapshot.cleanup_artifact()
    original_unlink = snapshot_module.os.unlink
    failed = False

    def fail_header_once(path, *args, **kwargs) -> None:
        nonlocal failed
        if os.fspath(path).endswith(cleanup.header_path.name) and not failed:
            failed = True
            raise OSError("injected cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(snapshot_module.os, "unlink", fail_header_once)
    with pytest.raises(OSError, match="injected cleanup failure"):
        cleanup.cleanup()
    assert not cleanup.data_path.exists()
    assert cleanup.header_pending
    assert cleanup.directory.exists()

    cleanup.cleanup()
    assert not cleanup.directory.exists()


def test_disk_rollback_snapshot_cleanup_retries_fd_finalization(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _create_disk_snapshot(tmp_path, torch.arange(17, dtype=torch.float32))
    cleanup = snapshot.cleanup_artifact()
    original_prepare = snapshot_module._prepare_fd_close
    failed = False

    def fail_directory_close_once(owner) -> None:
        nonlocal failed
        if owner.fd == cleanup._directory_fd and not failed:
            failed = True
            raise OSError("injected cleanup fd close failure")
        original_prepare(owner)

    monkeypatch.setattr(snapshot_module, "_prepare_fd_close", fail_directory_close_once)
    with pytest.raises(OSError, match="cleanup fd close failure"):
        cleanup.cleanup()

    monkeypatch.setattr(snapshot_module, "_prepare_fd_close", original_prepare)
    cleanup.cleanup()
    assert cleanup._directory_fd == -1
    assert cleanup._root is None


@pytest.mark.parametrize("target", ["directory", "root"])
def test_disk_rollback_cleanup_close_then_error_is_not_replayed(
    tmp_path: Path, monkeypatch, target: str
) -> None:
    snapshot = _create_disk_snapshot(tmp_path, torch.arange(17, dtype=torch.float32))
    cleanup = snapshot.cleanup_artifact()
    original_close = snapshot_module._close_fd
    target_fd = cleanup._directory_fd if target == "directory" else cleanup._root.fd
    calls = 0

    def close_then_fail_once(fd: int) -> None:
        nonlocal calls
        if fd == target_fd:
            calls += 1
            original_close(fd)
            if calls == 1:
                raise OSError(f"injected {target} post-close failure")
            pytest.fail(f"closed finalized {target} FD more than once")
        original_close(fd)

    monkeypatch.setattr(snapshot_module, "_close_fd", close_then_fail_once)
    with pytest.raises(OSError, match="post-close failure"):
        cleanup.cleanup()
    with pytest.raises(RuntimeError, match="consumed descriptor"):
        cleanup.cleanup()
    assert calls == 1
    assert cleanup._directory_fd == -1
    owner = (
        cleanup._directory_fd_owner
        if target == "directory"
        else cleanup._root._fd_owner
    )
    assert owner is not None
    owner.close_diagnostic = None
    cleanup.cleanup()


@pytest.mark.parametrize(
    ("target", "failures"), [("directory", 2), ("root", 1), ("root", 3)]
)
def test_disk_rollback_cleanup_retries_only_fd_that_remains_open(
    tmp_path: Path, monkeypatch, target: str, failures: int
) -> None:
    snapshot = _create_disk_snapshot(tmp_path, torch.arange(17, dtype=torch.float32))
    cleanup = snapshot.cleanup_artifact()
    original_prepare = snapshot_module._prepare_fd_close
    directory_fd = cleanup._directory_fd
    root_fd = cleanup._root.fd
    target_fd = directory_fd if target == "directory" else root_fd
    attempts = 0
    successful_closes: list[int] = []

    def fail_before_close(owner) -> None:
        nonlocal attempts
        if owner.fd == target_fd and attempts < failures:
            attempts += 1
            raise OSError(f"injected {target} pre-close failure")
        successful_closes.append(owner.fd)
        original_prepare(owner)

    monkeypatch.setattr(snapshot_module, "_prepare_fd_close", fail_before_close)
    for _ in range(failures):
        with pytest.raises(OSError, match="pre-close failure"):
            cleanup.cleanup()
    cleanup.cleanup()

    assert attempts == failures
    assert successful_closes.count(directory_fd) == 1
    assert successful_closes.count(root_fd) == 1
    assert cleanup._directory_fd == -1
    assert cleanup._root is None


def test_disk_rollback_cleanup_revalidates_fd_after_preclose_failure(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _create_disk_snapshot(tmp_path, torch.arange(17, dtype=torch.float32))
    cleanup = snapshot.cleanup_artifact()
    original_prepare = snapshot_module._prepare_fd_close
    original_close = snapshot_module._close_fd
    directory_fd = cleanup._directory_fd
    replacement_path = tmp_path / "replacement-preclose-fd-target"
    replacement_source_fd = os.open(
        replacement_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
    )
    injected = False

    def reuse_then_fail_before_close(owner) -> None:
        nonlocal injected, replacement_source_fd
        if owner.fd == directory_fd and not injected:
            injected = True
            original_close(owner.fd)
            os.dup2(replacement_source_fd, owner.fd)
            original_close(replacement_source_fd)
            replacement_source_fd = -1
            os.write(owner.fd, b"replacement")
            raise OSError("injected pre-close failure after FD reuse")
        original_prepare(owner)

    monkeypatch.setattr(
        snapshot_module, "_prepare_fd_close", reuse_then_fail_before_close
    )
    try:
        with pytest.raises(
            OSError, match="pre-close failure after FD reuse"
        ) as exc_info:
            cleanup.cleanup()
        owner = cleanup._directory_fd_owner
        assert owner is not None
        assert owner.state is snapshot_module._FDState.OWNERSHIP_LOST
        assert owner.fd == -1
        assert "original_preclose=OSError" in owner.ownership_diagnostic
        assert "expected=" in owner.ownership_diagnostic
        assert "replacement=" in owner.ownership_diagnostic
        assert any("ownership changed" in note for note in exc_info.value.__notes__)

        monkeypatch.setattr(snapshot_module, "_prepare_fd_close", original_prepare)
        for _ in range(3):
            with pytest.raises(RuntimeError, match="ownership changed"):
                cleanup.cleanup()
            os.lseek(directory_fd, 0, os.SEEK_SET)
            assert os.read(directory_fd, 11) == b"replacement"
    finally:
        monkeypatch.setattr(snapshot_module, "_prepare_fd_close", original_prepare)
        if replacement_source_fd >= 0:
            original_close(replacement_source_fd)
        try:
            original_close(directory_fd)
        except OSError as error:
            if error.errno != errno.EBADF:
                raise
        cleanup._directory_fd = -1
        cleanup._directory_fd_owner = None
        cleanup.cleanup()


def test_disk_rollback_cleanup_rejects_reused_fd_without_closing_new_object(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _create_disk_snapshot(tmp_path, torch.arange(17, dtype=torch.float32))
    cleanup = snapshot.cleanup_artifact()
    original_close = snapshot_module._close_fd
    directory_fd = cleanup._directory_fd
    replacement_path = tmp_path / "replacement-fd-target"
    replacement_source_fd = os.open(
        replacement_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
    )
    injected = False

    def close_reuse_then_fail(fd: int) -> None:
        nonlocal injected, replacement_source_fd
        if fd == directory_fd and not injected:
            injected = True
            original_close(fd)
            os.dup2(replacement_source_fd, directory_fd)
            original_close(replacement_source_fd)
            replacement_source_fd = -1
            os.write(directory_fd, b"replacement")
            raise OSError("injected close failure after FD reuse")
        original_close(fd)

    monkeypatch.setattr(snapshot_module, "_close_fd", close_reuse_then_fail)
    try:
        with pytest.raises(OSError, match="close failure after FD reuse"):
            cleanup.cleanup()
        with pytest.raises(RuntimeError, match="consumed descriptor"):
            cleanup.cleanup()

        os.lseek(directory_fd, 0, os.SEEK_SET)
        assert os.read(directory_fd, 11) == b"replacement"
    finally:
        monkeypatch.setattr(snapshot_module, "_close_fd", original_close)
        if replacement_source_fd >= 0:
            original_close(replacement_source_fd)
        try:
            original_close(directory_fd)
        except OSError as error:
            if error.errno != errno.EBADF:
                raise
        owner = cleanup._directory_fd_owner
        assert owner is not None
        owner.close_diagnostic = None
        cleanup.cleanup()


def test_owned_fd_preclose_ebadf_detaches_without_retry(
    tmp_path: Path, monkeypatch
) -> None:
    fd = os.open(tmp_path / "owned", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    owner = snapshot_module._OwnedFD.capture(fd)
    original_prepare = snapshot_module._prepare_fd_close
    original_close = snapshot_module._close_fd
    injected = False

    def close_then_fail(current) -> None:
        nonlocal injected
        if not injected:
            injected = True
            original_close(current.fd)
            raise OSError("injected pre-close EBADF")
        original_prepare(current)

    monkeypatch.setattr(snapshot_module, "_prepare_fd_close", close_then_fail)
    with pytest.raises(OSError, match="pre-close EBADF") as exc_info:
        owner.close()
    assert owner.state is snapshot_module._FDState.OWNERSHIP_LOST
    assert owner.fd == -1
    assert "original_preclose=OSError" in owner.ownership_diagnostic
    assert "expected=" in owner.ownership_diagnostic
    assert "replacement='EBADF'" in owner.ownership_diagnostic
    assert any("ownership changed" in note for note in exc_info.value.__notes__)
    with pytest.raises(RuntimeError, match="ownership changed"):
        owner.close()


def test_owned_fd_preclose_error_with_same_owner_is_retryable(
    tmp_path: Path, monkeypatch
) -> None:
    fd = os.open(tmp_path / "owned", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    owner = snapshot_module._OwnedFD.capture(fd)
    original_prepare = snapshot_module._prepare_fd_close
    injected = False

    def fail_once(current) -> None:
        nonlocal injected
        if not injected:
            injected = True
            raise OSError("injected retryable pre-close")
        original_prepare(current)

    monkeypatch.setattr(snapshot_module, "_prepare_fd_close", fail_once)
    with pytest.raises(OSError, match="retryable pre-close"):
        owner.close()
    assert owner.state is snapshot_module._FDState.CLOSE_PENDING
    assert owner.fd == fd
    owner.close()
    assert owner.state is snapshot_module._FDState.CLOSED
    assert owner.fd == -1


def test_disk_rollback_cleanup_rejects_same_inode_fd_reuse(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _create_disk_snapshot(tmp_path, torch.arange(17, dtype=torch.float32))
    cleanup = snapshot.cleanup_artifact()
    original_close = snapshot_module._close_fd
    directory_fd = cleanup._directory_fd
    duplicate_fd = os.dup(directory_fd)
    injected = False

    def close_reuse_same_inode_then_fail(fd: int) -> None:
        nonlocal duplicate_fd, injected
        if fd == directory_fd and not injected:
            injected = True
            original_close(fd)
            os.dup2(duplicate_fd, directory_fd)
            original_close(duplicate_fd)
            duplicate_fd = -1
            raise OSError("injected close failure after same-inode FD reuse")
        original_close(fd)

    monkeypatch.setattr(snapshot_module, "_close_fd", close_reuse_same_inode_then_fail)
    try:
        with pytest.raises(OSError, match="same-inode FD reuse"):
            cleanup.cleanup()
        with pytest.raises(RuntimeError, match="consumed descriptor"):
            cleanup.cleanup()
        os.fstat(directory_fd)
    finally:
        monkeypatch.setattr(snapshot_module, "_close_fd", original_close)
        if duplicate_fd >= 0:
            original_close(duplicate_fd)
        try:
            original_close(directory_fd)
        except OSError as error:
            if error.errno != errno.EBADF:
                raise
        owner = cleanup._directory_fd_owner
        assert owner is not None
        owner.close_diagnostic = None
        cleanup.cleanup()


def test_disk_rollback_snapshot_cleanup_rejects_replaced_directory(
    tmp_path: Path, monkeypatch
) -> None:
    """A retry must not delete a new directory installed at an owned path."""
    snapshot = _create_disk_snapshot(tmp_path, torch.arange(17, dtype=torch.float32))
    cleanup = snapshot.cleanup_artifact()
    original_unlink = snapshot_module.os.unlink
    failed = False

    def fail_header_once(path, *args, **kwargs) -> None:
        nonlocal failed
        if os.fspath(path).endswith(cleanup.header_path.name) and not failed:
            failed = True
            raise OSError("injected cleanup interruption")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(snapshot_module.os, "unlink", fail_header_once)
    with pytest.raises(OSError, match="cleanup interruption"):
        cleanup.cleanup()
    monkeypatch.setattr(snapshot_module.os, "unlink", original_unlink)

    moved_directory = tmp_path / "original-moved"
    cleanup.directory.rename(moved_directory)
    cleanup.directory.mkdir(mode=0o700)
    shutil.copy2(moved_directory / "owner.json", cleanup.directory / "owner.json")

    with pytest.raises(RuntimeError, match="ownership changed"):
        cleanup.cleanup()
    assert cleanup.directory.exists()

    # The production cleanup deliberately refuses both directories.  The test
    # owns these exact paths, so restore the original inode before retrying.
    shutil.rmtree(cleanup.directory)
    moved_directory.rename(cleanup.directory)
    cleanup.cleanup()


def test_disk_rollback_snapshot_cleanup_rejects_replaced_root(
    tmp_path: Path,
) -> None:
    snapshot = _create_disk_snapshot(tmp_path, torch.arange(17, dtype=torch.float32))
    cleanup = snapshot.cleanup_artifact()
    moved_root = tmp_path.parent / f"{tmp_path.name}-original"
    tmp_path.rename(moved_root)
    tmp_path.mkdir(mode=0o700)

    with pytest.raises(RuntimeError, match="root ownership changed"):
        cleanup.cleanup()
    assert (moved_root / snapshot.directory.name / snapshot.data_path.name).exists()

    tmp_path.rmdir()
    moved_root.rename(tmp_path)
    cleanup.cleanup()


@pytest.mark.parametrize("replacement_kind", ["symlink", "regular"])
def test_disk_rollback_snapshot_cleanup_rejects_replaced_file(
    tmp_path: Path, replacement_kind: str
) -> None:
    snapshot = _create_disk_snapshot(tmp_path, torch.arange(17, dtype=torch.float32))
    cleanup = snapshot.cleanup_artifact()
    original = tmp_path / f"original-{replacement_kind}.data"
    snapshot.data_path.rename(original)
    if replacement_kind == "symlink":
        snapshot.data_path.symlink_to(original)
    else:
        snapshot.data_path.write_bytes(original.read_bytes())

    with pytest.raises(RuntimeError, match="file ownership changed"):
        cleanup.cleanup()
    assert snapshot.data_path.exists()

    snapshot.data_path.unlink()
    original.rename(snapshot.data_path)
    cleanup.cleanup()


def test_disk_rollback_snapshot_directory_cleanup_retries_after_marker_removed(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _create_disk_snapshot(tmp_path, torch.arange(7, dtype=torch.float32))
    cleanup = snapshot.cleanup_artifact()
    original_rmdir = snapshot_module.os.rmdir
    failed = False

    def fail_directory_once(path, *args, **kwargs) -> None:
        nonlocal failed
        if os.fspath(path).endswith(cleanup.directory.name) and not failed:
            failed = True
            raise OSError("injected directory cleanup failure")
        original_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(snapshot_module.os, "rmdir", fail_directory_once)
    with pytest.raises(OSError, match="directory cleanup failure"):
        snapshot.cleanup()
    assert not cleanup.directory.exists()
    assert cleanup._directory_quarantine_name is not None
    assert (tmp_path / cleanup._directory_quarantine_name).exists()
    assert not cleanup.data_pending
    assert not cleanup.header_pending
    assert not cleanup.marker_pending
    assert cleanup.directory_pending

    snapshot.cleanup()
    assert not cleanup.directory.exists()


@pytest.mark.parametrize("rename_before_error", [False, True])
def test_disk_rollback_snapshot_directory_rename_failure_is_retryable(
    tmp_path: Path, monkeypatch, rename_before_error: bool
) -> None:
    snapshot = _create_disk_snapshot(tmp_path, torch.arange(7, dtype=torch.float32))
    cleanup = snapshot.cleanup_artifact()
    original_rename = snapshot_module.os.rename
    failed = False

    def fail_directory_rename_once(src, dst, *args, **kwargs) -> None:
        nonlocal failed
        if os.fspath(src) == cleanup.directory.name and not failed:
            failed = True
            if rename_before_error:
                original_rename(src, dst, *args, **kwargs)
            raise OSError("injected directory rename failure")
        original_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(snapshot_module.os, "rename", fail_directory_rename_once)
    with pytest.raises(OSError, match="directory rename failure"):
        cleanup.cleanup()

    monkeypatch.setattr(snapshot_module.os, "rename", original_rename)
    cleanup.cleanup()
    assert not cleanup.directory.exists()


def test_disk_rollback_snapshot_cleanup_retry_does_not_replay_restore(
    tmp_path: Path, monkeypatch
) -> None:
    source = torch.arange(11, dtype=torch.float32)
    expected = source.clone()
    snapshot = _create_disk_snapshot(tmp_path, source)
    source.zero_()
    original_cleanup = DiskSnapshotCleanup.cleanup
    failed = False

    def fail_once(cleanup: DiskSnapshotCleanup) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected rollback cleanup failure")
        original_cleanup(cleanup)

    monkeypatch.setattr(DiskSnapshotCleanup, "cleanup", fail_once)
    with pytest.raises(OSError, match="rollback cleanup failure"):
        snapshot.restore_into(source)
    torch.testing.assert_close(source, expected, rtol=0.0, atol=0.0)
    restored_version = source._version
    assert snapshot.restore_complete

    monkeypatch.setattr(DiskSnapshotCleanup, "cleanup", original_cleanup)
    snapshot.restore_into(source)
    assert source._version == restored_version
    assert not snapshot.directory.exists()


def test_disk_rollback_orphan_discovery_is_read_only(tmp_path: Path) -> None:
    snapshot = _create_disk_snapshot(tmp_path, torch.ones(5, dtype=torch.float32))
    assert discover_orphaned_snapshot_directories(tmp_path) == (snapshot.directory,)
    assert snapshot.data_path.exists()
    snapshot.cleanup()


@pytest.mark.slow
def test_disk_rollback_snapshot_workspace_rss_is_bounded_across_sizes(
    tmp_path: Path,
) -> None:
    script = r"""
import json
import pathlib
import sys
import torch
from areal.engine.megatron_utils.checkpoint_snapshot import DiskTensorRollbackSnapshot

def memory_kib(name):
    for line in pathlib.Path('/proc/self/status').read_text().splitlines():
        if line.startswith(name + ':'):
            return int(line.split()[1])
    raise RuntimeError(name)

numel = int(sys.argv[1])
root = pathlib.Path(sys.argv[2])
source = torch.arange(numel, dtype=torch.float32)
before_rss = memory_kib('VmRSS')
before_hwm = memory_kib('VmHWM')
snapshot = DiskTensorRollbackSnapshot.create(
    source,
    parent=root,
    leaf_identity={'tree_path': [0], 'signature': 'rss'},
    slab_key='master',
    chunk_bytes=1024 * 1024,
    rank=0,
)
after_hwm = memory_kib('VmHWM')
snapshot.cleanup()
print(json.dumps({
    'workspace_peak': max(0, after_hwm - max(before_rss, before_hwm)) * 1024,
    'payload': source.numel() * source.element_size(),
}))
"""

    results = []
    for label, numel in (
        ("small", 64 * 1024 * 1024),
        ("large", 128 * 1024 * 1024),
    ):
        root = tmp_path / label
        root.mkdir()
        process = subprocess.run(
            [sys.executable, "-c", script, str(numel), str(root)],
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(process.stdout.strip()))

    assert results[0]["payload"] == 256 * 1024 * 1024
    assert results[1]["payload"] == 512 * 1024 * 1024
    # The serializer owns at most the fixed 1 MiB chunk plus bounded Python
    # buffers.  Leave room for allocator/import noise without allowing a
    # second full-state clone.
    assert max(result["workspace_peak"] for result in results) < 24 * 1024 * 1024


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_staged_checkpoint_rollback_uses_disk_without_slab_clone(
    tmp_path: Path,
) -> None:
    _, optimizer = _direct_checkpoint_optimizer(
        snapshot_root=str(tmp_path), snapshot_chunk_mb=1
    )
    assert optimizer.cpu_slabs is not None
    originals = {
        name: getattr(optimizer.cpu_slabs, name).clone()
        for name in ("master", "exp_avg", "exp_avg_sq")
    }
    pointers = {
        name: getattr(optimizer.cpu_slabs, name).untyped_storage().data_ptr()
        for name in originals
    }

    optimizer.begin_checkpoint_load()

    rollback = optimizer._checkpoint_rollback
    slab_actions = [
        action for action in rollback.actions if action.name.startswith("slab.")
    ]
    assert len(slab_actions) == 3
    assert all(
        isinstance(action.snapshot, DiskTensorRollbackSnapshot)
        for action in slab_actions
    )
    assert len(discover_orphaned_snapshot_directories(tmp_path)) == 3
    for action in slab_actions:
        header = json.loads(action.snapshot.header_path.read_text())
        assert header["leaf_identity"]["kind"] == "local-layout"
        assert header["dtype"] == "torch.float32"

    for name in originals:
        getattr(optimizer.cpu_slabs, name).fill_(99)
    optimizer.abort_checkpoint_load(RuntimeError("injected post-DCP failure"))

    assert optimizer.checkpoint_lifecycle == "CLEAN"
    assert optimizer._checkpoint_rollback is None
    assert discover_orphaned_snapshot_directories(tmp_path) == ()
    for name, expected in originals.items():
        slab = getattr(optimizer.cpu_slabs, name)
        torch.testing.assert_close(slab, expected, rtol=0.0, atol=0.0)
        assert slab.untyped_storage().data_ptr() == pointers[name]
        assert slab.is_pinned()
    assert optimizer.cuda_state_numel == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_staged_checkpoint_snapshot_preflight_failure_has_zero_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    _, optimizer = _direct_checkpoint_optimizer(snapshot_root=str(tmp_path))
    assert optimizer.cpu_slabs is not None
    slabs = (
        optimizer.cpu_slabs.master,
        optimizer.cpu_slabs.exp_avg,
        optimizer.cpu_slabs.exp_avg_sq,
    )
    values = tuple(slab.clone() for slab in slabs)
    versions = tuple(slab._version for slab in slabs)
    pointers = tuple(slab.untyped_storage().data_ptr() for slab in slabs)
    groups = [dict(group) for group in optimizer.param_groups]
    monkeypatch.setattr(snapshot_module, "_filesystem_free_bytes", lambda root: 0)

    with pytest.raises(OSError, match="insufficient rollback snapshot capacity"):
        optimizer.begin_checkpoint_load()

    assert optimizer.checkpoint_lifecycle == "CLEAN"
    assert optimizer._checkpoint_rollback is None
    assert optimizer.param_groups == groups
    for slab, value, version, pointer in zip(
        slabs, values, versions, pointers, strict=True
    ):
        torch.testing.assert_close(slab, value, rtol=0.0, atol=0.0)
        assert slab._version == version
        assert slab.untyped_storage().data_ptr() == pointer
    assert discover_orphaned_snapshot_directories(tmp_path) == ()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_staged_checkpoint_corrupt_snapshot_retains_poisoned_recovery(
    tmp_path: Path,
) -> None:
    param, optimizer = _direct_checkpoint_optimizer(snapshot_root=str(tmp_path))
    optimizer.begin_checkpoint_load()
    rollback = optimizer._checkpoint_rollback
    exp_avg_action = next(
        action for action in rollback.actions if action.name == "slab.exp_avg"
    )
    snapshot = exp_avg_action.snapshot
    original_payload = snapshot.data_path.read_bytes()
    corrupted = bytearray(original_payload)
    corrupted[0] ^= 0xFF
    snapshot.data_path.write_bytes(corrupted)

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        optimizer.abort_checkpoint_load(RuntimeError("load failed"))

    assert optimizer.checkpoint_lifecycle == "RECOVERY_PENDING"
    assert exp_avg_action.pending
    assert exp_avg_action.snapshot is snapshot
    param.decoupled_grad = torch.ones_like(param)
    with pytest.raises(RuntimeError, match="failed checkpoint load"):
        optimizer.step()

    snapshot.data_path.write_bytes(original_payload)
    optimizer.retry_checkpoint_recovery()
    assert optimizer.checkpoint_lifecycle == "POISONED"
    optimizer.prepare_checkpoint_recovery()
    assert optimizer.checkpoint_lifecycle == "RELOAD_REQUIRED"
    assert discover_orphaned_snapshot_directories(tmp_path) == ()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_staged_checkpoint_postcommit_cleanup_retries_only_files(
    tmp_path: Path, monkeypatch
) -> None:
    param, optimizer = _direct_checkpoint_optimizer(snapshot_root=str(tmp_path))
    transaction = begin_managed_checkpoint_load(SimpleNamespace(optimizer=optimizer))
    optimizer.load_state_dict(_detached_state_dict(optimizer))
    prepare_managed_checkpoint_load(transaction)
    original_cleanup = DiskSnapshotCleanup.cleanup
    failed = False

    def fail_once(cleanup: DiskSnapshotCleanup) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected postcommit unlink failure")
        original_cleanup(cleanup)

    monkeypatch.setattr(DiskSnapshotCleanup, "cleanup", fail_once)
    with pytest.raises(OSError, match="postcommit unlink failure"):
        commit_managed_checkpoint_load(transaction)

    assert transaction.committed
    assert optimizer.checkpoint_lifecycle == "CLEANUP_PENDING"
    assert optimizer._checkpoint_rollback is None
    assert optimizer._checkpoint_cleanup is not None
    before = optimizer.cpu_slabs.master.clone()
    param.decoupled_grad = torch.ones_like(param)
    optimizer.step()
    optimizer.drain()
    assert not torch.equal(optimizer.cpu_slabs.master, before)

    monkeypatch.setattr(DiskSnapshotCleanup, "cleanup", original_cleanup)
    retry_managed_checkpoint_cleanup(transaction)
    assert optimizer.checkpoint_lifecycle == "CLEAN"
    assert discover_orphaned_snapshot_directories(tmp_path) == ()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_staged_optimizer_checkpoint_rejects_wrong_state_dtype() -> None:
    """A corrupted CPU state dtype must fail instead of being silently cast."""
    _, optimizer = _direct_checkpoint_optimizer()
    state_dict = _detached_state_dict(optimizer)
    state_dict["state"][0]["exp_avg"] = state_dict["state"][0]["exp_avg"].double()

    with pytest.raises((TypeError, ValueError, RuntimeError), match="dtype"):
        optimizer.load_state_dict(state_dict)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_staged_optimizer_checkpoint_rejects_extra_state_field() -> None:
    """Unknown optimizer tensor state must not be silently dropped on load."""
    _, optimizer = _direct_checkpoint_optimizer()
    state_dict = _detached_state_dict(optimizer)
    state_dict["state"][0]["unexpected_state"] = torch.ones(1)

    with pytest.raises((KeyError, ValueError, RuntimeError), match="unexpected_state"):
        optimizer.load_state_dict(state_dict)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_staged_optimizer_checkpoint_rejects_empty_bound_state() -> None:
    """A bound optimizer must not combine old moments with new group metadata."""
    _, optimizer = _direct_checkpoint_optimizer()
    state_dict = _detached_state_dict(optimizer)
    state_dict["state"] = {}
    state_dict["param_groups"][0]["lr"] = 0.25

    with pytest.raises((KeyError, ValueError, RuntimeError), match="state"):
        optimizer.load_state_dict(state_dict)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_staged_optimizer_model_only_reset_failure_is_atomic(
    monkeypatch,
) -> None:
    """A failed model-only reset must restore state and poison further steps."""
    param, optimizer = _direct_checkpoint_optimizer()
    optimizer.cpu_slabs.exp_avg.fill_(4)
    optimizer.cpu_slabs.exp_avg_sq.fill_(9)
    optimizer.param_groups[0]["step"] = 17
    before_avg = optimizer.cpu_slabs.exp_avg.clone()
    before_avg_sq = optimizer.cpu_slabs.exp_avg_sq.clone()

    def fail_initialization(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected model-only reset failure")

    monkeypatch.setattr(
        optimizer, "_schedule_master_initialization", fail_initialization
    )
    with pytest.raises(RuntimeError, match="injected model-only reset failure"):
        reset_managed_optimizer_from_model(SimpleNamespace(optimizer=optimizer))

    torch.testing.assert_close(
        optimizer.cpu_slabs.exp_avg, before_avg, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        optimizer.cpu_slabs.exp_avg_sq, before_avg_sq, rtol=0.0, atol=0.0
    )
    assert optimizer.param_groups[0]["step"] == 17
    assert optimizer.residency == "CPU_RESIDENT"
    param.decoupled_grad = torch.ones_like(param)
    with pytest.raises(RuntimeError, match="failed checkpoint load"):
        optimizer.step()


def test_begin_managed_checkpoint_load_rolls_back_all_begun_leaves() -> None:
    """One rollback failure must not mask begin failure or skip earlier leaves."""
    events: list[str] = []

    class Managed:
        manages_cpu_residency = True

        def __init__(self, name, *, begin_error=False, abort_error=False):
            self.name = name
            self.begin_error = begin_error
            self.abort_error = abort_error

        def begin_checkpoint_load(self):
            events.append(f"{self.name}.begin")
            if self.begin_error:
                raise RuntimeError("begin-error")

        def abort_checkpoint_load(self, error):
            del error
            events.append(f"{self.name}.abort")
            if self.abort_error:
                raise RuntimeError("abort-error")

    first = Managed("first")
    second = Managed("second", abort_error=True)
    third = Managed("third", begin_error=True)
    root = SimpleNamespace(
        chained_optimizers=[
            SimpleNamespace(optimizer=first),
            SimpleNamespace(optimizer=second),
            SimpleNamespace(optimizer=third),
        ]
    )

    with pytest.raises(RuntimeError, match="begin-error") as exc_info:
        begin_managed_checkpoint_load(root)

    assert "first.abort" in events
    assert any("abort-error" in note for note in exc_info.value.__notes__)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda state: state.update(extra=1), "top-level"),
        (lambda state: state["param_groups"][0].update(extra=1), "param_group"),
        (lambda state: state["state"][0].pop("exp_avg_sq"), "exp_avg_sq"),
        (
            lambda state: state["state"][0].update(
                exp_avg=torch.zeros(6, dtype=torch.float32)
            ),
            "shape",
        ),
        (lambda state: state["param_groups"][0].update(step=-1), "step"),
    ],
)
def test_gpu_staged_optimizer_checkpoint_schema_is_strict_before_mutation(
    mutate, match
) -> None:
    _, optimizer = _direct_checkpoint_optimizer()
    state_dict = _detached_state_dict(optimizer)
    before_groups = [dict(group) for group in optimizer.param_groups]
    before_master = optimizer.cpu_slabs.master.clone()
    mutate(state_dict)

    with pytest.raises((KeyError, TypeError, ValueError, RuntimeError), match=match):
        optimizer.load_state_dict(state_dict)

    assert optimizer.param_groups == before_groups
    torch.testing.assert_close(
        optimizer.cpu_slabs.master, before_master, rtol=0.0, atol=0.0
    )


def test_mcore_checkpoint_rejects_duplicate_outer_group() -> None:
    """MCore must not collapse duplicate ownership groups before strict validation."""
    group = {
        "params": [0],
        "lr": 0.1,
        "step": 3,
        "wd_mult": 1.0,
        "lr_mult": 1.0,
        "is_expert_parallel": False,
        "is_decoupled_lr": False,
    }

    class Inner:
        manages_cpu_residency = True

        def __init__(self):
            self.state = {0: {"exp_avg": torch.zeros(1)}}
            self.param_groups = [dict(group), dict(group, lr_mult=2.0)]
            self.loads = []

        def state_dict(self):
            return {"state": self.state, "param_groups": self.param_groups}

        def load_state_dict(self, state_dict):
            self.loads.append(state_dict)

    inner = Inner()
    distributed = SimpleNamespace(
        ddp_config=SimpleNamespace(use_megatron_fsdp=False),
        optimizer=inner,
        grad_scaler=None,
        config=SimpleNamespace(fp16=False),
        opt_group_ranges=[],
    )
    distributed.opt_group_ranges = [
        {"orig_group": inner.param_groups[0]},
        {"orig_group": inner.param_groups[1]},
    ]
    duplicate = {
        key: value for key, value in dict(group, lr=0.3).items() if key != "params"
    }
    checkpoint = {
        "optimizer": {
            "param_groups": [
                {key: value for key, value in group.items() if key != "params"},
                duplicate,
            ]
        },
        "managed_checkpoint_identity": _outer_identity((), "duplicate"),
    }

    with pytest.raises((KeyError, ValueError, RuntimeError), match="param.group"):
        validate_managed_optimizer_outer_state(
            distributed,
            checkpoint,
            {(): _outer_identity((), "duplicate")},
        )

    assert inner.loads == []


def _managed_outer_test_leaf(groups: list[dict]) -> SimpleNamespace:
    class Inner:
        manages_cpu_residency = True

        def __init__(self):
            self.param_groups = groups

    inner = Inner()
    return SimpleNamespace(
        optimizer=inner,
        opt_group_ranges=[{"orig_group": group} for group in groups],
    )


def _outer_group(lr_mult: float) -> dict:
    return {
        "params": [object()],
        "lr": 0.1,
        "initial_lr": 0.0,
        "max_lr": 0.1,
        "min_lr": 0.0,
        "weight_decay": 0.01,
        "betas": (0.9, 0.99),
        "eps": 1e-8,
        "step": 3,
        "wd_mult": 1.0,
        "lr_mult": lr_mult,
        "is_expert_parallel": False,
        "is_decoupled_lr": False,
    }


def _outer_identity(path: tuple[int, ...], buffer: str) -> dict:
    return {
        "version": 1,
        "tree_path": list(path),
        "data_parallel_group_index": 0,
        "buffer_signature": buffer,
        "group_schema_signature": "groups",
    }


def _outer_leaf_state(groups: list[dict], path: tuple[int, ...], buffer: str) -> dict:
    return {
        "optimizer": {"param_groups": groups},
        "managed_checkpoint_identity": _outer_identity(path, buffer),
    }


@pytest.mark.parametrize("case", ["missing", "extra", "malformed"])
def test_mcore_checkpoint_rejects_invalid_outer_group_ownership(case: str) -> None:
    runtime_groups = [_outer_group(1.0), _outer_group(2.0)]
    leaf = _managed_outer_test_leaf(runtime_groups)
    checkpoint_groups = [
        {key: value for key, value in group.items() if key != "params"}
        for group in runtime_groups
    ]
    if case == "missing":
        checkpoint_groups.pop()
    elif case == "extra":
        checkpoint_groups.append(
            {key: value for key, value in _outer_group(3.0).items() if key != "params"}
        )
    else:
        checkpoint_groups[1].pop("lr_mult")

    with pytest.raises((KeyError, TypeError, ValueError), match="param.group"):
        validate_managed_optimizer_outer_state(
            leaf,
            _outer_leaf_state(checkpoint_groups, (), "leaf"),
            {(): _outer_identity((), "leaf")},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lr", -1.0),
        ("lr", float("nan")),
        ("initial_lr", -1.0),
        ("max_lr", float("nan")),
        ("min_lr", float("inf")),
        ("weight_decay", -0.1),
        ("eps", -1.0),
        ("eps", 0.0),
        ("eps", float("inf")),
        ("betas", (-0.1, 1.5)),
        ("betas", (0.9, float("nan"))),
        ("betas", (0.9,)),
        ("step", -1),
        ("step", True),
    ],
)
def test_mcore_checkpoint_outer_preflight_rejects_invalid_hyperparameters(
    field: str, value
) -> None:
    runtime_group = _outer_group(1.0)
    runtime_group["betas"] = (0.9, 0.99)
    leaf = _managed_outer_test_leaf([runtime_group])
    checkpoint_group = {
        key: item for key, item in runtime_group.items() if key != "params"
    }
    checkpoint_group[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        validate_managed_optimizer_outer_state(
            leaf,
            _outer_leaf_state([checkpoint_group], (), "leaf"),
            {(): _outer_identity((), "leaf")},
        )


def test_mcore_checkpoint_rejects_ambiguous_multi_leaf_order() -> None:
    first_group = _outer_group(1.0)
    second_group = _outer_group(1.0)
    first_group["lr"] = 0.1
    second_group["lr"] = 0.2
    root = SimpleNamespace(
        chained_optimizers=[
            _managed_outer_test_leaf([first_group]),
            _managed_outer_test_leaf([second_group]),
        ]
    )
    first_state = {key: value for key, value in first_group.items() if key != "params"}
    second_state = {
        key: value for key, value in second_group.items() if key != "params"
    }

    with pytest.raises((ValueError, RuntimeError), match="leaf|order|ambiguous"):
        validate_managed_optimizer_outer_state(
            root,
            {
                0: _outer_leaf_state([second_state], (1,), "second"),
                1: _outer_leaf_state([first_state], (0,), "first"),
            },
            {
                (0,): _outer_identity((0,), "first"),
                (1,): _outer_identity((1,), "second"),
            },
        )


def test_mcore_checkpoint_outer_preflight_accepts_nested_chain_schema() -> None:
    groups = [_outer_group(value) for value in (1.0, 2.0, 3.0)]
    leaves = [_managed_outer_test_leaf([group]) for group in groups]
    root = SimpleNamespace(
        chained_optimizers=[
            SimpleNamespace(chained_optimizers=leaves[:2]),
            leaves[2],
        ]
    )

    def state(group, path, buffer):
        metadata = {key: value for key, value in group.items() if key != "params"}
        return _outer_leaf_state([metadata], path, buffer)

    validated = validate_managed_optimizer_outer_state(
        root,
        {
            0: {
                0: state(groups[0], (0, 0), "a"),
                1: state(groups[1], (0, 1), "b"),
            },
            1: state(groups[2], (1,), "c"),
        },
        {
            (0, 0): _outer_identity((0, 0), "a"),
            (0, 1): _outer_identity((0, 1), "b"),
            (1,): _outer_identity((1,), "c"),
        },
    )

    assert validated == 3


@pytest.mark.parametrize("case", ["missing", "extra", "malformed"])
def test_mcore_checkpoint_outer_preflight_rejects_malformed_nested_schema(
    case: str,
) -> None:
    groups = [_outer_group(value) for value in (1.0, 2.0, 3.0)]
    leaves = [_managed_outer_test_leaf([group]) for group in groups]
    root = SimpleNamespace(
        chained_optimizers=[
            SimpleNamespace(chained_optimizers=leaves[:2]),
            leaves[2],
        ]
    )

    def state(index: int, path: tuple[int, ...], buffer: str) -> dict:
        metadata = {
            key: value for key, value in groups[index].items() if key != "params"
        }
        return _outer_leaf_state([metadata], path, buffer)

    checkpoint_state: dict = {
        0: {
            0: state(0, (0, 0), "a"),
            1: state(1, (0, 1), "b"),
        },
        1: state(2, (1,), "c"),
    }
    if case == "missing":
        del checkpoint_state[0][1]
    elif case == "extra":
        checkpoint_state[0][2] = checkpoint_state[0][0]
    else:
        checkpoint_state[0] = "not-an-indexed-node"

    with pytest.raises((TypeError, ValueError), match="tree path"):
        validate_managed_optimizer_outer_state(
            root,
            checkpoint_state,
            {
                (0, 0): _outer_identity((0, 0), "a"),
                (0, 1): _outer_identity((0, 1), "b"),
                (1,): _outer_identity((1,), "c"),
            },
        )


def test_mcore_checkpoint_accepts_reordered_unique_outer_groups_and_multi_leaf() -> (
    None
):
    first_groups = [_outer_group(1.0), _outer_group(2.0)]
    second_groups = [_outer_group(4.0)]
    leaves = [
        _managed_outer_test_leaf(first_groups),
        _managed_outer_test_leaf(second_groups),
    ]
    root = SimpleNamespace(chained_optimizers=leaves)
    first_checkpoint = [
        {key: value for key, value in group.items() if key != "params"}
        for group in reversed(first_groups)
    ]
    second_checkpoint = [
        {key: value for key, value in second_groups[0].items() if key != "params"}
    ]

    validated = validate_managed_optimizer_outer_state(
        root,
        {
            0: _outer_leaf_state(first_checkpoint, (0,), "first"),
            1: _outer_leaf_state(second_checkpoint, (1,), "second"),
        },
        {
            (0,): _outer_identity((0,), "first"),
            (1,): _outer_identity((1,), "second"),
        },
    )

    assert validated == 2


def test_managed_checkpoint_prepare_failure_aborts_every_leaf_in_reverse() -> None:
    events: list[str] = []

    class Managed:
        manages_cpu_residency = True

        def __init__(self, name: str, *, prepare_error: bool = False):
            self.name = name
            self.value = name
            self.snapshot = None
            self.prepare_error = prepare_error
            self.poisoned = False

        def begin_checkpoint_load(self):
            events.append(f"{self.name}.begin")
            self.snapshot = self.value

        def prepare_checkpoint_load(self):
            events.append(f"{self.name}.prepare")
            assert self.snapshot is not None
            if self.prepare_error:
                raise RuntimeError("prepare-error")

        def prepare_checkpoint_commit(self, commit_token=None):
            del commit_token
            events.append(f"{self.name}.commit")
            self.snapshot = None

        def decide_checkpoint_commit(self):
            pass

        def discard_checkpoint_snapshot(self):
            self.snapshot = None

        def abort_checkpoint_load(self, error):
            del error
            events.append(f"{self.name}.abort")
            self.value = self.snapshot
            self.snapshot = None

        def mark_checkpoint_poisoned(self, error):
            del error
            self.poisoned = True

    leaves = [Managed("a"), Managed("b", prepare_error=True), Managed("c")]
    root = SimpleNamespace(
        chained_optimizers=[SimpleNamespace(optimizer=leaf) for leaf in leaves]
    )
    transaction = begin_managed_checkpoint_load(root)
    for leaf in leaves:
        leaf.value = f"new-{leaf.name}"

    with pytest.raises(RuntimeError, match="prepare-error"):
        try:
            prepare_managed_checkpoint_load(transaction)
        except BaseException as error:
            abort_managed_checkpoint_load(transaction, error)
            raise

    assert [leaf.value for leaf in leaves] == ["a", "b", "c"]
    assert events[-3:] == ["c.abort", "b.abort", "a.abort"]
    assert all(leaf.snapshot is None for leaf in leaves)


def test_managed_checkpoint_global_commit_failure_rolls_back_every_leaf() -> None:
    """A commit failure must not strand an already-committed leaf without rollback."""
    events: list[str] = []

    class Managed:
        manages_cpu_residency = True

        def __init__(self, name: str, *, commit_error: bool = False):
            self.name = name
            self.value = f"old-{name}"
            self.snapshot = None
            self.commit_error = commit_error

        def begin_checkpoint_load(self):
            events.append(f"{self.name}.begin")
            self.snapshot = self.value

        def prepare_checkpoint_load(self):
            events.append(f"{self.name}.prepare")

        def prepare_checkpoint_commit(self, commit_token=None):
            del commit_token
            events.append(f"{self.name}.commit")
            if self.commit_error:
                raise RuntimeError("commit-error")

        def decide_checkpoint_commit(self):
            pass

        def discard_checkpoint_snapshot(self):
            self.snapshot = None

        def abort_checkpoint_load(self, error):
            del error
            events.append(f"{self.name}.abort")
            self.value = self.snapshot
            self.snapshot = None

        def mark_checkpoint_poisoned(self, error):
            del error

    leaves = [Managed("a"), Managed("b", commit_error=True), Managed("c")]
    root = SimpleNamespace(
        chained_optimizers=[SimpleNamespace(optimizer=leaf) for leaf in leaves]
    )
    transaction = begin_managed_checkpoint_load(root)
    for leaf in leaves:
        leaf.value = f"new-{leaf.name}"
    prepare_managed_checkpoint_load(transaction)

    with pytest.raises(RuntimeError, match="commit-error"):
        try:
            commit_managed_checkpoint_load(transaction)
        except BaseException as error:
            abort_managed_checkpoint_load(transaction, error, poison=True)
            raise

    assert [leaf.value for leaf in leaves] == ["old-a", "old-b", "old-c"]
    assert events[-3:] == ["c.abort", "b.abort", "a.abort"]
    assert all(leaf.snapshot is None for leaf in leaves)


def test_managed_checkpoint_cleanup_failure_retries_without_rollback() -> None:
    events: list[str] = []

    class Managed:
        manages_cpu_residency = True

        def __init__(self, name: str, *, cleanup_error: bool = False):
            self.name = name
            self.value = f"old-{name}"
            self.snapshot = None
            self.cleanup_error = cleanup_error

        def begin_checkpoint_load(self):
            self.snapshot = self.value

        def prepare_checkpoint_load(self):
            pass

        def prepare_checkpoint_commit(self, commit_token=None):
            del commit_token
            events.append(f"{self.name}.prepare-commit")

        def decide_checkpoint_commit(self):
            pass

        def discard_checkpoint_snapshot(self):
            events.append(f"{self.name}.cleanup")
            if self.cleanup_error:
                raise RuntimeError(f"cleanup-{self.name}")
            self.snapshot = None

        def abort_checkpoint_load(self, error):
            del error
            events.append(f"{self.name}.abort")
            self.value = self.snapshot

        def mark_checkpoint_poisoned(self, error):
            del error

    leaves = [Managed("a"), Managed("b", cleanup_error=True), Managed("c")]
    root = SimpleNamespace(
        chained_optimizers=[SimpleNamespace(optimizer=leaf) for leaf in leaves]
    )
    transaction = begin_managed_checkpoint_load(root)
    for leaf in leaves:
        leaf.value = f"new-{leaf.name}"
    prepare_managed_checkpoint_load(transaction)

    with pytest.raises(RuntimeError, match="cleanup-b"):
        commit_managed_checkpoint_load(transaction)

    assert transaction.committed
    assert transaction.phase.name == "CLEANUP_PENDING"
    assert transaction.cleanup_pending == [leaves[1]]
    assert [leaf.value for leaf in leaves] == ["new-a", "new-b", "new-c"]
    assert not any(event.endswith("abort") for event in events)

    leaves[1].cleanup_error = False
    retry_managed_checkpoint_cleanup(transaction)

    assert transaction.cleanup_pending == []
    assert transaction.phase.name == "CLEAN"
    assert [event for event in events if event.endswith("cleanup")] == [
        "a.cleanup",
        "b.cleanup",
        "c.cleanup",
        "b.cleanup",
    ]
    assert all(leaf.snapshot is None for leaf in leaves)


def test_post_commit_leaf_transition_failure_continues_and_retries_only_pending() -> (
    None
):
    events: list[str] = []

    class Managed:
        manages_cpu_residency = True

        def __init__(self, name: str, fail_decision: bool = False):
            self.name = name
            self.value = f"old-{name}"
            self.snapshot = None
            self.fail_decision = fail_decision
            self.commit_token = None

        def begin_checkpoint_load(self):
            self.snapshot = self.value

        def prepare_checkpoint_load(self):
            pass

        def prepare_checkpoint_commit(self, commit_token):
            self.commit_token = commit_token

        def decide_checkpoint_commit(self):
            events.append(f"{self.name}.decision")
            assert self.commit_token.decided
            if self.fail_decision:
                raise RuntimeError(f"decision-{self.name}")

        def discard_checkpoint_snapshot(self):
            events.append(f"{self.name}.cleanup")
            self.snapshot = None

        def abort_checkpoint_load(self, error):
            del error
            events.append(f"{self.name}.abort")
            self.value = self.snapshot

        def mark_checkpoint_poisoned(self, error):
            del error

    leaves = [Managed("a"), Managed("b", True), Managed("c")]
    root = SimpleNamespace(
        chained_optimizers=[SimpleNamespace(optimizer=leaf) for leaf in leaves]
    )
    transaction = begin_managed_checkpoint_load(root)
    for leaf in leaves:
        leaf.value = f"new-{leaf.name}"
    prepare_managed_checkpoint_load(transaction)

    with pytest.raises(RuntimeError, match="decision-b"):
        commit_managed_checkpoint_load(transaction)

    assert transaction.committed
    assert transaction.phase is ManagedCheckpointTransactionPhase.CLEANUP_PENDING
    assert transaction.cleanup_pending == [leaves[1]]
    assert [leaf.value for leaf in leaves] == ["new-a", "new-b", "new-c"]
    assert leaves[0].snapshot is None
    assert leaves[1].snapshot == "old-b"
    assert leaves[2].snapshot is None
    assert events == [
        "a.decision",
        "a.cleanup",
        "b.decision",
        "c.decision",
        "c.cleanup",
    ]
    with pytest.raises(RuntimeError, match="global commit decision"):
        abort_managed_checkpoint_load(transaction, RuntimeError("late abort"))

    leaves[1].fail_decision = False
    retry_managed_checkpoint_cleanup(transaction)
    assert transaction.phase is ManagedCheckpointTransactionPhase.CLEAN
    assert events[-2:] == ["b.decision", "b.cleanup"]
    assert not any(event.endswith("abort") for event in events)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cleanup_journal_cannot_be_reclassified_as_rollback_recovery(
    monkeypatch,
) -> None:
    """A committed cleanup snapshot must never become rollback authority again."""
    _, optimizer = _direct_checkpoint_optimizer()
    optimizer.cpu_slabs.master.fill_(1.0)
    optimizer.cpu_slabs.exp_avg.fill_(2.0)
    optimizer.cpu_slabs.exp_avg_sq.fill_(3.0)
    transaction = begin_managed_checkpoint_load(SimpleNamespace(optimizer=optimizer))
    loaded_state = _detached_state_dict(optimizer)
    loaded_state["state"][0]["master_param"].fill_(11.0)
    loaded_state["state"][0]["exp_avg"].fill_(12.0)
    loaded_state["state"][0]["exp_avg_sq"].fill_(13.0)
    optimizer.load_state_dict(loaded_state)
    prepare_managed_checkpoint_load(transaction)

    original_discard = optimizer.discard_checkpoint_snapshot
    monkeypatch.setattr(
        optimizer,
        "discard_checkpoint_snapshot",
        lambda: (_ for _ in ()).throw(RuntimeError("injected cleanup failure")),
    )
    with pytest.raises(RuntimeError, match="injected cleanup failure"):
        commit_managed_checkpoint_load(transaction)
    assert transaction.committed
    assert optimizer.checkpoint_lifecycle == "CLEANUP_PENDING"
    assert optimizer._checkpoint_rollback is None
    assert optimizer._checkpoint_cleanup is not None
    assert transaction.cleanup_pending == [optimizer]
    monkeypatch.setattr(optimizer, "discard_checkpoint_snapshot", original_discard)

    control = create_managed_checkpoint_load_transaction(
        SimpleNamespace(optimizer=optimizer)
    )
    poison_managed_checkpoint_transaction(control, RuntimeError("request mismatch"))
    assert optimizer.checkpoint_lifecycle == "CLEANUP_PENDING"
    with pytest.raises(RuntimeError, match="cannot enter rollback recovery"):
        prepare_managed_checkpoint_recovery(control)
    with pytest.raises(RuntimeError, match="irreversible checkpoint commit"):
        optimizer.abort_checkpoint_load(RuntimeError("late abort"))

    torch.testing.assert_close(
        optimizer.cpu_slabs.master,
        torch.full_like(optimizer.cpu_slabs.master, 11.0),
        rtol=0.0,
        atol=0.0,
    )
    retry_managed_checkpoint_cleanup(transaction)
    assert optimizer.checkpoint_lifecycle == "CLEAN"
    assert optimizer._checkpoint_cleanup is None
    assert optimizer._checkpoint_load_error is None
    torch.testing.assert_close(
        optimizer.cpu_slabs.exp_avg,
        torch.full_like(optimizer.cpu_slabs.exp_avg, 12.0),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        optimizer.cpu_slabs.exp_avg_sq,
        torch.full_like(optimizer.cpu_slabs.exp_avg_sq, 13.0),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_leaf_abort_rejects_global_decision_before_cleanup_transition() -> None:
    _, optimizer = _direct_checkpoint_optimizer()
    transaction = begin_managed_checkpoint_load(SimpleNamespace(optimizer=optimizer))
    optimizer.load_state_dict(_detached_state_dict(optimizer))
    prepare_managed_checkpoint_load(transaction)
    prepare_managed_checkpoint_commit(transaction)
    decide_managed_checkpoint_commit(transaction)

    # The manager decision is authoritative before the fallible per-leaf
    # transition runs. A direct leaf API must honor the shared decision token.
    assert optimizer.checkpoint_lifecycle == "COMMIT_DECIDED"
    with pytest.raises(RuntimeError, match="irreversible checkpoint commit"):
        optimizer.abort_checkpoint_load(RuntimeError("late direct abort"))

    retry_managed_checkpoint_cleanup(transaction)
    assert optimizer.checkpoint_lifecycle == "CLEAN"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cleanup_failure_retries_without_accumulating_snapshot_journals(
    monkeypatch,
) -> None:
    _, optimizer = _direct_checkpoint_optimizer()
    transaction = begin_managed_checkpoint_load(SimpleNamespace(optimizer=optimizer))
    loaded_state = _detached_state_dict(optimizer)
    loaded_state["state"][0]["master_param"].fill_(11.0)
    optimizer.load_state_dict(loaded_state)
    prepare_managed_checkpoint_load(transaction)
    snapshot_ref = weakref.ref(optimizer._checkpoint_rollback.master)
    original_discard = optimizer.discard_checkpoint_snapshot
    attempts = 0

    def fail_cleanup() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError(f"cleanup-round-{attempts}")

    monkeypatch.setattr(optimizer, "discard_checkpoint_snapshot", fail_cleanup)
    with pytest.raises(RuntimeError, match="cleanup-round-1"):
        commit_managed_checkpoint_load(transaction)
    cleanup_journal_identity = id(optimizer._checkpoint_cleanup)
    cleanup_refs = [
        weakref.ref(reference)
        for reference in optimizer._checkpoint_cleanup.references
        if isinstance(reference, DiskSnapshotCleanup)
    ]
    with pytest.raises(RuntimeError, match="cleanup-round-2"):
        retry_managed_checkpoint_cleanup(transaction)

    assert id(optimizer._checkpoint_cleanup) == cleanup_journal_identity
    assert snapshot_ref() is None
    assert cleanup_refs and all(reference() is not None for reference in cleanup_refs)
    assert transaction.phase is ManagedCheckpointTransactionPhase.CLEANUP_PENDING
    monkeypatch.setattr(optimizer, "discard_checkpoint_snapshot", original_discard)
    retry_managed_checkpoint_cleanup(transaction)

    assert all(reference() is None for reference in cleanup_refs)
    assert optimizer._checkpoint_cleanup is None
    assert optimizer.checkpoint_lifecycle == "CLEAN"
    assert transaction.phase is ManagedCheckpointTransactionPhase.CLEAN
    torch.testing.assert_close(
        optimizer.cpu_slabs.master,
        torch.full_like(optimizer.cpu_slabs.master, 11.0),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_repeated_commit_cleanup_releases_snapshots_without_rss_staircase(
    tmp_path: Path,
) -> None:
    statm = Path("/proc/self/statm")
    if not statm.exists():
        pytest.skip("Linux /proc RSS accounting is required")

    numel = 512 * 1024
    param = torch.nn.Parameter(torch.zeros(numel, device="cuda", dtype=torch.bfloat16))
    optimizer = GPUStagedAdamW(
        [param],
        staged_config=GPUStagedAdamWConfig(
            buffer_count=1,
            bucket_size_mb=1,
            checkpoint_snapshot_root=str(tmp_path),
            checkpoint_snapshot_chunk_mb=1,
        ),
    )
    optimizer.bind_owned_params(optimizer.param_groups)
    loaded_state = _detached_state_dict(optimizer)

    def rss_bytes() -> int:
        resident_pages = int(statm.read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")

    def checkpoint_cycle() -> list[weakref.ReferenceType[torch.Tensor]]:
        transaction = begin_managed_checkpoint_load(
            SimpleNamespace(optimizer=optimizer)
        )
        assert optimizer._checkpoint_rollback is not None
        snapshot_refs = [
            weakref.ref(action.snapshot)
            for action in optimizer._checkpoint_rollback.actions
            if isinstance(action.snapshot, DiskTensorRollbackSnapshot)
        ]
        optimizer.load_state_dict(loaded_state)
        prepare_managed_checkpoint_load(transaction)
        commit_managed_checkpoint_load(transaction)
        assert transaction.cleanup_journal is None
        assert optimizer._checkpoint_cleanup is None
        return snapshot_refs

    # Warm the pinned allocator so the measured loop distinguishes retained
    # journals from allocator initialization/caching.
    for _ in range(2):
        assert all(reference() is None for reference in checkpoint_cycle())
    gc.collect()
    baseline_rss = rss_bytes()
    fd_root = Path("/proc/self/fd")
    baseline_fds = len(tuple(fd_root.iterdir())) if fd_root.exists() else None

    snapshot_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    rss_samples = []
    for _ in range(8):
        snapshot_refs.extend(checkpoint_cycle())
        gc.collect()
        rss_samples.append(rss_bytes())

    assert all(reference() is None for reference in snapshot_refs)
    # A retained three-slab snapshot would grow by 6 MiB per cycle. Allow up
    # to four snapshot sets for allocator noise while rejecting a staircase.
    snapshot_bytes = 3 * numel * torch.tensor([], dtype=torch.float32).element_size()
    assert max(rss_samples) - baseline_rss < 4 * snapshot_bytes
    assert discover_orphaned_snapshot_directories(tmp_path) == ()
    if baseline_fds is not None:
        assert len(tuple(fd_root.iterdir())) <= baseline_fds + 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_optimizer_step_is_safe_while_snapshot_cleanup_is_pending(monkeypatch) -> None:
    param, optimizer = _direct_checkpoint_optimizer()
    transaction = begin_managed_checkpoint_load(SimpleNamespace(optimizer=optimizer))
    loaded_state = _detached_state_dict(optimizer)
    loaded_state["state"][0]["master_param"].fill_(11.0)
    optimizer.load_state_dict(loaded_state)
    prepare_managed_checkpoint_load(transaction)
    original_discard = optimizer.discard_checkpoint_snapshot
    monkeypatch.setattr(
        optimizer,
        "discard_checkpoint_snapshot",
        lambda: (_ for _ in ()).throw(RuntimeError("cleanup-pending")),
    )
    with pytest.raises(RuntimeError, match="cleanup-pending"):
        commit_managed_checkpoint_load(transaction)

    before = optimizer.cpu_slabs.master.clone()
    param.decoupled_grad = torch.ones_like(param)
    optimizer.step()
    optimizer.drain()
    assert not torch.equal(optimizer.cpu_slabs.master, before)

    monkeypatch.setattr(optimizer, "discard_checkpoint_snapshot", original_discard)
    retry_managed_checkpoint_cleanup(transaction)
    assert optimizer.checkpoint_lifecycle == "CLEAN"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_optimizer_step_is_safe_after_leaf_commit_transition_failure(
    monkeypatch,
) -> None:
    """The shared decision token makes new state usable before leaf cleanup retry."""
    param, optimizer = _direct_checkpoint_optimizer()
    transaction = begin_managed_checkpoint_load(SimpleNamespace(optimizer=optimizer))
    loaded_state = _detached_state_dict(optimizer)
    loaded_state["state"][0]["master_param"].fill_(11.0)
    optimizer.load_state_dict(loaded_state)
    prepare_managed_checkpoint_load(transaction)
    prepare_managed_checkpoint_commit(transaction)
    decide_managed_checkpoint_commit(transaction)

    original_decide = optimizer.decide_checkpoint_commit
    monkeypatch.setattr(
        optimizer,
        "decide_checkpoint_commit",
        lambda: (_ for _ in ()).throw(RuntimeError("injected leaf transition failure")),
    )
    with pytest.raises(RuntimeError, match="injected leaf transition failure"):
        retry_managed_checkpoint_cleanup(transaction)
    assert transaction.phase is ManagedCheckpointTransactionPhase.CLEANUP_PENDING

    before = optimizer.cpu_slabs.master.clone()
    param.decoupled_grad = torch.ones_like(param)
    optimizer.step()
    optimizer.drain()
    assert not torch.equal(optimizer.cpu_slabs.master, before)

    monkeypatch.setattr(optimizer, "decide_checkpoint_commit", original_decide)
    retry_managed_checkpoint_cleanup(transaction)
    assert optimizer.checkpoint_lifecycle == "CLEAN"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_committed_token_projects_leaf_and_cleanup_releases_all_references(
    monkeypatch,
) -> None:
    param, optimizer = _direct_checkpoint_optimizer()
    transaction = begin_managed_checkpoint_load(SimpleNamespace(optimizer=optimizer))
    loaded_state = _detached_state_dict(optimizer)
    loaded_state["state"][0]["master_param"].fill_(11.0)
    optimizer.load_state_dict(loaded_state)
    prepare_managed_checkpoint_load(transaction)
    assert optimizer._checkpoint_rollback is not None
    snapshot_refs = [
        weakref.ref(action.snapshot)
        for action in optimizer._checkpoint_rollback.actions
        if isinstance(action.snapshot, DiskTensorRollbackSnapshot)
    ]
    prepare_managed_checkpoint_commit(transaction)
    shared_token = optimizer._checkpoint_commit_token
    decide_managed_checkpoint_commit(transaction)

    original_decide = optimizer.decide_checkpoint_commit
    transition_attempts = 0

    def fail_transition() -> None:
        nonlocal transition_attempts
        transition_attempts += 1
        raise RuntimeError(f"transition-round-{transition_attempts}")

    monkeypatch.setattr(optimizer, "decide_checkpoint_commit", fail_transition)
    for round_index in (1, 2):
        with pytest.raises(RuntimeError, match=f"transition-round-{round_index}"):
            retry_managed_checkpoint_cleanup(transaction)
        assert optimizer.checkpoint_lifecycle == "COMMIT_DECIDED"
        assert optimizer._checkpoint_load_error is None
        assert optimizer._checkpoint_commit_token is shared_token
        assert len(transaction.cleanup_journal.entries[0].diagnostics) == 1

        before = optimizer.cpu_slabs.master.clone()
        param.decoupled_grad = torch.ones_like(param)
        optimizer.step()
        optimizer.drain()
        assert not torch.equal(optimizer.cpu_slabs.master, before)

    for operation in (
        optimizer.prepare_checkpoint_save,
        optimizer.state_dict,
        optimizer.begin_checkpoint_load,
        optimizer.prepare_checkpoint_load,
        optimizer.apply_model_checkpoint_reset,
        lambda: optimizer.prepare_checkpoint_commit(object()),
    ):
        with pytest.raises(RuntimeError, match="cleanup|commit"):
            operation()
    assert optimizer._checkpoint_commit_token is shared_token
    with pytest.raises(RuntimeError, match="irreversible checkpoint commit"):
        optimizer.abort_checkpoint_load(RuntimeError("late abort"))
    with pytest.raises(RuntimeError, match="rollback recovery"):
        optimizer.prepare_checkpoint_recovery()

    optimizer.mark_checkpoint_poisoned(RuntimeError("post-commit control failure"))
    assert optimizer._checkpoint_load_error is None
    assert optimizer._checkpoint_cleanup_error == (
        "RuntimeError: post-commit control failure"
    )

    monkeypatch.setattr(optimizer, "decide_checkpoint_commit", original_decide)
    retry_managed_checkpoint_cleanup(transaction)
    gc.collect()

    assert optimizer.checkpoint_lifecycle == "CLEAN"
    assert optimizer._checkpoint_load_error is None
    assert optimizer._checkpoint_cleanup_error is None
    assert optimizer._checkpoint_rollback is None
    assert optimizer._checkpoint_cleanup is None
    assert optimizer._checkpoint_prepared_cleanup is None
    assert optimizer._checkpoint_commit_token is None
    assert transaction.cleanup_journal is None
    assert transaction.post_commit_error is None
    assert transaction.commit_token is None
    assert all(reference() is None for reference in snapshot_refs)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_unpublished_commit_token_retains_precommit_rollback_authority() -> None:
    _, optimizer = _direct_checkpoint_optimizer()
    old_master = optimizer.cpu_slabs.master.clone()
    transaction = begin_managed_checkpoint_load(SimpleNamespace(optimizer=optimizer))
    loaded_state = _detached_state_dict(optimizer)
    loaded_state["state"][0]["master_param"].fill_(11.0)
    optimizer.load_state_dict(loaded_state)
    prepare_managed_checkpoint_load(transaction)
    prepare_managed_checkpoint_commit(transaction)

    with pytest.raises(RuntimeError, match="has not been published"):
        optimizer.decide_checkpoint_commit()
    optimizer.abort_checkpoint_load(RuntimeError("pre-commit transition failure"))

    assert optimizer.checkpoint_lifecycle == "CLEAN"
    torch.testing.assert_close(
        optimizer.cpu_slabs.master, old_master, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize(
    ("operation", "request_phase"),
    [("load", "request"), ("save", "save_request")],
)
def test_manager_cleanup_retry_precedes_request_mismatch_without_poison(
    operation: str, request_phase: str
) -> None:
    events: list[str] = []
    manager = object.__new__(MegatronCheckpointManager)
    manager.managed_checkpoint_enabled = True
    manager.optimizer = None
    manager._managed_checkpoint_poisoned_error = None
    manager._retry_managed_checkpoint_cleanup = lambda: events.append("cleanup")
    request_error = ManagedCheckpointPhaseError(
        request_phase,
        [
            {
                "global_rank": 1,
                "error_type": "CheckpointRequestMismatch",
                "message": "different request",
            }
        ],
    )

    def vote(phase, *args, **kwargs):
        del args, kwargs
        events.append(phase)
        return request_error if phase == request_phase else None

    manager._vote_managed_phase = vote
    with pytest.raises(ManagedCheckpointPhaseError, match="different request"):
        if operation == "load":
            manager.load_checkpoint("unused")
        else:
            manager.save_checkpoint("unused")

    assert events == ["cleanup", "recovery_required", request_phase]
    assert manager._managed_checkpoint_poisoned_error is None


@pytest.mark.parametrize("operation", ["load", "save"])
def test_manager_cleanup_failure_blocks_request_vote(operation: str) -> None:
    events: list[str] = []
    manager = object.__new__(MegatronCheckpointManager)
    manager.managed_checkpoint_enabled = True
    manager.optimizer = None

    def fail_cleanup() -> None:
        events.append("cleanup")
        raise RuntimeError("cleanup-still-pending")

    manager._retry_managed_checkpoint_cleanup = fail_cleanup
    manager._vote_managed_phase = lambda *args, **kwargs: events.append("vote")
    with pytest.raises(RuntimeError, match="cleanup-still-pending"):
        if operation == "load":
            manager.load_checkpoint("unused")
        else:
            manager.save_checkpoint("unused")
    assert events == ["cleanup"]


def test_post_commit_vote_failure_does_not_poison_leaf(monkeypatch) -> None:
    class Managed:
        def __init__(self):
            self.poison_calls = 0

        def mark_checkpoint_poisoned(self, error):
            del error
            self.poison_calls += 1

    leaf = Managed()
    transaction = create_managed_checkpoint_load_transaction(None)
    transaction.leaves = (leaf,)
    transaction.committed = True
    transaction.phase = ManagedCheckpointTransactionPhase.CLEANUP_PENDING
    manager = object.__new__(MegatronCheckpointManager)
    manager.checkpoint_process_group = object()
    manager._require_managed_checkpoint_group = lambda: manager.checkpoint_process_group

    def fail_vote(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("post-commit vote failure")

    monkeypatch.setattr(checkpointer_module, "vote_managed_checkpoint_phase", fail_vote)
    with pytest.raises(RuntimeError, match="post-commit vote failure"):
        manager._vote_managed_phase("cleanup", None, transaction)

    assert not transaction.poisoned
    assert isinstance(transaction.post_commit_error, RuntimeError)
    assert leaf.poison_calls == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_post_commit_control_vote_failure_keeps_optimizer_trainable(
    monkeypatch,
) -> None:
    param, optimizer = _direct_checkpoint_optimizer()
    transaction = begin_managed_checkpoint_load(SimpleNamespace(optimizer=optimizer))
    optimizer.load_state_dict(_detached_state_dict(optimizer))
    prepare_managed_checkpoint_load(transaction)
    prepare_managed_checkpoint_commit(transaction)
    decide_managed_checkpoint_commit(transaction)

    manager = object.__new__(MegatronCheckpointManager)
    manager.checkpoint_process_group = object()
    manager._managed_checkpoint_cleanup_recovery = transaction
    manager._managed_checkpoint_poisoned_error = None
    manager._managed_checkpoint_control_error = None
    manager._require_managed_checkpoint_group = lambda: manager.checkpoint_process_group

    monkeypatch.setattr(
        checkpointer_module,
        "vote_managed_checkpoint_phase",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("post-commit control failure")
        ),
    )
    with pytest.raises(RuntimeError, match="post-commit control failure"):
        manager._vote_managed_phase("cleanup", None, transaction)

    assert manager._managed_checkpoint_poisoned_error is None
    assert manager._managed_checkpoint_cleanup_recovery is transaction
    assert optimizer._checkpoint_load_error is None
    param.decoupled_grad = torch.ones_like(param)
    optimizer.step()
    optimizer.drain()

    monkeypatch.setattr(
        checkpointer_module,
        "vote_managed_checkpoint_phase",
        lambda *args, **kwargs: None,
    )
    manager._retry_managed_checkpoint_cleanup()
    assert optimizer.checkpoint_lifecycle == "CLEAN"
    assert transaction.post_commit_error is None
    assert manager._managed_checkpoint_cleanup_recovery is None
    assert manager._managed_checkpoint_control_error is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_partial_multileaf_transition_failure_keeps_every_leaf_trainable(
    monkeypatch,
) -> None:
    params_and_optimizers = [_direct_checkpoint_optimizer() for _ in range(2)]
    root = SimpleNamespace(
        chained_optimizers=[
            SimpleNamespace(optimizer=optimizer)
            for _, optimizer in params_and_optimizers
        ]
    )
    transaction = begin_managed_checkpoint_load(root)
    for _, optimizer in params_and_optimizers:
        optimizer.load_state_dict(_detached_state_dict(optimizer))
    prepare_managed_checkpoint_load(transaction)
    prepare_managed_checkpoint_commit(transaction)
    decide_managed_checkpoint_commit(transaction)

    failing_optimizer = params_and_optimizers[0][1]
    original_decide = failing_optimizer.decide_checkpoint_commit
    monkeypatch.setattr(
        failing_optimizer,
        "decide_checkpoint_commit",
        lambda: (_ for _ in ()).throw(RuntimeError("leaf-0 transition failure")),
    )
    with pytest.raises(RuntimeError, match="leaf-0 transition failure"):
        retry_managed_checkpoint_cleanup(transaction)

    for param, optimizer in params_and_optimizers:
        before = optimizer.cpu_slabs.master.clone()
        param.decoupled_grad = torch.ones_like(param)
        optimizer.step()
        optimizer.drain()
        assert not torch.equal(optimizer.cpu_slabs.master, before)

    monkeypatch.setattr(failing_optimizer, "decide_checkpoint_commit", original_decide)
    retry_managed_checkpoint_cleanup(transaction)
    assert all(
        optimizer.checkpoint_lifecycle == "CLEAN"
        for _, optimizer in params_and_optimizers
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_staged_checkpoint_commit_releases_snapshot_and_clears_poison() -> None:
    _, optimizer = _direct_checkpoint_optimizer()
    root = SimpleNamespace(optimizer=optimizer)
    failed_state = _detached_state_dict(optimizer)
    original_schedule = optimizer._schedule_master_initialization
    optimizer._schedule_master_initialization = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("reset-failure"))
    with pytest.raises(RuntimeError, match="reset-failure"):
        reset_managed_optimizer_from_model(root)
    optimizer._schedule_master_initialization = original_schedule
    assert optimizer.checkpoint_lifecycle == "POISONED"

    optimizer.prepare_checkpoint_recovery()
    assert optimizer.checkpoint_lifecycle == "RELOAD_REQUIRED"
    transaction = begin_managed_checkpoint_load(root)
    snapshot_ref = weakref.ref(optimizer._checkpoint_rollback.master)
    optimizer.load_state_dict(failed_state)
    prepare_managed_checkpoint_load(transaction)
    commit_managed_checkpoint_load(transaction)

    assert optimizer.checkpoint_lifecycle == "CLEAN"
    assert optimizer._checkpoint_rollback is None
    assert snapshot_ref() is None


def test_awex_checkpoint_residency_keeps_managed_optimizer_released() -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []

    class Adapter:
        def __init__(self):
            self._released_tags = {"optimizer", "weights"}
            self._optimizer_rollback_recovery = None

        def resume_memory(self, tags):
            calls.append(("resume", tuple(tags)))
            self._released_tags.difference_update(tags)

        def release_memory(self, tags):
            calls.append(("release", tuple(tags)))
            self._released_tags.update(tags)

    managed = SimpleNamespace(manages_cpu_residency=True)
    root = SimpleNamespace(optimizer=managed)
    adapter = Adapter()

    with checkpoint_awex_residency(adapter, root, with_model=True, with_optimizer=True):
        assert adapter._released_tags == {"optimizer"}

    assert calls == [("resume", ("weights",)), ("release", ("weights",))]
    assert adapter._released_tags == {"optimizer", "weights"}


def test_managed_checkpoint_multiple_abort_failures_do_not_stop_rollback() -> None:
    events: list[str] = []

    class Managed:
        manages_cpu_residency = True

        def __init__(self, name, *, fail_prepare=False, fail_abort=False):
            self.name = name
            self.fail_prepare = fail_prepare
            self.fail_abort = fail_abort
            self.poisoned = False

        def begin_checkpoint_load(self):
            events.append(f"{self.name}.begin")

        def prepare_checkpoint_load(self):
            events.append(f"{self.name}.prepare")
            if self.fail_prepare:
                raise RuntimeError("prepare-primary")

        def abort_checkpoint_load(self, error):
            del error
            events.append(f"{self.name}.abort")
            if self.fail_abort:
                raise RuntimeError(f"abort-{self.name}")

        def mark_checkpoint_poisoned(self, error):
            del error
            self.poisoned = True

    leaves = [
        Managed("a"),
        Managed("b", fail_abort=True),
        Managed("c", fail_prepare=True, fail_abort=True),
    ]
    root = SimpleNamespace(
        chained_optimizers=[SimpleNamespace(optimizer=leaf) for leaf in leaves]
    )
    transaction = begin_managed_checkpoint_load(root)

    with pytest.raises(RuntimeError, match="prepare-primary") as exc_info:
        try:
            prepare_managed_checkpoint_load(transaction)
        except BaseException as error:
            abort_managed_checkpoint_load(transaction, error)
            raise

    assert events[-3:] == ["c.abort", "b.abort", "a.abort"]
    assert len(exc_info.value.__notes__) == 2
    assert all(leaf.poisoned for leaf in leaves)
    assert transaction.begun == []


def test_manager_scheduler_failure_restores_scheduler_and_all_leaves(
    tmp_path: Path, monkeypatch
) -> None:
    class Managed:
        manages_cpu_residency = True

        def __init__(self):
            self.value = "old"
            self.snapshot = None
            self.poisoned = False

        def begin_checkpoint_load(self):
            self.snapshot = self.value

        def prepare_checkpoint_load(self):
            if self.snapshot is None:
                raise RuntimeError("snapshot released before scheduler finalize")

        def prepare_checkpoint_commit(self, commit_token=None):
            del commit_token
            pass

        def decide_checkpoint_commit(self):
            pass

        def discard_checkpoint_snapshot(self):
            self.snapshot = None

        def abort_checkpoint_load(self, error):
            del error
            self.value = self.snapshot
            self.snapshot = None

        def mark_checkpoint_poisoned(self, error):
            del error
            self.poisoned = True

    class Root:
        def __init__(self, leaf):
            self.leaf = leaf
            self.chained_optimizers = [SimpleNamespace(optimizer=leaf)]

        def load_state_dict(self, state):
            self.leaf.value = state["value"]

    class Scheduler:
        def __init__(self):
            self.epoch = 3

        def state_dict(self):
            return {"epoch": self.epoch}

        def load_state_dict(self, state):
            self.epoch = state["epoch"]
            if self.epoch == 9:
                raise RuntimeError("scheduler-finalize")

    leaf = Managed()
    manager = object.__new__(MegatronCheckpointManager)
    manager.optimizer = Root(leaf)
    manager.lr_scheduler = Scheduler()
    manager.model = []
    manager.rank = 0
    manager.use_dist_checkpointing = True
    manager.use_checkpoint_opt_param_scheduler = True
    manager.async_save = False
    manager.managed_checkpoint_enabled = True
    manager.checkpoint_process_group = object()
    manager._async_queue = None
    manager.wait_async_saves = lambda: None
    manager._require_managed_checkpoint_group = lambda: manager.checkpoint_process_group
    manager.generate_state_dict = lambda *args, **kwargs: {}

    def local_vote(group, phase, error, **kwargs):
        del group, kwargs
        if error is None:
            return None
        return ManagedCheckpointPhaseError(
            phase,
            [
                {
                    "global_rank": 0,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            ],
            error,
        )

    monkeypatch.setattr(
        checkpointer_module,
        "vote_managed_checkpoint_phase",
        local_vote,
    )
    monkeypatch.setattr(
        checkpointer_module,
        "load_dist_checkpointing",
        lambda *args, **kwargs: {
            "optimizer": {"value": "new"},
            "lr_scheduler": {"epoch": 9},
        },
    )

    with pytest.raises(RuntimeError, match="scheduler-finalize"):
        manager.load_checkpoint(
            str(tmp_path), with_model=False, with_optimizer=True, with_rng=False
        )

    assert leaf.value == "old"
    assert leaf.snapshot is None
    assert not leaf.poisoned
    assert manager.lr_scheduler.epoch == 3


def test_poisoned_manager_full_load_recovers_to_clean(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[str] = []

    class Managed:
        manages_cpu_residency = True

        def __init__(self):
            self.value = "old"
            self.snapshot = None
            self.lifecycle = "POISONED"

        def prepare_checkpoint_recovery(self):
            calls.append("recover")
            self.lifecycle = "RELOAD_REQUIRED"

        def begin_checkpoint_load(self):
            assert self.lifecycle == "RELOAD_REQUIRED"
            calls.append("begin")
            self.snapshot = self.value
            self.lifecycle = "LOAD_ACTIVE"

        def prepare_checkpoint_load(self):
            calls.append("validate")

        def prepare_checkpoint_commit(self, commit_token=None):
            del commit_token
            calls.append("prepare-commit")

        def decide_checkpoint_commit(self):
            calls.append("decision")

        def discard_checkpoint_snapshot(self):
            calls.append("cleanup")
            self.snapshot = None
            self.lifecycle = "CLEAN"

        def abort_checkpoint_load(self, error):
            del error
            self.value = self.snapshot
            self.snapshot = None
            self.lifecycle = "POISONED"

        def mark_checkpoint_poisoned(self, error):
            del error
            self.lifecycle = "POISONED"

    leaf = Managed()
    manager = object.__new__(MegatronCheckpointManager)
    manager.optimizer = SimpleNamespace(
        chained_optimizers=[SimpleNamespace(optimizer=leaf)]
    )
    manager.lr_scheduler = None
    manager.model = []
    manager.managed_checkpoint_enabled = True
    manager.checkpoint_process_group = object()
    manager._managed_checkpoint_poisoned_error = RuntimeError("old rollback fault")
    manager._async_queue = None
    manager.wait_async_saves = lambda: None
    manager._require_managed_checkpoint_group = lambda: manager.checkpoint_process_group
    manager._build_checkpoint_load_template = lambda **kwargs: {}
    manager._load_checkpoint_data = lambda *args, **kwargs: {"value": "new"}
    manager._apply_checkpoint_state = lambda state, *args, **kwargs: setattr(
        leaf, "value", state["value"]
    )
    manager.get_rng_state = lambda **kwargs: {"rng": "old"}
    manager.load_rng_states = lambda state: None

    def local_vote(group, phase, error, **kwargs):
        del group, kwargs
        if error is None:
            return None
        return ManagedCheckpointPhaseError(
            phase,
            [
                {
                    "global_rank": 0,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            ],
            error,
        )

    monkeypatch.setattr(
        checkpointer_module, "vote_managed_checkpoint_phase", local_vote
    )

    manager.load_checkpoint(
        str(tmp_path), with_model=True, with_optimizer=True, with_rng=True
    )

    assert calls == [
        "recover",
        "begin",
        "validate",
        "prepare-commit",
        "decision",
        "cleanup",
    ]
    assert leaf.value == "new"
    assert leaf.lifecycle == "CLEAN"
    assert leaf.snapshot is None
    assert manager._managed_checkpoint_poisoned_error is None
    assert manager._managed_checkpoint_recovery_transaction is None
    assert manager._managed_checkpoint_cleanup_recovery is None


def test_checkpoint_recovery_retries_all_leaves_and_retains_poison_on_failure() -> None:
    calls: list[str] = []

    class Managed:
        manages_cpu_residency = True

        def __init__(self, name: str, fail: bool):
            self.name = name
            self.fail = fail
            self.poisoned = True

        def prepare_checkpoint_recovery(self):
            calls.append(self.name)
            if self.fail:
                raise RuntimeError(f"recover-{self.name}")
            self.poisoned = False

        def mark_checkpoint_poisoned(self, error):
            del error
            self.poisoned = True

    leaves = (Managed("a", True), Managed("b", False), Managed("c", True))
    transaction = create_managed_checkpoint_load_transaction(None)
    transaction.leaves = leaves
    with pytest.raises(RuntimeError, match="recover-a") as exc_info:
        prepare_managed_checkpoint_recovery(transaction)
    checkpointer_module.poison_managed_checkpoint_transaction(
        transaction, exc_info.value
    )

    assert calls == ["a", "b", "c"]
    assert all(leaf.poisoned for leaf in leaves)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_checkpoint_recovery_retries_only_pending_slab_actions() -> None:
    """A retained rollback must not replay slab copies that already succeeded."""
    _, optimizer = _direct_checkpoint_optimizer()
    optimizer.begin_checkpoint_load()
    assert optimizer.cpu_slabs is not None
    calls: list[tuple[int, str]] = []
    current_round = 1

    def observe_restore(name: str) -> None:
        calls.append((current_round, name))
        if (current_round, name) in {(1, "exp_avg"), (2, "master")}:
            raise RuntimeError(f"copy-{current_round}-{name}")

    _instrument_slab_rollback_actions(optimizer, observe_restore)
    load_error = RuntimeError("injected load failure")
    with pytest.raises(RuntimeError, match="copy-1-exp_avg"):
        optimizer.abort_checkpoint_load(load_error)
    assert optimizer.checkpoint_lifecycle == "RECOVERY_PENDING"
    actions = {action.name: action for action in optimizer._checkpoint_rollback.actions}
    assert actions["slab.master"].status.name == "COMPLETED"
    assert actions["slab.master"].snapshot is None
    assert actions["slab.exp_avg"].status.name == "PENDING"
    assert isinstance(actions["slab.exp_avg"].snapshot, DiskTensorRollbackSnapshot)
    assert actions["slab.exp_avg_sq"].status.name == "COMPLETED"
    assert actions["slab.exp_avg_sq"].snapshot is None

    current_round = 2
    optimizer.retry_checkpoint_recovery()

    assert calls == [
        (1, "master"),
        (1, "exp_avg"),
        (1, "exp_avg_sq"),
        (2, "exp_avg"),
    ]
    assert optimizer.checkpoint_lifecycle == "POISONED"
    assert optimizer._checkpoint_rollback is None
    assert any("slab.exp_avg" in note for note in load_error.__notes__)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("failed_name", ["master", "exp_avg_sq"])
def test_checkpoint_recovery_handles_first_and_last_slab_action_failure(
    failed_name: str,
) -> None:
    _, optimizer = _direct_checkpoint_optimizer()
    optimizer.begin_checkpoint_load()
    assert optimizer.cpu_slabs is not None
    calls: list[tuple[int, str]] = []
    current_round = 1

    def observe_restore(name: str) -> None:
        calls.append((current_round, name))
        if current_round == 1 and name == failed_name:
            raise RuntimeError(f"copy-{name}")

    _instrument_slab_rollback_actions(optimizer, observe_restore)
    with pytest.raises(RuntimeError, match=f"copy-{failed_name}"):
        optimizer.abort_checkpoint_load(RuntimeError("load-failure"))

    current_round = 2
    optimizer.retry_checkpoint_recovery()

    assert [name for round_index, name in calls if round_index == 2] == [failed_name]
    assert optimizer._checkpoint_rollback is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_checkpoint_recovery_retries_only_multiple_failed_slab_actions() -> None:
    _, optimizer = _direct_checkpoint_optimizer()
    optimizer.begin_checkpoint_load()
    assert optimizer.cpu_slabs is not None
    calls: list[tuple[int, str]] = []
    current_round = 1

    def observe_restore(name: str) -> None:
        calls.append((current_round, name))
        if current_round == 1 and name in {"master", "exp_avg_sq"}:
            raise RuntimeError(f"copy-{name}")

    _instrument_slab_rollback_actions(optimizer, observe_restore)
    load_error = RuntimeError("load-failure")
    with pytest.raises(RuntimeError, match="copy-master"):
        optimizer.abort_checkpoint_load(load_error)
    assert len(load_error.__notes__) == 2

    current_round = 2
    optimizer.retry_checkpoint_recovery()

    assert [name for round_index, name in calls if round_index == 2] == [
        "master",
        "exp_avg_sq",
    ]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_checkpoint_recovery_retries_only_pending_param_group() -> None:
    optimizer = _direct_checkpoint_optimizer_with_groups(3)
    current_round = 1
    calls: list[tuple[int, str]] = []

    class RestoreFailingGroup(dict):
        def __init__(self, name: str, values: dict, fail_first: bool):
            super().__init__(values)
            self.name = name
            self.fail_first = fail_first

        def clear(self) -> None:
            calls.append((current_round, self.name))
            if current_round == 1 and self.fail_first:
                raise RuntimeError(f"group-{self.name}")
            super().clear()

    optimizer.param_groups = [
        RestoreFailingGroup(str(index), group, fail_first=index == 1)
        for index, group in enumerate(optimizer.param_groups)
    ]
    original_lrs = [group["lr"] for group in optimizer.param_groups]
    optimizer.begin_checkpoint_load()
    for group in optimizer.param_groups:
        group["lr"] = 9.0

    with pytest.raises(RuntimeError, match="group-1"):
        optimizer.abort_checkpoint_load(RuntimeError("load-failure"))
    actions = {action.name: action for action in optimizer._checkpoint_rollback.actions}
    assert actions["param_group.0"].snapshot is None
    assert actions["param_group.1"].status.name == "PENDING"
    assert actions["param_group.2"].snapshot is None

    current_round = 2
    optimizer.retry_checkpoint_recovery()

    assert [name for round_index, name in calls if round_index == 2] == ["1"]
    assert [group["lr"] for group in optimizer.param_groups] == original_lrs


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_checkpoint_recovery_retries_pending_slab_and_group_independently() -> None:
    optimizer = _direct_checkpoint_optimizer_with_groups(2)
    assert optimizer.cpu_slabs is not None
    current_round = 1
    calls: list[tuple[int, str]] = []

    def observe_restore(name: str) -> None:
        if name == "exp_avg":
            calls.append((current_round, "slab"))
            if current_round == 1:
                raise RuntimeError("slab-failure")

    class RestoreFailingGroup(dict):
        def clear(self) -> None:
            calls.append((current_round, "group"))
            if current_round == 1:
                raise RuntimeError("group-failure")
            super().clear()

    optimizer.param_groups[1] = RestoreFailingGroup(optimizer.param_groups[1])
    optimizer.begin_checkpoint_load()
    _instrument_slab_rollback_actions(optimizer, observe_restore)
    load_error = RuntimeError("load-failure")
    with pytest.raises(RuntimeError, match="slab-failure"):
        optimizer.abort_checkpoint_load(load_error)
    assert len(load_error.__notes__) == 2

    current_round = 2
    optimizer.retry_checkpoint_recovery()

    assert [name for round_index, name in calls if round_index == 2] == [
        "slab",
        "group",
    ]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_checkpoint_recovery_retains_one_action_across_three_failed_rounds() -> None:
    _, optimizer = _direct_checkpoint_optimizer()
    optimizer.begin_checkpoint_load()
    assert optimizer.cpu_slabs is not None
    attempts = 0

    def observe_restore(name: str) -> None:
        if name == "exp_avg":
            nonlocal attempts
            attempts += 1
            if attempts <= 3:
                raise RuntimeError(f"exp-avg-round-{attempts}")

    _instrument_slab_rollback_actions(optimizer, observe_restore)
    original_error = RuntimeError("load-failure")
    with pytest.raises(RuntimeError, match="round-1"):
        optimizer.abort_checkpoint_load(original_error)
    for expected_round in (2, 3):
        with pytest.raises(RuntimeError, match=f"round-{expected_round}"):
            optimizer.retry_checkpoint_recovery()

    optimizer.retry_checkpoint_recovery()
    optimizer.prepare_checkpoint_recovery()

    assert attempts == 4
    assert optimizer.checkpoint_lifecycle == "RELOAD_REQUIRED"
    assert optimizer._checkpoint_rollback is None
    assert len(original_error.__notes__) == 3


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_checkpoint_recovery_fences_slabs_after_pending_drain() -> None:
    """A failed fence must retain every slab snapshot until D2H is quiescent."""
    _, optimizer = _direct_checkpoint_optimizer()
    optimizer.begin_checkpoint_load()
    assert optimizer.cpu_slabs is not None
    original_slabs = optimizer.cpu_slabs
    original_values = {
        name: getattr(original_slabs, name).clone()
        for name in ("master", "exp_avg", "exp_avg_sq")
    }
    for slab in (
        original_slabs.master,
        original_slabs.exp_avg,
        original_slabs.exp_avg_sq,
    ):
        slab.fill_(17)
    calls: list[tuple[int, str]] = []
    current_round = 1

    def observe_restore(name: str) -> None:
        calls.append((current_round, name))

    _instrument_slab_rollback_actions(optimizer, observe_restore)
    original_drain = optimizer.drain
    drain_attempts = 0

    def failing_drain() -> None:
        nonlocal drain_attempts
        drain_attempts += 1
        if drain_attempts == 1:
            raise RuntimeError("injected-drain-failure")
        original_drain()

    optimizer.drain = failing_drain
    with pytest.raises(RuntimeError, match="injected-drain-failure"):
        optimizer.abort_checkpoint_load(RuntimeError("load-failure"))

    actions = {action.name: action for action in optimizer._checkpoint_rollback.actions}
    assert calls == []
    for name in ("master", "exp_avg", "exp_avg_sq"):
        action = actions[f"slab.{name}"]
        assert action.status.name == "PENDING"
        assert isinstance(action.snapshot, DiskTensorRollbackSnapshot)

    # Model an old asynchronous D2H overwriting the authoritative slabs after
    # the first, failed fence. The retained snapshots must repair this write.
    for slab in (
        original_slabs.master,
        original_slabs.exp_avg,
        original_slabs.exp_avg_sq,
    ):
        slab.fill_(23)
    current_round = 2
    optimizer.retry_checkpoint_recovery()

    assert calls == [
        (2, "master"),
        (2, "exp_avg"),
        (2, "exp_avg_sq"),
    ]
    for name, expected in original_values.items():
        torch.testing.assert_close(
            getattr(original_slabs, name), expected, rtol=0.0, atol=0.0
        )
    assert optimizer._checkpoint_rollback is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_checkpoint_recovery_retries_runtime_metadata_action_only() -> None:
    _, optimizer = _direct_checkpoint_optimizer()
    optimizer.begin_checkpoint_load()
    rollback = optimizer._checkpoint_rollback
    metadata_action = next(
        action for action in rollback.actions if action.name == "runtime.metadata"
    )
    original_restore = metadata_action.restore
    attempts = 0

    def fail_once(target, snapshot) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("metadata-restore-failure")
        original_restore(target, snapshot)

    metadata_action.restore = fail_once
    with pytest.raises(RuntimeError, match="metadata-restore-failure"):
        optimizer.abort_checkpoint_load(RuntimeError("load-failure"))

    assert all(
        action.status.name == "COMPLETED"
        for action in rollback.actions
        if action is not metadata_action
    )
    assert all(
        action.snapshot is None
        for action in rollback.actions
        if action is not metadata_action
    )
    assert metadata_action.status.name == "PENDING"
    assert metadata_action.snapshot is not None

    optimizer.retry_checkpoint_recovery()

    assert attempts == 2
    assert optimizer._checkpoint_rollback is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_checkpoint_recovery_rejects_state_writes_until_full_reload() -> None:
    _, optimizer = _direct_checkpoint_optimizer()
    optimizer.begin_checkpoint_load()
    checkpoint = _detached_state_dict(optimizer)
    assert optimizer.cpu_slabs is not None

    def fail_exp_avg(name: str) -> None:
        if name == "exp_avg":
            raise RuntimeError("retained-copy")

    originals = _instrument_slab_rollback_actions(optimizer, fail_exp_avg)
    with pytest.raises(RuntimeError, match="retained-copy"):
        optimizer.abort_checkpoint_load(RuntimeError("load-failure"))

    with pytest.raises(RuntimeError, match="rollback is pending"):
        optimizer.load_state_dict(checkpoint)
    param = optimizer.param_groups[0]["params"][0]
    with pytest.raises(RuntimeError, match="rollback is pending"):
        optimizer.set_scaled_state(
            param, "exp_avg", torch.zeros_like(optimizer.state[param]["exp_avg"])
        )
    exp_avg_action = next(
        action
        for action in optimizer._checkpoint_rollback.actions
        if action.name == "slab.exp_avg"
    )
    exp_avg_action.restore = originals["exp_avg"]
    optimizer.retry_checkpoint_recovery()


def test_managed_checkpoint_vote_rejects_rank_request_mismatch(monkeypatch) -> None:
    group = object()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)

    def gather(statuses, status, *, group):
        assert group is not None
        statuses[:] = [
            status,
            {
                **status,
                "global_rank": 1,
                "group_rank": 1,
                "details": {"path": "/different"},
            },
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    error = vote_managed_checkpoint_phase(
        group,
        "request",
        None,
        details={"path": "/expected"},
        require_consistent_details=True,
    )

    assert error is not None
    assert error.phase == "request"
    assert error.failures[0]["global_rank"] == 1
    assert error.failures[0]["error_type"] == "CheckpointRequestMismatch"


def test_managed_checkpoint_preflight_only_allows_known_mcore_tensor_keys() -> None:
    prefix = "chained_0.optimizer.distributed.dp_group_idx_0"
    for state_name in ("param", "exp_avg", "exp_avg_sq"):
        assert is_managed_optimizer_tensor_checkpoint_key(
            f"{prefix}.gbuf_idx_0.dtype_x.bucket_idx_0.{state_name}"
        )
    assert is_managed_optimizer_tensor_checkpoint_key(
        f"{prefix}.per_bucket_numel/shard_0_1"
    )
    assert is_managed_optimizer_tensor_checkpoint_key(
        f"{prefix}.per_bucket_numel_unpadded/shard_0_1"
    )
    assert not is_managed_optimizer_tensor_checkpoint_key(
        f"{prefix}.unexpected_outer_field/shard_0_1"
    )


@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float64])
def test_managed_checkpoint_source_metadata_rejects_non_fp32_before_load(
    monkeypatch, source_dtype: torch.dtype
) -> None:
    from torch.distributed.checkpoint.metadata import (
        ChunkStorageMetadata,
        Metadata,
        TensorProperties,
        TensorStorageMetadata,
    )

    prefix = (
        "optimizer.distributed.dp_group_idx_0.gbuf_idx_0."
        "dtype_(torch.bfloat16, torch.bfloat16).bucket_idx_0"
    )
    manifest = {
        f"{prefix}.{name}": ((8,), "torch.float32")
        for name in ("param", "exp_avg", "exp_avg_sq")
    }
    entries = {
        key: TensorStorageMetadata(
            TensorProperties(dtype=source_dtype),
            torch.Size([8]),
            [ChunkStorageMetadata(torch.Size([0]), torch.Size([8]))],
        )
        for key in manifest
    }
    monkeypatch.setattr(
        "torch.distributed.checkpoint.FileSystemReader.read_metadata",
        lambda self: Metadata(entries),
    )

    with pytest.raises(TypeError, match=f"dtype.*{source_dtype}"):
        validate_managed_optimizer_source_tensor_metadata("unused", manifest)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("shape", "global shape mismatch"),
        ("missing", "source tensor key mismatch"),
        ("extra", "source tensor key mismatch"),
        ("out_of_bounds", "out of bounds"),
        ("partition", "different source chunk partitions"),
    ],
)
def test_managed_checkpoint_source_metadata_rejects_manifest_corruption(
    monkeypatch, corruption: str, match: str
) -> None:
    from torch.distributed.checkpoint.metadata import (
        ChunkStorageMetadata,
        Metadata,
        TensorProperties,
        TensorStorageMetadata,
    )

    prefix = (
        "optimizer.distributed.dp_group_idx_0.gbuf_idx_0."
        "dtype_(torch.bfloat16, torch.bfloat16).bucket_idx_0"
    )
    manifest = {
        f"{prefix}.{name}": ((8,), "torch.float32")
        for name in ("param", "exp_avg", "exp_avg_sq")
    }

    def entry(
        *,
        shape: int = 8,
        chunks: tuple[tuple[int, int], ...] = ((0, 8),),
    ) -> TensorStorageMetadata:
        return TensorStorageMetadata(
            TensorProperties(dtype=torch.float32),
            torch.Size([shape]),
            [
                ChunkStorageMetadata(torch.Size([offset]), torch.Size([size]))
                for offset, size in chunks
            ],
        )

    entries = {key: entry() for key in manifest}
    if corruption == "shape":
        entries[f"{prefix}.param"] = entry(shape=9, chunks=((0, 9),))
    elif corruption == "missing":
        del entries[f"{prefix}.exp_avg_sq"]
    elif corruption == "extra":
        entries[f"{prefix}.unexpected"] = entry()
    elif corruption == "out_of_bounds":
        entries[f"{prefix}.param"] = entry(chunks=((0, 7), (7, 2)))
    else:
        entries[f"{prefix}.exp_avg"] = entry(chunks=((0, 4), (4, 4)))

    monkeypatch.setattr(
        "torch.distributed.checkpoint.FileSystemReader.read_metadata",
        lambda self: Metadata(entries),
    )

    with pytest.raises((KeyError, TypeError, ValueError), match=match):
        validate_managed_optimizer_source_tensor_metadata("unused", manifest)


@pytest.mark.parametrize(
    "chunks,match",
    [
        ([(0, 3), (4, 4)], "gap"),
        ([(0, 5), (4, 4)], "overlap"),
        ([(0, 7)], "ends"),
    ],
)
def test_managed_checkpoint_source_metadata_rejects_incomplete_coverage(
    monkeypatch, chunks: list[tuple[int, int]], match: str
) -> None:
    from torch.distributed.checkpoint.metadata import (
        ChunkStorageMetadata,
        Metadata,
        TensorProperties,
        TensorStorageMetadata,
    )

    prefix = (
        "optimizer.distributed.dp_group_idx_0.gbuf_idx_0."
        "dtype_(torch.bfloat16, torch.bfloat16).bucket_idx_0"
    )
    manifest = {
        f"{prefix}.{name}": ((8,), "torch.float32")
        for name in ("param", "exp_avg", "exp_avg_sq")
    }
    entries = {
        key: TensorStorageMetadata(
            TensorProperties(dtype=torch.float32),
            torch.Size([8]),
            [
                ChunkStorageMetadata(torch.Size([offset]), torch.Size([size]))
                for offset, size in chunks
            ],
        )
        for key in manifest
    }
    monkeypatch.setattr(
        "torch.distributed.checkpoint.FileSystemReader.read_metadata",
        lambda self: Metadata(entries),
    )

    with pytest.raises(ValueError, match=match):
        validate_managed_optimizer_source_tensor_metadata("unused", manifest)


def test_managed_checkpoint_manifest_merge_allows_dp_chunk_layout_change() -> None:
    key = (
        "optimizer.distributed.dp_group_idx_0.gbuf_idx_0."
        "dtype_(torch.bfloat16, torch.bfloat16).bucket_idx_0.param"
    )
    manifest = {key: ((16,), "torch.float32")}
    assert merge_managed_optimizer_tensor_manifests([manifest, manifest]) == manifest


def test_managed_checkpoint_vote_failure_poisons_manager_and_transaction(
    monkeypatch,
) -> None:
    class Managed:
        def __init__(self):
            self.poisoned = False

        def mark_checkpoint_poisoned(self, error):
            del error
            self.poisoned = True

    leaf = Managed()
    transaction = create_managed_checkpoint_load_transaction(None)
    transaction.leaves = (leaf,)
    manager = object.__new__(MegatronCheckpointManager)
    manager.checkpoint_process_group = object()
    manager._require_managed_checkpoint_group = lambda: manager.checkpoint_process_group

    def fail_vote(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("status vote timed out")

    monkeypatch.setattr(checkpointer_module, "vote_managed_checkpoint_phase", fail_vote)

    with pytest.raises(RuntimeError, match="status vote timed out"):
        manager._vote_managed_phase("prepare_commit", None, transaction)

    assert transaction.poisoned
    assert leaf.poisoned
    assert isinstance(manager._managed_checkpoint_poisoned_error, RuntimeError)


def _run_checkpoint_phase(
    *,
    world_size: int,
    mode: str,
    checkpoint_dir: Path,
    output_dir: Path,
    numel: int = 96,
    inject_prepare_rank: int = -1,
    inject_local_validate_rank: int = -1,
    inject_preflight_rank: int = -1,
    inject_abort_rank: int = -1,
    inject_pending_action_rank: int = -1,
    inject_cleanup_rank: int = -1,
    inject_transition_rank: int = -1,
    inject_snapshot_preflight_rank: int = -1,
    inject_snapshot_write_rank: int = -1,
    inject_snapshot_rename_rank: int = -1,
    inject_snapshot_read_rank: int = -1,
    inject_restore_fd_preclose_rank: int = -1,
    inject_snapshot_fd_close_rank: int = -1,
    inject_snapshot_fd_preclose_reuse_rank: int = -1,
    inject_shared_capacity: bool = False,
    inject_filesystem_identity_conflict: bool = False,
    inject_snapshot_directory_replacement_rank: int = -1,
    cleanup_request_mismatch: bool = False,
    check_template_no_mutation: bool = False,
    expect_source_metadata_error: bool = False,
    recover_after_abort: bool = False,
    continuation_steps: int = 1,
) -> None:
    port = find_free_ports(1)[0]
    env = os.environ.copy()
    env["NCCL_DEBUG"] = "WARN"
    command = [
        "torchrun",
        f"--nproc_per_node={world_size}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        "tests/torchrun/run_gpu_staged_optimizer_checkpoint.py",
        "--mode",
        mode,
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output-dir",
        str(output_dir),
        "--numel",
        str(numel),
        "--continuation-steps",
        str(continuation_steps),
    ]
    if inject_prepare_rank >= 0:
        command.extend(("--inject-prepare-rank", str(inject_prepare_rank)))
    if inject_local_validate_rank >= 0:
        command.extend(
            ("--inject-local-validate-rank", str(inject_local_validate_rank))
        )
    if inject_preflight_rank >= 0:
        command.extend(("--inject-preflight-rank", str(inject_preflight_rank)))
    if inject_abort_rank >= 0:
        command.extend(("--inject-abort-rank", str(inject_abort_rank)))
    if inject_pending_action_rank >= 0:
        command.extend(
            ("--inject-pending-action-rank", str(inject_pending_action_rank))
        )
    if inject_cleanup_rank >= 0:
        command.extend(("--inject-cleanup-rank", str(inject_cleanup_rank)))
    if inject_transition_rank >= 0:
        command.extend(("--inject-transition-rank", str(inject_transition_rank)))
    if inject_snapshot_preflight_rank >= 0:
        command.extend(
            ("--inject-snapshot-preflight-rank", str(inject_snapshot_preflight_rank))
        )
    if inject_snapshot_write_rank >= 0:
        command.extend(
            ("--inject-snapshot-write-rank", str(inject_snapshot_write_rank))
        )
    if inject_snapshot_rename_rank >= 0:
        command.extend(
            ("--inject-snapshot-rename-rank", str(inject_snapshot_rename_rank))
        )
    if inject_snapshot_read_rank >= 0:
        command.extend(("--inject-snapshot-read-rank", str(inject_snapshot_read_rank)))
    if inject_restore_fd_preclose_rank >= 0:
        command.extend(
            (
                "--inject-restore-fd-preclose-rank",
                str(inject_restore_fd_preclose_rank),
            )
        )
    if inject_snapshot_fd_close_rank >= 0:
        command.extend(
            ("--inject-snapshot-fd-close-rank", str(inject_snapshot_fd_close_rank))
        )
    if inject_snapshot_fd_preclose_reuse_rank >= 0:
        command.extend(
            (
                "--inject-snapshot-fd-preclose-reuse-rank",
                str(inject_snapshot_fd_preclose_reuse_rank),
            )
        )
    if inject_shared_capacity:
        command.append("--inject-shared-capacity")
    if inject_filesystem_identity_conflict:
        command.append("--inject-filesystem-identity-conflict")
    if inject_snapshot_directory_replacement_rank >= 0:
        command.extend(
            (
                "--inject-snapshot-directory-replacement-rank",
                str(inject_snapshot_directory_replacement_rank),
            )
        )
    if cleanup_request_mismatch:
        command.append("--cleanup-request-mismatch")
    if check_template_no_mutation:
        command.append("--check-template-no-mutation")
    if expect_source_metadata_error:
        command.append("--expect-source-metadata-error")
    if recover_after_abort:
        command.append("--recover-after-abort")
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )


def _rewrite_managed_source_tensor_dtype(
    checkpoint_dir: Path, dtype: torch.dtype
) -> None:
    metadata_path = checkpoint_dir / ".metadata"
    with metadata_path.open("rb") as stream:
        metadata = pickle.load(stream)
    for key, entry in tuple(metadata.state_dict_metadata.items()):
        if (
            "optimizer.distributed." in key
            and ".gbuf_idx_" in key
            and key.endswith((".param", ".exp_avg", ".exp_avg_sq"))
        ):
            metadata.state_dict_metadata[key] = replace(
                entry, properties=replace(entry.properties, dtype=dtype)
            )
    temporary_path = checkpoint_dir / ".metadata.rewrite"
    with temporary_path.open("wb") as stream:
        pickle.dump(metadata, stream)
    temporary_path.replace(metadata_path)


def _assert_load_results(
    output_dir: Path, world_size: int, *, expected_step: int = 4
) -> None:
    results = [
        json.loads((output_dir / f"load_dp{world_size}_rank{rank}.json").read_text())
        for rank in range(world_size)
    ]
    for result in results:
        assert result["step"] == expected_step
        assert result["scheduler_last_epoch"] == expected_step
        assert result["scheduler_lr"] == [pytest.approx(3e-3 * 0.9**expected_step)]
        assert result["residency"] == "CPU_RESIDENT"
        assert result["cuda_state_numel"] == 0
        assert max(result["errors"].values()) <= 2e-6
        assert result["allocated_peak"] - result["allocated_before"] < 64 * 1024
        assert result["checkpoint_bytes"] > 0
        assert result["rollback_directories"] == 0


def _assert_fault_results(
    output_dir: Path,
    world_size: int,
    *,
    success: bool,
    committed: bool,
    lifecycle: str,
) -> None:
    results = json.loads(
        (output_dir / f"failure_dp{world_size}_rank0.json").read_text()
    )
    assert len(results) == world_size
    assert {result["rank"] for result in results} == set(range(world_size))
    assert {result["success"] for result in results} == {success}
    assert {result["committed"] for result in results} == {committed}
    assert {result["lifecycle"] for result in results} == {lifecycle}
    if lifecycle != "POISONED":
        assert {result["rollback_directories"] for result in results} == {0}


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_dcp_fixed_dp2(tmp_path: Path) -> None:
    """Real MCore/NCCL torch_dist checkpoint resumes at fixed DP=2."""
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        continuation_steps=3,
    )
    _assert_load_results(output_dir, 2, expected_step=6)


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float64])
def test_gpu_staged_optimizer_dcp_rejects_non_fp32_source_metadata_dp2(
    tmp_path: Path, source_dtype: torch.dtype
) -> None:
    """Raw torch_dist metadata must reject casts before any CPU slab write."""
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _rewrite_managed_source_tensor_dtype(checkpoint_dir, source_dtype)
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        expect_source_metadata_error=True,
    )
    for rank in range(2):
        results = json.loads((output_dir / f"metadata_dp2_rank{rank}.json").read_text())
        assert all(not result["success"] for result in results)
        assert all("dtype" in result["error"] for result in results)
        assert all(result["versions_unchanged"] for result in results)
        assert all(result["lifecycle"] == "CLEAN" for result in results)


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_outer_template_does_not_mutate_slabs(
    tmp_path: Path,
) -> None:
    """Building the metadata-only preflight template must not write CPU slabs."""
    if current_platform.device_count() < 1:
        pytest.skip("GPU-staged optimizer DCP test requires a GPU")
    _run_checkpoint_phase(
        world_size=1,
        mode="load",
        checkpoint_dir=tmp_path / "unused-checkpoint",
        output_dir=tmp_path / "output",
        check_template_no_mutation=True,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_dcp_prepare_failure_has_rank_consensus(
    tmp_path: Path,
) -> None:
    """A single-rank commit failure must make every rank abort the DCP load."""
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        inject_prepare_rank=1,
    )
    _assert_fault_results(
        output_dir, 2, success=False, committed=False, lifecycle="CLEAN"
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_dcp_local_validate_failure_has_rank_consensus(
    tmp_path: Path,
) -> None:
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        inject_local_validate_rank=1,
    )
    _assert_fault_results(
        output_dir, 2, success=False, committed=False, lifecycle="CLEAN"
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_dcp_rollback_failure_recovers_by_full_load_dp2(
    tmp_path: Path,
) -> None:
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        inject_prepare_rank=1,
        inject_abort_rank=1,
        recover_after_abort=True,
    )
    for rank in range(2):
        results = json.loads((output_dir / f"failure_dp2_rank{rank}.json").read_text())
        assert all(result["success"] for result in results)
        assert all(result["lifecycle"] == "CLEAN" for result in results)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_dcp_pending_action_recovers_by_full_load_dp2(
    tmp_path: Path,
) -> None:
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        inject_prepare_rank=1,
        inject_pending_action_rank=1,
        recover_after_abort=True,
    )
    for rank in range(2):
        results = json.loads((output_dir / f"failure_dp2_rank{rank}.json").read_text())
        assert all(result["success"] for result in results)
        assert all(result["lifecycle"] == "CLEAN" for result in results)
        assert all(result["post_recovery_step"] for result in results)
        assert results[0]["pending_action_calls"] == []
        assert results[0]["pending_action_names"] == []
        assert results[1]["pending_action_calls"] == [
            "1:slab.master",
            "1:slab.exp_avg",
            "1:slab.exp_avg_sq",
            "2:slab.exp_avg",
        ]
        assert results[1]["pending_action_names"] == [
            "slab.exp_avg",
            "runtime.metadata",
        ]


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_restore_preclose_recovers_without_fd_leak_dp2(
    tmp_path: Path,
) -> None:
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        inject_prepare_rank=1,
        inject_restore_fd_preclose_rank=1,
        recover_after_abort=True,
    )
    for rank in range(2):
        results = json.loads((output_dir / f"failure_dp2_rank{rank}.json").read_text())
        assert all(result["success"] for result in results)
        assert all(result["lifecycle"] == "CLEAN" for result in results)
        assert all(result["post_recovery_step"] for result in results)
        assert all(result["rollback_directories"] == 0 for result in results)
        assert not results[0]["restore_preclose_retained"]
        assert not results[0]["restore_preclose_finalized"]
        assert results[1]["restore_preclose_retained"]
        assert results[1]["restore_preclose_finalized"]


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_disk_read_failure_recovers_by_full_load_dp2(
    tmp_path: Path,
) -> None:
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        inject_prepare_rank=1,
        inject_snapshot_read_rank=1,
        recover_after_abort=True,
    )
    for rank in range(2):
        results = json.loads((output_dir / f"failure_dp2_rank{rank}.json").read_text())
        assert all(result["success"] for result in results)
        assert all(result["lifecycle"] == "CLEAN" for result in results)
        assert all(result["rollback_directories"] == 0 for result in results)


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "fault", ["snapshot_preflight", "snapshot_write", "snapshot_rename"]
)
def test_gpu_staged_optimizer_disk_snapshot_fault_has_rank_consensus_dp2(
    tmp_path: Path, fault: str
) -> None:
    """One rank's disk fault aborts every rank before DCP without a hang."""
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        inject_snapshot_preflight_rank=(1 if fault == "snapshot_preflight" else -1),
        inject_snapshot_write_rank=(1 if fault == "snapshot_write" else -1),
        inject_snapshot_rename_rank=(1 if fault == "snapshot_rename" else -1),
    )
    _assert_fault_results(
        output_dir, 2, success=False, committed=False, lifecycle="CLEAN"
    )
    if fault == "snapshot_rename":
        for rank in range(2):
            results = json.loads(
                (output_dir / f"failure_dp2_rank{rank}.json").read_text()
            )
            assert all(
                "partial rename failure" in result["error"] for result in results
            )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_filesystem_identity_conflict_has_consensus_dp2(
    tmp_path: Path,
) -> None:
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        inject_filesystem_identity_conflict=True,
    )
    _assert_fault_results(
        output_dir, 2, success=False, committed=False, lifecycle="CLEAN"
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_shared_capacity_is_aggregated_dp2(
    tmp_path: Path,
) -> None:
    """Per-rank capacity is insufficient when the shared sum is considered."""
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        inject_shared_capacity=True,
    )
    _assert_fault_results(
        output_dir, 2, success=False, committed=False, lifecycle="CLEAN"
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_replaced_cleanup_directory_is_safe_dp2(
    tmp_path: Path,
) -> None:
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        inject_snapshot_directory_replacement_rank=1,
    )
    _assert_fault_results(
        output_dir, 2, success=True, committed=True, lifecycle="CLEAN"
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize("preclose_reuse", [False, True])
def test_gpu_staged_optimizer_snapshot_fd_close_failure_is_safe_dp2(
    tmp_path: Path, preclose_reuse: bool
) -> None:
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        inject_snapshot_fd_close_rank=(-1 if preclose_reuse else 1),
        inject_snapshot_fd_preclose_reuse_rank=(1 if preclose_reuse else -1),
        cleanup_request_mismatch=True,
    )
    for rank in range(2):
        results = json.loads((output_dir / f"failure_dp2_rank{rank}.json").read_text())
        assert not any(result["success"] for result in results)
        assert all(result["committed"] for result in results)
        assert {result["lifecycle"] for result in results} <= {
            "CLEAN",
            "CLEANUP_PENDING",
        }
        error_fragment = (
            "pre-close FD reuse failure" if preclose_reuse else "FD close failure"
        )
        assert all(error_fragment in result["error"] for result in results)
        assert all(result["committed_state_preserved"] for result in results)
        assert all(result["rollback_directories"] == 0 for result in results)
        assert results[1]["replacement_fd_alive"]


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize("fault", ["preflight", "abort", "cleanup"])
def test_gpu_staged_optimizer_dcp_fault_matrix_has_rank_consensus(
    tmp_path: Path, fault: str
) -> None:
    """Every checkpoint phase reaches one rank-global outcome without hangs."""
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        inject_preflight_rank=1 if fault == "preflight" else -1,
        inject_prepare_rank=1 if fault == "abort" else -1,
        inject_abort_rank=1 if fault == "abort" else -1,
        inject_cleanup_rank=1 if fault == "cleanup" else -1,
    )
    if fault == "preflight":
        _assert_fault_results(
            output_dir, 2, success=False, committed=False, lifecycle="CLEAN"
        )
    elif fault == "abort":
        _assert_fault_results(
            output_dir, 2, success=False, committed=False, lifecycle="POISONED"
        )
    else:
        _assert_fault_results(
            output_dir, 2, success=True, committed=True, lifecycle="CLEAN"
        )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_dcp_cleanup_then_request_mismatch_dp2(
    tmp_path: Path,
) -> None:
    """A new rank-mismatched request cannot revive post-commit rollback state."""
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        inject_cleanup_rank=1,
        cleanup_request_mismatch=True,
    )
    for rank in range(2):
        results = json.loads((output_dir / f"failure_dp2_rank{rank}.json").read_text())
        assert all(result["success"] for result in results)
        assert all(result["committed"] for result in results)
        assert all(result["lifecycle"] == "CLEAN" for result in results)
        assert all(result["request_mismatch_seen"] for result in results)
        assert all(result["committed_state_preserved"] for result in results)
        assert all(result["post_cleanup_step"] for result in results)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_dcp_leaf_transition_failure_is_trainable_dp2(
    tmp_path: Path,
) -> None:
    """A rank-local bookkeeping failure cannot undo or block a global commit."""
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        inject_transition_rank=1,
    )
    for rank in range(2):
        results = json.loads((output_dir / f"failure_dp2_rank{rank}.json").read_text())
        assert all(result["success"] for result in results)
        assert all(result["committed"] for result in results)
        assert all(result["lifecycle"] == "CLEAN" for result in results)
        assert all(result["post_commit_step"] for result in results)
        assert all(result["load_request_mismatch_seen"] for result in results)
        assert all(result["save_request_mismatch_seen"] for result in results)
        assert all(result["committed_state_preserved"] for result in results)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_dcp_reshard_dp1_to_dp2(tmp_path: Path) -> None:
    """MCore dp_reshardable repartitions CPU slabs from DP=1 to DP=2."""
    if current_platform.device_count() < 2:
        pytest.skip("GPU-staged optimizer DCP test requires 2 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=1,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=2,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        continuation_steps=3,
    )
    _assert_load_results(output_dir, 2, expected_step=6)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_dcp_reshard_dp2_to_dp4(tmp_path: Path) -> None:
    """MCore dp_reshardable repartitions CPU slabs from DP=2 to DP=4."""
    if current_platform.device_count() < 4:
        pytest.skip("GPU-staged optimizer DCP test requires 4 GPUs")
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    _run_checkpoint_phase(
        world_size=2,
        mode="save",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    _run_checkpoint_phase(
        world_size=4,
        mode="load",
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        continuation_steps=3,
    )
    _assert_load_results(output_dir, 4, expected_step=6)


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_checkpoint_peak_is_slot_bounded(tmp_path: Path) -> None:
    """Checkpoint load GPU growth stays fixed as CPU optimizer state grows."""
    if current_platform.device_count() < 1:
        pytest.skip("GPU-staged optimizer DCP test requires a GPU")

    deltas = []
    sizes = []
    for numel in (96, 9600):
        checkpoint_dir = tmp_path / f"checkpoint-{numel}"
        output_dir = tmp_path / f"output-{numel}"
        _run_checkpoint_phase(
            world_size=1,
            mode="save",
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            numel=numel,
        )
        _run_checkpoint_phase(
            world_size=1,
            mode="load",
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            numel=numel,
        )
        _assert_load_results(output_dir, 1)
        result = json.loads((output_dir / "load_dp1_rank0.json").read_text())
        deltas.append(result["allocated_peak"] - result["allocated_before"])
        sizes.append(result["checkpoint_bytes"])

    assert sizes[1] > sizes[0] * 5
    assert deltas[1] <= deltas[0] + 16 * 1024


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize("world_size", [1, 2])
def test_gpu_staged_optimizer_managed_async_save_wait_load_and_step_dp(
    tmp_path: Path, world_size: int
) -> None:
    """Real MCore async DCP keeps slabs fenced until foreground finalization."""
    if current_platform.device_count() < world_size:
        pytest.skip(f"managed async DCP test requires {world_size} GPU(s)")
    checkpoint_dir = tmp_path / f"async-dp{world_size}"
    output_dir = tmp_path / f"async-output-dp{world_size}"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    command = [
        "torchrun",
        f"--nproc_per_node={world_size}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={find_free_ports(1)[0]}",
        "tests/torchrun/run_gpu_staged_optimizer_async_checkpoint.py",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output-dir",
        str(output_dir),
        "--numel",
        "96",
    ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
    )
    results = [
        json.loads((output_dir / f"rank{rank}.json").read_text())
        for rank in range(world_size)
    ]
    assert {result["async_state"] for result in results} == {"COMPLETE"}
    assert {result["cuda_state_numel"] for result in results} == {0}
    assert {result["checkpoint_complete"] for result in results} == {True}
    assert {result["slab_storage_preserved"] for result in results} == {True}
    assert all(result["save_schedule_seconds"] >= 0.0 for result in results)
    assert all(result["step_fence_seconds"] >= 0.0 for result in results)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_managed_async_marker_publish_failure_dp2(
    tmp_path: Path,
) -> None:
    """A rank-0 marker failure is voted to every DCP participant."""
    if current_platform.device_count() < 2:
        pytest.skip("managed async marker failure test requires 2 GPUs")
    checkpoint_dir = tmp_path / "async-marker-failure-dp2"
    output_dir = tmp_path / "async-marker-failure-output-dp2"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    command = [
        "torchrun",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={find_free_ports(1)[0]}",
        "tests/torchrun/run_gpu_staged_optimizer_async_checkpoint.py",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output-dir",
        str(output_dir),
        "--numel",
        "96",
        "--inject-marker-publish-failure",
    ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
    )
    results = [
        json.loads((output_dir / f"rank{rank}.json").read_text()) for rank in range(2)
    ]
    assert {result["async_state"] for result in results} == {"FAILED"}
    assert {result["incomplete"] for result in results} == {True}
    assert {result["complete"] for result in results} == {False}
    assert all("async_complete_marker" in result["error"] for result in results)


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize("fault_phase", ["unlink-after-effect", "authority-close"])
def test_gpu_staged_optimizer_managed_async_postcommit_fault_dp2(
    tmp_path: Path, fault_phase: str
) -> None:
    """Post-commit marker faults never revoke a DP-wide completed save."""
    if current_platform.device_count() < 2:
        pytest.skip("managed async post-commit marker test requires 2 GPUs")
    checkpoint_dir = tmp_path / f"async-postcommit-{fault_phase}-dp2"
    output_dir = tmp_path / f"async-postcommit-output-{fault_phase}-dp2"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    command = [
        "torchrun",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={find_free_ports(1)[0]}",
        "tests/torchrun/run_gpu_staged_optimizer_async_checkpoint.py",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output-dir",
        str(output_dir),
        "--numel",
        "96",
        "--inject-marker-postcommit-fault",
        fault_phase,
    ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
    )
    results = [
        json.loads((output_dir / f"rank{rank}.json").read_text()) for rank in range(2)
    ]
    assert {result["async_state"] for result in results} == {"COMPLETE"}
    assert {result["checkpoint_complete"] for result in results} == {True}
    assert {result["cuda_state_numel"] for result in results} == {0}


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_rank_local_mcore_finalize_failure_has_consensus_dp2(
    tmp_path: Path,
) -> None:
    """A rank-local finalize callback must not strand a peer in WORLD all-reduce."""
    if current_platform.device_count() < 2:
        pytest.skip("managed async finalize fault test requires 2 GPUs")
    checkpoint_dir = tmp_path / "async-finalize-failure-dp2"
    output_dir = tmp_path / "async-finalize-failure-output-dp2"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    command = [
        "torchrun",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={find_free_ports(1)[0]}",
        "tests/torchrun/run_gpu_staged_optimizer_async_checkpoint.py",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output-dir",
        str(output_dir),
        "--numel",
        "96",
        "--inject-rank-local-finalize-failure",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=75)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode, command, output=stdout, stderr=stderr
        )
    results = [
        json.loads((output_dir / f"rank{rank}.json").read_text()) for rank in range(2)
    ]
    assert {result["async_state"] for result in results} == {"FAILED"}
    assert {result["collective_healthy"] for result in results} == {True}
    assert {result["queue_depth"] for result in results} == {0}
    assert all("finalize" in result["error"].lower() for result in results)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_callback_audit_failure_has_consensus_dp2(
    tmp_path: Path,
) -> None:
    """Rank-local callback introspection errors must enter the same phase vote."""
    if current_platform.device_count() < 2:
        pytest.skip("managed async callback audit fault test requires 2 GPUs")
    output_dir = tmp_path / "async-callback-audit-failure-output-dp2"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    command = [
        "torchrun",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={find_free_ports(1)[0]}",
        "tests/torchrun/run_gpu_staged_optimizer_async_checkpoint.py",
        "--checkpoint-dir",
        str(tmp_path / "async-callback-audit-failure-dp2"),
        "--output-dir",
        str(output_dir),
        "--numel",
        "96",
        "--inject-callback-detail-failure",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=75)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    assert process.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    results = [
        json.loads(path.read_text()) for path in sorted(output_dir.glob("*.json"))
    ]
    assert len(results) == 2
    assert {result["async_state"] for result in results} == {"FAILED"}
    assert {result["collective_healthy"] for result in results} == {True}
    assert {result["queue_depth"] for result in results} == {0}


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_unreaped_worker_recovery_has_consensus_dp2(
    tmp_path: Path,
) -> None:
    """An unreaped rank keeps every rank pending until the same worker recovers."""
    if current_platform.device_count() < 2:
        pytest.skip("managed async worker recovery test requires 2 GPUs")
    output_dir = tmp_path / "async-worker-recovery-output-dp2"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    command = [
        "torchrun",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={find_free_ports(1)[0]}",
        "tests/torchrun/run_gpu_staged_optimizer_async_checkpoint.py",
        "--checkpoint-dir",
        str(tmp_path / "async-worker-recovery-dp2"),
        "--output-dir",
        str(output_dir),
        "--numel",
        "96",
        "--inject-unreaped-worker",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=75)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    assert process.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    results = [
        json.loads(path.read_text()) for path in sorted(output_dir.glob("*.json"))
    ]
    assert len(results) == 2
    assert {result["async_state"] for result in results} == {"FAILED"}
    assert [result["first_queue_depth"] for result in results] == [0, 1]
    assert {result["queue_depth"] for result in results} == {0}
    assert {result["worker_close_count"] for result in results} == {1}
    assert {result["collective_healthy"] for result in results} == {True}
    assert results[1]["worker_kill_count"] == 1


@pytest.mark.parametrize("worker_rank", (0, 1))
@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_partial_unbound_schedule_recovery_has_consensus_dp2(
    tmp_path: Path,
    worker_rank: int,
) -> None:
    """A one-rank schedule keeps both ranks fenced until exact worker recovery."""
    if current_platform.device_count() < 2:
        pytest.skip("managed async partial schedule test requires 2 GPUs")
    output_dir = tmp_path / f"async-partial-schedule-output-rank{worker_rank}-dp2"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    command = [
        "torchrun",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={find_free_ports(1)[0]}",
        "tests/torchrun/run_gpu_staged_optimizer_async_checkpoint.py",
        "--checkpoint-dir",
        str(tmp_path / f"async-partial-schedule-rank{worker_rank}-dp2"),
        "--output-dir",
        str(output_dir),
        "--numel",
        "96",
        "--inject-partial-unbound-schedule-owner",
        str(worker_rank),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=75)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    assert process.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    results = [
        json.loads(path.read_text()) for path in sorted(output_dir.glob("*.json"))
    ]
    assert len(results) == 2
    assert {result["async_state"] for result in results} == {"FAILED"}
    assert {result["first_queue_depth"] for result in results} == {0, 1}
    assert {result["queue_depth"] for result in results} == {0}
    assert {result["recovery_visible_first"] for result in results} == {True}
    assert {result["recovery_process_identity_preserved"] for result in results} == {
        True
    }
    assert {result["collective_healthy"] for result in results} == {True}
    assert {result["manager_fence_retained"] for result in results} == {True}
    assert {result["manager_lease_retained"] for result in results} == {True}
    assert {result["manager_marker_retained"] for result in results} == {True}
    assert {result["manager_release_count"] for result in results} == {1}
    assert {result["manager_leaf_fail_count"] for result in results} == {1}
    assert {result["incomplete"] for result in results} == {True}
    assert {result["complete"] for result in results} == {False}
    assert results[worker_rank]["worker_kill_count"] == 1
    assert results[worker_rank]["worker_close_count"] == 1


@pytest.mark.parametrize("failure_rank", (0, 1))
@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_failure_queue_pop_has_consensus_dp2(
    tmp_path: Path,
    failure_rank: int,
) -> None:
    """A locally popped rank holds publication through two peer pop failures."""

    if current_platform.device_count() < 2:
        pytest.skip("managed async failure queue-pop test requires 2 GPUs")
    output_dir = tmp_path / f"async-failure-pop-output-rank{failure_rank}-dp2"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    command = [
        "torchrun",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={find_free_ports(1)[0]}",
        "tests/torchrun/run_gpu_staged_optimizer_async_checkpoint.py",
        "--checkpoint-dir",
        str(tmp_path / f"async-failure-pop-rank{failure_rank}-dp2"),
        "--output-dir",
        str(output_dir),
        "--numel",
        "96",
        "--inject-failure-pop-owner",
        str(failure_rank),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=75)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    assert process.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    results = [
        json.loads(path.read_text()) for path in sorted(output_dir.glob("*.json"))
    ]
    assert len(results) == 2
    assert results[0]["phase_trace"] == results[1]["phase_trace"]
    for trace in results[0]["phase_trace"].values():
        assert trace == sorted(trace, key=lambda item: item[0])
    assert {result["async_state"] for result in results} == {"FAILED"}
    assert results[failure_rank]["queue_depths"] == [1, 1, 0]
    assert results[1 - failure_rank]["queue_depths"] == [0, 0, 0]
    assert {tuple(result["recovery_visibility"]) for result in results} == {
        (True, True, False)
    }
    assert {result["pop_attempt_count"] for result in results} == {3}
    assert {result["pop_remove_count"] for result in results} == {1}
    for result in results:
        assert result["manager_retention"][:2] == [
            {"fence": True, "lease": True, "marker": True},
            {"fence": True, "lease": True, "marker": True},
        ]
        assert result["manager_retention"][-1] == {
            "fence": False,
            "lease": False,
            "marker": False,
        }
        assert result["queue_depth"] == 0
        assert result["manager_release_count"] == 1
        assert result["manager_leaf_fail_count"] == 1
        assert result["incomplete"] is True
        assert result["complete"] is False
        assert result["collective_healthy"] is True


@pytest.mark.parametrize("failure_mode", ("post", "clear"))
@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_failure_queue_terminal_cleanup_has_consensus_dp2(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    """Post-effect pop and publication-clear faults retain global recovery."""

    if current_platform.device_count() < 2:
        pytest.skip("managed async terminal cleanup test requires 2 GPUs")
    output_dir = tmp_path / f"async-failure-{failure_mode}-output-dp2"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    command = [
        "torchrun",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={find_free_ports(1)[0]}",
        "tests/torchrun/run_gpu_staged_optimizer_async_checkpoint.py",
        "--checkpoint-dir",
        str(tmp_path / f"async-failure-{failure_mode}-dp2"),
        "--output-dir",
        str(output_dir),
        "--numel",
        "96",
        "--inject-failure-pop-owner",
        "1",
        "--inject-failure-pop-mode",
        failure_mode,
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=75)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    assert process.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    results = [
        json.loads(path.read_text()) for path in sorted(output_dir.glob("*.json"))
    ]
    assert len(results) == 2
    assert results[0]["phase_trace"] == results[1]["phase_trace"]
    assert {tuple(result["queue_depths"]) for result in results} == {(0, 0)}
    assert {tuple(result["recovery_visibility"]) for result in results} == {
        (True, False)
    }
    assert {result["pop_attempt_count"] for result in results} == {2}
    assert {result["pop_remove_count"] for result in results} == {1}
    for result in results:
        assert result["manager_retention"] == [
            {"fence": True, "lease": True, "marker": True},
            {"fence": False, "lease": False, "marker": False},
        ]
        assert result["queue_depth"] == 0
        assert result["manager_release_count"] == 1
        assert result["manager_leaf_fail_count"] == 1
        assert result["incomplete"] is True
        assert result["complete"] is False
        assert result["collective_healthy"] is True


@pytest.mark.parametrize("failure_mode", ("publish-after", "hold", "republish"))
@pytest.mark.multi_gpu
@pytest.mark.slow
def test_gpu_staged_optimizer_recovery_publication_is_atomic_dp2(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    """Publication mutation faults retain the manager transaction on every rank."""

    if current_platform.device_count() < 2:
        pytest.skip("managed async publication transaction test requires 2 GPUs")
    output_dir = tmp_path / f"async-publication-{failure_mode}-output-dp2"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    command = [
        "torchrun",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={find_free_ports(1)[0]}",
        "tests/torchrun/run_gpu_staged_optimizer_async_checkpoint.py",
        "--checkpoint-dir",
        str(tmp_path / f"async-publication-{failure_mode}-dp2"),
        "--output-dir",
        str(output_dir),
        "--numel",
        "96",
        "--inject-failure-pop-owner",
        "0",
        "--inject-failure-pop-mode",
        failure_mode,
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=75)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    assert process.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    results = [
        json.loads(path.read_text()) for path in sorted(output_dir.glob("*.json"))
    ]
    assert len(results) == 2
    assert results[0]["phase_trace"] == results[1]["phase_trace"]
    assert {result["recovery_token_state"] for result in results} == {"CLEARED"}
    assert {result["queue_depth"] for result in results} == {0}
    assert {result["pop_remove_count"] for result in results} == {1}
    assert {result["manager_release_count"] for result in results} == {1}
    assert {result["manager_leaf_fail_count"] for result in results} == {1}
    assert {result["incomplete"] for result in results} == {True}
    assert {result["complete"] for result in results} == {False}
    assert {result["collective_healthy"] for result in results} == {True}

    if failure_mode == "publish-after":
        assert {tuple(result["queue_depths"]) for result in results} == {(0,)}
        assert {tuple(result["recovery_visibility"]) for result in results} == {
            (False,)
        }
        expected_retention = [{"fence": False, "lease": False, "marker": False}]
        assert {result["pop_attempt_count"] for result in results} == {1}
    elif failure_mode == "hold":
        assert results[0]["queue_depths"] == [1, 1, 0]
        assert results[1]["queue_depths"] == [0, 0, 0]
        assert {tuple(result["recovery_visibility"]) for result in results} == {
            (True, True, False)
        }
        expected_retention = [
            {"fence": True, "lease": True, "marker": True},
            {"fence": True, "lease": True, "marker": True},
            {"fence": False, "lease": False, "marker": False},
        ]
        assert {result["pop_attempt_count"] for result in results} == {3}
    else:
        assert {tuple(result["queue_depths"]) for result in results} == {(0, 0)}
        assert results[0]["recovery_visibility"] == [False, False]
        assert results[1]["recovery_visibility"] == [True, False]
        expected_retention = [
            {"fence": True, "lease": True, "marker": True},
            {"fence": False, "lease": False, "marker": False},
        ]
        assert {result["pop_attempt_count"] for result in results} == {2}
    for result in results:
        assert result["manager_retention"] == expected_retention
