# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from areal.utils import fs


def _partition(mountpoint: Path | str, fstype: str) -> SimpleNamespace:
    return SimpleNamespace(
        mountpoint=str(mountpoint), fstype=fstype, device="/dev/test"
    )


@pytest.fixture
def fs_warnings(caplog):
    fs.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING):
            yield caplog
    finally:
        fs.logger.removeHandler(caplog.handler)


@pytest.mark.parametrize(
    "root_type,mount_type,target_name,expected_warning",
    [
        ("ext4", "nfs", "database/run", True),
        ("nfs", "ext4", "database/run", False),
        ("ext4", "nfs", "data", False),
        ("nfs", "ext4", "data", True),
        ("ext4", "nfs", "data/run", False),
        ("nfs", "ext4", "data/run", True),
    ],
)
def test_shared_path_mount_boundaries(
    tmp_path,
    monkeypatch,
    fs_warnings,
    root_type,
    mount_type,
    target_name,
    expected_warning,
):
    """Select exact and descendant mounts, ignoring similarly named siblings."""
    base = tmp_path.resolve()
    target = base / target_name
    target.mkdir(parents=True)
    partitions = [
        _partition(base.anchor, root_type),
        _partition(base / "data", mount_type),
    ]
    monkeypatch.setattr(fs.psutil, "disk_partitions", lambda **kwargs: partitions)

    fs.validate_shared_path(str(target))

    assert bool(fs_warnings.records) is expected_warning


@pytest.mark.parametrize("fstype,expected_warning", [("ext4", True), ("nfs", False)])
def test_shared_path_root_mount(
    tmp_path, monkeypatch, fs_warnings, fstype, expected_warning
):
    """The root mount contains a path when no more specific mount exists."""
    target = tmp_path.resolve()
    partitions = [_partition(target.anchor, fstype)]
    monkeypatch.setattr(fs.psutil, "disk_partitions", lambda **kwargs: partitions)

    fs.validate_shared_path(str(target))

    assert bool(fs_warnings.records) is expected_warning


@pytest.mark.parametrize(
    "inner_type,expected_warning", [("ext4", True), ("nfs", False)]
)
def test_shared_path_longest_nested_mount(
    tmp_path, monkeypatch, fs_warnings, inner_type, expected_warning
):
    """A real nested mount overrides its containing mount."""
    base = tmp_path.resolve()
    target = base / "data" / "nested" / "run"
    target.mkdir(parents=True)
    outer_type = "nfs" if inner_type == "ext4" else "ext4"
    partitions = [
        _partition(base.anchor, outer_type),
        _partition(base / "data", outer_type),
        _partition(base / "data" / "nested", inner_type),
    ]
    monkeypatch.setattr(fs.psutil, "disk_partitions", lambda **kwargs: partitions)

    fs.validate_shared_path(str(target))

    assert bool(fs_warnings.records) is expected_warning


def test_shared_path_without_partition_warns(tmp_path, monkeypatch, fs_warnings):
    """No containing partition retains the advisory warning."""
    monkeypatch.setattr(fs.psutil, "disk_partitions", lambda **kwargs: [])
    fs.validate_shared_path(str(tmp_path))
    assert fs_warnings.records


@pytest.mark.parametrize("empty", [False, True])
def test_shared_path_disabled_or_empty_avoids_partition_query(
    tmp_path, monkeypatch, fs_warnings, empty
):
    """Disabled warnings and empty paths do not inspect the mount table."""

    def unexpected_query(**kwargs):
        pytest.fail("partition query was not expected")

    monkeypatch.setattr(fs.psutil, "disk_partitions", unexpected_query)
    fs.validate_shared_path("" if empty else str(tmp_path), warn_nfs=empty)
    assert not fs_warnings.records


def test_shared_path_missing_raises_with_warnings_disabled(tmp_path):
    """Existence validation remains independent of advisory warnings."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        fs.validate_shared_path(str(tmp_path / "missing"), warn_nfs=False)
