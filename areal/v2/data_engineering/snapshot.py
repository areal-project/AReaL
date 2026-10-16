# SPDX-License-Identifier: Apache-2.0

"""Content-verified task snapshots and explicit training-data materialization."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any

FORMAT = "areal.data.snapshot.v1"


def json_bytes(value: Any) -> bytes:
    """Fixed snapshot encoding; these bytes determine content identities."""
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def decode_jsonl(data: bytes) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL rows must be objects")
    return rows


def write_file(path: Path, data: bytes | Iterable[dict[str, Any]]) -> None:
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        if isinstance(data, bytes):
            stream.write(data)
        else:
            for row in data:
                stream.write(json_bytes(row))
        stream.flush()
        os.fsync(stream.fileno())


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Stream a derived JSONL export and replace it atomically."""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.unlink()
        write_file(temporary, rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def file_digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return "sha256:" + sha.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL rows must be objects")
                yield row


def _verified_manifest(root: Path) -> dict[str, Any]:
    """Verify large snapshots without loading their payloads into memory."""
    manifest = json.loads((root / "manifest.json").read_bytes())
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError("Unsupported data snapshot format")
    payload = {key: value for key, value in manifest.items() if key != "dataset_id"}
    if manifest.get("dataset_id") != digest(json_bytes(payload)):
        raise ValueError("Snapshot identity mismatch")
    required = {
        "public.jsonl",
        "references.jsonl",
        "train.jsonl",
        "probes.jsonl",
        "report.json",
    }
    if (
        not isinstance(manifest.get("files"), dict)
        or set(manifest["files"]) != required
    ):
        raise ValueError("Snapshot has unexpected or missing files")
    for name, expected in manifest["files"].items():
        path = root / name
        if path.is_symlink():
            raise ValueError("Snapshot files must be regular files")
        if file_digest(path) != expected:
            raise ValueError("Snapshot digest mismatch: " + name)
    ids = set()
    for public, reference in zip_longest(
        iter_jsonl(root / "public.jsonl"), iter_jsonl(root / "references.jsonl")
    ):
        task_id = public.get("task_id") if public is not None else None
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in ids
            or reference is None
            or task_id != reference.get("task_id")
        ):
            raise ValueError("Snapshot public/reference task identities differ")
        ids.add(task_id)
    if len(ids) != manifest["count"] or sum(
        1 for _ in iter_jsonl(root / "train.jsonl")
    ) != len(ids):
        raise ValueError("Snapshot row count mismatch")
    return manifest


@dataclass(frozen=True)
class DatasetSnapshot:
    root: Path
    dataset_id: str

    @property
    def public_path(self) -> Path:
        return self.root / "public.jsonl"

    @property
    def training_path(self) -> Path:
        return self.root / "train.jsonl"

    def training_rows(self) -> list[dict[str, Any]]:
        manifest = _verified_manifest(self.root)
        if manifest["dataset_id"] != self.dataset_id:
            raise ValueError("Snapshot differs from the requested dataset identity")
        data = self.training_path.read_bytes()
        if digest(data) != manifest["files"]["train.jsonl"]:
            raise ValueError("Snapshot digest mismatch: train.jsonl")
        return decode_jsonl(data)


def load_snapshot(root: str | Path) -> DatasetSnapshot:
    path = Path(root).resolve()
    manifest = _verified_manifest(path)
    return DatasetSnapshot(path, manifest["dataset_id"])


def publish_snapshot(
    destination: Path,
    *,
    public: Iterable[dict[str, Any]],
    references: Iterable[dict[str, Any]],
    training: Iterable[dict[str, Any]],
    probes: Iterable[dict[str, Any]],
    report: dict[str, Any],
    provenance: dict[str, Any],
) -> DatasetSnapshot:
    if destination.exists():
        raise FileExistsError(f"Snapshot already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dataset-", dir=destination.parent))
    try:
        values = {
            "public.jsonl": public,
            "references.jsonl": references,
            "train.jsonl": training,
            "probes.jsonl": probes,
            "report.json": json_bytes(report),
        }
        for name, value in values.items():
            write_file(staging / name, value)
        count = sum(1 for _ in iter_jsonl(staging / "public.jsonl"))
        manifest = {
            "format": FORMAT,
            "count": count,
            "files": {name: file_digest(staging / name) for name in values},
            "provenance": provenance,
        }
        manifest["dataset_id"] = digest(json_bytes(manifest))
        write_file(staging / "manifest.json", json_bytes(manifest))
        _verified_manifest(staging)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return DatasetSnapshot(destination.resolve(), manifest["dataset_id"])


def export_hf_dataset(snapshot: DatasetSnapshot, destination: str | Path) -> Path:
    """Materialize a task recipe's training view for existing AReaL dataset loaders.

    Tokenization and RL/SFT/OPD-specific tensor conversion remain downstream.
    """
    from datasets import Dataset

    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / f".{destination.name}.export.lock"
    with lock.open("x"):
        staging: Path | None = None
        try:
            if destination.exists():
                raise FileExistsError(f"Training dataset already exists: {destination}")
            rows = snapshot.training_rows()
            staging = Path(tempfile.mkdtemp(prefix=".hf-data-", dir=destination.parent))
            Dataset.from_list(rows).save_to_disk(str(staging))
            write_file(
                staging / "areal-source.json",
                json_bytes({"dataset_id": snapshot.dataset_id}),
            )
            staging.rename(destination)
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
            lock.unlink(missing_ok=True)
    return destination
