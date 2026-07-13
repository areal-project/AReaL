# SPDX-License-Identifier: Apache-2.0

"""Tests for immutable Memory Service release values."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from hashlib import sha256

import pytest

from areal.v2.memory_service.release_types import MemoryRelease, ReleaseManifest
from areal.v2.memory_service.types import MemoryScope


class StringSubclass(str):
    pass


class TupleSubclass(tuple[str, ...]):
    def __iter__(self):
        return iter(("overridden",))


class MemoryScopeSubclass(MemoryScope):
    pass


class ReleaseManifestSubclass(ReleaseManifest):
    pass


class MutableTimezone(tzinfo):
    def __init__(self) -> None:
        self.offset = timedelta(hours=1)

    def utcoffset(self, value: datetime | None) -> timedelta:
        return self.offset

    def dst(self, value: datetime | None) -> timedelta:
        return timedelta(0)

    def tzname(self, value: datetime | None) -> str:
        return "MUTABLE"


class StatefulDatetime(datetime):
    def astimezone(self, timezone: tzinfo | None = None) -> StatefulDatetime:
        return self


LONE_SURROGATE = "\ud800"


def make_scope(**overrides: str) -> MemoryScope:
    values = {
        "tenant_id": "tenant-1",
        "namespace": "assistant-memory",
        "subject_id": "user-1",
    }
    values.update(overrides)
    return MemoryScope(**values)


def make_manifest(**overrides: object) -> ReleaseManifest:
    values: dict[str, object] = {
        "scope": make_scope(),
        "revision_ids": ("rev_a", "rev_b"),
    }
    values.update(overrides)
    return ReleaseManifest(**values)  # type: ignore[arg-type]


def release_identity(
    manifest: ReleaseManifest,
    release_graph_sha256: str = "b" * 64,
) -> tuple[str, str, bytes]:
    manifest_sha256 = sha256(manifest.canonical_bytes()).hexdigest()
    commitment_bytes = json.dumps(
        {
            "manifest_sha256": manifest_sha256,
            "record_kind": "memory_release_commitment",
            "release_graph_sha256": release_graph_sha256,
            "schema_version": 1,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    content_hash = sha256(commitment_bytes).hexdigest()
    return f"rel_{content_hash[:24]}", content_hash, commitment_bytes


def make_release(**overrides: object) -> MemoryRelease:
    manifest = overrides.pop("manifest", make_manifest())
    assert isinstance(manifest, ReleaseManifest)
    graph_hash = overrides.pop("release_graph_sha256", "b" * 64)
    assert isinstance(graph_hash, str)
    release_id, content_hash, _ = release_identity(manifest, graph_hash)
    values: dict[str, object] = {
        "release_id": release_id,
        "manifest": manifest,
        "content_hash": content_hash,
        "release_graph_sha256": graph_hash,
        "created_at": datetime.now(UTC),
    }
    values.update(overrides)
    return MemoryRelease(**values)  # type: ignore[arg-type]


def test_release_manifest_canonical_schema_v1_is_frozen() -> None:
    manifest = make_manifest()

    assert manifest.canonical_bytes() == (
        b'{"revision_ids":["rev_a","rev_b"],"schema_version":1,'
        b'"scope":{"namespace":"assistant-memory","subject_id":"user-1",'
        b'"tenant_id":"tenant-1"}}'
    )
    value = json.loads(manifest.canonical_bytes())
    assert tuple(value) == ("revision_ids", "schema_version", "scope")
    assert "idempotency_key" not in value


def test_empty_release_manifest_has_stable_memory_off_wire_value() -> None:
    manifest = make_manifest(revision_ids=())

    assert manifest.revision_ids == ()
    assert manifest.canonical_bytes() == (
        b'{"revision_ids":[],"schema_version":1,'
        b'"scope":{"namespace":"assistant-memory","subject_id":"user-1",'
        b'"tenant_id":"tenant-1"}}'
    )


def test_release_manifest_canonical_schema_preserves_literal_utf8() -> None:
    manifest = ReleaseManifest(
        scope=make_scope(
            tenant_id="租户一",
            namespace="智能体记忆",
            subject_id="用户🙂",
        ),
        revision_ids=("修订一", "rev_🙂"),
    )

    canonical_bytes = manifest.canonical_bytes()
    expected = (
        '{"revision_ids":["修订一","rev_🙂"],"schema_version":1,'
        '"scope":{"namespace":"智能体记忆","subject_id":"用户🙂",'
        '"tenant_id":"租户一"}}'
    ).encode()

    assert canonical_bytes == expected
    assert b"\\u" not in canonical_bytes


def test_release_manifest_snapshots_tuple_and_string_subclasses() -> None:
    manifest = make_manifest(
        revision_ids=TupleSubclass(
            (StringSubclass(" rev_a "), StringSubclass(" rev_b "))
        )
    )

    assert type(manifest.revision_ids) is tuple
    assert manifest.revision_ids == (" rev_a ", " rev_b ")
    assert all(type(item) is str for item in manifest.revision_ids)


@pytest.mark.parametrize("revision_ids", [["rev_a"], {"rev_a"}, object()])
def test_release_manifest_rejects_non_tuple_revision_ids(
    revision_ids: object,
) -> None:
    with pytest.raises(TypeError, match="revision_ids"):
        make_manifest(revision_ids=revision_ids)


def test_release_manifest_rejects_duplicate_revision_ids() -> None:
    with pytest.raises(ValueError, match="revision_ids"):
        make_manifest(revision_ids=("rev_a", "rev_a"))


@pytest.mark.parametrize("revision_id", ["", " \t\n", LONE_SURROGATE])
def test_release_manifest_rejects_invalid_revision_id(revision_id: str) -> None:
    with pytest.raises(ValueError, match="revision_ids"):
        make_manifest(revision_ids=(revision_id,))


@pytest.mark.parametrize("revision_id", [None, 7, b"rev_a"])
def test_release_manifest_rejects_non_string_revision_id(
    revision_id: object,
) -> None:
    with pytest.raises(TypeError, match="revision_ids"):
        make_manifest(revision_ids=(revision_id,))


def test_release_manifest_uses_base_tuple_iterator_before_duplicate_check() -> None:
    with pytest.raises(ValueError, match="revision_ids"):
        make_manifest(revision_ids=TupleSubclass(("rev_a", "rev_a")))


@pytest.mark.parametrize("scope", [object(), None])
def test_release_manifest_requires_memory_scope(scope: object) -> None:
    with pytest.raises(TypeError, match="scope"):
        make_manifest(scope=scope)


def test_release_manifest_rejects_memory_scope_subclass() -> None:
    scope = MemoryScopeSubclass("tenant-1", "assistant-memory", "user-1")

    with pytest.raises(TypeError, match="scope must be a MemoryScope"):
        make_manifest(scope=scope)


def test_memory_release_snapshots_identifiers_and_utc_datetime() -> None:
    manifest = make_manifest()
    release_id, content_hash, _ = release_identity(manifest)
    release = MemoryRelease(
        release_id=StringSubclass(release_id),
        manifest=manifest,
        content_hash=StringSubclass(content_hash),
        release_graph_sha256=StringSubclass("b" * 64),
        created_at=datetime(
            2026,
            7,
            7,
            12,
            5,
            6,
            789000,
            tzinfo=timezone(timedelta(hours=8)),
        ),
    )

    assert type(release.release_id) is str
    assert release.release_id == release_id
    assert type(release.content_hash) is str
    assert release.content_hash == content_hash
    assert type(release.release_graph_sha256) is str
    assert release.release_graph_sha256 == "b" * 64
    assert type(release.created_at) is datetime
    assert release.created_at == datetime(2026, 7, 7, 4, 5, 6, 789000, tzinfo=UTC)
    assert release.created_at.tzinfo is UTC


def test_memory_release_detaches_mutable_timezone_and_datetime_subclass() -> None:
    mutable_timezone = MutableTimezone()
    mutable_source = datetime(2026, 7, 7, 5, tzinfo=mutable_timezone)
    mutable_release = make_release(created_at=mutable_source)
    subclass_source = StatefulDatetime(2026, 7, 7, 4, tzinfo=UTC)
    subclass_release = make_release(
        manifest=make_manifest(revision_ids=("rev_c",)),
        created_at=subclass_source,
    )
    mutable_timezone.offset = timedelta(hours=2)

    assert type(mutable_release.created_at) is datetime
    assert mutable_release.created_at == datetime(2026, 7, 7, 4, tzinfo=UTC)
    assert type(subclass_release.created_at) is datetime
    assert subclass_release.created_at == datetime(2026, 7, 7, 4, tzinfo=UTC)


@pytest.mark.parametrize(
    "created_at",
    [
        datetime(2026, 7, 7),
        datetime.min.replace(tzinfo=timezone(timedelta(hours=1))),
    ],
)
def test_memory_release_rejects_naive_or_non_normalizable_datetime(
    created_at: datetime,
) -> None:
    with pytest.raises(ValueError, match="created_at"):
        make_release(created_at=created_at)


def test_memory_release_requires_exact_manifest() -> None:
    manifest_subclass = ReleaseManifestSubclass(make_scope(), ("rev_a",))

    with pytest.raises(TypeError, match="manifest must be a ReleaseManifest"):
        MemoryRelease(
            "rel_a",
            manifest_subclass,
            "a" * 64,
            "b" * 64,
            datetime.now(UTC),
        )
    with pytest.raises(TypeError, match="manifest must be a ReleaseManifest"):
        MemoryRelease(
            "rel_a",
            object(),  # type: ignore[arg-type]
            "a" * 64,
            "b" * 64,
            datetime.now(UTC),
        )


@pytest.mark.parametrize(
    "field", ["release_id", "content_hash", "release_graph_sha256"]
)
@pytest.mark.parametrize("value", ["", " \t\n", LONE_SURROGATE])
def test_memory_release_rejects_invalid_text(field: str, value: str) -> None:
    manifest = make_manifest()
    release_id, content_hash, _ = release_identity(manifest)
    values: dict[str, object] = {
        "release_id": release_id,
        "manifest": manifest,
        "content_hash": content_hash,
        "release_graph_sha256": "b" * 64,
        "created_at": datetime.now(UTC),
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        MemoryRelease(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field", ["release_id", "content_hash", "release_graph_sha256"]
)
@pytest.mark.parametrize("value", [None, 7, b"value"])
def test_memory_release_rejects_non_string_text(field: str, value: object) -> None:
    manifest = make_manifest()
    release_id, content_hash, _ = release_identity(manifest)
    values: dict[str, object] = {
        "release_id": release_id,
        "manifest": manifest,
        "content_hash": content_hash,
        "release_graph_sha256": "b" * 64,
        "created_at": datetime.now(UTC),
    }
    values[field] = value

    with pytest.raises(TypeError, match=field):
        MemoryRelease(**values)  # type: ignore[arg-type]


def test_release_values_are_frozen_slotted_and_have_exact_fields() -> None:
    manifest = make_manifest()
    release = make_release(manifest=manifest)

    assert tuple(field.name for field in fields(ReleaseManifest)) == (
        "scope",
        "revision_ids",
    )
    assert tuple(field.name for field in fields(MemoryRelease)) == (
        "release_id",
        "manifest",
        "content_hash",
        "release_graph_sha256",
        "created_at",
    )
    assert not hasattr(manifest, "__dict__")
    assert not hasattr(release, "__dict__")
    with pytest.raises(FrozenInstanceError):
        manifest.revision_ids = ()
    with pytest.raises(FrozenInstanceError):
        release.release_id = "rel_changed"


def test_memory_release_commitment_has_stable_domain_separated_wire_value() -> None:
    manifest = make_manifest(revision_ids=())
    release_id, content_hash, expected = release_identity(manifest, "c" * 64)
    release = MemoryRelease(
        release_id=release_id,
        manifest=manifest,
        content_hash=content_hash,
        release_graph_sha256="c" * 64,
        created_at=datetime.now(UTC),
    )

    assert release.commitment_bytes() == expected
    assert release.commitment_bytes() == (
        b'{"manifest_sha256":"b3acef085dae55b17b05e651e1071f20961db204'
        b'acad245d5285b4f27cf5f5f1","record_kind":"memory_release_commitment",'
        b'"release_graph_sha256":"cccccccccccccccccccccccccccccccccccccccc'
        b'cccccccccccccccccccccccc","schema_version":1}'
    )
    assert release.content_hash == sha256(expected).hexdigest()
    assert release.release_id == f"rel_{release.content_hash[:24]}"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("content_hash", "0" * 64, "content_hash"),
        ("release_id", "rel_" + "0" * 24, "release_id"),
        ("content_hash", "A" * 64, "lowercase SHA-256"),
        ("release_graph_sha256", "g" * 64, "lowercase SHA-256"),
    ],
)
def test_memory_release_rejects_incoherent_commitments(
    field: str,
    value: str,
    message: str,
) -> None:
    manifest = make_manifest()
    release_id, content_hash, _ = release_identity(manifest)
    values: dict[str, object] = {
        "release_id": release_id,
        "manifest": manifest,
        "content_hash": content_hash,
        "release_graph_sha256": "b" * 64,
        "created_at": datetime.now(UTC),
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        MemoryRelease(**values)  # type: ignore[arg-type]
