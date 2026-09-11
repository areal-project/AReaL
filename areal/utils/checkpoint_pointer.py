# SPDX-License-Identifier: Apache-2.0
"""Crash-safe publication of immutable recovery checkpoints."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
from dataclasses import dataclass

from areal.utils.logging import getLogger

logger = getLogger("CheckpointPointer")

FORMAT_VERSION = 1
GENERATIONS_DIRNAME = "checkpoint_generations"
GENERATION_PREFIX = "generation_step"
GENERATION_STEP_WIDTH = 8
MANIFEST_DIRNAME = "manifest"
PAYLOADS_DIRNAME = "payloads"
LATEST_FILENAME = "LATEST"
LEGACY_MANIFEST_DIRNAME = "recover_info"
LEGACY_PAYLOAD_DIRNAME = "recover_checkpoint"


class CheckpointConsistencyError(RuntimeError):
    """Checkpoint artifacts exist, but do not identify a usable snapshot."""


@dataclass(frozen=True)
class PointerRecord:
    format_version: int
    generation: str
    global_step: int
    engines: list[str]

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> PointerRecord:
        payload = json.loads(text)
        record = cls(
            format_version=payload["format_version"],
            generation=payload["generation"],
            global_step=payload["global_step"],
            engines=payload["engines"],
        )
        record.validate()
        return record

    def validate(self) -> None:
        if not isinstance(self.format_version, int) or isinstance(
            self.format_version, bool
        ):
            raise ValueError("format_version must be an integer")
        if self.format_version != FORMAT_VERSION:
            raise ValueError(f"unsupported format_version {self.format_version}")
        if not isinstance(self.global_step, int) or isinstance(self.global_step, bool):
            raise ValueError("global_step must be an integer")
        if not isinstance(self.generation, str):
            raise ValueError("generation must be a string")
        if self.generation != generation_dirname(self.global_step):
            raise ValueError(
                f"generation {self.generation!r} does not match "
                f"global_step {self.global_step}"
            )
        if not isinstance(self.engines, list) or not self.engines:
            raise ValueError("engines must be a non-empty list of strings")
        if any(not isinstance(name, str) for name in self.engines):
            raise ValueError("engines must be a non-empty list of strings")
        if any(
            not name or name in (".", "..") or os.path.basename(name) != name
            for name in self.engines
        ):
            raise ValueError("engine names must be single path components")
        if len(set(self.engines)) != len(self.engines):
            raise ValueError("engines contains duplicate names")


@dataclass(frozen=True)
class ResumeSource:
    manifest: str
    payloads: dict[str, str]
    label: str
    transactional: bool


def generation_dirname(global_step: int) -> str:
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    return f"{GENERATION_PREFIX}{global_step:0{GENERATION_STEP_WIDTH}d}"


def generations_root(save_root: str) -> str:
    return os.path.join(save_root, GENERATIONS_DIRNAME)


def generation_dir(save_root: str, global_step: int) -> str:
    return os.path.join(generations_root(save_root), generation_dirname(global_step))


def manifest_dir(generation: str) -> str:
    return os.path.join(generation, MANIFEST_DIRNAME)


def payload_dir(generation: str, engine_name: str) -> str:
    return os.path.join(generation, PAYLOADS_DIRNAME, engine_name)


def latest_path(save_root: str) -> str:
    return os.path.join(save_root, LATEST_FILENAME)


def make_record(global_step: int, engine_names: list[str]) -> PointerRecord:
    record = PointerRecord(
        format_version=FORMAT_VERSION,
        generation=generation_dirname(global_step),
        global_step=global_step,
        engines=list(engine_names),
    )
    record.validate()
    return record


def read_latest(save_root: str) -> PointerRecord | None:
    path = latest_path(save_root)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return PointerRecord.from_json(f.read())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        raise CheckpointConsistencyError(
            f"Invalid recovery checkpoint pointer at {path}: {e}"
        ) from e


def _source_from_record(
    save_root: str,
    record: PointerRecord,
    engine_names: list[str] | None,
    *,
    label: str,
) -> ResumeSource:
    names = record.engines if engine_names is None else engine_names
    if set(names) != set(record.engines):
        raise CheckpointConsistencyError(
            f"Checkpoint {record.generation} contains engines "
            f"{sorted(record.engines)}, but this run requires {sorted(names)}"
        )
    generation = os.path.join(generations_root(save_root), record.generation)
    manifest = manifest_dir(generation)
    payloads = {name: payload_dir(generation, name) for name in names}
    missing = [
        path for path in [manifest, *payloads.values()] if not os.path.isdir(path)
    ]
    if missing:
        raise CheckpointConsistencyError(
            f"Checkpoint pointer {record.generation} references missing paths: {missing}"
        )
    return ResumeSource(
        manifest=manifest,
        payloads=payloads,
        label=label,
        transactional=True,
    )


def _legacy_source(
    save_root: str, engine_names: list[str] | None
) -> ResumeSource | None:
    manifest = os.path.join(save_root, LEGACY_MANIFEST_DIRNAME)
    if not os.path.isdir(manifest):
        return None
    if engine_names is None:
        engine_names = sorted(
            name
            for name in os.listdir(save_root)
            if os.path.isdir(os.path.join(save_root, name, LEGACY_PAYLOAD_DIRNAME))
        )
    payloads = {
        name: os.path.join(save_root, name, LEGACY_PAYLOAD_DIRNAME)
        for name in engine_names
    }
    if not payloads or any(not os.path.isdir(path) for path in payloads.values()):
        return None
    return ResumeSource(
        manifest=manifest,
        payloads=payloads,
        label=f"legacy checkpoint under {save_root}",
        transactional=False,
    )


def resolve_checkpoint(
    save_root: str, engine_names: list[str] | None
) -> ResumeSource | None:
    record = read_latest(save_root)
    if record is not None:
        return _source_from_record(
            save_root,
            record,
            engine_names,
            label=f"checkpoint pointer {latest_path(save_root)}",
        )

    return _legacy_source(save_root, engine_names)


def prepare_generation(
    save_root: str, global_step: int, engine_names: list[str]
) -> tuple[str, PointerRecord]:
    record = make_record(global_step, engine_names)
    current = read_latest(save_root)
    if current is not None and global_step <= current.global_step:
        raise CheckpointConsistencyError(
            f"Refusing to save step {global_step}: LATEST already points to "
            f"step {current.global_step}"
        )

    generation = generation_dir(save_root, global_step)
    if os.path.exists(generation):
        shutil.rmtree(generation)
    os.makedirs(manifest_dir(generation))
    for name in engine_names:
        os.makedirs(payload_dir(generation, name))
    return generation, record


def _fsync_directory(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def publish_latest(save_root: str, record_json: str) -> None:
    """Atomically publish a finalized generation, then retire its predecessor."""
    record = PointerRecord.from_json(record_json)
    _source_from_record(
        save_root,
        record,
        record.engines,
        label=f"generation {record.generation}",
    )
    previous = read_latest(save_root)
    if previous is not None:
        if previous.global_step > record.global_step:
            raise CheckpointConsistencyError(
                f"Refusing to move LATEST backwards from step "
                f"{previous.global_step} to {record.global_step}"
            )
        if previous.global_step == record.global_step:
            if previous == record:
                return
            raise CheckpointConsistencyError(
                f"Step {record.global_step} has conflicting checkpoint pointers"
            )

    os.makedirs(save_root, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".LATEST.", dir=save_root, text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(record_json)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, latest_path(save_root))
        _fsync_directory(save_root)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if previous is None or previous.generation == record.generation:
        return
    previous_dir = os.path.join(generations_root(save_root), previous.generation)
    try:
        shutil.rmtree(previous_dir)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning(
            "Published checkpoint step %s but could not remove the previous "
            "generation %s: %s",
            record.global_step,
            previous_dir,
            e,
        )
