# SPDX-License-Identifier: Apache-2.0

"""Atomic checkpoint generations and their durable consumption cursor."""

import json
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from areal.utils.atomic import atomic_json


class CheckpointStore:
    def __init__(self, root: Path, *, keep: int = 2, free_space_floor: int = 10 << 30):
        if keep < 1:
            raise ValueError("keep must be at least one")
        self.root = root.resolve()
        self.keep = keep
        self.free_space_floor = free_space_floor
        self.root.mkdir(parents=True, exist_ok=True)

    def commit(
        self,
        *,
        step: int,
        world_size: int,
        compatibility: dict[str, Any],
        write: Callable[[Path], None],
        estimated_bytes: int,
        data_cursor: dict[str, Any] | None = None,
        data_progress: dict[str, int] | None = None,
    ) -> Path:
        if step < 1 or world_size < 1 or estimated_bytes < 0:
            raise ValueError("Invalid checkpoint step, world size or size estimate")
        if shutil.disk_usage(self.root).free < self.free_space_floor + estimated_bytes:
            raise OSError("Insufficient space to commit another checkpoint safely")
        if data_cursor is not None:
            self._validate_cursor(data_cursor, step)
        if data_progress is not None:
            self._validate_progress(data_progress)
        previous = None
        if (self.root / "latest.json").exists():
            previous_path, previous_manifest = self.latest(compatibility)
            if step < previous_manifest["global_step"]:
                raise ValueError("Checkpoint cannot move published progress backwards")
            previous = previous_path.name
        # A crash after directory rename but before latest publication leaves an
        # orphan. A unique generation lets replay commit the same step again.
        target = self.root / f"step-{step:08d}-{uuid.uuid4().hex}"
        staging = self.root / f".step-{step:08d}-{uuid.uuid4().hex}.pending"
        staging.mkdir()
        # Failed/incomplete writes remain pending and are never selected by latest.
        write(staging)
        if not (staging / "dcp" / ".metadata").is_file():
            raise ValueError("Missing DCP metadata")
        for rank in range(world_size):
            state = json.loads((staging / f"runtime-{rank}.json").read_text())
            if (state["rank"], state["world_size"], state["policy_version"]) != (
                rank,
                world_size,
                step,
            ):
                raise ValueError("Incomplete or inconsistent rank runtime state")
        for file in staging.rglob("*"):
            if file.is_file():
                with file.open("rb") as stream:
                    os.fsync(stream.fileno())
        # Persist nested DCP directory entries before publishing the generation.
        for directory in sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        manifest = {
            "format_version": 1,
            "global_step": step,
            "next_batch_id": step,
            "world_size": world_size,
            "compatibility": compatibility,
            "previous_generation": previous,
        }
        if data_cursor is not None:
            manifest.pop("next_batch_id")
            manifest["data_cursor"] = data_cursor
        if data_progress is not None:
            manifest["data_progress"] = data_progress
        atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, target)
        atomic_json(self.root / "latest.json", {"generation": target.name})
        # Follow published history, not directory timestamps: a newer orphan is
        # not a substitute for the previous recoverable generation.
        retained = {target.name}
        while previous and len(retained) < self.keep:
            if Path(previous).name != previous or not previous.startswith("step-"):
                raise ValueError("Invalid checkpoint history")
            prior = self.root / previous / "manifest.json"
            if not prior.is_file() or previous in retained:
                break
            retained.add(previous)
            previous = json.loads(prior.read_text()).get("previous_generation")
        for old in self.root.glob("step-*"):
            if old.name not in retained and (old / "manifest.json").is_file():
                shutil.rmtree(old)
        return target

    @staticmethod
    def _validate_progress(progress: dict[str, int]) -> None:
        if not isinstance(progress, dict) or any(
            type(progress.get(key)) is not int or progress[key] < 0
            for key in ("samples", "effective_tokens")
        ):
            raise ValueError("Invalid logical data progress")

    @staticmethod
    def _validate_cursor(cursor: dict[str, Any], step: int) -> None:
        if (
            cursor.get("format_version") != 1
            or not isinstance(cursor.get("queue_id"), str)
            or not cursor["queue_id"]
            or type(cursor.get("applied_batches")) is not int
            or cursor["applied_batches"] != step
            or type(cursor.get("through_sequence")) is not int
            or cursor["through_sequence"] < step
            or not isinstance(cursor.get("chain_sha256"), str)
            or len(cursor["chain_sha256"]) != 64
        ):
            raise ValueError("Checkpoint data cursor disagrees with training progress")

    def latest(self, compatibility: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
        pointer = json.loads((self.root / "latest.json").read_text())
        name = pointer["generation"]
        if Path(name).name != name or not name.startswith("step-"):
            raise ValueError("Invalid checkpoint generation")
        path = self.root / name
        manifest = json.loads((path / "manifest.json").read_text())
        if (
            manifest["format_version"] != 1
            or manifest["compatibility"] != compatibility
        ):
            raise ValueError("Checkpoint is incompatible with this training run")
        if "data_progress" in manifest:
            self._validate_progress(manifest["data_progress"])
        if not (path / "dcp" / ".metadata").is_file():
            raise ValueError("Checkpoint DCP payload is missing")
        if "data_cursor" in manifest:
            self._validate_cursor(manifest["data_cursor"], manifest["global_step"])
        elif manifest["next_batch_id"] != manifest["global_step"]:
            raise ValueError(
                "Checkpoint data position disagrees with training progress"
            )
        for rank in range(manifest["world_size"]):
            if not (path / f"runtime-{rank}.json").is_file():
                raise ValueError("Checkpoint rank runtime state is missing")
            state = json.loads((path / f"runtime-{rank}.json").read_text())
            if (state["rank"], state["world_size"], state["policy_version"]) != (
                rank,
                manifest["world_size"],
                manifest["global_step"],
            ):
                raise ValueError("Checkpoint rank state disagrees with manifest")
        return path, manifest
