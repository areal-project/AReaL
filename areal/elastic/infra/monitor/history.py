# SPDX-License-Identifier: Apache-2.0

"""Bounded published-update and lifecycle measurements for native Elastic training."""

import json
from pathlib import Path

from areal.utils.atomic import atomic_json
from areal.utils.atomic import json_digest as digest


class RuntimeHistory:
    def __init__(self, path: Path, *, context: dict, capacity: int = 256):
        if type(capacity) is not int or capacity < 4:
            raise ValueError("Runtime history needs at least four committed updates")
        self.execution = json.loads(json.dumps(context, allow_nan=False))
        if not all(
            self.execution.get(k)
            for k in ("queue_id", "checkpoint_root", "compatibility")
        ):
            raise ValueError(
                "Runtime history requires queue, checkpoint and training identities"
            )
        self.path, self.context, self.capacity = path, digest(self.execution), capacity
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            atomic_json(
                path, {"format_version": 1, "context": self.context, "updates": []}
            )
        self._read()

    def _read(self):
        value = json.loads(self.path.read_text())
        if value["format_version"] != 1 or value["context"] != self.context:
            raise ValueError("Runtime history belongs to another execution")
        return value

    def append(self, record: dict):
        """Record a completed, published update; checkpointing may lag behind."""
        value = self._read()
        if (
            type(record.get("weights_version")) is not int
            or record["weights_version"] < 1
        ):
            raise ValueError("Runtime observations require committed training progress")
        prior = next(
            (
                r
                for r in value["updates"]
                if r.get("update_id", r["checkpoint_id"])
                == record.get("update_id", record["checkpoint_id"])
            ),
            None,
        )
        if prior is not None:
            if prior != record:
                raise ValueError("Committed runtime observation changed")
            return
        if (
            value["updates"]
            and record["weights_version"] <= value["updates"][-1]["weights_version"]
        ):
            raise ValueError("Runtime progress cannot move backwards")
        atomic_json(
            self.path,
            {**value, "updates": [*value["updates"], record][-self.capacity :]},
        )

    def validate_version(self, version: int):
        value = self._read()
        if any(
            r["weights_version"] > version
            for r in (*value["updates"], *value.get("boundaries", ()))
        ):
            raise ValueError("Runtime history is newer than the recovered checkpoint")

    def rollback(self, version: int):
        """Remove observations belonging to the abandoned, unsaved GPU branch."""
        value = self._read()
        retained = self.snapshot(version)
        if (value["updates"], value.get("boundaries", [])) != (
            retained["updates"],
            retained["boundaries"],
        ):
            atomic_json(self.path, {**value, **retained})

    def append_boundary(self, record: dict):
        """A committed lifecycle measurement, without inventing an optimizer update."""
        value = self._read()
        if (
            type(record.get("weights_version")) is not int
            or record["weights_version"] < 0
        ):
            raise ValueError("Boundary observation needs a committed weight version")
        sequence = value.get("boundary_sequence", 0) + 1
        row = {**record, "sequence": sequence}
        atomic_json(
            self.path,
            {
                **value,
                "boundary_sequence": sequence,
                "boundaries": [*value.get("boundaries", ()), row][-self.capacity :],
            },
        )
        return row

    def snapshot(self, through_version: int):
        value = self._read()
        updates = [
            r for r in value["updates"] if r["weights_version"] <= through_version
        ]
        return {
            **value,
            "updates": updates,
            "boundaries": [
                r
                for r in value.get("boundaries", ())
                if r["weights_version"] <= through_version
            ],
            "through_version": through_version,
        }
