# SPDX-License-Identifier: Apache-2.0

"""Self-contained batch snapshots for single-controller RL debugging."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

import torch

from areal.infra.rpc.rtensor import RTensor, clear_fetch_buffer, flatten_shard_ids
from areal.utils.data import RolloutGroup


class TrajectoryStore:
    """Own a versioned directory of complete, CPU-resident training batches.

    A single controller writes one file per global step. Published files are
    immutable; a failed/interrupted temporary file cannot appear as a valid step.
    Only tensors and primitive containers are deserialized, never remote handles
    or arbitrary Python classes. Callers must retain model/optimizer state
    separately to reproduce a particular update.
    """

    FORMAT_VERSION = 1

    def __init__(self, path: str, *, scope: str, identity: dict[str, Any]):
        self.path = Path(path)
        self.scope = scope
        self.identity = identity

    def initialize_dump(self, *, steps_per_epoch: int, batch_size: int) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format_version": self.FORMAT_VERSION,
            "scope": self.scope,
            "identity": self.identity,
            "steps_per_epoch": steps_per_epoch,
            "batch_size": batch_size,
        }
        self._publish("manifest.json", lambda f: f.write(json.dumps(manifest).encode()))

    def replay_manifest(self, *, batch_size: int) -> dict[str, Any]:
        with (self.path / "manifest.json").open() as f:
            manifest = json.load(f)
        expected = {
            "format_version": self.FORMAT_VERSION,
            "scope": self.scope,
            "identity": self.identity,
            "batch_size": batch_size,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"Trajectory manifest {key} does not match replay config"
                )
        if (
            type(manifest.get("steps_per_epoch")) is not int
            or manifest["steps_per_epoch"] < 1
        ):
            raise ValueError("Trajectory manifest needs positive steps_per_epoch")
        return manifest

    def validate_steps(self, start: int, stop: int) -> None:
        for step in range(start, stop):
            if not self._step_path(step).is_file():
                raise FileNotFoundError(f"Missing trajectory snapshot for step {step}")

    def save(self, batch: list[dict[str, Any]], step: int) -> Path:
        # localize() mutates wrappers and fills the process-local fetch buffer.
        # Work on a copied graph, retain aliases, and drain only this batch's
        # cached shards even if fetching or encoding fails. Storage stays live
        # until the trainer's normal clear_batches fan-out.
        shard_ids = flatten_shard_ids(batch)
        try:
            copied = self._copy_wrappers(batch, {})
            local = RTensor.localize(copied, preserve_tensor_aliases=True)
            encoded = self._encode(copied, {})
            self._validate_batch(local)
            payload = {
                "format_version": self.FORMAT_VERSION,
                "scope": self.scope,
                "step": step,
                "batch": encoded,
            }
            self._publish(self._step_path(step).name, lambda f: torch.save(payload, f))
        finally:
            clear_fetch_buffer(shard_ids)
        return self._step_path(step)

    def load(self, step: int) -> list[dict[str, Any]]:
        payload = torch.load(
            self._step_path(step), map_location="cpu", weights_only=True
        )
        if not isinstance(payload, dict) or any(
            payload.get(key) != value
            for key, value in (
                ("format_version", self.FORMAT_VERSION),
                ("scope", self.scope),
                ("step", step),
            )
        ):
            raise ValueError(f"Incompatible trajectory snapshot at step {step}")
        batch = self._decode(payload["batch"], {})
        self._validate_batch(batch)
        return batch

    def _step_path(self, step: int) -> Path:
        return self.path / f"step_{step:06d}.pt"

    def _publish(self, name: str, write: Callable[[BinaryIO], Any]) -> None:
        # A hard link publishes the completed file atomically without replacing
        # another run's snapshot. Both names live on the same filesystem.
        with tempfile.NamedTemporaryFile(dir=self.path, delete=False) as f:
            temporary = Path(f.name)
            try:
                write(f)
                f.flush()
                os.fsync(f.fileno())
                os.link(temporary, self.path / name)
            finally:
                temporary.unlink(missing_ok=True)

    @classmethod
    def _copy_wrappers(cls, value: Any, memo: dict[int, Any]) -> Any:
        if id(value) in memo:
            return memo[id(value)]
        if isinstance(value, RTensor):
            copied = RTensor(shard=value.shard, data=value.data)
        elif isinstance(value, dict):
            copied = {k: cls._copy_wrappers(v, memo) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            copied = type(value)(cls._copy_wrappers(v, memo) for v in value)
        else:
            return value
        memo[id(value)] = copied
        return copied

    @classmethod
    def _encode(cls, value: Any, memo: dict[int, Any]) -> Any:
        if id(value) in memo:
            return memo[id(value)]
        if isinstance(value, RTensor):
            encoded = cls._encode(value.data, memo)
        elif isinstance(value, torch.Tensor):
            if value.is_meta:
                raise ValueError(
                    "Trajectory snapshot contains an unresolved meta tensor"
                )
            encoded = value.detach().to(device="cpu", copy=True)
        elif isinstance(value, RolloutGroup):
            encoded = {
                "__areal_rollout_group__": True,
                "row_counts": value.row_counts,
                "rewards": value.rewards,
            }
        elif isinstance(value, dict):
            if "__areal_rollout_group__" in value:
                raise ValueError("Reserved trajectory metadata key in batch")
            encoded = {k: cls._encode(v, memo) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            encoded = type(value)(cls._encode(v, memo) for v in value)
        elif value is None or type(value) in (bool, int, float, str):
            return value
        else:
            raise TypeError(
                f"Unsupported trajectory snapshot value: {type(value).__name__}"
            )
        memo[id(value)] = encoded
        return encoded

    @classmethod
    def _decode(cls, value: Any, memo: dict[int, Any]) -> Any:
        if id(value) in memo:
            return memo[id(value)]
        if isinstance(value, dict):
            if value.get("__areal_rollout_group__") is True:
                decoded = RolloutGroup(value["row_counts"], value["rewards"])
            else:
                decoded = {k: cls._decode(v, memo) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            decoded = type(value)(cls._decode(v, memo) for v in value)
        else:
            return value
        memo[id(value)] = decoded
        return decoded

    @staticmethod
    def _validate_batch(batch: Any) -> None:
        if not isinstance(batch, list) or not batch:
            raise ValueError("Trajectory snapshot must contain a nonempty batch list")
        token_fields = (
            "input_ids",
            "attention_mask",
            "loss_mask",
            "logprobs",
            "versions",
        )
        required = (*token_fields, "rewards")
        for trajectory in batch:
            if not isinstance(trajectory, dict) or any(
                not isinstance(trajectory.get(key), torch.Tensor) for key in required
            ):
                raise ValueError(f"Trajectory snapshot needs tensor fields {required}")
            shape = trajectory["input_ids"].shape
            if len(shape) != 2 or any(
                trajectory[key].shape != shape for key in token_fields
            ):
                raise ValueError(
                    "Trajectory token fields must have matching [rows, tokens] shapes"
                )
            rewards = trajectory["rewards"]
            if rewards.ndim < 1 or rewards.shape[0] != shape[0]:
                raise ValueError("Trajectory rewards must match the token row count")
            group = trajectory.get("rollout_group")
            if group is not None:
                if not isinstance(group, RolloutGroup):
                    raise ValueError("Invalid rollout_group in trajectory snapshot")
                group.validate_rows(trajectory["input_ids"].shape[0])
