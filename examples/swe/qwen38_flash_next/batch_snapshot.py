# SPDX-License-Identifier: Apache-2.0
"""Opt-in, lossless CPU snapshots for diagnosing a collected training batch."""

import copy
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from areal.infra.rpc.rtensor import RTensor
from areal.utils.data import RolloutGroup, TrajBatchMeta

_TYPE_KEY = "__areal_snapshot_type__"
_METADATA_TYPES = {cls.__name__: cls for cls in (RolloutGroup, TrajBatchMeta)}


def save_batch_snapshot(path: Path, batch: Any, metadata: dict[str, Any]) -> None:
    """Fetch copied remote wrappers without changing the live batch or its leases."""
    localized = RTensor.localize(copy.deepcopy(batch), preserve_tensor_aliases=True)
    memo: dict[int, torch.Tensor] = {}

    def cpu(value):
        if isinstance(value, torch.Tensor):
            if id(value) not in memo:
                memo[id(value)] = value.detach().to(device="cpu", copy=True)
            return memo[id(value)]
        if type(value) in _METADATA_TYPES.values():
            return {
                _TYPE_KEY: type(value).__name__,
                "fields": {
                    field.name: cpu(getattr(value, field.name))
                    for field in fields(value)
                },
            }
        if isinstance(value, dict):
            if _TYPE_KEY in value:
                raise ValueError("Reserved snapshot metadata key in batch")
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cpu(v) for v in value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"Unsupported batch snapshot value: {type(value).__name__}")

    payload = {"schema_version": 2, "metadata": cpu(metadata), "batch": cpu(localized)}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=".batch-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication refuses to overwrite an existing snapshot.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def load_batch_snapshot(path: Path) -> dict[str, Any]:
    """Load safe tensor data and reconstruct the supported rollout metadata types."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported batch snapshot schema")

    def restore(value):
        if isinstance(value, dict):
            if _TYPE_KEY in value:
                kind = value[_TYPE_KEY]
                if kind not in _METADATA_TYPES or set(value) != {_TYPE_KEY, "fields"}:
                    raise ValueError("Unsupported snapshot metadata record")
                return _METADATA_TYPES[kind](**restore(value["fields"]))
            return {k: restore(v) for k, v in value.items()}
        if isinstance(value, list):
            return [restore(v) for v in value]
        if isinstance(value, tuple):
            return tuple(restore(v) for v in value)
        return value

    return restore(payload) if payload["schema_version"] == 2 else payload


@contextmanager
def capture_training_batches(actor, directory: Path, metadata: dict[str, Any]):
    """Capture preparation and advantage calls; indices are calls, not global steps.

    This preserves inputs for replay experiments, not optimizer/RNG state. It does
    not enable training replay or make repeated off-policy updates safe.
    """
    originals = {
        name: getattr(actor, name) for name in ("prepare_batch", "compute_advantages")
    }
    own = {name: name in vars(actor) for name in originals}
    counts = {name: 0 for name in originals}

    def wrap(name):
        def call(*args, **kwargs):
            index = counts[name]
            prefix = directory / f"{name}-{index:04d}"
            info = {**metadata, "method": name, "call_index": index}
            if name == "compute_advantages":
                batch = args[0] if args else kwargs["data"]
                save_batch_snapshot(prefix.with_suffix(".input.pt"), batch, info)
            result = originals[name](*args, **kwargs)
            save_batch_snapshot(prefix.with_suffix(".output.pt"), result, info)
            counts[name] += 1
            return result

        return call

    try:
        for name in originals:
            setattr(actor, name, wrap(name))
        yield
    finally:
        for name, original in originals.items():
            if own[name]:
                setattr(actor, name, original)
            else:
                delattr(actor, name)


def resolve_replay_paths(path: str | None, paths_json: str | None) -> list[Path]:
    """Read either a single snapshot path or an ordered JSON array of paths."""
    if path and paths_json:
        raise ValueError(
            "Set only one of QWEN_BATCH_REPLAY_PATH and QWEN_BATCH_REPLAY_PATHS"
        )
    if paths_json:
        paths = json.loads(paths_json)
        if (
            not isinstance(paths, list)
            or not paths
            or any(not isinstance(item, str) or not item for item in paths)
        ):
            raise ValueError(
                "QWEN_BATCH_REPLAY_PATHS requires a nonempty JSON array of paths"
            )
        return [Path(item) for item in paths]
    return [Path(path)] if path else []


def validate_diagnostic_replay(config, batch_count: int = 1) -> None:
    """Limit replay to one update per supplied batch without recovery or evaluation."""
    if batch_count < 1 or config.total_train_steps != batch_count:
        raise ValueError(
            "Batch replay requires total_train_steps equal to snapshot count"
        )
    if config.recover.mode not in ("off", "disabled"):
        raise ValueError("Batch replay requires recovery disabled")
    if config.evaluator.eval_before_train:
        raise ValueError("Batch replay requires eval_before_train=false")


@contextmanager
def replay_training_batch(actor, path: Path, expected_metadata: dict[str, Any]):
    """Supply one captured batch once without invoking live generation."""
    with replay_training_batches(actor, [path], expected_metadata):
        yield


@contextmanager
def replay_training_batches(
    actor, paths: list[Path], expected_metadata: dict[str, Any]
):
    """Supply captured batches in order, exercising normal updates between calls.

    Multiple batches must start at call zero and be contiguous from one source
    trial. Initial weights, optimizer and RNG are not recovered from snapshots.
    Exhaustion fails rather than recollecting rollout or reusing an old batch.
    """
    if not paths:
        raise ValueError("Replay requires at least one snapshot")
    batches = []
    origin = None
    for position, path in enumerate(paths):
        payload = load_batch_snapshot(path)
        metadata = payload["metadata"]
        index = metadata.get("call_index")
        if (
            metadata.get("method") != "prepare_batch"
            or type(index) is not int
            or index < 0
        ):
            raise ValueError(
                "Replay requires a prepare_batch snapshot with a valid call index"
            )
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise ValueError(f"Replay metadata mismatch: {key}")
        if len(paths) > 1:
            source = (metadata.get("experiment"), metadata.get("trial"))
            if not all(isinstance(item, str) and item for item in source):
                raise ValueError("Replay sequence requires source experiment and trial")
            if origin is None:
                origin = source
            if source != origin or index != position:
                raise ValueError(
                    "Replay sequence must be contiguous from call zero in one trial"
                )
        batch = payload["batch"]
        if not isinstance(batch, list) or not batch:
            raise ValueError("Replay requires a nonempty prepared batch list")
        batches.append(batch)
    original = actor.prepare_batch
    own = "prepare_batch" in vars(actor)
    consumed = 0

    def prepare(*args, **kwargs):
        nonlocal consumed
        if consumed == len(batches):
            raise RuntimeError("Diagnostic replay batch already consumed")
        batch = batches[consumed]
        batches[consumed] = None
        consumed += 1
        return batch

    try:
        actor.prepare_batch = prepare
        yield
    finally:
        if own:
            actor.prepare_batch = original
        else:
            delattr(actor, "prepare_batch")
