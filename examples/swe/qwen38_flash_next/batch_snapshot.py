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
        if value is None or isinstance(value, str | int | float | bool):
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


def load_batch_snapshot(path: Path, *, mmap: bool = False) -> dict[str, Any]:
    """Load safe tensor data and reconstruct the supported rollout metadata types."""
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
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
    actor,
    paths: list[Path],
    expected_metadata: dict[str, Any],
    *,
    worker_profile: str | None = None,
):
    """Supply captured batches in order, exercising normal updates between calls.

    Multiple batches must start at call zero and be contiguous from one source
    trial. Initial weights, optimizer and RNG are not recovered from snapshots.
    Exhaustion fails rather than recollecting rollout or reusing an old batch.
    """
    if not paths:
        raise ValueError("Replay requires at least one snapshot")
    if worker_profile not in (None, "fixed", "unfixed"):
        raise ValueError("Unknown replay memory profile")
    batches = []
    origin = None
    for position, path in enumerate(paths):
        payload = load_batch_snapshot(path, mmap=worker_profile is not None)
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
        batches.append(path if worker_profile is not None else batch)
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
        if worker_profile is not None:
            results = actor._custom_function_call_all_dp_heads(
                "load_diagnostic_replay", str(batch), expected_metadata, worker_profile
            )
            if len(results) != 1 or not isinstance(results[0], list) or not results[0]:
                raise ValueError(
                    "Diagnostic worker replay requires exactly one data-parallel head"
                )
            return results[0]
        return batch

    try:
        actor.prepare_batch = prepare
        yield
    finally:
        if own:
            actor.prepare_batch = original
        else:
            delattr(actor, "prepare_batch")


def split_replay_image_aliases(batch):
    """Copy each trajectory's image record while retaining all semantic tensors."""

    def clone_record(value, memo):
        if isinstance(value, torch.Tensor):
            if id(value) not in memo:
                memo[id(value)] = value.clone()
            return memo[id(value)]
        if isinstance(value, dict):
            return {key: clone_record(item, memo) for key, item in value.items()}
        if isinstance(value, list):
            return [clone_record(item, memo) for item in value]
        if isinstance(value, tuple):
            return tuple(clone_record(item, memo) for item in value)
        return value

    result = []
    for group in batch:
        group = dict(group)
        for key, records in group.items():
            if str(key).startswith("multi_modal_input"):
                if not isinstance(records, list):
                    raise ValueError(
                        "Replay image records must be a per-trajectory list"
                    )
                group[key] = [clone_record(record, {}) for record in records]
        result.append(group)
    return result


def replay_image_storage_stats(batch):
    """Count diagnostic image aliases without reading tensor values."""
    tensors = []

    def visit(value):
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list | tuple):
            for item in value:
                visit(item)

    for group in batch:
        for key, value in group.items():
            if str(key).startswith("multi_modal_input"):
                visit(value)
    storages = {
        (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
        for tensor in tensors
    }
    return {
        "references": len(tensors),
        "unique_objects": len({id(tensor) for tensor in tensors}),
        "unique_storages": len(storages),
        "unique_storage_bytes": sum(size for _, size in storages),
    }


def load_diagnostic_replay(
    engine, path: str, expected_metadata: dict[str, Any], profile: str
):
    """Read once on the DP head and send shard references through normal PPO RPCs."""
    if not engine.is_data_parallel_head():
        return None
    if profile not in ("fixed", "unfixed"):
        raise ValueError("Unknown replay memory profile")
    payload = load_batch_snapshot(Path(path))
    metadata = payload["metadata"]
    if metadata.get("method") != "prepare_batch":
        raise ValueError("Worker replay requires an unprocessed prepare_batch snapshot")
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"Replay metadata mismatch: {key}")
    batch = payload["batch"]
    if not isinstance(batch, list) or not batch:
        raise ValueError("Replay requires a nonempty prepared batch list")
    before = replay_image_storage_stats(batch)
    if profile == "unfixed":
        batch = split_replay_image_aliases(batch)
    after = replay_image_storage_stats(batch)
    if getattr(engine, "logger", None) is not None:
        engine.logger.info(
            f"DiagnosticReplay profile={profile} image_storage_before={json.dumps(before)} "
            f"image_storage_after={json.dumps(after)}"
        )
    # The RPC guard remotizes this result in the HTTP application context.
    return batch
