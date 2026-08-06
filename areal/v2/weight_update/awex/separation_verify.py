# SPDX-License-Identifier: Apache-2.0
"""Opt-in post-apply verification for separated-card AWEX transfers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from typing import Any, Literal

import torch
import torch.distributed as dist

from areal.utils import logging

logger = logging.getLogger("AwexSeparationVerify")

_VERIFY_ENV = "AREAL_DTE_POST_APPLY_VERIFY"
_FINGERPRINT_LANES = 64
_CUDA_SYNC_INTERVAL = 16


def separation_post_apply_verify_enabled() -> bool:
    """Return whether the correctness-only post-apply verifier is enabled."""
    return os.environ.get(_VERIFY_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _slice_key(slices: tuple[slice, ...]) -> list[list[int | None]]:
    return [[item.start, item.stop, item.step] for item in slices]


def _metadata_key(op: Any) -> str:
    """Build a role-independent identity for one reshard operation."""
    payload = {
        "send_rank": int(op.send_rank),
        "recv_rank": int(op.recv_rank),
        "send_name": str(op.send_shard_meta.name),
        "recv_name": str(op.recv_shard_meta.name),
        "send_shape": list(op.send_shard_meta.shape),
        "recv_shape": list(op.recv_shard_meta.shape),
        "send_offset": list(op.send_offset),
        "recv_offset": list(op.recv_offset),
        "overlap_shape": list(op.overlap_shape),
        "train_slices": _slice_key(tuple(op.train_slices)),
        "inf_slices": _slice_key(tuple(op.inf_slices)),
        "send_dtype": str(op.send_shard_meta.dtype),
        "recv_dtype": str(op.recv_shard_meta.dtype),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _slice_bounds(
    slices: tuple[slice, ...], shape: tuple[int, ...], name: str
) -> tuple[tuple[int, int], ...]:
    if len(slices) != len(shape):
        raise ValueError(
            f"receiver slice rank mismatch for {name}: "
            f"slices={len(slices)} shape={shape}"
        )
    bounds = []
    for dim, (item, size) in enumerate(zip(slices, shape, strict=True)):
        step = 1 if item.step is None else item.step
        start = 0 if item.start is None else item.start
        stop = size if item.stop is None else item.stop
        if step != 1 or start < 0 or stop < start or stop > size:
            raise ValueError(
                f"receiver slice out of bounds for {name} dim={dim}: "
                f"slice={item} size={size}"
            )
        bounds.append((start, stop))
    return tuple(bounds)


def _regions_overlap(
    left: tuple[tuple[int, int], ...], right: tuple[tuple[int, int], ...]
) -> bool:
    return all(
        max(a_start, b_start) < min(a_stop, b_stop)
        for (a_start, a_stop), (b_start, b_stop) in zip(left, right, strict=True)
    )


def _validate_infer_coverage(
    params: dict[str, torch.Tensor], operations: list[Any]
) -> None:
    """Require every planned receiver shard to be covered exactly once."""
    by_name: dict[str, list[Any]] = {}
    for op in operations:
        by_name.setdefault(str(op.recv_shard_meta.name), []).append(op)

    for name, ops in by_name.items():
        tensor = params.get(name)
        if tensor is None:
            raise ValueError(f"receiver plan references missing parameter {name}")
        shape = tuple(tensor.shape)
        regions = [_slice_bounds(tuple(op.inf_slices), shape, name) for op in ops]
        unique_regions: list[tuple[tuple[int, int], ...]] = []
        for region in regions:
            # Replicated training ranks can legitimately produce multiple
            # sender ops for the exact same receiver region. Keep every op in
            # the fingerprint report so all replicas must match the final
            # receiver value, but count the region once for coverage.
            if region in unique_regions:
                continue
            for existing in unique_regions:
                if _regions_overlap(existing, region):
                    raise ValueError(
                        f"receiver plan overlap for {name}: {existing} vs {region}"
                    )
            unique_regions.append(region)
        covered = sum(
            math.prod(stop - start for start, stop in region)
            for region in unique_regions
        )
        if covered != tensor.numel():
            raise ValueError(
                f"receiver plan coverage gap for {name}: "
                f"covered={covered} expected={tensor.numel()}"
            )

    alias_pairs = {
        "lm_head.weight": "model.embed_tokens.weight",
        "model.embed_tokens.weight": "lm_head.weight",
    }
    for alias, counterpart in alias_pairs.items():
        if alias not in params or alias in by_name or counterpart not in by_name:
            continue
        alias_tensor = params[alias]
        counterpart_tensor = params.get(counterpart)
        if counterpart_tensor is None:
            raise ValueError(f"receiver alias {alias} has no local {counterpart}")
        same_storage = (
            alias_tensor.untyped_storage().data_ptr()
            == counterpart_tensor.untyped_storage().data_ptr()
            and alias_tensor.storage_offset() == counterpart_tensor.storage_offset()
            and tuple(alias_tensor.shape) == tuple(counterpart_tensor.shape)
            and tuple(alias_tensor.stride()) == tuple(counterpart_tensor.stride())
        )
        if not same_storage:
            raise ValueError(
                f"receiver alias {alias} does not share storage with {counterpart}"
            )


def _allowed_infer_alias(
    name: str, infer_names: set[str], recv_op_names: set[str]
) -> bool:
    pairs = {
        "lm_head.weight": "model.embed_tokens.weight",
        "model.embed_tokens.weight": "lm_head.weight",
    }
    counterpart = pairs.get(name)
    return counterpart in infer_names and counterpart in recv_op_names


def _tensor_fingerprint_lanes(tensor: torch.Tensor) -> torch.Tensor:
    """Return position-sensitive byte-lane sums without copying weights to CPU."""
    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    if raw.numel() == 0:
        raise ValueError("post-apply verification does not accept empty tensors")

    lane_count = min(_FINGERPRINT_LANES, raw.numel())
    main_size = raw.numel() - raw.numel() % lane_count
    if main_size:
        blocks = raw[:main_size].reshape(-1, lane_count)
        even = blocks[0::2].sum(dim=0, dtype=torch.int64)
        odd = blocks[1::2].sum(dim=0, dtype=torch.int64)
    else:  # pragma: no cover - lane_count <= numel makes this unreachable
        even = torch.zeros(lane_count, dtype=torch.int64, device=raw.device)
        odd = torch.zeros_like(even)

    tail = raw[main_size:]
    if tail.numel():
        target = even if (main_size // lane_count) % 2 == 0 else odd
        target[: tail.numel()] += tail.to(torch.int64)
    return torch.cat((even, odd))


@torch.no_grad()
def _build_local_report(
    params: dict[str, torch.Tensor],
    transfer_plan: Any,
    *,
    role: Literal["train", "infer"],
) -> dict[str, Any]:
    """Fingerprint this rank's plan slices and return a small CPU report."""
    pending: list[tuple[str, int, torch.Tensor]] = []
    op_names: list[tuple[str, str]] = []
    local_error: str | None = None
    try:
        operations = [
            op
            for peer_rank in sorted(transfer_plan.operations)
            for op in transfer_plan.operations[peer_rank]
        ]
        if role == "infer":
            _validate_infer_coverage(params, operations)
        for op in operations:
            op_names.append(
                (str(op.send_shard_meta.name), str(op.recv_shard_meta.name))
            )
            if role == "train":
                name = op.send_shard_meta.name
                slices = tuple(op.train_slices)
            else:
                name = op.recv_shard_meta.name
                slices = tuple(op.inf_slices)
            tensor = params[name][slices]
            recv_dtype = op.recv_shard_meta.dtype
            if tensor.dtype != recv_dtype:
                tensor = tensor.to(recv_dtype)
            expected_shape = tuple(op.overlap_shape)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{role} slice shape mismatch for {name}: "
                    f"actual={tuple(tensor.shape)} expected={expected_shape}"
                )
            pending.append(
                (
                    _metadata_key(op),
                    int(tensor.numel()),
                    _tensor_fingerprint_lanes(tensor),
                )
            )
            # Non-contiguous receiver slices require a temporary contiguous
            # view. Bound queued CUDA temporaries in this correctness-only
            # path instead of allowing a full-model backlog to accumulate.
            if tensor.is_cuda and len(pending) % _CUDA_SYNC_INTERVAL == 0:
                torch.cuda.current_stream(tensor.device).synchronize()
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    entries: list[tuple[str, str, int]] = []
    if pending and local_error is None:
        lane_matrix = torch.stack([item[2] for item in pending]).cpu()
        for (op_key, numel, _), lanes in zip(pending, lane_matrix, strict=True):
            digest = hashlib.sha256(
                op_key.encode("ascii") + lanes.numpy().tobytes()
            ).hexdigest()[:32]
            entries.append((op_key, digest, numel))
    return {
        "role": role,
        "entries": entries,
        "error": local_error,
        "param_names": sorted(params),
        "op_names": op_names,
    }


def _validate_reports(reports: list[dict[str, Any]]) -> tuple[int, int]:
    """Validate global structure and fingerprints; return op/element counts."""
    errors = [report["error"] for report in reports if report.get("error")]
    if errors:
        raise RuntimeError(
            f"post-apply fingerprint failed on at least one rank: {errors}"
        )

    train_entries = [
        tuple(entry)
        for report in reports
        if report["role"] == "train"
        for entry in report["entries"]
    ]
    infer_entries = [
        tuple(entry)
        for report in reports
        if report["role"] == "infer"
        for entry in report["entries"]
    ]
    if not train_entries or not infer_entries:
        raise RuntimeError(
            "post-apply verification requires non-empty train and infer plans"
        )

    train_param_names = {
        name
        for report in reports
        if report["role"] == "train"
        for name in report["param_names"]
    }
    infer_param_names = {
        name
        for report in reports
        if report["role"] == "infer"
        for name in report["param_names"]
    }
    send_op_names = {
        send_name for report in reports for send_name, _ in report["op_names"]
    }
    recv_op_names = {
        recv_name for report in reports for _, recv_name in report["op_names"]
    }
    missing_train = train_param_names - send_op_names
    extra_train = send_op_names - train_param_names
    missing_infer = {
        name
        for name in infer_param_names - recv_op_names
        if not _allowed_infer_alias(name, infer_param_names, recv_op_names)
    }
    extra_infer = recv_op_names - infer_param_names
    if missing_train or extra_train or missing_infer or extra_infer:
        raise RuntimeError(
            "post-apply parameter coverage mismatch: "
            f"missing_train={sorted(missing_train)[:3]} "
            f"extra_train={sorted(extra_train)[:3]} "
            f"missing_infer={sorted(missing_infer)[:3]} "
            f"extra_infer={sorted(extra_infer)[:3]}"
        )

    train_structure = Counter((op_key, numel) for op_key, _, numel in train_entries)
    infer_structure = Counter((op_key, numel) for op_key, _, numel in infer_entries)
    if train_structure != infer_structure:
        missing = list((train_structure - infer_structure).elements())[:3]
        extra = list((infer_structure - train_structure).elements())[:3]
        raise RuntimeError(
            "post-apply transfer-plan mismatch: "
            f"missing_on_infer={missing} extra_on_infer={extra}"
        )

    train_values = Counter(train_entries)
    infer_values = Counter(infer_entries)
    if train_values != infer_values:
        mismatched = list((train_values - infer_values).elements())[:3]
        raise RuntimeError(
            f"post-apply weight mismatch: mismatched_train_fingerprints={mismatched}"
        )
    return len(train_entries), sum(int(entry[2]) for entry in train_entries)


@torch.no_grad()
def verify_separation_post_apply(
    params: dict[str, torch.Tensor],
    transfer_plan: Any,
    process_group: Any,
    *,
    role: Literal["train", "infer"],
    version: int,
    mode: Literal["full", "delta"],
) -> None:
    """Fail all transfer ranks if receiver weights differ after an update."""
    local_report = _build_local_report(params, transfer_plan, role=role)
    reports: list[dict[str, Any] | None] = [
        None for _ in range(dist.get_world_size(process_group))
    ]
    dist.all_gather_object(reports, local_report, group=process_group)
    complete_reports = [report for report in reports if report is not None]
    op_count, element_count = _validate_reports(complete_reports)
    logger.info(
        "separation post-apply verify v%d OK: mode=%s ops=%d elements=%d "
        "digest_bits=128",
        version,
        mode,
        op_count,
        element_count,
    )
