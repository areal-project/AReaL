# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

import torch


class OptimizerShardRef(NamedTuple):
    name: str
    tensor: torch.Tensor


class StepDirtyShardSnapshot(NamedTuple):
    name: str
    tensor: torch.Tensor
    before_bf16: torch.Tensor
    numel: int
    snapshot_bytes: int


class StepDirtyCapture(NamedTuple):
    shards: list[StepDirtyShardSnapshot]
    total_params: int
    captured_params: int
    skipped_by_cap: int
    total_snapshot_bytes: int
    captured_snapshot_bytes: int
    capture_ms: float


class StepDirtyCompareResult(NamedTuple):
    captured_params: int
    captured_elements: int
    changed_elements: int
    changed_ratio: float
    snapshot_bytes: int
    bitset_bytes: int
    indices_elements: int
    indices_bytes: int
    compare_ms: float
    pack_ms: float
    indices_ms: float


class CopybackDirtyBitsetResult(NamedTuple):
    records: list[dict[str, Any]]
    complete: bool
    collect_ms: float


NameResolver = Callable[[torch.Tensor], str | None]


def iter_optimizer_shard_refs(
    inner_optimizers: Iterable[object],
    *,
    name_resolver: NameResolver | None = None,
) -> list[OptimizerShardRef]:
    """Return Megatron distributed-optimizer main shard tensor refs.

    The function intentionally depends only on Megatron optimizer attributes used
    elsewhere in the AWEX inversion detector. It is safe for unit tests with fake
    optimizers and for dry-run profiling because it only returns tensor refs.
    """
    refs: list[OptimizerShardRef] = []
    seen: set[int] = set()
    for opt_idx, opt in enumerate(inner_optimizers):
        fp32_groups = getattr(opt, "shard_fp32_from_float16_groups", None)
        model_groups = getattr(opt, "model_float16_groups", None)
        if fp32_groups is None or model_groups is None:
            continue
        for group_idx, (fp32_group, model_group) in enumerate(
            zip(fp32_groups, model_groups)
        ):
            for param_idx, (main_shard, model_param) in enumerate(
                zip(fp32_group, model_group)
            ):
                if main_shard is None or not isinstance(main_shard, torch.Tensor):
                    continue
                if main_shard.numel() == 0:
                    continue
                ptr = main_shard.data_ptr()
                if ptr in seen:
                    continue
                seen.add(ptr)
                name = name_resolver(model_param) if name_resolver is not None else None
                if name is None:
                    name = f"optimizer{opt_idx}.group{group_idx}.param{param_idx}"
                refs.append(OptimizerShardRef(name=name, tensor=main_shard))
    return refs


def capture_bf16_optimizer_shards(
    shard_refs: Iterable[OptimizerShardRef],
    *,
    max_snapshot_bytes: int = 0,
    storage: str = "cpu",
    sync_cuda: bool = False,
) -> StepDirtyCapture:
    """Snapshot optimizer main shards as bf16 for dry-run dirty detection.

    Args:
        shard_refs: optimizer main shard tensor refs.
        max_snapshot_bytes: byte cap for captured bf16 snapshots. ``0`` means no
            cap. Positive values keep profiling safe on large models.
        storage: ``"cpu"`` or ``"gpu"``. CPU is safer for colocate memory;
            GPU is closer to a kernel-level implementation but can OOM.
        sync_cuda: synchronize around timing to get blocking GPU cost.
    """
    if storage not in {"cpu", "gpu"}:
        raise ValueError(f"storage must be 'cpu' or 'gpu', got {storage!r}")
    if max_snapshot_bytes < 0:
        raise ValueError("max_snapshot_bytes must be non-negative")

    refs = list(shard_refs)
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    snapshots: list[StepDirtyShardSnapshot] = []
    total_snapshot_bytes = 0
    captured_snapshot_bytes = 0
    skipped_by_cap = 0
    for ref in refs:
        snapshot_bytes = (
            ref.tensor.numel() * torch.tensor([], dtype=torch.bfloat16).element_size()
        )
        total_snapshot_bytes += snapshot_bytes
        if (
            max_snapshot_bytes > 0
            and captured_snapshot_bytes + snapshot_bytes > max_snapshot_bytes
        ):
            skipped_by_cap += 1
            continue
        device = ref.tensor.device if storage == "gpu" else torch.device("cpu")
        before = (
            ref.tensor.detach().to(device=device, dtype=torch.bfloat16).contiguous()
        )
        if before.data_ptr() == ref.tensor.data_ptr():
            before = before.clone()
        snapshots.append(
            StepDirtyShardSnapshot(
                name=ref.name,
                tensor=ref.tensor,
                before_bf16=before,
                numel=ref.tensor.numel(),
                snapshot_bytes=snapshot_bytes,
            )
        )
        captured_snapshot_bytes += snapshot_bytes
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    capture_ms = (time.perf_counter() - start) * 1000
    return StepDirtyCapture(
        shards=snapshots,
        total_params=len(refs),
        captured_params=len(snapshots),
        skipped_by_cap=skipped_by_cap,
        total_snapshot_bytes=total_snapshot_bytes,
        captured_snapshot_bytes=captured_snapshot_bytes,
        capture_ms=capture_ms,
    )


def compare_bf16_optimizer_shards(
    capture: StepDirtyCapture,
    *,
    pack_bitset: bool = False,
    materialize_indices: bool = False,
    sync_cuda: bool = False,
) -> StepDirtyCompareResult:
    """Compare current optimizer shards with a prior bf16 snapshot.

    Equality is bitwise, not numerical: bf16 tensors are viewed as int16 before
    comparing so cases like ``+0.0`` vs ``-0.0`` keep the payload oracle's
    semantics.
    """
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    pack_ms = 0.0
    indices_ms = 0.0
    captured_elements = 0
    changed_elements = 0
    bitset_bytes = 0
    indices_elements = 0
    indices_bytes = 0
    pack_bool_mask_to_uint8 = None
    packed_bool_mask_to_indices = None
    if pack_bitset or materialize_indices:
        from dte.core import pack_bool_mask_to_uint8
    if materialize_indices:
        from dte.core import packed_bool_mask_to_indices

    for shard in capture.shards:
        before = shard.before_bf16.reshape(-1)
        after = (
            shard.tensor.detach()
            .to(device=before.device, dtype=torch.bfloat16)
            .contiguous()
            .reshape(-1)
        )
        changed = before.view(torch.int16) != after.view(torch.int16)
        captured_elements += shard.numel
        changed_elements += int(changed.sum().item())
        if pack_bitset or materialize_indices:
            pack_start = time.perf_counter()
            packed = pack_bool_mask_to_uint8(changed)
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            pack_ms += (time.perf_counter() - pack_start) * 1000
            bitset_bytes += packed.numel()
            if materialize_indices:
                indices_start = time.perf_counter()
                indices = packed_bool_mask_to_indices(
                    packed,
                    shard.numel,
                    dtype=torch.int32,
                )
                if sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                indices_ms += (time.perf_counter() - indices_start) * 1000
                indices_elements += indices.numel()
                indices_bytes += indices.numel() * indices.element_size()
        else:
            bitset_bytes += (shard.numel + 7) // 8

    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    compare_ms = (time.perf_counter() - start) * 1000
    return StepDirtyCompareResult(
        captured_params=capture.captured_params,
        captured_elements=captured_elements,
        changed_elements=changed_elements,
        changed_ratio=changed_elements / max(captured_elements, 1),
        snapshot_bytes=capture.captured_snapshot_bytes,
        bitset_bytes=bitset_bytes,
        indices_elements=indices_elements,
        indices_bytes=indices_bytes,
        compare_ms=compare_ms,
        pack_ms=pack_ms,
        indices_ms=indices_ms,
    )


def collect_copyback_dirty_bitsets(
    optimizer: object,
    *,
    name_resolver: NameResolver | None = None,
) -> CopybackDirtyBitsetResult:
    """Collect dirty bitsets before Megatron copies main shards to model params.

    This is an experimental B1.5/B2 bridge: Megatron's distributed optimizer
    copy-back path still has the old model-visible payload slice resident in
    the grad buffer and the updated fp32 main shard in hand. Comparing them
    here can produce optimizer-shard dirty bitsets without AdamW inversion and
    without a CPU snapshot. The caller must invoke this immediately before the
    optimizer's original ``_copy_main_params_to_model_params`` method.
    """
    from dte.core import bitwise_changed_mask, pack_bool_mask_to_uint8

    start = time.perf_counter()
    records: list[dict[str, Any]] = []
    complete = True

    def collect_group(shard_main_groups, model_groups) -> None:
        nonlocal complete
        for shard_main_group, model_group in zip(shard_main_groups, model_groups):
            for shard_main_param, model_param in zip(shard_main_group, model_group):
                if shard_main_param is None or not isinstance(
                    shard_main_param, torch.Tensor
                ):
                    complete = False
                    continue
                name = name_resolver(model_param) if name_resolver is not None else None
                if name is None:
                    complete = False
                    continue

                try:
                    param_range_map = optimizer._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    world_range = param_range_map["gbuf_world_in_bucket"]
                    gbuf_index, _, bucket_id = optimizer.model_param_gbuf_map[
                        model_param
                    ]
                    model_param_buffer = (
                        optimizer.buffers[gbuf_index].buckets[bucket_id].param_data
                    )
                except Exception:
                    complete = False
                    continue

                old_payload = model_param_buffer.view(-1)[
                    world_range.start : world_range.end
                ].detach()
                new_payload = (
                    shard_main_param.detach()
                    .to(device=old_payload.device, dtype=old_payload.dtype)
                    .reshape(-1)
                )
                if old_payload.numel() != new_payload.numel():
                    complete = False
                    continue

                changed = bitwise_changed_mask(
                    new_payload,
                    old_payload.reshape(-1),
                )
                packed = pack_bool_mask_to_uint8(changed).contiguous()
                records.append(
                    {
                        "name": name,
                        "packed_bitset": packed,
                        "shape": tuple(model_param.shape),
                        "shard_start": int(param_range.start),
                        "shard_numel": int(param_range.size),
                    }
                )

    fp32_from_float16_groups = getattr(
        optimizer,
        "shard_fp32_from_float16_groups",
        None,
    )
    model_float16_groups = getattr(optimizer, "model_float16_groups", None)
    if fp32_from_float16_groups is None or model_float16_groups is None:
        complete = False
    else:
        collect_group(fp32_from_float16_groups, model_float16_groups)

    shard_fp32_groups = getattr(optimizer, "shard_fp32_groups", None)
    model_fp32_groups = getattr(optimizer, "model_fp32_groups", None)
    if shard_fp32_groups is not None and model_fp32_groups is not None:
        collect_group(shard_fp32_groups, model_fp32_groups)

    return CopybackDirtyBitsetResult(
        records=records,
        complete=complete,
        collect_ms=(time.perf_counter() - start) * 1000,
    )
