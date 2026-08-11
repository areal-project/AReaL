# SPDX-License-Identifier: Apache-2.0

"""Environment-gated finite checks for distributed model weights."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

WEIGHT_FINITE_CHECK_ENV = "AREAL_CHECK_WEIGHTS_FINITE"
WEIGHT_FINITE_CHUNK_NUMEL_ENV = "AREAL_WEIGHT_FINITE_CHUNK_NUMEL"
DEFAULT_WEIGHT_FINITE_CHUNK_NUMEL = 16 * 1024 * 1024
_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "y", "on"})


@dataclass(frozen=True)
class WeightFiniteReport:
    """Summary of one successful local finite-weight scan."""

    stage: str
    version: int | None
    tensor_count: int
    numel: int


def weight_finite_check_enabled() -> bool:
    """Return whether the finite-weight diagnostic is enabled."""

    return (
        os.environ.get(WEIGHT_FINITE_CHECK_ENV, "").strip().lower() in _TRUE_ENV_VALUES
    )


def iter_module_named_tensors(
    modules: nn.Module | Sequence[nn.Module],
    *,
    include_parameters: bool = True,
    include_buffers: bool = True,
    extra_tensor_attrs: Sequence[str] = (),
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield registered weights and selected derived tensor attributes."""

    module_list = [modules] if isinstance(modules, nn.Module) else list(modules)
    multi_module = len(module_list) > 1
    for index, module in enumerate(module_list):
        prefix = f"model[{index}]." if multi_module else ""
        if include_parameters:
            for name, parameter in module.named_parameters():
                yield f"{prefix}{name}", parameter
        if include_buffers:
            for name, buffer in module.named_buffers():
                yield f"{prefix}{name}", buffer
        if extra_tensor_attrs:
            for module_name, submodule in module.named_modules():
                submodule_prefix = f"{module_name}." if module_name else ""
                for attr_name in extra_tensor_attrs:
                    value = getattr(submodule, attr_name, None)
                    if torch.is_tensor(value):
                        yield f"{prefix}{submodule_prefix}{attr_name}", value


def _chunk_numel(logger: Any) -> int:
    value = os.environ.get(WEIGHT_FINITE_CHUNK_NUMEL_ENV, "").strip()
    if not value:
        return DEFAULT_WEIGHT_FINITE_CHUNK_NUMEL
    try:
        chunk_numel = int(value)
    except ValueError:
        chunk_numel = 0
    if chunk_numel <= 0:
        logger.warning(
            "Invalid %s=%r; using %d",
            WEIGHT_FINITE_CHUNK_NUMEL_ENV,
            value,
            DEFAULT_WEIGHT_FINITE_CHUNK_NUMEL,
        )
        return DEFAULT_WEIGHT_FINITE_CHUNK_NUMEL
    return chunk_numel


def _iter_tensor_chunks(tensor: torch.Tensor, max_numel: int) -> Iterator[torch.Tensor]:
    """Yield no-copy tensor views bounded by ``max_numel`` elements."""

    pending = [tensor]
    while pending:
        current = pending.pop()
        if current.numel() <= max_numel:
            yield current
            continue

        split_dim = max(range(current.ndim), key=current.size)
        split_size = current.size(split_dim)
        elements_per_index = current.numel() // split_size
        indices_per_chunk = max(1, max_numel // elements_per_index)
        for start in range(split_size, 0, -indices_per_chunk):
            chunk_start = max(0, start - indices_per_chunk)
            pending.append(current.narrow(split_dim, chunk_start, start - chunk_start))


def _tensor_is_finite(tensor: torch.Tensor, chunk_numel: int) -> torch.Tensor:
    finite = torch.ones((), dtype=torch.bool, device=tensor.device)
    for chunk in _iter_tensor_chunks(tensor.detach(), chunk_numel):
        finite.logical_and_(torch.isfinite(chunk).all())
    return finite


def _nonfinite_counts(tensor: torch.Tensor, chunk_numel: int) -> tuple[int, int]:
    nan_count = torch.zeros((), dtype=torch.int64, device=tensor.device)
    inf_count = torch.zeros((), dtype=torch.int64, device=tensor.device)
    for chunk in _iter_tensor_chunks(tensor.detach(), chunk_numel):
        nan_count.add_(torch.isnan(chunk).sum())
        inf_count.add_(torch.isinf(chunk).sum())
    counts = torch.stack((nan_count, inf_count)).cpu().tolist()
    return int(counts[0]), int(counts[1])


def _collective_any(
    local_bad: bool,
    *,
    process_group: dist.ProcessGroup | None,
    cuda_device: torch.device | None,
) -> bool:
    if process_group is None or not dist.is_initialized():
        return local_bad

    backend = str(dist.get_backend(process_group)).lower()
    if "nccl" in backend:
        if cuda_device is None:
            cuda_device = torch.device("cuda", torch.cuda.current_device())
        device = cuda_device
    else:
        device = torch.device("cpu")
    bad = torch.tensor(int(local_bad), dtype=torch.int32, device=device)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=process_group)
    return bool(bad.item())


@torch.no_grad()
def check_named_tensors_finite(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    stage: str,
    logger: Any,
    version: int | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> WeightFiniteReport | None:
    """Check floating tensors and fail all participating ranks on non-finite data.

    The environment gate is checked before consuming ``named_tensors``. Each tensor
    is scanned through bounded views so a large model shard does not allocate a
    model-sized boolean temporary.
    """

    if not weight_finite_check_enabled():
        return None

    chunk_numel = _chunk_numel(logger)
    checked: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    meta_tensors: list[tuple[str, torch.Tensor]] = []
    tensor_count = 0
    total_numel = 0
    cuda_device: torch.device | None = None
    for name, tensor in named_tensors:
        if not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        if tensor.device.type == "meta":
            meta_tensors.append((name, tensor))
            continue
        if tensor.numel() == 0:
            continue
        finite = _tensor_is_finite(tensor, chunk_numel)
        checked.append((name, tensor, finite))
        tensor_count += 1
        total_numel += tensor.numel()
        if tensor.device.type == "cuda":
            cuda_device = tensor.device

    flags_by_device: dict[torch.device, list[tuple[int, torch.Tensor]]] = {}
    for index, (_, tensor, finite) in enumerate(checked):
        flags_by_device.setdefault(tensor.device, []).append((index, finite))

    local_flags = [True] * len(checked)
    for device_flags in flags_by_device.values():
        indices, flags = zip(*device_flags, strict=True)
        values = torch.stack(flags).cpu().tolist()
        for index, value in zip(indices, values, strict=True):
            local_flags[index] = bool(value)

    bad_indices = [index for index, finite in enumerate(local_flags) if not finite]
    local_bad = bool(bad_indices or meta_tensors)
    global_bad = _collective_any(
        local_bad,
        process_group=process_group,
        cuda_device=cuda_device,
    )

    rank = dist.get_rank() if dist.is_initialized() else 0
    context = f"stage={stage} version={version} rank={rank}"
    if global_bad:
        if local_bad:
            details = [
                f"{name}:shape={tuple(tensor.shape)},dtype={tensor.dtype},device=meta"
                for name, tensor in meta_tensors[:20]
            ]
            for index in bad_indices[:20]:
                if len(details) >= 20:
                    break
                name, tensor, _ = checked[index]
                nan_count, inf_count = _nonfinite_counts(tensor, chunk_numel)
                details.append(
                    f"{name}:shape={tuple(tensor.shape)},dtype={tensor.dtype},"
                    f"nan={nan_count},inf={inf_count}"
                )
            omitted = len(meta_tensors) + len(bad_indices) - len(details)
            if omitted > 0:
                details.append(f"... and {omitted} more tensors")
            message = f"Non-finite weights detected: {context}; " + "; ".join(details)
        else:
            message = (
                f"Non-finite weights detected on another distributed rank: {context}"
            )
        logger.error("[WeightFinite] FAIL %s", message)
        raise FloatingPointError(message)

    logger.info(
        "[WeightFinite] PASS %s tensors=%d numel=%d chunk_numel=%d",
        context,
        tensor_count,
        total_numel,
        chunk_numel,
    )
    return WeightFiniteReport(
        stage=stage,
        version=version,
        tensor_count=tensor_count,
        numel=total_numel,
    )
