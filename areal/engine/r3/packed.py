# SPDX-License-Identifier: Apache-2.0

"""CPU routing storage with cheap row permutations and explicit token offsets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from operator import index
from typing import Any

import torch

from areal.engine.r3.asserts import r3_error

ROUTING_OFFSETS_KEY = "r3_cu_seqlens"
ROUTING_LAYERS_KEY = "r3_layer_indices"


@dataclass(frozen=True)
class PackedRouting:
    """A row-indexed view; selecting rows never copies the routing payload.

    ``row_indices == -1`` represents a transport-only row without allocating
    dummy token routing. Layer IDs refer to global MoE indices, including VPP.
    All metadata and values stay on CPU until a local microbatch is consumed.
    """

    values: torch.Tensor
    cu_seqlens: torch.Tensor
    row_indices: torch.Tensor
    layer_indices: tuple[int, ...]

    @classmethod
    def from_tensors(
        cls,
        values: torch.Tensor,
        cu_seqlens: torch.Tensor,
        layer_indices: Sequence[int] | None = None,
    ) -> PackedRouting:
        if values.ndim != 3 or values.device.type != "cpu":
            raise r3_error("Packed routing must be a CPU [tokens, layers, topk] tensor")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise r3_error("Packed routing offsets must use an integer dtype")
        offsets = cu_seqlens.to(device="cpu", dtype=torch.long)
        if (
            offsets.ndim != 1
            or offsets.numel() == 0
            or int(offsets[0]) != 0
            or int(offsets[-1]) != values.shape[0]
            or bool(torch.any(offsets[1:] < offsets[:-1]))
        ):
            raise r3_error("Invalid packed routing token offsets")
        if values.dtype not in (torch.int16, torch.int32, torch.int64):
            raise r3_error("Packed routing values must use a signed integer dtype")
        try:
            layers = (
                tuple(range(values.shape[1]))
                if layer_indices is None
                else tuple(index(x) for x in layer_indices)
            )
        except TypeError as exc:
            raise r3_error("Packed routing layer IDs must be integers") from exc
        if (
            len(layers) != values.shape[1]
            or len(set(layers)) != len(layers)
            or any(x < 0 for x in layers)
        ):
            raise r3_error("Packed routing layer metadata does not match values")
        return cls(values, offsets, torch.arange(len(offsets) - 1), layers)

    def __len__(self) -> int:
        return self.row_indices.numel()

    @property
    def device(self) -> torch.device:
        return self.values.device

    def select_rows(self, indices: torch.Tensor | Sequence[int]) -> PackedRouting:
        indices = torch.as_tensor(indices, device="cpu", dtype=torch.long)
        if indices.ndim != 1 or bool(torch.any((indices < 0) | (indices >= len(self)))):
            raise r3_error("Packed routing row index is out of range")
        return PackedRouting(
            self.values, self.cu_seqlens, self.row_indices[indices], self.layer_indices
        )

    def append_dummy_rows(self, count: int) -> PackedRouting:
        return PackedRouting(
            self.values,
            self.cu_seqlens,
            torch.cat([self.row_indices, torch.full((count,), -1, dtype=torch.long)]),
            self.layer_indices,
        )

    def select_layers(self, layers: Sequence[int]) -> PackedRouting:
        layers = tuple(layers)
        if layers == self.layer_indices:
            return self
        try:
            indices = [self.layer_indices.index(layer) for layer in layers]
        except ValueError as exc:
            raise r3_error("Requested MoE layer is absent from packed routing") from exc
        # An independent allocation is intentional: a small view must not keep
        # the full global-layer storage alive on a pipeline stage.
        values = self.values.index_select(1, torch.tensor(indices, dtype=torch.long))
        return PackedRouting(values, self.cu_seqlens, self.row_indices, layers)

    def row(self, index: int) -> torch.Tensor:
        source = int(self.row_indices[index])
        if source < 0:
            return self.values[:0]
        start, end = self.cu_seqlens[source : source + 2].tolist()
        return self.values[start:end]


def compact_routing_dtype(values: torch.Tensor) -> torch.Tensor:
    """Use signed int16 when lossless, retaining int32 for larger expert IDs."""
    if values.dtype not in (torch.int16, torch.int32, torch.int64):
        raise r3_error("Expert IDs must use a signed integer dtype")
    if values.numel():
        lo, hi = torch.aminmax(values)
        if int(lo) < 0:
            raise r3_error("Expert IDs must be nonnegative")
        if int(hi) > torch.iinfo(torch.int32).max:
            raise r3_error("Expert ID exceeds int32 capacity")
        dtype = torch.int16 if int(hi) <= torch.iinfo(torch.int16).max else torch.int32
    else:
        dtype = torch.int16
    return values.to(dtype=dtype)


def normalize_legacy_routing_cache(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade cached dense routes without requiring the original response.

    Deserialized interactions can retain only their tensor cache. Preserve
    historical routes and validity instead of rebuilding them from a response.
    """
    values = data.get("routed_experts")
    if ROUTING_OFFSETS_KEY in data or not isinstance(values, torch.Tensor):
        return data
    if values.ndim != 4:
        raise r3_error("Legacy routing cache must have [batch, sequence, layers, topk]")
    if values.dtype not in (torch.int16, torch.int32, torch.int64):
        raise r3_error("Legacy routing cache must use a signed integer dtype")
    batch, sequence, layers, topk = values.shape
    attention = data.get("attention_mask")
    if not isinstance(attention, torch.Tensor) or attention.shape != (batch, sequence):
        raise r3_error("Legacy routing cache requires a matching attention mask")
    attention = attention.detach().to(device="cpu", dtype=torch.bool)
    lengths = attention.sum(-1)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
    dtype = torch.int16
    if values.numel():
        lo, hi = torch.aminmax(values)
        if int(lo) < 0 or int(hi) > torch.iinfo(torch.int32).max:
            raise r3_error("Legacy routing cache contains invalid expert IDs")
        if int(hi) > torch.iinfo(torch.int16).max:
            dtype = torch.int32
    packed = torch.empty((int(offsets[-1]), layers, topk), dtype=dtype, device="cpu")
    for row in range(batch):
        start, end = offsets[row : row + 2].tolist()
        mask = attention[row].to(device=values.device)
        packed[start:end].copy_(values[row, mask])
    return {**data, "routed_experts": packed, ROUTING_OFFSETS_KEY: offsets}
