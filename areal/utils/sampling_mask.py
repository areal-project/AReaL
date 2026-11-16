# SPDX-License-Identifier: Apache-2.0
"""Frozen sampling supports, uint24 wire format, and restricted policy math.

Offsets count IDs (not bytes). Rows correspond to *sampled* tokens; callers must
apply the same next-token shift as behavior logprobs before training. This module
does not reconstruct supports from updated logits.
"""

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist

PROTOCOL_VERSION = 1
UINT24_LIMIT = 1 << 24


def pack_uint24(ids: torch.Tensor) -> torch.Tensor:
    """Pack integer IDs into three little-endian bytes, without a device sync."""
    if ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("Sampling mask IDs must have int32 or int64 dtype")
    torch._assert_async(
        ((ids >= 0) & (ids < UINT24_LIMIT)).all(),
        "Sampling mask ID is outside the uint24 range",
    )
    return torch.stack((ids, ids >> 8, ids >> 16), dim=-1).to(torch.uint8)


def unpack_uint24(packed: torch.Tensor) -> torch.Tensor:
    """Decode little-endian uint24 IDs to int32 on the input device."""
    if packed.dtype != torch.uint8 or packed.ndim < 1 or packed.shape[-1] != 3:
        raise ValueError("Packed sampling mask IDs must have uint8 shape [..., 3]")
    parts = packed.to(torch.int32)
    return parts[..., 0] | (parts[..., 1] << 8) | (parts[..., 2] << 16)


@dataclass(frozen=True)
class SamplingMask:
    """CPU-owned CSR support for one immutable rollout fragment.

    A fragment contains only effective generated positions, including EOS.
    Tensor ownership is transferred to this object; consumers must not mutate it.
    Validation deliberately happens at the CPU transport boundary, not per epoch.
    """

    packed_ids: torch.Tensor
    offsets: torch.Tensor
    vocab_size: int
    weight_version: int
    protocol_version: int = PROTOCOL_VERSION

    @property
    def num_rows(self) -> int:
        return self.offsets.numel() - 1

    def validate(self, tokens: torch.Tensor, *, weight_version: int) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Sampling mask protocol version mismatch")
        if self.weight_version != weight_version:
            raise ValueError("Sampling mask model weight version mismatch")
        if not 0 < self.vocab_size <= UINT24_LIMIT:
            raise ValueError("Sampling mask vocabulary does not fit uint24")
        if self.packed_ids.device.type != "cpu" or self.offsets.device.type != "cpu":
            raise ValueError("Sampling mask validation requires CPU transport tensors")
        if self.offsets.dtype != torch.int64 or self.offsets.ndim != 1:
            raise ValueError("Sampling mask offsets must be a vector of int64")
        if self.packed_ids.ndim != 2:
            raise ValueError("Sampling mask packed IDs must have shape [candidates, 3]")
        ids = unpack_uint24(self.packed_ids)
        if self.offsets.numel() == 0 or self.offsets[0] != 0:
            raise ValueError("Sampling mask offsets must start at zero")
        if self.offsets[-1] != ids.numel():
            raise ValueError("Sampling mask final offset differs from candidate count")
        counts = self.offsets.diff()
        if (counts <= 0).any():
            raise ValueError("Empty or invalid sampling mask row")
        if tokens.device.type != "cpu" or tokens.shape != (self.num_rows,):
            raise ValueError("Sampling mask rows and generated tokens are misaligned")
        if (ids >= self.vocab_size).any():
            raise ValueError("Sampling mask ID exceeds the model vocabulary")
        rows = torch.repeat_interleave(torch.arange(self.num_rows), counts)
        keys = rows * self.vocab_size + ids.long()
        if torch.unique(keys).numel() != keys.numel():
            raise ValueError("Duplicate ID in sampling mask row")
        matches = torch.zeros(self.num_rows, dtype=torch.int64)
        matches.scatter_add_(0, rows, (ids == tokens[rows]).long())
        if (matches != 1).any():
            raise ValueError("Generated token is missing from its sampling mask")

    def select_rows(self, rows: torch.Tensor) -> "SamplingMask":
        """Select/reorder CPU rows without padding to a maximum candidate width."""
        if rows.device.type != "cpu" or rows.dtype != torch.int64 or rows.ndim != 1:
            raise ValueError("Sampling mask row selection requires a CPU int64 vector")
        if ((rows < 0) | (rows >= self.num_rows)).any():
            raise ValueError("Sampling mask row index out of bounds")
        counts = self.offsets.diff()[rows]
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        starts = torch.repeat_interleave(self.offsets[rows] - offsets[:-1], counts)
        indices = starts + torch.arange(int(offsets[-1]))
        return SamplingMask(
            self.packed_ids[indices],
            offsets,
            self.vocab_size,
            self.weight_version,
            self.protocol_version,
        )

    def statistics(self) -> dict[str, int | float]:
        """Payload accounting excludes framing; transports count network bytes."""
        counts = self.offsets.diff().float()
        quantiles = (
            torch.quantile(counts, torch.tensor([0.95, 0.99]))
            if counts.numel()
            else torch.zeros(2)
        )
        return {
            "generated_tokens": self.num_rows,
            "candidates_mean": float(counts.mean()) if counts.numel() else 0.0,
            "candidates_p95": float(quantiles[0]),
            "candidates_p99": float(quantiles[1]),
            "candidates_max": int(counts.max()) if counts.numel() else 0,
            "id_bytes": self.packed_ids.numel(),
            "offset_bytes": self.offsets.numel() * 8,
        }


def restricted_policy_terms(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ids: torch.Tensor,
    offsets: torch.Tensor,
    *,
    temperature: float,
    vocab_size: int,
    tp_group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable CSR softmax with no dense vocabulary mask.

    Rows and supports are replicated in the explicit tensor-parallel group.
    Every TP rank evaluates the same loss, so collective SUM uses an identity
    local backward, matching Megatron's vocabulary-parallel loss convention.
    Empty rows are permitted only for non-supervised positions; they return zero.
    Callers must validate that every supervised row is nonempty at ingestion.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Sampling mask temperature must be finite and positive")
    if logits.ndim != 2 or labels.shape != logits.shape[:1]:
        raise ValueError("Expected logits [tokens, local_vocab] and labels [tokens]")
    if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Sampling mask candidate IDs must be an integer vector")
    if offsets.dtype != torch.int64 or offsets.shape != (logits.shape[0] + 1,):
        raise ValueError("Sampling mask offsets do not align with logits")
    if not (ids.device == offsets.device == labels.device == logits.device):
        raise ValueError("Sampling mask tensors and logits must share a device")
    world = dist.get_world_size(tp_group) if tp_group is not None else 1
    rank = dist.get_rank(tp_group) if tp_group is not None else 0
    width = logits.shape[1]
    if not 0 < vocab_size <= world * width:
        raise ValueError("Sampling mask vocabulary does not fit the TP shards")
    counts = offsets.diff()
    torch._assert_async(
        (offsets[0] == 0) & (offsets[-1] == ids.numel()) & (counts >= 0).all(),
        "Malformed sampling mask CSR offsets",
    )
    torch._assert_async(
        ((ids >= 0) & (ids < vocab_size)).all(), "Sampling mask ID out of vocabulary"
    )
    rows = torch.repeat_interleave(
        torch.arange(logits.shape[0], device=logits.device),
        counts,
        output_size=ids.numel(),
    )
    local_ids = ids.long() - rank * width
    owned = (local_ids >= 0) & (local_ids < width)
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    selected = logits[rows, local_ids.clamp(0, width - 1)].to(dtype) / temperature
    selected = selected.masked_fill(~owned, -torch.inf)
    maximum = torch.full(
        (logits.shape[0],), -torch.inf, dtype=dtype, device=logits.device
    )
    maximum.scatter_reduce_(
        0, rows, selected.detach(), reduce="amax", include_self=True
    )
    if world > 1:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=tp_group)
    maximum = torch.where(counts > 0, maximum, 0)

    def sum_across_tp(value: torch.Tensor) -> torch.Tensor:
        if world == 1:
            return value
        total = value.detach().clone()
        dist.all_reduce(total, group=tp_group)
        return total + (value - value.detach())

    shifted = selected - maximum[rows]
    weights = shifted.exp()
    denominator = sum_across_tp(torch.zeros_like(maximum).scatter_add(0, rows, weights))
    log_denominator = torch.where(counts > 0, denominator, 1).log()
    target = (ids == labels[rows]) & owned
    target_logits = sum_across_tp(
        torch.zeros_like(maximum).scatter_add(0, rows, shifted.masked_fill(~target, 0))
    )
    logprobs = target_logits - log_denominator
    # Entropy of the same restricted policy, with a valid differentiable path.
    weighted_logits = torch.zeros_like(maximum).scatter_add(
        0,
        rows,
        weights * shifted.masked_fill(~owned, 0),
    )
    entropy = log_denominator - sum_across_tp(weighted_logits) / torch.where(
        counts > 0, denominator, 1
    )
    return logprobs, entropy, maximum, denominator


def restricted_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ids: torch.Tensor,
    offsets: torch.Tensor,
    *,
    temperature: float,
    vocab_size: int,
    tp_group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the frozen-support policy with differentiable normalization."""
    logprobs, entropy, _, _ = restricted_policy_terms(
        logits,
        labels,
        ids,
        offsets,
        temperature=temperature,
        vocab_size=vocab_size,
        tp_group=tp_group,
    )
    return logprobs, entropy


def restricted_logits_backward_(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ids: torch.Tensor,
    offsets: torch.Tensor,
    maximum: torch.Tensor,
    denominator: torch.Tensor,
    grad: torch.Tensor | None,
    *,
    temperature: float,
    tp_rank: int,
    logit_scale: float = 1.0,
) -> torch.Tensor:
    """Overwrite a recomputed chunk with its exact sparse logprob VJP.

    ``maximum`` and ``denominator`` are global TP statistics from the forward.
    No collective or full-vocabulary mask is needed during this backward.
    """
    rows = torch.repeat_interleave(
        torch.arange(logits.shape[0], device=logits.device),
        offsets.diff(),
        output_size=ids.numel(),
    )
    local = ids.long() - tp_rank * logits.shape[-1]
    owned = (local >= 0) & (local < logits.shape[-1])
    columns = local.clamp(0, logits.shape[-1] - 1)
    values = logits[rows, columns].float() / temperature
    values = values.masked_fill(~owned, -torch.inf)
    probabilities = (values - maximum[rows]).exp() / denominator[rows]
    weights = (ids == labels[rows]).to(probabilities.dtype) - probabilities
    weights *= (grad[rows] if grad is not None else 0) * (logit_scale / temperature)
    weights.masked_fill_(~owned, 0)
    logits.zero_()
    logits.index_put_((rows, columns), weights.to(logits.dtype), accumulate=True)
    return logits


def estimate_mask_memory(
    *,
    max_tokens: int,
    top_k: int,
    vocab_size: int,
    data_parallel_size: int,
    chunk_tokens: int,
    buffer_bytes: int,
    budget_bytes: int = 30_000_000_000,
) -> dict[str, int]:
    """Conservative preflight bound, including packed host and unpacked GPU data.

    There is no implicit top-k cap. Without one, use the whole vocabulary bound.
    DP shards are estimated with ceiling division; callers must separately budget
    any rollout-owner concentration and trainer redistribution staging buffers.
    """
    if (
        min(
            max_tokens,
            vocab_size,
            data_parallel_size,
            chunk_tokens,
            buffer_bytes,
            budget_bytes,
        )
        <= 0
    ):
        raise ValueError("Sampling mask memory estimate requires positive capacities")
    if vocab_size > UINT24_LIMIT:
        raise ValueError("Sampling mask vocabulary exceeds uint24 capacity")
    candidates = min(top_k, vocab_size) if top_k > 0 else vocab_size
    tokens = (max_tokens + data_parallel_size - 1) // data_parallel_size
    # Per-row offsets, token/version metadata and numeric cache locators; count
    # the sparse index against the global token bound even on a DP shard.
    host_bytes = tokens * (candidates * 3 + 64) + max_tokens * 8 + 8
    host_bytes += 2 * min(tokens, chunk_tokens) * candidates * 3
    # CSR gather/reordering scratch (int64 indices plus compact copies).
    host_bytes += min(tokens, chunk_tokens) * candidates * 48
    # Double-buffered registered wire buffers + int32 candidates and row IDs.
    device_bytes = 2 * buffer_bytes + min(tokens, chunk_tokens) * candidates * 80
    result = {
        "candidates_bound": candidates,
        "tokens_per_rank": tokens,
        "host_bytes": host_bytes,
        "device_bytes": device_bytes,
        "total_bytes": host_bytes + device_bytes,
    }
    if result["total_bytes"] > budget_bytes:
        raise ValueError(
            f"sampling_mask_transfer preflight exceeds the {budget_bytes} byte budget: "
            f"{result}. Reduce explicit sampling top_k, batch/length, or shard size."
        )
    return result
