# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pybase64
import torch

from areal.engine.r3.asserts import r3_error


def decode_sglang_routed_experts(
    raw_routed_experts: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    source: str,
) -> np.ndarray:
    """Decode SGLang base64 int32 routed_experts into ``[tokens, flat_dim]``."""

    num_sgl_tokens = int(prompt_tokens) + int(completion_tokens) - 1
    if num_sgl_tokens < 0:
        raise r3_error(
            "Invalid SGLang token counts for routed_experts",
            source=source,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    try:
        decoded = pybase64.b64decode(raw_routed_experts.encode("utf-8"))
    except Exception as exc:
        raise r3_error(
            "Failed to base64 decode SGLang routed_experts",
            source=source,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ) from exc

    if len(decoded) % np.dtype(np.int32).itemsize != 0:
        raise r3_error(
            "Decoded SGLang routed_experts byte length is not int32-aligned",
            source=source,
            byte_len=len(decoded),
        )
    flat = np.frombuffer(decoded, dtype=np.int32)
    if num_sgl_tokens == 0:
        if flat.size != 0:
            raise r3_error(
                "SGLang routed_experts returned data for an empty token range",
                source=source,
                flat_size=flat.size,
            )
        return flat.reshape(0, 0)
    if flat.size % num_sgl_tokens != 0:
        raise r3_error(
            "SGLang routed_experts cannot be reshaped by token count",
            source=source,
            flat_size=flat.size,
            num_sgl_tokens=num_sgl_tokens,
        )
    return flat.reshape(num_sgl_tokens, -1)


def _reshape_routed_experts(
    routed_experts: np.ndarray,
    *,
    num_moe_layers: int,
    topk: int,
) -> np.ndarray:
    arr = np.asarray(routed_experts, dtype=np.int32)
    if arr.ndim == 2:
        expected_flat_dim = num_moe_layers * topk
        if arr.shape[1] != expected_flat_dim:
            raise r3_error(
                "routed_experts flat_dim does not match Megatron MoE config",
                raw_shape=arr.shape,
                flat_dim=arr.shape[1],
                num_moe_layers=num_moe_layers,
                topk=topk,
                expected_flat_dim=expected_flat_dim,
            )
        return arr.reshape(arr.shape[0], num_moe_layers, topk)
    if arr.ndim == 3:
        if arr.shape[1:] != (num_moe_layers, topk):
            raise r3_error(
                "routed_experts layer/topk dims do not match Megatron MoE config",
                raw_shape=arr.shape,
                num_moe_layers=num_moe_layers,
                topk=topk,
            )
        return arr
    raise r3_error("Unsupported routed_experts ndim", raw_shape=arr.shape)


def _nearest_valid_row(rows: np.ndarray, before: int) -> np.ndarray | None:
    for idx in range(before - 1, -1, -1):
        row = rows[idx]
        if not np.all(row == 0):
            return row
    return None


def _preprocess_one(
    routed_experts: np.ndarray | None,
    *,
    seq_len: int,
    num_moe_layers: int,
    topk: int,
) -> tuple[np.ndarray, bool]:
    out = np.zeros((seq_len, num_moe_layers, topk), dtype=np.int32)
    if routed_experts is None:
        return out, False

    arr = _reshape_routed_experts(
        routed_experts,
        num_moe_layers=num_moe_layers,
        topk=topk,
    )
    if arr.shape[0] > seq_len:
        raise r3_error(
            "routed_experts has more token rows than the training sequence",
            raw_shape=arr.shape,
            seq_len=seq_len,
        )

    copied = arr.shape[0]
    if copied > 0:
        out[:copied] = arr

    valid = copied >= max(seq_len - 1, 0)
    for pos in range(copied, seq_len):
        replacement = _nearest_valid_row(out, pos)
        if replacement is not None:
            out[pos] = replacement

    real_rows_to_check = max(seq_len - 1, 0)
    if real_rows_to_check > 0:
        zero_rows = np.all(out[:real_rows_to_check] == 0, axis=(1, 2))
        if bool(np.any(zero_rows)):
            valid = False
            for pos in np.nonzero(zero_rows)[0]:
                replacement = _nearest_valid_row(out, int(pos))
                if replacement is not None:
                    out[pos] = replacement

    if seq_len > 0 and np.all(out[-1] == 0):
        replacement = _nearest_valid_row(out, seq_len - 1)
        if replacement is not None:
            out[-1] = replacement

    return out, valid


def preprocess_routed_experts_batch(
    routed_experts: Sequence[np.ndarray | None],
    *,
    seq_lens: Sequence[int],
    num_moe_layers: int,
    topk: int,
    max_seq_len: int | None = None,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    if len(routed_experts) != len(seq_lens):
        raise r3_error(
            "routed_experts batch size does not match seq_lens",
            routed_batch_size=len(routed_experts),
            seq_lens_size=len(seq_lens),
        )
    if max_seq_len is None:
        max_seq_len = max(seq_lens, default=0)

    batch = np.zeros(
        (len(routed_experts), max_seq_len, num_moe_layers, topk),
        dtype=np.int32,
    )
    valid: list[bool] = []
    for idx, (sample_routing, seq_len) in enumerate(zip(routed_experts, seq_lens)):
        if seq_len > max_seq_len:
            raise r3_error(
                "seq_len exceeds max_seq_len",
                sample_idx=idx,
                seq_len=seq_len,
                max_seq_len=max_seq_len,
            )
        sample, sample_valid = _preprocess_one(
            sample_routing,
            seq_len=int(seq_len),
            num_moe_layers=num_moe_layers,
            topk=topk,
        )
        batch[idx, : int(seq_len)] = sample
        valid.append(sample_valid)

    return {
        "routed_experts": torch.as_tensor(batch, dtype=torch.int32, device=device),
        "r3_routing_valid": torch.as_tensor(valid, dtype=torch.bool, device=device),
    }
