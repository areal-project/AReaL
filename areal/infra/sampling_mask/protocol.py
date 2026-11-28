# SPDX-License-Identifier: Apache-2.0
"""Small control-plane references; candidate IDs are exclusively GIN payloads."""

import hashlib
import ipaddress
from itertools import chain

import numpy as np
import torch

from areal.utils.sampling_mask import PROTOCOL_VERSION

MAX_FETCH_ROWS = 65536
REFERENCE_KEYS = ("sampling_mask_source", "sampling_mask_nonce", "sampling_mask_row")


def batch_identity(data: dict[str, torch.Tensor]) -> bytes:
    """Identify a rollout shard independently of epoch order and padding."""
    for key in REFERENCE_KEYS:
        if data[key].device.type != "cpu":
            raise ValueError("Sampling-mask batch identity requires CPU staging")
    refs = np.stack([data[key].reshape(-1).numpy() for key in REFERENCE_KEYS], axis=1)
    refs = np.unique(refs[refs[:, 0] != 0], axis=0)
    return hashlib.sha256(memoryview(refs)).digest()


def encode_endpoint(host: str, port: int) -> int:
    if not 0 < port < 65536:
        raise ValueError("Invalid sampling-mask service port")
    return (int(ipaddress.IPv4Address(host)) << 16) | port


def decode_endpoint(value: int) -> str:
    return f"http://{ipaddress.IPv4Address(value >> 16)}:{value & 65535}"


def validate_references(
    references, tokens: list[int], versions: list[int] | None = None
) -> None:
    if references is None or len(references) != len(tokens):
        raise ValueError("Missing or misaligned sampling-mask references")
    for index, (reference, token) in enumerate(zip(references, tokens, strict=True)):
        if not isinstance(reference, list) or len(reference) != 5:
            raise ValueError("Malformed sampling-mask reference")
        source, nonce, row, protocol, version = reference
        if protocol != PROTOCOL_VERSION:
            raise ValueError("Sampling-mask protocol version mismatch")
        if any(
            not isinstance(value, int) or isinstance(value, bool) for value in reference
        ):
            raise ValueError("Sampling-mask references must contain integer metadata")
        if source <= 0 or nonce <= 0 or row <= 0:
            raise ValueError("Invalid sampling-mask source or row identity")
        if versions is not None and version != versions[index]:
            raise ValueError("Sampling-mask weight version mismatch")


def trajectory_references(response) -> dict[str, torch.Tensor]:
    validate_references(
        response.sampling_mask_refs, response.output_tokens, response.output_versions
    )
    length = response.input_len + response.output_len
    values = torch.zeros((length, 3), dtype=torch.int64)
    if response.output_len:
        values[response.input_len :] = torch.tensor(
            [r[:3] for r in response.sampling_mask_refs], dtype=torch.int64
        )
    return {
        key: values[:, index].contiguous() for index, key in enumerate(REFERENCE_KEYS)
    }


def pack_row_ranges(rows: np.ndarray) -> list:
    """Sorted unique handles: one integer per singleton, [first,last] per run."""
    bounds = np.r_[0, np.flatnonzero(np.diff(rows) != 1) + 1, len(rows)]
    return [
        int(rows[a]) if b == a + 1 else [int(rows[a]), int(rows[b - 1])]
        for a, b in zip(bounds[:-1], bounds[1:], strict=True)
        if b > a
    ]


def unpack_row_ranges(ranges: list) -> np.ndarray:
    runs = [
        range(span, span + 1) if isinstance(span, int) else range(span[0], span[1] + 1)
        for span in ranges
    ]
    n = sum(map(len, runs))
    if not 0 < n <= MAX_FETCH_ROWS:
        raise ValueError("Sampling-mask fetch row count exceeds request bounds")
    return np.fromiter(chain.from_iterable(runs), dtype=np.int64, count=n)


def pack_frame(packed, offsets, tokens, versions) -> torch.Tensor:
    counts = offsets.diff().numpy()
    width = np.min_scalar_type(int(counts.max())).itemsize
    # All fields are little-endian; rows follow the fetch request order.
    return torch.cat(
        [
            torch.from_numpy(
                np.array([len(counts), width], dtype="<u4").view(np.uint8)
            ),
            torch.from_numpy(counts.astype(f"<u{width}").view(np.uint8)),
            torch.from_numpy(tokens.numpy().astype("<i4").view(np.uint8)),
            torch.from_numpy(versions.numpy().astype("<i8", copy=False).view(np.uint8)),
            packed.flatten(),
        ]
    )


def unpack_frame(frame: torch.Tensor):
    data = frame.numpy()
    n, width = map(int, np.frombuffer(data, "<u4", 2))
    counts = np.frombuffer(data, f"<u{width}", n, 8).astype(np.int64)
    start = 8 + width * n
    tokens = np.frombuffer(data, "<i4", n, start).astype(np.int64)
    versions = np.frombuffer(data, "<i8", n, start + 4 * n).copy()
    offsets = np.r_[np.int64(0), counts.cumsum()]
    # Detach the compact IDs from the transient frame and its metadata.
    packed = frame[start + 12 * n :].reshape(int(offsets[-1]), 3).clone()
    return (
        packed,
        torch.from_numpy(offsets),
        torch.from_numpy(tokens),
        torch.from_numpy(versions),
    )
