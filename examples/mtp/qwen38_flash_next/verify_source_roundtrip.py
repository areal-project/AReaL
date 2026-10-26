# SPDX-License-Identifier: Apache-2.0
"""Compare every source/export safetensors payload byte with bounded memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from functools import lru_cache
from pathlib import Path

CHUNK_BYTES = 16 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=512)
def _header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length <= 0 or length > 64 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header size: {path}: {length}")
        header = json.loads(stream.read(length))
    return 8 + length, header


def _shard(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"invalid or missing shard: {root / name}")
    return path


def _tensor_span(root: Path, shard_name: str, key: str) -> tuple[Path, int, int, dict]:
    path = _shard(root, shard_name)
    start, header = _header(path)
    info = header[key]
    begin, end = info["data_offsets"]
    if begin < 0 or end < begin or start + end > path.stat().st_size:
        raise ValueError(f"invalid tensor span: {key} in {path}")
    return path, start + begin, end - begin, info


def _equal_bytes(
    left: Path, left_at: int, right: Path, right_at: int, length: int
) -> bool:
    left_fd = os.open(left, os.O_RDONLY)
    right_fd = os.open(right, os.O_RDONLY)
    try:
        for offset in range(0, length, CHUNK_BYTES):
            amount = min(CHUNK_BYTES, length - offset)
            lhs = os.pread(left_fd, amount, left_at + offset)
            rhs = os.pread(right_fd, amount, right_at + offset)
            if len(lhs) != amount or len(rhs) != amount or lhs != rhs:
                return False
    finally:
        os.close(left_fd)
        os.close(right_fd)
    return True


def verify(
    source: Path, exported: Path, *, excluded_keys: set[str] | None = None
) -> dict:
    source_index = source / "model.safetensors.index.json"
    export_index = exported / "model.safetensors.index.json"
    left_map = json.loads(source_index.read_text())["weight_map"]
    right_map = json.loads(export_index.read_text())["weight_map"]
    missing = sorted(left_map.keys() - right_map.keys())
    extra = sorted(right_map.keys() - left_map.keys())
    if missing or extra:
        raise ValueError(f"key mismatch: missing={missing[:20]} extra={extra[:20]}")

    mismatches = []
    checked_bytes = 0
    selected_keys = left_map.keys() - (excluded_keys or set())
    for key in sorted(selected_keys):
        left, left_at, left_size, left_info = _tensor_span(source, left_map[key], key)
        right, right_at, right_size, right_info = _tensor_span(
            exported, right_map[key], key
        )
        if (
            left_info["dtype"] != right_info["dtype"]
            or left_info["shape"] != right_info["shape"]
            or left_size != right_size
        ):
            mismatches.append({"key": key, "reason": "shape_dtype_or_length"})
            continue
        if not _equal_bytes(left, left_at, right, right_at, left_size):
            mismatches.append({"key": key, "reason": "value_bytes"})
        checked_bytes += left_size
    return {
        "source": str(source),
        "exported": str(exported),
        "source_index_sha256": _sha256(source_index),
        "export_index_sha256": _sha256(export_index),
        "keys": len(selected_keys),
        "excluded_key_count": len(excluded_keys or set()),
        "checked_payload_bytes": checked_bytes,
        "exact_byte_mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:30],
        "result": "PASS" if not mismatches else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("exported", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = verify(args.source, args.exported)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if result["result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
