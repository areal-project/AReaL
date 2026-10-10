# SPDX-License-Identifier: Apache-2.0
"""Apply the validated upstream #38346 fix to a disposable SGLang installation."""

import argparse
import hashlib
import importlib.util
from pathlib import Path

EXPECTED_SHA256 = "bb57ce1e9abc4fbfcba2c9aaaf125b9e625983966497df57165b6d4c6461afe2"
PATCHED_SHA256 = "72c5d471a67729446ec9b1c0d810bc87bc85149c381c213adfb4119eddde9a7c"
TARGET_RELATIVE = Path("srt/layers/attention/qsa/qsa_indexer.py")
OLD = "            source_keys = token_k\n            source_rope = metadata.extend_rope_matrix\n"
NEW = (
    "            source_keys = token_k\n"
    "            group_locs = group_locs.clamp_max(source_keys.shape[0] - 1)\n"
    "            source_rope = metadata.extend_rope_matrix\n"
)


def patched_source(source: bytes) -> bytes:
    """Reject unknown or already-patched sources before any write."""
    before = hashlib.sha256(source).hexdigest()
    if before != EXPECTED_SHA256:
        raise RuntimeError(
            "Unknown or already-patched SGLang source; refusing to modify"
        )
    text = source.decode()
    if text.count(OLD) != 1:
        raise RuntimeError("Expected exactly one unpatched compress-gather site")
    patched = text.replace(OLD, NEW).encode()
    after = hashlib.sha256(patched).hexdigest()
    if after != PATCHED_SHA256:
        raise RuntimeError("Patched source does not match the validated runtime")
    compile(patched, str(TARGET_RELATIVE), "exec")
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, help="Disposable qsa_indexer.py copy")
    args = parser.parse_args()
    target = args.target
    if target is None:
        spec = importlib.util.find_spec("sglang")
        if spec is None or spec.origin is None:
            raise RuntimeError("SGLang is not installed; pass --target")
        target = Path(spec.origin).parent / TARGET_RELATIVE
    target.write_bytes(patched_source(target.read_bytes()))


if __name__ == "__main__":
    main()
