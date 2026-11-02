# SPDX-License-Identifier: Apache-2.0

"""Reference implementation and profile identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path


def fingerprint(*values: object) -> str:
    def encode(value):
        if is_dataclass(value):
            return asdict(value)
        raise TypeError(f"Cannot fingerprint {type(value).__name__}")

    payload = json.dumps(values, default=encode, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@lru_cache(maxsize=1)
def code_revision() -> str:
    """Hash this reference implementation once per interpreter/model session."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    digest.update((root.parents[1] / "contracts.py").read_bytes())
    digest.update((root.parents[1] / "simulation_primitives.py").read_bytes())
    return digest.hexdigest()
