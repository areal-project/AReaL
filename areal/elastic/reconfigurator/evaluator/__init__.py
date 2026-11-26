# SPDX-License-Identifier: Apache-2.0

"""Predict serving and training performance for candidate layouts."""

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
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


# Backend models import the fingerprint helpers during evaluator initialization.
from .evaluator import (  # noqa: E402
    CapacityEvaluator,
    Catalog,
    LoadedProfiles,
    load_profiles,
)

__all__ = [
    "Catalog",
    "LoadedProfiles",
    "load_profiles",
    "CapacityEvaluator",
    "code_revision",
    "fingerprint",
]
