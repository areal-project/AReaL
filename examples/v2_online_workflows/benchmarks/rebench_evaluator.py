# SPDX-License-Identifier: Apache-2.0
"""Load the same pinned upstream evaluator in local and Arena reward workers."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

REVISION = "c71902a8cf8d2b725f63d51f199f4d3e56f68d2d"
SOURCE_HASHES = {
    "scripts/eval.py": "4768c0c3e2adf3540c2228f819f4b073e4665ada06fa00f2234a1f7620d69eda",
    "lib/agent/log_parsers.py": "a717b03efde1cb79dfb11e2a57d0262c0057d352a347a9fb09667ef6e5f6f20c",
    "lib/agent/swe_constants.py": "823dd1ef512d363ed5d4dce05d70f22d7f93b25722cda5b0971f17010f5168a5",
}


def load_evaluator(root: Path) -> Any:
    """Packaged workers need no Git checkout or dependency installation."""
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(
                "Upstream evaluator source differs from the pinned revision"
            )
    spec = importlib.util.spec_from_file_location(
        "swe_rebench_upstream_eval", root / "scripts/eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
