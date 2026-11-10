# SPDX-License-Identifier: Apache-2.0
"""Load AReaL's existing Anthropic adapter without initializing training engines."""

import importlib.util
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def anthropic_adapter():
    source = Path(
        os.environ.get(
            "AUTOPILOT_ANTHROPIC_ADAPTER_PATH",
            str(
                Path(__file__).resolve().parents[3]
                / "areal/experimental/openai/anthropic.py"
            ),
        )
    )
    spec = importlib.util.spec_from_file_location("_areal_cc_anthropic_adapter", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing pinned AReaL protocol adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
