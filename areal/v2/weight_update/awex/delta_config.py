# SPDX-License-Identifier: Apache-2.0
"""Runtime gates and factories for separation AdamW delta transfer.

The delta algorithm itself lives in the standalone ``dte`` package; this module
only decides whether the separated-card AWEX adapters invoke it and constructs
the writer tracker (with a clear error if DTE is not installed). It
accepts the new ``DTE_*`` runtime environment emitted from ``actor.dte.*`` CLI
config, while preserving the old ``AWEX_*`` names as a compatibility fallback.

DTE is imported lazily: a default AReaL install does not require it unless the
separation delta path is explicitly enabled.

Switches:
    DTE_DELTA_TRANSFER        enable sparse incremental transfer (default off)
    DTE_SEPARATION_WEIGHT_UPDATE
                              allow the separation-only sparse P2P path
    DTE_DELTA_ANCHOR_INTERVAL force a full sync every N deltas (0 = never)
"""

from __future__ import annotations

import os

_DTE_MISSING_MSG = (
    "DTE separation delta transfer requires the 'dte' package "
    "(delta-transfer-engine), which the AWEX adapters import "
    "lazily. Install it with `pip install -e <path>/delta-transfer-engine` "
    "(local dev) or add DTE_SRC to PYTHONPATH."
)


def _env_value(name: str, legacy_name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is not None and value.strip() != "":
        return value
    value = os.environ.get(legacy_name)
    if value is not None and value.strip() != "":
        return value
    return default


def _env_bool(name: str, legacy_name: str, default: bool = False) -> bool:
    value = _env_value(name, legacy_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def delta_transfer_enabled() -> bool:
    """Master switch for sparse incremental transfer."""
    return _env_bool("DTE_DELTA_TRANSFER", "AWEX_DELTA_TRANSFER")


def separation_weight_update_enabled() -> bool:
    """Whether the configured topology permits separation-only DTE code."""
    return _env_bool("DTE_SEPARATION_WEIGHT_UPDATE", "AWEX_SEPARATION_WEIGHT_UPDATE")


def separation_delta_transfer_enabled() -> bool:
    """Whether sparse separated-card transfer is explicitly enabled."""
    return separation_weight_update_enabled() and delta_transfer_enabled()


def delta_anchor_interval() -> int:
    """Force a full sync every N deltas (0 = never; rely on seed + chain-break)."""
    return int(
        _env_value(
            "DTE_DELTA_ANCHOR_INTERVAL",
            "AWEX_DELTA_ANCHOR_INTERVAL",
            "0",
        )
    )


def make_delta_tracker():
    """Sender-side dte ``DeltaTracker``, configured from env.

    Raises a clear ``ImportError`` if dte is not installed.
    """
    try:
        from dte.core import DeltaTracker
    except ImportError as e:  # pragma: no cover - exercised only without dte
        raise ImportError(_DTE_MISSING_MSG) from e
    return DeltaTracker(anchor_interval=delta_anchor_interval())


def cuda_mem_stats_mb(reset_peak: bool = True) -> tuple[float, float]:
    """Return ``(allocated_mb, peak_mb)`` for the current CUDA device.

    ``peak_mb`` is the high-water mark since the previous call (the peak
    counter is reset afterwards by default), which lets [dte-perf] stage
    marks attribute allocation spikes to individual weight-sync stages.
    Returns ``(-1.0, -1.0)`` when CUDA is unavailable (CPU tests).
    """
    import torch

    if not torch.cuda.is_available():
        return -1.0, -1.0
    allocated = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    peak = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    return allocated, peak
