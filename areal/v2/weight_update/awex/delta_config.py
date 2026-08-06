# SPDX-License-Identifier: Apache-2.0
"""Configuration gates + factories for colocate delta weight transfer.

The delta algorithm itself lives in the standalone ``dte`` package; this module
only decides *whether* and *how* the awex colocate adapters invoke it, and
constructs the dte objects (with a clear error if dte is not installed). It
accepts the new ``DTE_*`` runtime environment emitted from ``actor.dte.*`` CLI
config, while preserving the old ``AWEX_*`` names as a compatibility fallback.

dte is imported lazily inside the factories: a default AReaL install does not
require dte unless ``DTE_DELTA_TRANSFER=1`` actually exercises a colocate
transfer, OR the receiver detects a delta payload on the wire (see
``payload_carries_delta``).

Switches:
    DTE_DELTA_TRANSFER        enable sparse incremental transfer (default off)
    DTE_SEPARATION_WEIGHT_UPDATE
                              allow the separation-only sparse P2P path
    DTE_DELTA_ANCHOR_INTERVAL force a full sync every N deltas (0 = never)
    DTE_DELTA_BYTES_RATIO     per-tensor sparse-vs-dense fallback threshold
"""

from __future__ import annotations

import os

_DTE_MISSING_MSG = (
    "DTE_DELTA_TRANSFER=1 (or a delta payload on the wire) requires the 'dte' "
    "package (delta-transfer-engine), which the awex colocate adapters import "
    "lazily. Install it with `pip install -e <path>/delta-transfer-engine` "
    "(local dev) or add DTE_SRC to PYTHONPATH."
)

# Mirror of ``dte.core.DELTA_HEADER_NAME`` so the receiver can detect a delta
# payload WITHOUT importing dte (dte is an optional lazy dependency, and a plain
# full-weight transfer must not require it). The factories below assert this
# stays in sync with dte's own constant once dte is actually imported.
DELTA_HEADER_NAME = "__awex_delta_header__"


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
    """Master switch: enable sparse incremental colocate weight transfer."""
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


def delta_bytes_ratio() -> float:
    """Per-tensor sparse-vs-dense fallback threshold (bf16 break-even ~0.33)."""
    return float(_env_value("DTE_DELTA_BYTES_RATIO", "AWEX_DELTA_BYTES_RATIO", "0.9"))


def payload_carries_delta(names) -> bool:
    """Whether a deserialized payload's names carry a dte delta header.

    Lets the inference adapter defensively reconstruct a delta payload even when
    its own ``DTE_DELTA_TRANSFER`` is unset (e.g. the env var did not propagate
    to the sglang worker while the trainer enabled delta) — a mismatch would
    otherwise feed the sparse ``...@delta_idx``/header names straight into the
    weight apply and corrupt/crash it. Pure string check: never imports dte, so
    plain full-weight transfers stay dte-free.
    """
    return DELTA_HEADER_NAME in names


def _check_header_constant() -> None:
    """Guard against ``DELTA_HEADER_NAME`` drifting from dte's own definition.

    Called from the factories (dte already imported there). If dte ever renames
    its wire header, this surfaces immediately instead of silently breaking
    ``payload_carries_delta``'s dte-free detection.
    """
    from dte.core import DELTA_HEADER_NAME as _dte_name

    if _dte_name != DELTA_HEADER_NAME:
        raise RuntimeError(
            f"delta_config.DELTA_HEADER_NAME ({DELTA_HEADER_NAME!r}) is out of "
            f"sync with dte.core.DELTA_HEADER_NAME ({_dte_name!r}); update the "
            f"mirror in delta_config.py."
        )


def make_delta_tracker():
    """Sender-side dte ``DeltaTracker``, configured from env.

    Raises a clear ``ImportError`` if dte is not installed.
    """
    try:
        from dte.core import DeltaTracker
    except ImportError as e:  # pragma: no cover - exercised only without dte
        raise ImportError(_DTE_MISSING_MSG) from e
    _check_header_constant()
    return DeltaTracker(delta_anchor_interval(), delta_bytes_ratio())


def make_delta_engine(device):
    """Receiver-side dte ``DeltaEngine`` (transport-free, IPC-fed), from env.

    Raises a clear ``ImportError`` if dte is not installed.
    """
    try:
        from dte.engine import DeltaEngine
    except ImportError as e:  # pragma: no cover - exercised only without dte
        raise ImportError(_DTE_MISSING_MSG) from e
    _check_header_constant()
    return DeltaEngine(
        transport=None,
        mode="delta",
        anchor_interval=delta_anchor_interval(),
        sparse_bytes_ratio=delta_bytes_ratio(),
        device=device,
    )


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
