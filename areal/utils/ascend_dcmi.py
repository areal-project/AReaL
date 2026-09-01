# SPDX-License-Identifier: Apache-2.0

"""Minimal ctypes binding for Ascend DCMI, the library behind ``npu-smi``.

``npu-smi`` is a thin CLI over ``libdcmi.so``. Calling the library directly keeps
telemetry in-process, which matters because every ``npu-smi`` invocation forks
the training process: a dataloader worker forking concurrently inherits the
subprocess' ``errpipe_write``, so the parent's read never sees EOF and the
calling thread wedges permanently. Reading the same counters through DCMI cannot
fork, and is ~25x faster besides (~1.1s vs ~28s for a 16-chip sweep).

Everything here is best effort. Any missing library, symbol, or non-zero return
code degrades to ``None`` rather than raising, so telemetry can never break
training. Because this binds a vendor ABI, treat a ``None`` reader as normal.
"""

from __future__ import annotations

import ctypes
import functools

from areal.utils import logging

logger = logging.getLogger("AscendDCMI")

# DCMI reports power in units of 0.1W (raw 1624 == 162.4W, cross-checked against
# ``npu-smi info`` reporting 162.6W for the same chip moments later).
_POWER_SCALE = 10.0

# ``dcmi_interface_api.h``: DCMI_UTILIZATION_RATE_AICORE
_UTILIZATION_RATE_AICORE = 2

_MAX_CARDS = 64


class _HbmInfo(ctypes.Structure):
    """``struct dcmi_hbm_info`` from ``dcmi_interface_api.h``."""

    _fields_ = [
        ("memory_size", ctypes.c_ulonglong),
        ("freq", ctypes.c_uint),
        ("memory_usage", ctypes.c_ulonglong),
        ("temp", ctypes.c_int),
        ("bandwith_util_rate", ctypes.c_uint),
    ]


class DcmiReader:
    """Reads per-chip Ascend telemetry without spawning a process.

    Enumerates chips the way ``npu-smi`` does, so no device context is created
    and chips busy with other processes are still readable.
    """

    def __init__(self, lib: ctypes.CDLL, chips: tuple[tuple[int, int], ...]):
        self._lib = lib
        self.chips = chips

    def aicore_utilization(self, card_id: int, device_id: int) -> float | None:
        """AICore utilization (%)."""
        value = ctypes.c_uint()
        rc = self._lib.dcmi_get_device_utilization_rate(
            card_id, device_id, _UTILIZATION_RATE_AICORE, ctypes.byref(value)
        )
        return float(value.value) if rc == 0 else None

    def hbm_mb(self, card_id: int, device_id: int) -> tuple[float, float] | None:
        """HBM ``(used_mb, total_mb)``."""
        info = _HbmInfo()
        rc = self._lib.dcmi_get_device_hbm_info(card_id, device_id, ctypes.byref(info))
        if rc != 0:
            return None
        return float(info.memory_usage), float(info.memory_size)

    def temperature_celsius(self, card_id: int, device_id: int) -> float | None:
        value = ctypes.c_int()
        rc = self._lib.dcmi_get_device_temperature(
            card_id, device_id, ctypes.byref(value)
        )
        return float(value.value) if rc == 0 else None

    def power_watts(self, card_id: int, device_id: int) -> float | None:
        value = ctypes.c_int()
        rc = self._lib.dcmi_get_device_power_info(
            card_id, device_id, ctypes.byref(value)
        )
        return float(value.value) / _POWER_SCALE if rc == 0 else None


def _enumerate_chips(lib: ctypes.CDLL) -> tuple[tuple[int, int], ...]:
    """Return ``(card_id, device_id)`` for every chip the driver reports."""
    card_count = ctypes.c_int()
    card_list = (ctypes.c_int * _MAX_CARDS)()
    if lib.dcmi_get_card_list(ctypes.byref(card_count), card_list, _MAX_CARDS) != 0:
        return ()

    chips: list[tuple[int, int]] = []
    for index in range(card_count.value):
        card_id = card_list[index]
        device_max = ctypes.c_int()
        mcu_id = ctypes.c_int()
        cpu_id = ctypes.c_int()
        rc = lib.dcmi_get_device_id_in_card(
            card_id,
            ctypes.byref(device_max),
            ctypes.byref(mcu_id),
            ctypes.byref(cpu_id),
        )
        if rc != 0:
            continue
        chips.extend((card_id, device_id) for device_id in range(device_max.value))
    return tuple(chips)


@functools.cache
def get_reader() -> DcmiReader | None:
    """Load DCMI once, or return None when it is unusable on this host."""
    try:
        lib = ctypes.CDLL("libdcmi.so")
    except OSError as exc:
        logger.debug(f"libdcmi.so is not loadable: {exc}")
        return None

    try:
        if lib.dcmi_init() != 0:
            logger.debug("dcmi_init() failed.")
            return None
        chips = _enumerate_chips(lib)
    except AttributeError as exc:  # symbol missing on this DCMI version
        logger.debug(f"DCMI is missing an expected symbol: {exc}")
        return None
    except OSError as exc:
        logger.debug(f"DCMI call failed: {exc}")
        return None

    if not chips:
        logger.debug("DCMI reported no chips.")
        return None

    logger.info(f"Using DCMI for Ascend telemetry ({len(chips)} chips).")
    return DcmiReader(lib, chips)
