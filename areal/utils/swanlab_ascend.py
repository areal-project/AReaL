# SPDX-License-Identifier: Apache-2.0

"""Back SwanLab's Ascend hardware charts with DCMI instead of ``npu-smi``.

SwanLab's stock ``AscendCollector`` shells out to ``npu-smi`` three times per
chip on every monitor tick (usage, temperature, power). On a 16-chip A3 node
that is 48 ``fork+exec`` of the training process per sweep, a sweep takes ~28s,
and the first monitor ticks are only 10s apart -- so the monitor thread forks
essentially continuously while the trainer forks its dataloader workers. A
worker forking mid-``Popen`` inherits ``errpipe_write``, the parent's read never
sees EOF, and the monitor thread wedges for good; ``swanlab.finish()`` then
joins it without a timeout and shutdown never returns.

This module subclasses the stock collector and overrides only those three
methods, so every chart key, name, and config is inherited unchanged -- the
dashboards look exactly as they did. Measured against ``npu-smi`` on the same
node: identical temperature, power, and HBM readings, and 1.1s per sweep instead
of 28s with no subprocess in the recurring monitor path. SwanLab's one-off
device discovery still runs ``npu-smi`` once at init; that call is
single-threaded and precedes the dataloader workers, so it cannot be raced.

Note that SwanLab's ``mode="disabled"`` only stops uploading -- the hardware
monitor thread still runs -- so this must be installed regardless of mode.

When DCMI is unavailable the Ascend collector is dropped instead, which keeps
the CPU/memory/disk/network charts (all in-process psutil) and still avoids the
hang. If SwanLab's internals move so far that neither is possible, the caller is
told to switch hardware monitoring off entirely rather than silently leave the
forking collector in place.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

from areal.utils import logging
from areal.utils.ascend_dcmi import DcmiReader, get_reader

logger = logging.getLogger("SwanlabAscend")

_SENTINEL = "_areal_ascend_collector_replaced"


def _build_collector_class(base: type) -> type:
    """Subclass SwanLab's AscendCollector, swapping npu-smi for DCMI."""

    class DcmiAscendCollector(base):  # type: ignore[misc, valid-type]
        """SwanLab Ascend collector that reads DCMI in-process."""

        def __init__(self, npu_map, max_hbm_value: int, reader: DcmiReader):
            super().__init__(npu_map, max_hbm_value)
            self._reader = reader

        @staticmethod
        def _ids(npu_id: str, chip_id: str) -> tuple[int, int] | None:
            """SwanLab labels chips by ``npu_id``/``chip_id``; DCMI by card/device."""
            try:
                return int(npu_id), int(chip_id)
            except (TypeError, ValueError):
                return None

        def get_usage(self, npu_id: str, chip_id: str) -> list[dict[str, Any]]:
            _id, metric_name = self.get_label(npu_id, chip_id)
            util_info = {
                "key": self.util_key.format(npu_index=_id),
                "name": f"{metric_name} Utilization (%)",
                "value": math.nan,
                "config": self.per_util_configs[metric_name],
            }
            hbm_info = {
                "key": self.hbm_rate_key.format(npu_index=_id),
                "name": f"{metric_name} Memory Allocated (%)",
                "value": math.nan,
                "config": self.per_hbm_configs[metric_name],
            }
            hbm_value_info = {
                "key": self.hbm_value_key.format(npu_index=_id),
                "name": f"{metric_name} Memory Allocated (MB)",
                "value": math.nan,
                "config": self.per_hbm_value_configs[metric_name],
            }

            ids = self._ids(npu_id, chip_id)
            if ids is not None:
                utilization = self._reader.aicore_utilization(*ids)
                if utilization is not None:
                    util_info["value"] = utilization
                hbm = self._reader.hbm_mb(*ids)
                if hbm is not None:
                    used_mb, total_mb = hbm
                    # DCMI gives absolute MB, so report it directly rather than
                    # deriving it from a rounded percentage as npu-smi parsing did.
                    hbm_value_info["value"] = used_mb
                    if total_mb > 0:
                        hbm_info["value"] = 100.0 * used_mb / total_mb
            return [util_info, hbm_info, hbm_value_info]

        def get_chip_temp(self, npu_id: str, chip_id: str) -> dict[str, Any]:
            _id, metric_name = self.get_label(npu_id, chip_id)
            ids = self._ids(npu_id, chip_id)
            temperature = self._reader.temperature_celsius(*ids) if ids else None
            return {
                "key": self.temp_key.format(npu_index=_id),
                "name": f"{metric_name} Temperature (℃)",
                "value": math.nan if temperature is None else temperature,
                "config": self.per_temp_configs[metric_name],
            }

        def get_chip_power(self, npu_id: str, chip_id: str) -> dict[str, Any]:
            _id, metric_name = self.get_label(npu_id, chip_id)
            ids = self._ids(npu_id, chip_id)
            power = self._reader.power_watts(*ids) if ids else None
            return {
                "key": self.power_key.format(npu_index=_id),
                "name": f"{metric_name} Power Usage (W)",
                "value": math.nan if power is None else power,
                "config": self.per_power_config[metric_name],
            }

    return DcmiAscendCollector


def install_dcmi_ascend_collector() -> str:
    """Route SwanLab's Ascend charts through DCMI, or drop them if it is absent.

    Wraps ``get_ascend_npu_info`` because ``get_hardware_info`` only registers a
    collector when that function returns one. Returns ``"dcmi"``, ``"dropped"``,
    or ``"unavailable"``. Idempotent; never raises.
    """
    try:
        swanlab_hardware = importlib.import_module("swanlab.data.run.metadata.hardware")
    except ImportError:
        logger.warning(
            "SwanLab's hardware module has moved; cannot neutralize its npu-smi "
            "polling. Hardware monitoring must be disabled instead, or shutdown "
            "may hang."
        )
        return "unavailable"

    original = getattr(swanlab_hardware, "get_ascend_npu_info", None)
    if original is None:
        logger.warning(
            "SwanLab's Ascend hardware entry point has moved; cannot neutralize "
            "its npu-smi polling. Hardware monitoring must be disabled instead, "
            "or shutdown may hang."
        )
        return "unavailable"
    if getattr(original, _SENTINEL, False):
        return getattr(original, "_areal_ascend_mode", "dcmi")

    reader = get_reader()
    collector_cls = None
    if reader is not None:
        try:
            ascend = importlib.import_module(
                "swanlab.data.run.metadata.hardware.npu.ascend"
            )
            collector_cls = _build_collector_class(ascend.AscendCollector)
        except (AttributeError, ImportError) as exc:
            logger.debug(f"Could not subclass SwanLab's AscendCollector: {exc}")

    mode = "dcmi" if collector_cls is not None else "dropped"

    def get_ascend_npu_info_via_dcmi():
        info, collector = original()
        if collector is None or collector_cls is None:
            # Withhold the npu-smi collector; the info half still reaches the run
            # page, and CPU/memory/disk/network keep monitoring as usual.
            return info, None
        try:
            return info, collector_cls(
                collector.npu_map, collector.max_hbm_value, reader
            )
        except Exception as exc:  # noqa: BLE001 - telemetry must never break a run
            logger.warning(f"DCMI Ascend collector unavailable ({exc}); dropping it.")
            return info, None

    setattr(get_ascend_npu_info_via_dcmi, _SENTINEL, True)
    setattr(get_ascend_npu_info_via_dcmi, "_areal_ascend_mode", mode)
    swanlab_hardware.get_ascend_npu_info = get_ascend_npu_info_via_dcmi

    if mode == "dcmi":
        logger.info(
            "SwanLab Ascend charts now read DCMI in-process; npu-smi polling is off."
        )
    else:
        logger.info(
            "DCMI unavailable; dropped SwanLab's npu-smi Ascend polling. "
            "CPU, memory, disk and network metrics are unaffected."
        )
    return mode
