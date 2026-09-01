# SPDX-License-Identifier: Apache-2.0

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

import areal.utils.swanlab_ascend as swanlab_ascend
from areal.utils.swanlab_ascend import _build_collector_class


class FakeConfig:
    def clone(self, metric_name):
        return f"config:{metric_name}"


class FakeAscendCollector:
    """Stands in for SwanLab's collector: same attributes the subclass inherits."""

    def __init__(self, npu_map, max_hbm_value):
        self.npu_map = npu_map
        self.max_hbm_value = max_hbm_value
        self.util_key = "npu.{npu_index}.ptc"
        self.hbm_rate_key = "npu.{npu_index}.mem.ptc"
        self.hbm_value_key = "npu.{npu_index}.mem.value"
        self.temp_key = "npu.{npu_index}.temp"
        self.power_key = "npu.{npu_index}.power"
        names = [f"NPU {npu}-{chip}" for npu in npu_map for chip in npu_map[npu]]
        # SwanLab stores per-metric configs cloned from a shared template.
        cloned = {n: FakeConfig().clone(metric_name=n) for n in names}
        self.per_util_configs = dict(cloned)
        self.per_hbm_configs = dict(cloned)
        self.per_hbm_value_configs = dict(cloned)
        self.per_temp_configs = dict(cloned)
        self.per_power_config = dict(cloned)

    @staticmethod
    def get_label(npu_id, chip_id):
        _id = f"{npu_id}-{chip_id}"
        return _id, f"NPU {_id}"


class FakeReader:
    def __init__(self, **overrides):
        self._overrides = overrides

    def aicore_utilization(self, card, device):
        return self._overrides.get("util", 42.0)

    def hbm_mb(self, card, device):
        return self._overrides.get("hbm", (16384.0, 65536.0))

    def temperature_celsius(self, card, device):
        return self._overrides.get("temp", 49.0)

    def power_watts(self, card, device):
        return self._overrides.get("power", 162.4)


NPU_MAP = {"0": {"0": {}, "1": {}}}


def _collector(**reader_overrides):
    cls = _build_collector_class(FakeAscendCollector)
    return cls(NPU_MAP, 65536, FakeReader(**reader_overrides))


def test_usage_reports_absolute_hbm_and_percentage():
    util, hbm_pct, hbm_mb = _collector().get_usage("0", "1")

    assert util["key"] == "npu.0-1.ptc"
    assert util["value"] == 42.0
    # Percentage is derived from absolute MB, not a rounded npu-smi percentage.
    assert hbm_pct["value"] == pytest.approx(25.0)
    assert hbm_mb["value"] == 16384.0
    assert hbm_mb["name"] == "NPU 0-1 Memory Allocated (MB)"


def test_temp_and_power_use_inherited_keys_and_configs():
    collector = _collector()

    temp = collector.get_chip_temp("0", "0")
    power = collector.get_chip_power("0", "0")

    assert (temp["key"], temp["value"]) == ("npu.0-0.temp", 49.0)
    assert (power["key"], power["value"]) == ("npu.0-0.power", 162.4)
    assert temp["config"] == "config:NPU 0-0"


def test_failed_reads_become_nan_not_exceptions():
    collector = _collector(util=None, hbm=None, temp=None, power=None)

    util, hbm_pct, hbm_mb = collector.get_usage("0", "0")
    assert all(math.isnan(entry["value"]) for entry in (util, hbm_pct, hbm_mb))
    assert math.isnan(collector.get_chip_temp("0", "0")["value"])
    assert math.isnan(collector.get_chip_power("0", "0")["value"])


def test_zero_total_hbm_does_not_divide_by_zero():
    _, hbm_pct, hbm_mb = _collector(hbm=(0.0, 0.0)).get_usage("0", "0")

    assert math.isnan(hbm_pct["value"])
    assert hbm_mb["value"] == 0.0


def test_non_numeric_chip_labels_degrade_to_nan():
    """Test that a chip id DCMI cannot map to card/device ids yields NaN."""
    # Arrange: a driver reporting non-numeric ids still populates npu_map, so the
    # inherited chart configs exist -- only the DCMI lookup is impossible.
    cls = _build_collector_class(FakeAscendCollector)
    collector = cls({"mock": {"x": {}}}, 65536, FakeReader())

    # Act / Assert
    assert math.isnan(collector.get_chip_temp("mock", "x")["value"])
    assert math.isnan(collector.get_chip_power("mock", "x")["value"])
    assert all(math.isnan(e["value"]) for e in collector.get_usage("mock", "x"))


def _install_fake_swanlab(monkeypatch, collector=None):
    hardware = ModuleType("swanlab.data.run.metadata.hardware")
    hardware.get_ascend_npu_info = lambda: ({"driver": "1.0"}, collector)
    monkeypatch.setitem(sys.modules, "swanlab.data.run.metadata.hardware", hardware)

    ascend = ModuleType("swanlab.data.run.metadata.hardware.npu.ascend")
    ascend.AscendCollector = FakeAscendCollector
    monkeypatch.setitem(
        sys.modules, "swanlab.data.run.metadata.hardware.npu.ascend", ascend
    )
    return hardware


def test_install_swaps_in_the_dcmi_collector(monkeypatch):
    stock = FakeAscendCollector(NPU_MAP, 65536)
    hardware = _install_fake_swanlab(monkeypatch, stock)
    monkeypatch.setattr(swanlab_ascend, "get_reader", lambda: FakeReader())

    assert swanlab_ascend.install_dcmi_ascend_collector() == "dcmi"
    info, collector = hardware.get_ascend_npu_info()
    assert info == {"driver": "1.0"}
    assert type(collector).__name__ == "DcmiAscendCollector"
    # The npu-smi collector's chart metadata is carried over untouched.
    assert collector.npu_map is stock.npu_map
    assert collector.max_hbm_value == stock.max_hbm_value


def test_install_drops_collector_without_dcmi(monkeypatch):
    hardware = _install_fake_swanlab(monkeypatch, FakeAscendCollector(NPU_MAP, 65536))
    monkeypatch.setattr(swanlab_ascend, "get_reader", lambda: None)

    assert swanlab_ascend.install_dcmi_ascend_collector() == "dropped"
    info, collector = hardware.get_ascend_npu_info()
    # Inventory survives so the run page still lists the NPUs.
    assert info == {"driver": "1.0"}
    assert collector is None


def test_install_is_idempotent(monkeypatch):
    hardware = _install_fake_swanlab(monkeypatch, FakeAscendCollector(NPU_MAP, 65536))
    monkeypatch.setattr(swanlab_ascend, "get_reader", lambda: FakeReader())

    assert swanlab_ascend.install_dcmi_ascend_collector() == "dcmi"
    wrapped = hardware.get_ascend_npu_info
    assert swanlab_ascend.install_dcmi_ascend_collector() == "dcmi"
    assert hardware.get_ascend_npu_info is wrapped


def test_install_reports_unavailable_when_internals_move(monkeypatch):
    hardware = ModuleType("swanlab.data.run.metadata.hardware")
    monkeypatch.setitem(sys.modules, "swanlab.data.run.metadata.hardware", hardware)

    assert swanlab_ascend.install_dcmi_ascend_collector() == "unavailable"


def test_collector_construction_failure_falls_back_to_dropping(monkeypatch):
    hardware = _install_fake_swanlab(monkeypatch, SimpleNamespace())  # no npu_map
    monkeypatch.setattr(swanlab_ascend, "get_reader", lambda: FakeReader())

    assert swanlab_ascend.install_dcmi_ascend_collector() == "dcmi"
    info, collector = hardware.get_ascend_npu_info()
    assert collector is None
