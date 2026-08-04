# SPDX-License-Identifier: Apache-2.0

from areal.utils.stats_tracker import DistributedStatsTracker


def test_export_single_key_with_reset():
    t = DistributedStatsTracker()
    t.scalar(foo=1.0)

    assert t.export(key="foo", reset=True) == {"foo": 1.0, "foo__count": 1}
    assert "foo" not in t.reduce_types
    assert "foo" not in t.stats
