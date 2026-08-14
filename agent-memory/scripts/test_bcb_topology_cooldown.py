#!/usr/bin/env python3
"""Focused regression tests for BCB epoch-level topology cooldown semantics."""
from types import SimpleNamespace
from memrl.run.bcb_region_runner import BCBRegionRunner


def runner(last_edit: int, cooldown: int):
    obj = object.__new__(BCBRegionRunner)
    obj.region_topology_cooldown_epochs = cooldown
    obj.mem = SimpleNamespace(region_manager=SimpleNamespace(topology_last_edit_section=last_edit))
    return obj


def test_initial_cluster_protects_its_epoch_but_allows_next_epoch_end():
    r = runner(last_edit=1, cooldown=1)
    assert not r._topology_edit_allowed(1)
    assert r._topology_edit_allowed(2)


def test_post_edit_cooldown_persists_through_mark():
    r = runner(last_edit=0, cooldown=1)
    r._mark_topology_edit(3, "test")
    assert r.mem.region_manager.topology_last_edit_section == 3
    assert not r._topology_edit_allowed(3)
    assert r._topology_edit_allowed(4)


def test_zero_cooldown_preserves_historical_behavior():
    r = runner(last_edit=9, cooldown=0)
    assert r._topology_edit_allowed(9)


if __name__ == "__main__":
    test_initial_cluster_protects_its_epoch_but_allows_next_epoch_end()
    test_post_edit_cooldown_persists_through_mark()
    test_zero_cooldown_preserves_historical_behavior()
    print("OK: BCB topology cooldown semantics")
