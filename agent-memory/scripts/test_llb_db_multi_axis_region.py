#!/usr/bin/env python3
"""Focused tests for LLB-DB weighted multi-axis Region supervision."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memrl.configs.task_hierarchy import get_db_multi_axis_subtasks

spec = importlib.util.spec_from_file_location(
    "region_manager_multi_axis_test", ROOT / "memrl/service/region_manager.py"
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Region = module.Region
RegionManager = module.RegionManager


def _manager():
    mgr = RegionManager(
        task_hierarchy={}, alpha=0.3, region_utility_mode="beta",
        propagation_enabled=False,
    )
    mgr._is_clustered = True
    mgr.regions = [Region(
        region_id=0, centroid=np.array([0.5]), member_ids=["m"],
        prior_alpha_by_subtask={}, prior_beta_by_subtask={},
    )]
    mgr.membership_weights = {"m": np.array([1.0])}
    return mgr


def test_label_axes_and_normalization():
    labels = get_db_multi_axis_subtasks([
        "select", "subquery_nested", "where_multiple_conditions",
        "table_alias", "limit_only",
    ])
    assert abs(sum(weight for _, weight in labels) - 1.0) < 1e-12
    names = {name for name, _ in labels}
    assert "llb_db/op/select" in names
    assert "llb_db/shape/subquery" in names
    assert "llb_db/mod/nested" in names
    assert "llb_db/mod/predicate" in names
    assert "llb_db/mod/pagination" in names
    assert "llb_db/complexity/medium" in names


def test_weighted_update_conserves_one_observation():
    mgr = _manager()
    labels = [("llb_db/op/select", 0.3), ("llb_db/shape/aggregate", 0.7)]
    for subtask, weight in labels:
        mgr.update_subtask_q(["m"], subtask, 1.0, evidence_weight=weight)

    direct_total = sum(mgr.memory_total_count_by_subtask["m"].values())
    direct_success = sum(mgr.memory_success_sum_by_subtask["m"].values())
    source_total = sum(
        value
        for by_mem in mgr.region_source_total_by_region.values()
        for by_subtask in by_mem.values()
        for value in by_subtask.values()
    )
    source_success = sum(
        value
        for by_mem in mgr.region_source_success_by_region.values()
        for by_subtask in by_mem.values()
        for value in by_subtask.values()
    )
    assert abs(direct_total - 1.0) < 1e-12
    assert abs(direct_success - 1.0) < 1e-12
    assert abs(source_total - 1.0) < 1e-12
    assert abs(source_success - 1.0) < 1e-12

    # Q step is alpha * axis weight, not a full alpha update per label.
    assert abs(mgr.subtask_q["m"]["llb_db/op/select"] - 0.545) < 1e-12
    assert abs(mgr.subtask_q["m"]["llb_db/shape/aggregate"] - 0.605) < 1e-12


def test_default_single_label_unchanged():
    mgr = _manager()
    mgr.update_subtask_q(["m"], "llb_db/select_simple", 1.0)
    assert abs(mgr.subtask_q["m"]["llb_db/select_simple"] - 0.65) < 1e-12
    assert mgr.subtask_q_counts["m"]["llb_db/select_simple"] == 1.0
    assert mgr.memory_total_count_by_subtask["m"]["llb_db/select_simple"] == 1.0


def test_propagation_does_not_touch_evidence():
    mgr = _manager()
    mgr.propagation_enabled = True
    mgr.propagation_eta = 0.2
    mgr.propagation_k = 1
    mgr.propagation_sim_min = 0.4
    mgr.subtask_q["neighbor"] = {"x": 0.5}
    mgr._embedding_lookup = lambda _mid: np.array([1.0, 0.0])
    mgr._find_similar_memories = lambda *_args: [("neighbor", 0.99)]
    before_direct = dict(mgr.memory_total_count_by_subtask)
    before_source = dict(mgr.region_source_total_by_region)
    mgr._propagate_q_to_neighbors(["m"], "x", 1.0, evidence_weight=0.25)
    assert mgr.memory_total_count_by_subtask == before_direct
    assert mgr.region_source_total_by_region == before_source
    assert mgr.subtask_q["neighbor"]["x"] > 0.5


if __name__ == "__main__":
    test_label_axes_and_normalization()
    test_weighted_update_conserves_one_observation()
    test_default_single_label_unchanged()
    test_propagation_does_not_touch_evidence()
    print("LLB_DB_MULTI_AXIS_REGION_TESTS_OK")
