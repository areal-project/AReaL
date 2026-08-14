#!/usr/bin/env python3
"""Focused regression tests for Region hard membership and FS argmax lookup."""
from pathlib import Path
from types import SimpleNamespace
import sys

import importlib.util
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# Load the module directly so this focused test does not require the full memos
# stack imported by memrl.service.__init__.
_spec = importlib.util.spec_from_file_location(
    "region_manager_under_test", PROJECT / "memrl/service/region_manager.py"
)
_region_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _region_module
_spec.loader.exec_module(_region_module)
Region = _region_module.Region
RegionManager = _region_module.RegionManager


def _manager():
    manager = RegionManager(task_hierarchy={}, temperature=1.0)
    manager._known_subtasks = ["alf/task"]
    manager._is_clustered = True
    manager.regions = [
        Region(region_id=0, centroid=np.array([0.0]), member_ids=["stale", "m2"]),
        Region(region_id=1, centroid=np.array([1.0]), member_ids=["stale", "m1"]),
    ]
    manager.subtask_q = {
        "m0": {"alf/task": 0.1},
        "m1": {"alf/task": 0.9},
        "m2": {"alf/task": 0.2},
    }
    manager.membership_weights = {
        "m0": np.array([0.8, 0.2]),
        "m1": np.array([0.1, 0.9]),
        "m2": np.array([0.7, 0.3]),
    }
    return manager



def test_initial_cluster_rebuilds_hard_members_from_soft_argmax():
    manager = RegionManager(task_hierarchy={}, temperature=1.0, min_cluster_size=2)
    manager._known_subtasks = ["alf/task"]
    manager.subtask_q = {
        "m0": {"alf/task": 0.0},
        "m1": {"alf/task": 0.1},
        "m2": {"alf/task": 0.9},
        "m3": {"alf/task": 1.0},
    }
    # Deliberately make seed labels disagree with distance-based soft argmax.
    manager._hdbscan_cluster_precomputed = lambda _d, _n: np.array([0, 1, 0, 1])
    manager.cluster_by_utility()

    flattened = [mid for region in manager.regions for mid in region.member_ids]
    assert sorted(flattened) == sorted(manager.membership_weights)
    assert len(flattened) == len(set(flattened))
    for mem_id, weights in manager.membership_weights.items():
        rid = int(np.argmax(weights))
        assert mem_id in manager.regions[rid].member_ids

def test_recompute_clears_stale_members():
    manager = _manager()
    manager._recompute_all_memberships()

    flattened = [mid for region in manager.regions for mid in region.member_ids]
    assert sorted(flattened) == ["m0", "m1", "m2"]
    assert len(flattened) == len(set(flattened)) == len(manager.membership_weights)
    for mem_id, weights in manager.membership_weights.items():
        rid = int(np.argmax(weights))
        assert mem_id in manager.regions[rid].member_ids


def test_restore_rebuild_uses_weight_argmax():
    manager = _manager()
    manager.rebuild_hard_memberships_from_weights()

    assert manager.regions[0].member_ids == ["m0", "m2"]
    assert manager.regions[1].member_ids == ["m1"]


def test_failure_summary_uses_weight_argmax_not_last_write():
    # Extract just the two runner methods so the test remains independent of
    # optional ALFWorld/TextWorld runtime packages.
    import ast

    runner_path = PROJECT / "memrl/run/llb_rl_runner.py"
    tree = ast.parse(runner_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LLBRunner")
    methods = [
        n for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name in {"_argmax_region_for_memory", "_replace_failure_with_region_summary"}
    ]
    mini_tree = ast.Module(body=[ast.ClassDef(
        name="MiniRunner", bases=[], keywords=[], body=methods, decorator_list=[]
    )], type_ignores=[])
    ast.fix_missing_locations(mini_tree)
    namespace = {"np": np, "List": list, "Dict": dict}
    exec(compile(mini_tree, str(runner_path), "exec"), namespace)
    MiniRunner = namespace["MiniRunner"]

    manager = _manager()
    # Deliberately duplicate m1; the old last-write mapping would select region 1.
    manager.regions[0].member_ids = ["m1"]
    manager.regions[1].member_ids = ["m1"]
    manager.membership_weights["m1"] = np.array([0.9, 0.1])
    manager.regions[0].failure_summary = "ARGMAX SUMMARY"
    manager.regions[1].failure_summary = "LAST-WRITE SUMMARY"

    runner = MiniRunner()
    runner.memory_service = SimpleNamespace(region_manager=manager)
    failed = [{"memory_id": "m1", "content": "raw failure"}]
    runner._replace_failure_with_region_summary(failed)

    assert failed[0]["content"] == "ARGMAX SUMMARY"
    assert failed[0]["_region_failure_summary"] is True
    assert failed[0]["_region_summary_region_id"] == 0


if __name__ == "__main__":
    test_initial_cluster_rebuilds_hard_members_from_soft_argmax()
    test_recompute_clears_stale_members()
    test_restore_rebuild_uses_weight_argmax()
    test_failure_summary_uses_weight_argmax_not_last_write()
    print("PASS: Region member rebuild and FS argmax regressions")
