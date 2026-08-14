#!/usr/bin/env python3
"""Focused regression tests for source-conserving Region split/merge evidence."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

# The focused test must not depend on the full sklearn installation. The fake
# KMeans deterministically partitions the synthetic low/high utility members.
cluster_mod = types.ModuleType("sklearn.cluster")
class _FakeKMeans:
    def __init__(self, n_clusters=2, random_state=None, n_init=None):
        self.n_clusters = n_clusters
    def fit_predict(self, X):
        return (np.asarray(X)[:, 0] >= np.median(np.asarray(X)[:, 0])).astype(int)
cluster_mod.KMeans = _FakeKMeans
sklearn_mod = types.ModuleType("sklearn")
sklearn_mod.cluster = cluster_mod
sys.modules.setdefault("sklearn", sklearn_mod)
sys.modules["sklearn.cluster"] = cluster_mod

spec = importlib.util.spec_from_file_location(
    "region_manager_under_test",
    Path(__file__).resolve().parents[1] / "memrl/service/region_manager.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Region = module.Region
RegionManager = module.RegionManager


def _manager(max_share: float, mode: str = "soft_source_conserving") -> RegionManager:
    mgr = RegionManager(
        task_hierarchy={}, min_cluster_size=3, min_samples=1,
        max_region_share=max_share, region_utility_mode="beta",
        region_split_evidence_migration_mode=mode,
    )
    mgr._known_subtasks = ["a", "b"]
    mgr._is_clustered = True
    members = [str(i) for i in range(12)]
    mgr.subtask_q, mgr.subtask_q_counts = {}, {}
    mgr.memory_success_sum_by_subtask, mgr.memory_total_count_by_subtask = {}, {}
    source_s, source_n = {}, {}
    for i, mid in enumerate(members):
        hi = i >= 6
        mgr.subtask_q[mid] = {"a": 0.9 if hi else 0.1, "b": 0.8 if hi else 0.2}
        # Deliberately huge; neither split mode may convert q*count into evidence.
        mgr.subtask_q_counts[mid] = {"a": 1_000_000, "b": 1_000_000}
        direct_s = {"a": float(i % 3), "b": float((i + 1) % 4)}
        direct_n = {"a": float(2 + (i % 2)), "b": float(3 + (i % 3))}
        mgr.memory_success_sum_by_subtask[mid] = direct_s
        mgr.memory_total_count_by_subtask[mid] = direct_n
        # In this synthetic source ledger every member previously contributed
        # direct evidence to the parent with weight 1.0. Split may reroute it
        # softly, but the sum across children must be exactly conserved.
        source_s[mid] = dict(direct_s)
        source_n[mid] = dict(direct_n)
    mgr.region_source_success_by_region = {0: source_s}
    mgr.region_source_total_by_region = {0: source_n}
    mgr.regions = [Region(
        region_id=0, member_ids=members, centroid=np.array([0.5, 0.5]),
        utility_by_subtask={"a": 0.5, "b": 0.6}, counts_by_subtask={"a": 12, "b": 12},
        # Deliberately stale aggregate: source/member ledgers are authoritative.
        success_sum_by_subtask={"a": 4000.0, "b": 6000.0},
        total_count_by_subtask={"a": 8000.0, "b": 10000.0},
        prior_alpha_by_subtask={"a": 2.5, "b": 3.0},
        prior_beta_by_subtask={"a": 2.5, "b": 2.0},
    )]
    return mgr


def _source_totals(mgr, st):
    s = sum(v.get(st, 0.0) for v in mgr.region_source_success_by_region[0].values())
    n = sum(v.get(st, 0.0) for v in mgr.region_source_total_by_region[0].values())
    return s, n


def _direct_totals(mgr, members, st):
    return (
        sum(mgr.memory_success_sum_by_subtask[mid].get(st, 0.0) for mid in members),
        sum(mgr.memory_total_count_by_subtask[mid].get(st, 0.0) for mid in members),
    )


def _assert_priors_and_no_q_pseudoevidence(mgr):
    assert len(mgr.regions) == 2
    assert {mid for r in mgr.regions for mid in r.member_ids} == set(mgr.subtask_q)
    for child in mgr.regions:
        assert child.prior_alpha_by_subtask == {"a": 2.5, "b": 3.0}
        assert child.prior_beta_by_subtask == {"a": 2.5, "b": 2.0}
        for st in ("a", "b"):
            # A million q-count observations per member must never freeze posterior.
            assert child.total_count_by_subtask.get(st, 0.0) < 100.0
            assert child.success_sum_by_subtask.get(st, 0.0) < 100.0


def _assert_soft_conservation(mgr, expected):
    _assert_priors_and_no_q_pseudoevidence(mgr)
    for st in ("a", "b"):
        got_s = sum(r.success_sum_by_subtask.get(st, 0.0) for r in mgr.regions)
        got_n = sum(r.total_count_by_subtask.get(st, 0.0) for r in mgr.regions)
        want_s, want_n = expected[st]
        assert abs(got_s - want_s) < 1e-9
        assert abs(got_n - want_n) < 1e-9


def test_dominant_split_soft_source_conserves_exact_evidence():
    mgr = _manager(max_share=0.30)
    expected = {st: _source_totals(mgr, st) for st in ("a", "b")}
    assert mgr.maybe_split_merge()
    _assert_soft_conservation(mgr, expected)


def test_variance_split_soft_source_conserves_exact_evidence():
    mgr = _manager(max_share=0.0)
    expected = {st: _source_totals(mgr, st) for st in ("a", "b")}
    assert mgr.maybe_split_merge()
    _assert_soft_conservation(mgr, expected)


def test_hard_member_rebase_uses_exact_member_evidence():
    mgr = _manager(max_share=0.30, mode="hard_member_rebase")
    assert mgr.maybe_split_merge()
    _assert_priors_and_no_q_pseudoevidence(mgr)
    for child in mgr.regions:
        for st in ("a", "b"):
            assert child.success_sum_by_subtask[st] == _direct_totals(mgr, child.member_ids, st)[0]
            assert child.total_count_by_subtask[st] == _direct_totals(mgr, child.member_ids, st)[1]


def test_propagation_does_not_create_direct_or_source_evidence():
    mgr = _manager(max_share=0.0)
    before_direct_s = {m: dict(v) for m, v in mgr.memory_success_sum_by_subtask.items()}
    before_direct_n = {m: dict(v) for m, v in mgr.memory_total_count_by_subtask.items()}
    before_source_s = {r: {m: dict(v) for m, v in d.items()} for r, d in mgr.region_source_success_by_region.items()}
    before_source_n = {r: {m: dict(v) for m, v in d.items()} for r, d in mgr.region_source_total_by_region.items()}
    mgr._embedding_lookup = lambda mid: np.array([1.0, 0.0]) if mid == "0" else np.array([0.99, 0.01])
    mgr._find_similar_memories = lambda *_args: [("1", 0.99)]
    mgr._propagate_q_to_neighbors(["0"], "a", 1.0)
    assert mgr.memory_success_sum_by_subtask == before_direct_s
    assert mgr.memory_total_count_by_subtask == before_direct_n
    assert mgr.region_source_success_by_region == before_source_s
    assert mgr.region_source_total_by_region == before_source_n


def test_merge_conserves_source_evidence():
    mgr = _manager(max_share=0.30)
    expected = {st: _source_totals(mgr, st) for st in ("a", "b")}
    assert mgr.maybe_split_merge()
    # Two regions differ slightly enough to avoid util_range early exit, but fall
    # below merge threshold so the merge branch executes.
    mgr.regions[0].utility_by_subtask = {"a": 0.40, "b": 0.60}
    mgr.regions[1].utility_by_subtask = {"a": 0.41, "b": 0.61}
    mgr.max_region_share = 0.0
    assert mgr.maybe_split_merge()
    assert len(mgr.regions) == 1
    merged = mgr.regions[0]
    for st in ("a", "b"):
        assert abs(merged.success_sum_by_subtask[st] - expected[st][0]) < 1e-9
        assert abs(merged.total_count_by_subtask[st] - expected[st][1]) < 1e-9


if __name__ == "__main__":
    test_dominant_split_soft_source_conserves_exact_evidence()
    test_variance_split_soft_source_conserves_exact_evidence()
    test_hard_member_rebase_uses_exact_member_evidence()
    test_propagation_does_not_create_direct_or_source_evidence()
    test_merge_conserves_source_evidence()
    print("OK: source-conserving split/merge preserves exact online evidence; q/count and propagation are excluded")


def test_legacy_precluster_checkpoint_initializes_empty_complete_soft_ledger():
    import json
    import tempfile
    legacy = {
        "task_hierarchy": {}, "alpha": 0.1, "min_cluster_size": 3,
        "temperature": 0.025, "shrinkage_top_n": 1,
        "region_utility_mode": "beta", "bayesian_smoothing_C": 0.5,
        "subtask_q": {"m": {"a": 0.7}}, "subtask_q_counts": {"m": {"a": 2}},
        "known_subtasks": ["a"], "is_clustered": False,
        "global_reward_sum": 0.0, "global_reward_count": 0,
        "membership_weights": {}, "regions": {}, "subtask_embeddings": {},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(legacy, f)
        path = f.name
    try:
        loaded = RegionManager.load(path)
        assert loaded._has_complete_region_source_evidence_ledger is True
        assert loaded.region_source_success_by_region == {}
        assert loaded.region_source_total_by_region == {}
    finally:
        Path(path).unlink(missing_ok=True)


def test_legacy_clustered_checkpoint_keeps_soft_ledger_incomplete():
    import json
    import tempfile
    legacy = {
        "task_hierarchy": {}, "alpha": 0.1, "min_cluster_size": 3,
        "temperature": 0.025, "shrinkage_top_n": 1,
        "region_utility_mode": "beta", "bayesian_smoothing_C": 0.5,
        "subtask_q": {}, "subtask_q_counts": {}, "known_subtasks": ["a"],
        "is_clustered": True, "global_reward_sum": 0.0, "global_reward_count": 0,
        "membership_weights": {}, "regions": [], "subtask_embeddings": {},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(legacy, f)
        path = f.name
    try:
        loaded = RegionManager.load(path)
        assert loaded._has_complete_region_source_evidence_ledger is False
    finally:
        Path(path).unlink(missing_ok=True)


def test_variance_split_cap_limits_fanout():
    mgr = RegionManager(
        task_hierarchy={}, min_cluster_size=3, min_samples=1,
        max_region_share=0.0, region_utility_mode="beta",
        region_max_variance_splits_per_epoch=1,
        region_split_range_fraction=0.15,
    )
    mgr._known_subtasks = ["a", "b"]
    mgr._is_clustered = True
    mgr.subtask_q, mgr.subtask_q_counts = {}, {}
    mgr.memory_success_sum_by_subtask, mgr.memory_total_count_by_subtask = {}, {}
    regs = []
    for rid, base in enumerate((0.1, 0.6)):
        members = [f"{rid}_{i}" for i in range(12)]
        for i, mid in enumerate(members):
            value = base if i < 6 else base + 0.3
            mgr.subtask_q[mid] = {"a": value, "b": value}
            mgr.subtask_q_counts[mid] = {"a": 1_000_000, "b": 1_000_000}
        regs.append(Region(
            region_id=rid, member_ids=members, centroid=np.array([base + 0.15, base + 0.15]),
            utility_by_subtask={"a": base + 0.15, "b": base + 0.15},
            counts_by_subtask={"a": 12, "b": 12},
            success_sum_by_subtask={"a": 20.0, "b": 20.0},
            total_count_by_subtask={"a": 40.0, "b": 40.0},
            prior_alpha_by_subtask={"a": 2.5, "b": 2.5},
            prior_beta_by_subtask={"a": 2.5, "b": 2.5},
        ))
    mgr.regions = regs
    assert mgr.maybe_split_merge()
    # Both parents qualify on variance, but at most one may split in this cycle.
    assert len(mgr.regions) == 3


def test_variance_split_requires_effective_evidence():
    mgr = _manager(max_share=0.0)
    mgr.region_split_min_effective_evidence = 100.0
    mgr.regions[0].success_sum_by_subtask = {"a": 1.0, "b": 1.0}
    mgr.regions[0].total_count_by_subtask = {"a": 2.0, "b": 2.0}
    assert not mgr.maybe_split_merge()
    assert len(mgr.regions) == 1


if __name__ == "__main__":
    test_variance_split_cap_limits_fanout()
    test_variance_split_requires_effective_evidence()


def _shrinkage_margin_manager(min_margin: float) -> RegionManager:
    mgr = RegionManager(
        task_hierarchy={}, min_cluster_size=2, min_samples=1,
        region_utility_mode="beta", shrinkage_top_n=1,
        shrinkage_min_utility_margin=min_margin,
    )
    mgr._known_subtasks = ["a"]
    mgr._is_clustered = True
    mgr.subtask_q = {"m0": {"a": 0.9}, "m1": {"a": 0.2}}
    mgr.subtask_q_counts = {"m0": {"a": 10.0}, "m1": {"a": 10.0}}
    mgr.membership_weights = {
        "m0": np.array([1.0, 0.0]),
        "m1": np.array([0.0, 1.0]),
    }
    mgr.regions = [
        Region(region_id=0, member_ids=["m0"], centroid=np.array([0.8]),
               utility_by_subtask={"a": 0.61}, counts_by_subtask={"a": 100}),
        Region(region_id=1, member_ids=["m1"], centroid=np.array([0.2]),
               utility_by_subtask={"a": 0.59}, counts_by_subtask={"a": 100}),
    ]
    mgr.shrinkage_lambda_max = 0.15
    mgr.shrinkage_confidence_k = 3.0
    return mgr


def test_low_margin_shrinkage_abstains_to_per_memory_q():
    mgr = _shrinkage_margin_manager(0.05)
    assert abs(mgr.get_subtask_utility_margin("a") - 0.02) < 1e-9
    assert abs(mgr.compute_shrinkage_q("m0", "a") - 0.9) < 1e-9
    assert abs(mgr.compute_shrinkage_q("m1", "a") - 0.2) < 1e-9


def test_high_margin_keeps_existing_shrinkage_behavior():
    mgr = _shrinkage_margin_manager(0.05)
    mgr.regions[0].utility_by_subtask["a"] = 0.80
    mgr.regions[1].utility_by_subtask["a"] = 0.55
    q = mgr.compute_shrinkage_q("m0", "a")
    assert 0.80 < q < 0.90


def test_shrinkage_margin_persists_across_checkpoint():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        mgr = _shrinkage_margin_manager(0.05)
        mgr.save(path)
        loaded = RegionManager.load(path)
        assert loaded.shrinkage_min_utility_margin == 0.05
    finally:
        Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_low_margin_shrinkage_abstains_to_per_memory_q()
    test_high_margin_keeps_existing_shrinkage_behavior()
    test_shrinkage_margin_persists_across_checkpoint()
    print("OK: Leaf-v4 low-margin shrinkage abstention")


def _precluster_backfill_manager(scale: float) -> RegionManager:
    mgr = RegionManager(
        task_hierarchy={}, min_cluster_size=2, min_samples=1,
        region_utility_mode="beta", region_precluster_evidence_mode="soft_source_backfill",
        region_precluster_evidence_scale=scale,
    )
    mgr._known_subtasks = ["a"]
    mgr._is_clustered = True
    mgr.regions = [
        Region(region_id=0, member_ids=["m0"], centroid=np.array([0.7]),
               utility_by_subtask={"a": 0.5}, counts_by_subtask={"a": 0},
               success_sum_by_subtask={"a": 0.0}, total_count_by_subtask={"a": 0.0},
               prior_alpha_by_subtask={"a": 2.5}, prior_beta_by_subtask={"a": 2.5}),
        Region(region_id=1, member_ids=["m1"], centroid=np.array([0.3]),
               utility_by_subtask={"a": 0.5}, counts_by_subtask={"a": 0},
               success_sum_by_subtask={"a": 0.0}, total_count_by_subtask={"a": 0.0},
               prior_alpha_by_subtask={"a": 2.5}, prior_beta_by_subtask={"a": 2.5}),
    ]
    mgr.membership_weights = {
        "m0": np.array([0.8, 0.2]),
        "m1": np.array([0.1, 0.9]),
    }
    mgr.precluster_success_sum_by_subtask = {"m0": {"a": 3.0}, "m1": {"a": 1.0}}
    mgr.precluster_total_count_by_subtask = {"m0": {"a": 4.0}, "m1": {"a": 2.0}}
    return mgr


def test_precluster_backfill_routes_raw_evidence_with_scale_and_conserves_total():
    mgr = _precluster_backfill_manager(0.75)
    mgr._backfill_precluster_evidence_once()
    total = sum(r.total_count_by_subtask.get("a", 0.0) for r in mgr.regions)
    success = sum(r.success_sum_by_subtask.get("a", 0.0) for r in mgr.regions)
    assert abs(total - 0.75 * 6.0) < 1e-9
    assert abs(success - 0.75 * 4.0) < 1e-9
    assert mgr.precluster_total_count_by_subtask == {}
    assert mgr._precluster_evidence_backfilled is True


def test_precluster_backfill_is_idempotent_and_persists():
    import tempfile
    mgr = _precluster_backfill_manager(0.75)
    mgr._backfill_precluster_evidence_once()
    before = sum(r.total_count_by_subtask.get("a", 0.0) for r in mgr.regions)
    mgr._backfill_precluster_evidence_once()
    assert sum(r.total_count_by_subtask.get("a", 0.0) for r in mgr.regions) == before
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        mgr.save(path)
        loaded = RegionManager.load(path)
        assert loaded.region_precluster_evidence_mode == "soft_source_backfill"
        assert loaded.region_precluster_evidence_scale == 0.75
        assert loaded._precluster_evidence_backfilled is True
    finally:
        Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_precluster_backfill_routes_raw_evidence_with_scale_and_conserves_total()
    test_precluster_backfill_is_idempotent_and_persists()
    print("OK: Leaf-v5 pre-cluster raw-evidence soft backfill")
