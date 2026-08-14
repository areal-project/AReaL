#!/usr/bin/env python3
"""
Verification script for region-based transfer implementation.

Checks:
1. RegionManager can be instantiated and methods work
2. Memory service accepts new parameters
3. Task hierarchy config is valid
4. Metadata includes source_subtask field
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_region_manager():
    """Test RegionManager instantiation and basic methods."""
    print("Testing RegionManager...")
    from memrl.service.region_manager import RegionManager
    from memrl.configs.task_hierarchy import TASK_HIERARCHY

    # Mock memory service
    class MockMemoryService:
        def __init__(self):
            self.query_embeddings = {}
            self.dict_memory = {}

    mem_service = MockMemoryService()
    region_manager = RegionManager(mem_service, TASK_HIERARCHY, K_global=5, K_local=3)

    print("✓ RegionManager instantiated successfully")

    # Test update_region_utility (should not crash even with no regions)
    region_manager.update_region_utility("mem_123", "bcb/Computation", 1.0)
    print("✓ update_region_utility works")

    # Test compute_region_gating_score (should return default prior)
    score = region_manager.compute_region_gating_score("mem_123", "bcb/Network", "intra", ["bcb/Computation"])
    assert 0.0 <= score <= 1.0, f"Invalid gating score: {score}"
    print(f"✓ compute_region_gating_score works (score={score:.3f})")

    return True


def test_task_hierarchy():
    """Test task hierarchy config."""
    print("\nTesting task hierarchy...")
    from memrl.configs.task_hierarchy import (
        TASK_HIERARCHY,
        get_primary_subtask,
        get_benchmark_from_subtask,
        BCB_INTRA_SPLITS,
        HLE_INTRA_SPLITS,
    )

    # Check BCB hierarchy
    assert "bigcodebench" in TASK_HIERARCHY
    assert len(TASK_HIERARCHY["bigcodebench"]["subtasks"]) == 7
    print(f"✓ BCB has {len(TASK_HIERARCHY['bigcodebench']['subtasks'])} subtasks")

    # Check HLE hierarchy
    assert "hle" in TASK_HIERARCHY
    assert len(TASK_HIERARCHY["hle"]["subtasks"]) == 10
    print(f"✓ HLE has {len(TASK_HIERARCHY['hle']['subtasks'])} subtasks")

    # Test get_primary_subtask
    subtask = get_primary_subtask("bigcodebench", {"domains": ["numpy", "pandas"]})
    assert subtask.startswith("bcb/"), f"Invalid BCB subtask: {subtask}"
    print(f"✓ get_primary_subtask works (example: {subtask})")

    # Test get_benchmark_from_subtask
    benchmark = get_benchmark_from_subtask("bcb/Computation")
    assert benchmark == "bcb"
    print(f"✓ get_benchmark_from_subtask works")

    # Check splits
    assert "split_1" in BCB_INTRA_SPLITS
    assert "source" in BCB_INTRA_SPLITS["split_1"]
    assert "target" in BCB_INTRA_SPLITS["split_1"]
    print(f"✓ BCB intra-splits defined ({len(BCB_INTRA_SPLITS)} splits)")

    assert "split_1" in HLE_INTRA_SPLITS
    print(f"✓ HLE intra-splits defined ({len(HLE_INTRA_SPLITS)} splits)")

    return True


def test_memory_service_signature():
    """Test that memory service retrieve_query accepts new parameters."""
    print("\nTesting memory service signature...")
    from memrl.service.memory_service import MemoryService
    import inspect

    sig = inspect.signature(MemoryService.retrieve_query)
    params = list(sig.parameters.keys())

    required_params = [
        "filter_source_subtasks",
        "filter_source_benchmark",
        "region_manager",
        "target_subtask",
        "eval_mode",
    ]

    for param in required_params:
        assert param in params, f"Missing parameter: {param}"
        print(f"✓ Parameter '{param}' exists")

    return True


def test_metadata_fields():
    """Test that runners include source_subtask in metadata."""
    print("\nTesting metadata fields...")

    # Check BCB runner
    with open(PROJECT_ROOT / "memrl/run/bcb_runner.py", "r") as f:
        bcb_content = f.read()

    assert '"source_subtask"' in bcb_content, "BCB runner missing source_subtask in metadata"
    assert 'get_primary_subtask' in bcb_content, "BCB runner not using get_primary_subtask"
    print("✓ BCB runner includes source_subtask")

    # Check HLE runner
    with open(PROJECT_ROOT / "memrl/run/hle_runner.py", "r") as f:
        hle_content = f.read()

    assert '"source_subtask"' in hle_content, "HLE runner missing source_subtask in metadata"
    print("✓ HLE runner includes source_subtask")

    # Check region utility tracking
    assert 'region_manager.update_region_utility' in bcb_content, "BCB runner missing region utility tracking"
    assert 'region_manager.update_region_utility' in hle_content, "HLE runner missing region utility tracking"
    print("✓ Both runners track region utility")

    return True


def test_region_gating_logic():
    """Test that region gating is applied in memory service."""
    print("\nTesting region gating logic...")

    with open(PROJECT_ROOT / "memrl/service/memory_service.py", "r") as f:
        content = f.read()

    # Check source filtering
    assert 'filter_source_subtasks' in content
    assert 'filter_source_benchmark' in content
    print("✓ Source filtering implemented")

    # Check region gating
    assert 'region_manager.compute_region_gating_score' in content
    assert 'eval_mode' in content
    print("✓ Region gating implemented")

    # Check fairness (eval_mode guard)
    assert 'if region_manager and eval_mode and target_subtask:' in content
    print("✓ Region gating guarded by eval_mode (fairness)")

    return True


def main():
    print("=" * 60)
    print("Region-Based Transfer Implementation Verification")
    print("=" * 60)

    tests = [
        ("RegionManager", test_region_manager),
        ("Task Hierarchy", test_task_hierarchy),
        ("Memory Service Signature", test_memory_service_signature),
        ("Metadata Fields", test_metadata_fields),
        ("Region Gating Logic", test_region_gating_logic),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ All verification tests passed!")
        print("\nNext steps:")
        print("1. Train BCB with region tracking:")
        print("   python scripts/run_region_transfer_experiment.py --mode train --benchmark bcb --config configs/bcb_train.yaml")
        print("\n2. Evaluate intra-transfer:")
        print("   python scripts/run_region_transfer_experiment.py --mode eval_intra --benchmark bcb --checkpoint results/bcb_train/final")
        print("\n3. Compare with baseline (no region gating):")
        print("   python scripts/run_region_transfer_experiment.py --mode eval_intra --benchmark bcb --checkpoint results/bcb_train/final --no-region-gating")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
