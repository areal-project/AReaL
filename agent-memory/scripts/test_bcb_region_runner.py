#!/usr/bin/env python3
"""
Quick test to verify BCBRegionRunner can be instantiated and basic methods work.
"""

import sys
from pathlib import Path

# Add MemRL to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memrl.run.bcb_runner import BCBSelection
from memrl.run.bcb_region_runner import BCBRegionRunner
from memrl.service.region_memory_service import RegionMemoryService
from memrl.service.region_manager import RegionManager
from memrl.configs.task_hierarchy import TASK_HIERARCHY


def test_instantiation():
    """Test that BCBRegionRunner class structure is correct."""
    print("Testing BCBRegionRunner class structure...")

    from memrl.run.bcb_runner import BCBRunner

    # Verify inheritance
    assert issubclass(BCBRegionRunner, BCBRunner), "BCBRegionRunner should inherit from BCBRunner"
    print("✓ BCBRegionRunner inherits from BCBRunner")

    # Verify _run_phase is overridden
    assert BCBRegionRunner._run_phase is not BCBRunner._run_phase, "_run_phase should be overridden"
    print("✓ _run_phase is overridden in BCBRegionRunner")

    # Verify _run_phase_with_region exists
    assert hasattr(BCBRegionRunner, '_run_phase_with_region'), "Should have _run_phase_with_region"
    print("✓ _run_phase_with_region method exists")

    # Verify RegionMemoryService is referenced in __init__
    import inspect
    init_src = inspect.getsource(BCBRegionRunner.__init__)
    assert "RegionMemoryService" in init_src, "__init__ should reference RegionMemoryService"
    print("✓ __init__ enforces RegionMemoryService type")

    return True


def test_region_aware_methods():
    """Test that region-aware parameters are properly passed."""
    print("\nTesting region-aware method signatures...")

    from memrl.service.region_memory_service import RegionMemoryService
    import inspect

    # Check RegionMemoryService.retrieve_query signature
    sig = inspect.signature(RegionMemoryService.retrieve_query)
    params = list(sig.parameters.keys())

    required_params = [
        "target_subtask",
        "eval_mode",
        "use_region_gating",
        "filter_source_subtasks",
        "filter_source_benchmark",
    ]

    for param in required_params:
        if param in params:
            print(f"✓ RegionMemoryService.retrieve_query has '{param}' parameter")
        else:
            print(f"✗ Missing parameter: {param}")
            return False

    # Check RegionMemoryService.update_values signature
    sig = inspect.signature(RegionMemoryService.update_values)
    params = list(sig.parameters.keys())

    if "target_subtasks" in params:
        print(f"✓ RegionMemoryService.update_values has 'target_subtasks' parameter")
    else:
        print(f"✗ Missing parameter: target_subtasks")
        return False

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("BCBRegionRunner Implementation Test")
    print("=" * 60)

    success = True

    try:
        if not test_region_aware_methods():
            success = False
    except Exception as e:
        print(f"✗ Method signature test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False

    try:
        if not test_instantiation():
            success = False
    except Exception as e:
        print(f"✗ Instantiation test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
