"""Unit tests for StalenessManager.

This module provides comprehensive test coverage for the StalenessManager class,
including capacity calculations, thread safety, and state transitions.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from areal.infra.staleness_manager import StalenessManager


class MockVersionProvider:
    """Mock version provider for testing."""

    def __init__(self, initial_version: int = 0):
        self._version = initial_version

    def get_version(self) -> int:
        return self._version

    def set_version(self, version: int) -> None:
        self._version = version


def enqueue_and_submit(manager: StalenessManager, count: int = 1) -> None:
    """Convenience helper to enqueue and submit rollouts sequentially."""
    for _ in range(count):
        manager.on_rollout_enqueued()
        manager.on_rollout_submitted()


class TestStalenessManagerBasics:
    """Test basic functionality of StalenessManager."""

    def test_initialization(self):
        """Test manager initialization with various parameters."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        assert manager.max_concurrent_rollouts == 10
        assert manager.consumer_batch_size == 4
        assert manager.max_staleness == 2

        stats = manager.get_stats()
        assert stats.enqueued == 0
        assert stats.accepted == 0
        assert stats.running == 0

    def test_initial_capacity_full(self):
        """Test that initial capacity equals max_concurrent_rollouts."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # At version 0, with no running tasks, capacity should be limited by concurrency
        capacity = manager.get_capacity()
        assert capacity == 10  # max_concurrent_rollouts

    def test_initial_capacity_with_large_staleness(self):
        """Test capacity with large staleness allowance."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=100,
            consumer_batch_size=32,
            max_staleness=10,
        )

        # Staleness capacity: (10 + 0 + 1) * 32 = 352
        # Concurrency capacity: 100
        # Should be limited by concurrency
        capacity = manager.get_capacity()
        assert capacity == 100


class TestCapacityCalculations:
    """Test capacity calculation logic under various scenarios."""

    def test_concurrency_limit(self):
        """Test that concurrency limit is enforced."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=5,
            consumer_batch_size=2,
            max_staleness=10,  # Very high, won't be limiting factor
        )

        # Submit 3 rollouts
        enqueue_and_submit(manager, count=3)

        capacity = manager.get_capacity()
        assert capacity == 2  # 5 - 3 = 2 remaining

    def test_staleness_limit(self):
        """Test that staleness limit is enforced."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=100,  # Very high, won't be limiting factor
            consumer_batch_size=4,
            max_staleness=2,
        )

        # At version 0: max_samples = (2 + 0 + 1) * 4 = 12
        # With 0 submitted/accepted, capacity should be 12
        capacity = manager.get_capacity()
        assert capacity == 12

    def test_staleness_increases_with_version(self):
        """Test that allowed capacity increases with version."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=1000,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # At version 0: (2 + 0 + 1) * 4 = 12
        capacity_v0 = manager.get_capacity()
        assert capacity_v0 == 12

        # At version 5: (2 + 5 + 1) * 4 = 32
        version_provider.set_version(5)

        capacity_v5 = manager.get_capacity()
        assert capacity_v5 == 32

        # Capacity should increase with version
        assert capacity_v5 > capacity_v0

    def test_capacity_with_running_rollouts(self):
        """Test capacity calculation with running rollouts."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Submit 3 rollouts
        enqueue_and_submit(manager, count=3)

        # Concurrency capacity: 10 - 3 = 7
        # Staleness capacity: (2 + 0 + 1) * 4 - 3 = 12 - 3 = 9
        # Should be limited by concurrency
        capacity = manager.get_capacity()
        assert capacity == 7

    def test_capacity_with_accepted_rollouts(self):
        """Test capacity calculation with accepted rollouts."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=20,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Submit and accept 5 rollouts
        for _ in range(5):
            manager.on_rollout_enqueued()
            manager.on_rollout_submitted()
            manager.on_rollout_accepted()

        # running = 0, accepted = 5
        # Concurrency capacity: 20 - 0 = 20
        # Staleness capacity: (2 + 0 + 1) * 4 - 5 = 12 - 5 = 7
        # Should be limited by staleness
        capacity = manager.get_capacity()
        assert capacity == 7

    def test_capacity_at_limit(self):
        """Test capacity when at exactly the limit."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=5,
            consumer_batch_size=2,
            max_staleness=5,  # Make staleness not limiting
        )

        # Submit 5 rollouts (at concurrency limit)
        enqueue_and_submit(manager, count=5)

        # Concurrency: 5 - 5 = 0
        # Staleness: (5 + 0 + 1) * 2 - 5 = 12 - 5 = 7 (not limiting)
        capacity = manager.get_capacity()
        assert capacity == 0  # Limited by concurrency

    def test_capacity_can_be_negative(self):
        """Test that capacity can be negative when over limit."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=3,
            consumer_batch_size=2,
            max_staleness=1,
        )

        # Submit 10 rollouts (way over limit)
        enqueue_and_submit(manager, count=10)

        capacity = manager.get_capacity()
        assert capacity < 0

    def test_min_values_are_enforced(self):
        """Test that minimum values of 1 are enforced."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=0,  # Should become 1
            consumer_batch_size=0,  # Should become 1
            max_staleness=0,
        )

        # With max(1, 0) = 1 for both params:
        # Capacity should still work correctly
        capacity = manager.get_capacity()
        assert capacity >= 0  # Should not crash


class TestRolloutLifecycle:
    """Test rollout state transitions through their lifecycle."""

    def test_submit_increments_counters(self):
        """Test that submitting a rollout increments both submitted and running."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        manager.on_rollout_enqueued()
        manager.on_rollout_submitted()

        stats = manager.get_stats()
        assert stats.enqueued == 0
        assert stats.running == 1
        assert stats.accepted == 0
        assert stats.rejected == 0

    def test_accept_updates_counters(self):
        """Test that accepting a rollout updates counters correctly."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        manager.on_rollout_enqueued()
        manager.on_rollout_submitted()
        manager.on_rollout_accepted()

        stats = manager.get_stats()
        assert stats.enqueued == 0
        assert stats.running == 0  # Decremented
        assert stats.accepted == 1  # Incremented
        assert stats.rejected == 0

    def test_reject_updates_counters(self):
        """Test that rejecting a rollout updates counters correctly."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        manager.on_rollout_enqueued()
        manager.on_rollout_submitted()
        manager.on_rollout_rejected()

        stats = manager.get_stats()
        assert stats.enqueued == 0
        assert stats.running == 0  # Decremented
        assert stats.accepted == 0  # NOT incremented (key difference from accept)
        assert stats.rejected == 1

    def test_multiple_rollouts_lifecycle(self):
        """Test multiple rollouts going through their lifecycle."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Submit 5 rollouts
        enqueue_and_submit(manager, count=5)

        # Accept 3 of them
        for _ in range(3):
            manager.on_rollout_accepted()

        # Reject 2 of them
        for _ in range(2):
            manager.on_rollout_rejected()

        stats = manager.get_stats()
        assert stats.enqueued == 0
        assert stats.running == 0  # All completed
        assert stats.accepted == 3  # Only 3 accepted
        assert stats.rejected == 2

    def test_accept_without_submit_is_invalid(self):
        """Test that accepting without submitting leads to incorrect state."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # This is invalid usage, but shouldn't crash
        manager.on_rollout_accepted()

        stats = manager.get_stats()
        assert stats.enqueued == 0
        assert stats.running == -1  # Incorrect state!
        assert stats.accepted == 1


class TestThreadSafety:
    """Test thread safety of StalenessManager operations."""

    def test_concurrent_submissions(self):
        """Test that concurrent submissions are thread-safe."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=1000,
            consumer_batch_size=32,
            max_staleness=10,
        )

        num_threads = 10
        submissions_per_thread = 100

        def submit_many():
            for _ in range(submissions_per_thread):
                manager.on_rollout_enqueued()
                manager.on_rollout_submitted()

        threads = [threading.Thread(target=submit_many) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = manager.get_stats()
        expected_total = num_threads * submissions_per_thread
        assert stats.enqueued == 0
        assert stats.running == expected_total
        assert stats.accepted == 0
        assert stats.rejected == 0

    def test_concurrent_mixed_operations(self):
        """Test concurrent mixed operations (submit, accept, reject)."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=1000,
            consumer_batch_size=32,
            max_staleness=10,
        )

        num_operations = 100

        def submit_operations():
            for _ in range(num_operations):
                manager.on_rollout_enqueued()
                manager.on_rollout_submitted()

        def accept_operations():
            for _ in range(num_operations // 2):
                manager.on_rollout_accepted()

        def reject_operations():
            for _ in range(num_operations // 2):
                manager.on_rollout_rejected()

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(submit_operations),
                executor.submit(accept_operations),
                executor.submit(reject_operations),
            ]
            for f in futures:
                f.result()

        stats = manager.get_stats()
        assert stats.enqueued == 0
        # running should be 0 (100 submitted, 50 accepted, 50 rejected)
        assert stats.running == 0
        assert stats.accepted == num_operations // 2
        assert stats.rejected == num_operations // 2

    def test_concurrent_capacity_checks(self):
        """Test that concurrent capacity checks don't cause race conditions."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=100,
            consumer_batch_size=32,
            max_staleness=10,
        )

        results = []

        def check_capacity():
            for _ in range(100):
                capacity = manager.get_capacity()
                results.append(capacity)

        threads = [threading.Thread(target=check_capacity) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All capacity checks should return the same value (no submissions)
        assert all(c == results[0] for c in results)

    def test_concurrent_get_stats(self):
        """Test that concurrent get_stats calls are thread-safe."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=100,
            consumer_batch_size=32,
            max_staleness=10,
        )

        # Submit some rollouts in background
        enqueue_and_submit(manager, count=10)

        results = []

        def get_stats_many():
            for _ in range(100):
                stats = manager.get_stats()
                results.append((stats.enqueued, stats.running, stats.accepted))

        threads = [threading.Thread(target=get_stats_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All stats should be consistent
        for submitted, running, accepted in results:
            assert submitted >= 0
            assert running >= 0
            assert accepted >= 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_max_staleness(self):
        """Test with zero staleness (immediate consumption required)."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=100,
            consumer_batch_size=8,
            max_staleness=0,
        )

        # At version 0: (0 + 0 + 1) * 8 = 8
        capacity = manager.get_capacity()
        assert capacity == 8

        # At version 5: (0 + 5 + 1) * 8 = 48
        version_provider.set_version(5)

        capacity = manager.get_capacity()
        assert capacity == 48

    def test_very_large_version(self):
        """Test with very large version numbers."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=10000,
            consumer_batch_size=64,
            max_staleness=10,
        )

        # At version 1000000: (10 + 1000000 + 1) * 64 = 64000704
        # Should be limited by concurrency
        version_provider.set_version(1000000)

        capacity = manager.get_capacity()
        assert capacity == 10000  # Limited by max_concurrent_rollouts

    def test_single_rollout_batch_size(self):
        """Test with batch size of 1."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=10,
            consumer_batch_size=1,
            max_staleness=2,
        )

        # At version 0: (2 + 0 + 1) * 1 = 3
        capacity = manager.get_capacity()
        assert capacity == 3

    def test_all_rollouts_rejected(self):
        """Test scenario where all rollouts are rejected."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Submit and reject 10 rollouts
        for _ in range(10):
            manager.on_rollout_enqueued()
            manager.on_rollout_submitted()
            manager.on_rollout_rejected()

        stats = manager.get_stats()
        assert stats.enqueued == 0
        assert stats.running == 0
        assert stats.accepted == 0
        assert stats.rejected == 10

        # Should have full capacity again
        capacity = manager.get_capacity()
        assert capacity == 10  # Back to max_concurrent_rollouts

    def test_mixed_acceptance_rate(self):
        """Test with a realistic mixed acceptance rate."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=20,
            consumer_batch_size=8,
            max_staleness=3,
        )

        # Simulate 20 rollouts: 15 accepted, 5 rejected
        for _ in range(20):
            manager.on_rollout_enqueued()
            manager.on_rollout_submitted()

        for _ in range(15):
            manager.on_rollout_accepted()

        for _ in range(5):
            manager.on_rollout_rejected()

        stats = manager.get_stats()
        assert stats.enqueued == 0
        assert stats.running == 0
        assert stats.accepted == 15
        assert stats.rejected == 5

        # Check capacity with the accepted rollouts
        # Staleness capacity: (3 + 0 + 1) * 8 - 15 = 32 - 15 = 17
        # Concurrency capacity: 20 - 0 = 20
        # Should be limited by staleness
        capacity = manager.get_capacity()
        assert capacity == 17


class TestRealWorldScenarios:
    """Test realistic scenarios that might occur in production."""

    def test_typical_training_scenario(self):
        """Test a typical training scenario with version progression."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=32,
            consumer_batch_size=16,
            max_staleness=2,
        )

        # Version 0: Submit initial batch
        enqueue_and_submit(manager, count=16)

        # Complete 14 successfully, 2 rejected
        for _ in range(14):
            manager.on_rollout_accepted()
        for _ in range(2):
            manager.on_rollout_rejected()

        # Check capacity at version 1 (after training step)
        # Staleness: (2 + 1 + 1) * 16 - 14 = 64 - 14 = 50
        # Concurrency: 32 - 0 = 32
        version_provider.set_version(1)

        capacity_v1 = manager.get_capacity()
        assert capacity_v1 == 32  # Limited by concurrency

        # Submit another batch
        enqueue_and_submit(manager, count=16)

        stats = manager.get_stats()
        assert stats.enqueued == 0
        assert stats.running == 16
        assert stats.accepted == 14
        assert stats.rejected == 2

    def test_burst_load_scenario(self):
        """Test handling burst load of rollouts."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=50,
            consumer_batch_size=32,
            max_staleness=5,
        )

        # Burst: Submit 100 rollouts quickly
        enqueue_and_submit(manager, count=100)

        stats = manager.get_stats()
        assert stats.enqueued == 0
        assert stats.running == 100

        # Capacity should be negative (over limit)
        capacity = manager.get_capacity()
        assert capacity < 0

        # Process them gradually
        for _ in range(80):
            manager.on_rollout_accepted()
        for _ in range(20):
            manager.on_rollout_rejected()

        # Should have capacity again
        capacity = manager.get_capacity()
        assert capacity > 0

    def test_slow_consumption_scenario(self):
        """Test scenario where rollouts are consumed slower than generated."""
        version_provider = MockVersionProvider(0)

        manager = StalenessManager(
            version_provider=version_provider,
            max_concurrent_rollouts=100,
            consumer_batch_size=8,
            max_staleness=3,  # Limited staleness
        )

        # Generate rollouts and accept them without version progression
        # Staleness limit: (3 + 0 + 1) * 8 = 32

        # Accept 30 rollouts (approaching staleness limit)
        for _ in range(30):
            manager.on_rollout_enqueued()
            manager.on_rollout_submitted()
            manager.on_rollout_accepted()

        # Capacity should be very limited now
        # Staleness: (3 + 0 + 1) * 8 - 30 = 32 - 30 = 2
        capacity = manager.get_capacity()
        assert capacity == 2  # Very constrained by staleness


# Parametrized tests for broader coverage
@pytest.mark.parametrize(
    "max_concurrent_rollouts,consumer_batch_size,max_staleness",
    [
        (10, 4, 2),
        (100, 32, 5),
        (1, 1, 0),
        (1000, 128, 10),
        (50, 16, 3),
    ],
)
def test_parametrized_initialization(
    max_concurrent_rollouts, consumer_batch_size, max_staleness
):
    """Test initialization with various parameter combinations."""
    version_provider = MockVersionProvider(0)

    manager = StalenessManager(
        version_provider=version_provider,
        max_concurrent_rollouts=max_concurrent_rollouts,
        consumer_batch_size=consumer_batch_size,
        max_staleness=max_staleness,
    )

    assert manager.max_concurrent_rollouts == max_concurrent_rollouts
    assert manager.consumer_batch_size == consumer_batch_size
    assert manager.max_staleness == max_staleness

    # Should always be able to get capacity without crashing
    capacity = manager.get_capacity()
    assert isinstance(capacity, int)


@pytest.mark.parametrize("version", [0, 1, 10, 100, 1000])
def test_parametrized_version_progression(version):
    """Test capacity calculation across different versions."""
    version_provider = MockVersionProvider(version)

    manager = StalenessManager(
        version_provider=version_provider,
        max_concurrent_rollouts=1000,
        consumer_batch_size=32,
        max_staleness=5,
    )

    capacity = manager.get_capacity()
    expected_staleness_capacity = (5 + version + 1) * 32

    # Capacity should be limited by concurrency (1000)
    assert capacity == min(1000, expected_staleness_capacity)


@pytest.mark.parametrize("recovered_version", [0, 5, 10, 50])
def test_on_version_recovered(recovered_version):
    """Test that on_version_recovered adjusts accepted so capacity stays bounded."""
    version_provider = MockVersionProvider(0)
    manager = StalenessManager(
        version_provider=version_provider,
        max_concurrent_rollouts=1000,
        consumer_batch_size=16,
        max_staleness=2,
    )

    # Simulate recovery: version jumps to recovered_version
    version_provider.set_version(recovered_version)
    manager.on_version_recovered(recovered_version)

    # After recovery, capacity should be (max_staleness + 1) * consumer_batch_size
    # regardless of the recovered version value.
    capacity = manager.get_capacity()
    assert capacity == (2 + 1) * 16


@pytest.mark.parametrize("running", [1, 5, 16])
def test_on_version_recovered_with_running_rollouts(running):
    """Test that on_version_recovered sets accepted correctly even when running > 0."""
    recovered_version = 10
    version_provider = MockVersionProvider(0)
    manager = StalenessManager(
        version_provider=version_provider,
        max_concurrent_rollouts=1000,
        consumer_batch_size=16,
        max_staleness=2,
    )

    # Simulate in-flight rollouts at recovery time
    for _ in range(running):
        manager.on_rollout_enqueued()
        manager.on_rollout_submitted()

    version_provider.set_version(recovered_version)
    manager.on_version_recovered(recovered_version)

    # Capacity formula: (max_staleness + 1) * consumer_bs - running
    capacity = manager.get_capacity()
    assert capacity == (2 + 1) * 16 - running


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
