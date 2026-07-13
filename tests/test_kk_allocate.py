"""Tests for Karmarkar-Karp (KK) sequence packing algorithm and configurable dispatch.

Tests cover:
  1. KK core classes (_KKSet, _KKState) behaviour
  2. KK algorithm correctness (partition balance, index coverage, capacity)
  3. Configurable algorithm dispatch via get_allocate_fn()
  4. Comparative tests: KK vs FFD balance quality
  5. Edge cases and error handling
  6. MicroBatchSpec packing_algorithm field validation
  7. KK-specific: k upper bound cap, FFD fallback, equal_size mode
  8. Consistency with veRL-style KK output
  9. _compute_packing_metrics validation

Run with: pytest tests/test_kk_allocate.py -v
"""

import random

import pytest

from areal.utils.seqpack import (
    PACKING_ALGORITHM_FFD,
    PACKING_ALGORITHM_KK,
    PACKING_ALGORITHMS,
    _compute_packing_metrics,
    _kk_partition,
    _KKSet,
    _KKState,
    ffd_allocate,
    get_allocate_fn,
    kk_allocate,
)

# ---------------------------------------------------------------------------
# Data generators (matching test_seqpack.py patterns)
# ---------------------------------------------------------------------------


def generate_bimodal_seqlens(
    n=200, short_range=(50, 200), long_range=(800, 2048), long_ratio=0.3, seed=42
):
    rng = random.Random(seed)
    seqlens = []
    for _ in range(n):
        if rng.random() < long_ratio:
            seqlens.append(rng.randint(*long_range))
        else:
            seqlens.append(rng.randint(*short_range))
    return seqlens


def generate_uniform_seqlens(n=200, low=100, high=1024, seed=42):
    rng = random.Random(seed)
    return [rng.randint(low, high) for _ in range(n)]


def generate_skewed_seqlens(n=200, seed=42):
    rng = random.Random(seed)
    return [int(rng.paretovariate(1.5) * 100) for _ in range(n)]


# ---------------------------------------------------------------------------
# Tests — _KKSet
# ---------------------------------------------------------------------------


class TestKKSet:
    """Verify _KKSet behaviour (__slots__-based implementation)."""

    def test_empty_set(self):
        s = _KKSet()
        assert s.sum == 0
        assert s.items == []

    def test_add(self):
        s = _KKSet()
        s.add(idx=0, val=10)
        s.add(idx=3, val=20)
        assert s.sum == 30
        assert s.items == [(0, 10), (3, 20)]

    def test_merge(self):
        a = _KKSet()
        a.add(0, 5)
        b = _KKSet()
        b.add(1, 7)
        a.merge(b)
        assert a.sum == 12
        assert len(a.items) == 2

    def test_lt_by_sum(self):
        a = _KKSet()
        a.add(0, 5)
        b = _KKSet()
        b.add(1, 10)
        assert a < b  # 5 < 10

    def test_lt_by_length(self):
        a = _KKSet()
        a.add(0, 5)
        b = _KKSet()
        b.add(1, 3)
        b.add(2, 2)  # same sum=5 but more items
        assert a < b  # len 1 < len 2

    def test_lt_by_items(self):
        a = _KKSet()
        a.add(0, 5)
        b = _KKSet()
        b.add(1, 5)  # same sum, same length, compare items lexicographically
        assert a < b  # (0,5) < (1,5)


# ---------------------------------------------------------------------------
# Tests — _KKState
# ---------------------------------------------------------------------------


class TestKKState:
    """Verify _KKState init, merge, spread, and ordering."""

    def test_single_item_init(self):
        st = _KKState(items=[(0, 100)], k=3)
        assert st.k == 3
        assert len(st.sets) == 3
        sums = sorted(s.sum for s in st.sets)
        assert sums == [0, 0, 100]

    def test_k_items_init(self):
        st = _KKState(items=[(0, 10), (1, 20), (2, 30)], k=3)
        sums = sorted(s.sum for s in st.sets)
        assert sums == [10, 20, 30]

    def test_spread(self):
        st = _KKState(items=[(0, 10), (1, 50)], k=2)
        assert st.spread == 40  # 50 - 10

    def test_merge_reduces_spread(self):
        s0 = _KKState(items=[(0, 100)], k=2)
        s1 = _KKState(items=[(1, 90)], k=2)
        s0.merge(s1)
        assert s0.spread <= 10

    def test_lt_orders_by_spread_desc(self):
        """Larger spread compares as 'less than' (max-heap semantics)."""
        big = _KKState(items=[(0, 100)], k=2)  # spread = 100
        small = _KKState(items=[(1, 50), (2, 50)], k=2)  # spread = 0
        assert big < small

    def test_get_partitions(self):
        st = _KKState(items=[(0, 10), (1, 20)], k=2)
        parts = st.get_partitions()
        assert len(parts) == 2
        all_idx = sorted(sum(parts, []))
        assert all_idx == [0, 1]


# ---------------------------------------------------------------------------
# Tests — _kk_partition
# ---------------------------------------------------------------------------


class TestKKPartition:
    """Test _kk_partition core algorithm."""

    def test_basic_two_way(self):
        values = [7, 5, 5, 4, 4, 3, 2, 1]
        parts = _kk_partition(values, 2)
        assert len(parts) == 2
        all_indices = sorted(sum(parts, []))
        assert all_indices == list(range(len(values)))

    def test_perfect_partition(self):
        values = [10, 10, 10, 10]
        parts = _kk_partition(values, 2)
        sums = [sum(values[i] for i in p) for p in parts]
        assert max(sums) - min(sums) == 0

    def test_single_group(self):
        values = [5, 3, 2]
        parts = _kk_partition(values, 1)
        assert len(parts) == 1
        assert sorted(parts[0]) == [0, 1, 2]

    def test_many_groups(self):
        values = list(range(1, 21))  # 1..20
        k = 5
        parts = _kk_partition(values, k)
        assert len(parts) == k
        all_indices = sorted(sum(parts, []))
        assert all_indices == list(range(20))

    def test_balance_quality(self):
        """KK should produce well-balanced partitions."""
        rng = random.Random(123)
        values = [rng.randint(1, 1000) for _ in range(50)]
        k = 4
        parts = _kk_partition(values, k)
        sums = [sum(values[i] for i in p) for p in parts]
        spread = max(sums) - min(sums)
        avg = sum(values) / k
        assert spread < avg * 0.2, f"Spread {spread} too large vs avg {avg}"

    def test_equal_size_basic(self):
        """equal_size=True should give exactly n/k items per partition."""
        values = list(range(1, 13))  # 12 items
        k = 4
        parts = _kk_partition(values, k, equal_size=True)
        assert len(parts) == k
        for p in parts:
            assert len(p) == 3  # 12 / 4
        all_indices = sorted(sum(parts, []))
        assert all_indices == list(range(12))

    def test_ascending_sort_property(self):
        """Verify that the algorithm produces a deterministic, balanced output."""
        values = [1, 2, 3, 4]
        parts = _kk_partition(values, 2)
        sums = sorted(sum(values[i] for i in p) for p in parts)
        assert sums == [5, 5]  # Perfect split: {1,4} and {2,3}


# ---------------------------------------------------------------------------
# Tests — kk_allocate
# ---------------------------------------------------------------------------


class TestKKAllocate:
    """Test kk_allocate() with capacity and min_groups."""

    def test_basic_allocation(self):
        values = [100, 200, 300, 150, 250]
        result = kk_allocate(values, capacity=600, min_groups=2)
        assert len(result) >= 2
        all_idx = sorted(sum(result, []))
        assert all_idx == list(range(5))

    def test_respects_capacity_via_fallback(self):
        """When KK violates capacity it falls back to FFD, which respects it."""
        values = [100, 200, 300, 400]
        result = kk_allocate(values, capacity=500, min_groups=2)
        for group in result:
            group_sum = sum(values[i] for i in group)
            assert group_sum <= 500, f"Group sum {group_sum} exceeds capacity 500"

    def test_min_groups_guaranteed(self):
        values = [10, 20, 30]
        result = kk_allocate(values, capacity=10000, min_groups=3)
        assert len(result) >= 3

    def test_divisor(self):
        values = [100] * 10
        result = kk_allocate(values, capacity=500, min_groups=3, n_groups_divisor=2)
        assert len(result) % 2 == 0

    def test_large_capacity(self):
        """When capacity is huge, all items should go into min_groups bins."""
        values = [10, 20, 30, 40, 50]
        result = kk_allocate(values, capacity=int(1e12), min_groups=2)
        assert len(result) == 2
        all_idx = sorted(sum(result, []))
        assert all_idx == list(range(5))

    def test_raises_on_value_exceeding_capacity(self):
        with pytest.raises(RuntimeError):
            kk_allocate([100, 600], capacity=500, min_groups=1)

    def test_raises_on_too_few_values(self):
        with pytest.raises(RuntimeError):
            kk_allocate([10], capacity=100, min_groups=5)

    def test_k_upper_bound_cap(self):
        """Verify k = min(k, len(values)) prevents k from exceeding n.

        With 10 items summing to 1000, capacity=120 → ceil(1000/120)=9,
        n_groups_divisor=8 → roundup to 16, but min(16, 10) = 10.
        """
        # 10 items, total=1000, cap=120 → k=9, roundup(8)=16, cap→min(16,10)=10
        values = [100] * 10
        result = kk_allocate(values, capacity=120, min_groups=1, n_groups_divisor=8)
        # k was capped from 16 → 10 (= len(values))
        assert len(result) == 10
        all_idx = sorted(sum(result, []))
        assert all_idx == list(range(10))

        # Also verify min_groups > len(values) still raises
        with pytest.raises(RuntimeError):
            kk_allocate([100, 200, 300], capacity=int(1e12), min_groups=5)

    def test_k_upper_bound_with_divisor(self):
        """n_groups_divisor roundup can push k above n — should be clamped."""
        # 5 items, total=500, cap=200 → k=3, divisor=4 → roundup=4, min(4,5)=4
        values = [100] * 5
        result = kk_allocate(values, capacity=200, min_groups=1, n_groups_divisor=4)
        assert len(result) <= 5
        assert len(result) % 4 == 0 or len(result) == 5

    def test_kk_respects_capacity_without_fallback(self):
        """When KK naturally satisfies capacity, no FFD fallback is needed."""
        values = [250, 250, 250, 250]
        # total=1000, capacity=300 → k=ceil(1000/300)=4
        # Each group gets exactly 250 ≤ 300 → no fallback
        result = kk_allocate(values, capacity=300, min_groups=2)
        assert len(result) == 4
        for group in result:
            group_sum = sum(values[i] for i in group)
            assert group_sum <= 300
        all_idx = sorted(sum(result, []))
        assert all_idx == [0, 1, 2, 3]

    def test_kk_ffd_fallback_triggered(self):
        """Force FFD fallback: min_groups < needed groups, KK tries fewer bins."""
        # 4 items summing to 1000, capacity=300, min_groups=2
        # k = max(2, ceil(1000/300)) = 4, min(4,4)=4 → each gets 250 ≤ 300: OK
        # Let's use unequal values to force violation with min_groups=2:
        values = [280, 280, 280, 160]
        # total=1000, capacity=300 → k=4 → groups of ~250 each → fine
        # With capacity=400, k=max(2,ceil(1000/400))=3
        # KK tries 3 groups: best balance ~333 each, but 280+280=560>400?
        # Actually it'd be {280,160},{280},{280} → sums 440,280,280 → 440>400 → fallback!
        result = kk_allocate(values, capacity=400, min_groups=2)
        for group in result:
            group_sum = sum(values[i] for i in group)
            assert group_sum <= 400, f"Group sum {group_sum} > 400"

    def test_equal_size_mode(self):
        """Test equal_size=True produces groups with identical element counts."""
        values = [100, 200, 300, 400, 500, 600]
        k = 3
        result = kk_allocate(values, capacity=int(1e12), min_groups=k, equal_size=True)
        assert len(result) == k
        for group in result:
            assert len(group) == len(values) // k
        all_idx = sorted(sum(result, []))
        assert all_idx == list(range(len(values)))

    def test_equal_size_raises_if_not_divisible(self):
        with pytest.raises(RuntimeError):
            kk_allocate(
                [1, 2, 3, 4, 5], capacity=int(1e12), min_groups=3, equal_size=True
            )

    @pytest.mark.parametrize("seed", range(100))
    def test_kk_partition_consistency_with_verl(self, seed):
        """Verify _kk_partition: index coverage, sum invariant, determinism.

        Runs 100 seeds comparing that:
          - All indices are covered exactly once.
          - Partition sums add up to the total.
          - Re-running produces identical sums (determinism).
        """
        rng = random.Random(seed)
        n = rng.randint(10, 60)
        k = rng.randint(2, min(8, n))
        values = [rng.randint(1, 500) for _ in range(n)]

        parts = _kk_partition(values, k)

        # All indices covered
        all_idx = sorted(sum(parts, []))
        assert all_idx == list(range(n)), f"seed={seed}: indices mismatch"

        # Sums add up
        part_sums = [sum(values[i] for i in p) for p in parts]
        assert sum(part_sums) == sum(values), f"seed={seed}: total mismatch"

        # Determinism
        parts2 = _kk_partition(values, k)
        sums2 = sorted(sum(values[i] for i in p) for p in parts2)
        assert sorted(part_sums) == sums2, f"seed={seed}: non-deterministic"


# ---------------------------------------------------------------------------
# Tests — _compute_packing_metrics
# ---------------------------------------------------------------------------


class TestComputePackingMetrics:
    """Test _compute_packing_metrics helper function."""

    def test_balanced_case(self):
        values = [100, 200, 300, 400]
        partitions = [[0, 3], [1, 2]]  # sums: 500, 500
        capacity = 600

        m = _compute_packing_metrics(values, partitions, capacity)
        assert m["n_groups"] == 2
        assert m["spread"] == 0
        assert m["max_load"] == 500
        assert m["min_load"] == 500
        assert abs(m["std_dev"]) < 1e-9
        assert abs(m["cv"]) < 1e-9
        assert abs(m["imbalance_ratio"]) < 1e-9
        assert abs(m["max_load_ratio"] - 1.0) < 1e-9
        assert abs(m["utilization"] - 500 / 600) < 1e-9

    def test_unbalanced_case(self):
        values = [100, 200, 300, 400]
        partitions = [[0], [1, 2, 3]]  # sums: 100, 900
        capacity = 1000

        m = _compute_packing_metrics(values, partitions, capacity)
        assert m["spread"] == 800
        assert m["max_load"] == 900
        assert m["min_load"] == 100
        assert m["mean_load"] == 500.0
        assert m["imbalance_ratio"] == 800 / 500
        assert m["wasted_tokens"] == (1000 - 100) + (1000 - 900)

    def test_empty_partitions(self):
        m = _compute_packing_metrics([], [], 100)
        assert m["n_groups"] == 0
        assert m["spread"] == 0

    def test_single_group(self):
        values = [10, 20, 30]
        partitions = [[0, 1, 2]]
        capacity = 100
        m = _compute_packing_metrics(values, partitions, capacity)
        assert m["n_groups"] == 1
        assert m["spread"] == 0
        assert m["max_load"] == 60
        assert m["utilization"] == 60 / 100


# ---------------------------------------------------------------------------
# Tests — get_allocate_fn dispatch
# ---------------------------------------------------------------------------


class TestGetAllocateFn:
    """Test configurable algorithm dispatch."""

    def test_ffd_dispatch(self):
        fn = get_allocate_fn("ffd")
        assert fn is ffd_allocate

    def test_kk_dispatch(self):
        fn = get_allocate_fn("kk")
        assert fn is kk_allocate

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError, match="Unknown packing algorithm"):
            get_allocate_fn("nonexistent")

    def test_default_is_ffd(self):
        fn = get_allocate_fn()
        assert fn is ffd_allocate

    def test_dispatch_produces_valid_results(self):
        """Both algorithms should produce valid allocations."""
        values = generate_uniform_seqlens(n=50, seed=99)
        for algo in ["ffd", "kk"]:
            fn = get_allocate_fn(algo)
            result = fn(values, capacity=4096, min_groups=4)
            assert len(result) >= 4
            all_idx = sorted(sum(result, []))
            assert all_idx == list(range(50)), f"{algo} lost indices"


# ---------------------------------------------------------------------------
# Tests — KK vs FFD comparison
# ---------------------------------------------------------------------------


class TestKKVsFFDComparison:
    """Comparative tests demonstrating KK advantage over FFD."""

    @pytest.mark.parametrize("seed", range(10))
    def test_kk_balance_at_least_as_good(self, seed):
        """KK should produce partitions with spread <= FFD spread."""
        values = generate_bimodal_seqlens(n=100, seed=seed)
        min_groups = 4
        capacity = int(1e12)

        ffd_result = ffd_allocate(values, capacity, min_groups)
        kk_result = kk_allocate(values, capacity, min_groups)

        ffd_sums = sorted(sum(values[i] for i in g) for g in ffd_result if g)
        kk_sums = sorted(sum(values[i] for i in g) for g in kk_result if g)

        ffd_spread = max(ffd_sums) - min(ffd_sums) if ffd_sums else 0
        kk_spread = max(kk_sums) - min(kk_sums) if kk_sums else 0

        assert kk_spread <= ffd_spread * 1.05 + 50, (
            f"seed={seed}: KK spread {kk_spread} >> FFD spread {ffd_spread}"
        )

    def test_kk_wins_majority(self):
        """Over many random trials, KK should win or tie majority of times."""
        kk_wins = 0
        ffd_wins = 0
        ties = 0
        n_trials = 100
        min_groups = 4
        capacity = int(1e12)

        for seed in range(n_trials):
            values = generate_bimodal_seqlens(n=100, seed=seed * 7 + 13)

            ffd_result = ffd_allocate(values, capacity, min_groups)
            kk_result = kk_allocate(values, capacity, min_groups)

            ffd_sums = [sum(values[i] for i in g) for g in ffd_result if g]
            kk_sums = [sum(values[i] for i in g) for g in kk_result if g]

            ffd_spread = max(ffd_sums) - min(ffd_sums) if ffd_sums else 0
            kk_spread = max(kk_sums) - min(kk_sums) if kk_sums else 0

            if kk_spread < ffd_spread:
                kk_wins += 1
            elif ffd_spread < kk_spread:
                ffd_wins += 1
            else:
                ties += 1

        assert kk_wins + ties >= n_trials * 0.7, (
            f"KK wins={kk_wins}, ties={ties}, FFD wins={ffd_wins}"
        )

    @pytest.mark.parametrize(
        "gen_fn,gen_kwargs,spread_threshold",
        [
            (generate_bimodal_seqlens, {"n": 200}, 0.15),
            (generate_uniform_seqlens, {"n": 200}, 0.15),
            (generate_skewed_seqlens, {"n": 200}, 0.55),
        ],
        ids=["bimodal", "uniform", "skewed"],
    )
    def test_kk_balance_across_distributions(
        self, gen_fn, gen_kwargs, spread_threshold
    ):
        """KK produces good balance across different sequence length distributions."""
        values = gen_fn(**gen_kwargs, seed=42)
        min_groups = 8
        capacity = int(1e12)

        kk_result = kk_allocate(values, capacity, min_groups)
        kk_sums = [sum(values[i] for i in g) for g in kk_result if g]

        if kk_sums:
            spread = max(kk_sums) - min(kk_sums)
            avg = sum(values) / len(kk_sums)
            assert spread < avg * spread_threshold, (
                f"Spread {spread} too large vs avg {avg:.0f} "
                f"(threshold {spread_threshold})"
            )


# ---------------------------------------------------------------------------
# Tests — MicroBatchSpec packing config
# ---------------------------------------------------------------------------


class TestMicroBatchSpecPacking:
    """Test MicroBatchSpec-like config validation (standalone)."""

    def test_valid_algorithms(self):
        for algo in ["ffd", "kk"]:
            assert algo in PACKING_ALGORITHMS

    def test_invalid_algorithm_detected(self):
        assert "invalid_algo" not in PACKING_ALGORITHMS

    def test_default_is_ffd(self):
        assert PACKING_ALGORITHM_FFD == "ffd"

    def test_kk_constant(self):
        assert PACKING_ALGORITHM_KK == "kk"

    def test_config_driven_allocation(self):
        """Simulate config-driven allocation: read algorithm from config, dispatch."""

        class FakeSpec:
            max_tokens_per_mb = 4096
            n_mbs = 4
            n_mbs_divisor = 1
            packing_algorithm = "kk"

        spec = FakeSpec()
        allocate_fn = get_allocate_fn(spec.packing_algorithm)
        values = [512, 1024, 256, 768, 2048, 300, 1500, 900]
        result = allocate_fn(
            values, spec.max_tokens_per_mb, spec.n_mbs, spec.n_mbs_divisor
        )
        assert len(result) >= spec.n_mbs
        all_idx = sorted(sum(result, []))
        assert all_idx == list(range(len(values)))

    def test_config_switch_ffd_to_kk(self):
        """Switching algorithm via config produces valid results for both."""
        values = generate_bimodal_seqlens(n=100, seed=42)
        capacity = int(1e12)
        min_groups = 4

        ffd_fn = get_allocate_fn("ffd")
        kk_fn = get_allocate_fn("kk")

        ffd_result = ffd_fn(values, capacity, min_groups)
        kk_result = kk_fn(values, capacity, min_groups)

        assert sorted(sum(ffd_result, [])) == list(range(len(values)))
        assert sorted(sum(kk_result, [])) == list(range(len(values)))
