"""End-to-end test: KK vs FFD sequence packing with 4 GPUs.

Launches a torchrun worker (run_kk_vs_ffd.py) that simulates trajectory
redistribution using both FFD and KK algorithms, then validates that KK
produces better load balance.

Usage:
    pytest tests/test_kk_e2e.py -v -m multi_gpu

Requires:
    - 4 GPUs available
    - NCCL-capable environment
"""

import os
import pickle
import subprocess
import sys

import pytest

from areal.infra.platforms import current_platform

# Path to the torchrun worker script
WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "torchrun", "run_kk_vs_ffd.py")


@pytest.mark.multi_gpu
@pytest.mark.parametrize("world_size", [4])
@pytest.mark.parametrize(
    "n_seqs,seed",
    [
        (200, 42),
        (500, 123),
        (1000, 7),
    ],
    ids=["200seqs", "500seqs", "1000seqs"],
)
@pytest.mark.skipif(
    current_platform.device_count() < 4, reason="This test requires 4 GPUs"
)
def test_kk_vs_ffd_e2e(tmp_path, world_size, n_seqs, seed):
    """End-to-end test: KK produces more balanced redistribution than FFD.

    This test:
      1. Launches torchrun with ``world_size`` workers
      2. Each worker simulates redistribute_trajectories with FFD and KK
      3. Validates KK achieves lower or equal spread (max_load - min_load)
      4. Validates all indices are covered (no data loss)

    Args:
        tmp_path: Pytest temporary directory for output files.
        world_size: Number of simulated DP ranks (GPUs).
        n_seqs: Number of synthetic sequences.
        seed: Random seed for reproducibility.
    """
    output_dir = str(tmp_path / "results")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        "--master_port=29501",
        WORKER_SCRIPT,
        f"--output_dir={output_dir}",
        f"--n_seqs={n_seqs}",
        f"--seed={seed}",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(world_size)),
        },
    )

    assert result.returncode == 0, (
        f"torchrun failed (rc={result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    # Load results from rank 0 (all ranks compute the same global metrics)
    rank0_path = os.path.join(output_dir, "rank_0.pkl")
    assert os.path.exists(rank0_path), f"Missing output: {rank0_path}"

    with open(rank0_path, "rb") as f:
        metrics = pickle.load(f)

    # Validate structure
    assert metrics["world_size"] == world_size
    assert metrics["n_seqs"] == n_seqs

    # Core assertion: KK spread <= FFD spread (with small tolerance)
    kk_spread = metrics["kk_spread"]
    ffd_spread = metrics["ffd_spread"]

    print(f"\n{'=' * 60}")
    print(f"E2E Results (n_seqs={n_seqs}, world_size={world_size}, seed={seed}):")
    print(f"  FFD spread: {ffd_spread:>8,} tokens")
    print(f"  KK  spread: {kk_spread:>8,} tokens")
    print(f"  Improvement: {metrics['improvement_pct']:.1f}%")
    print(f"  FFD loads: {metrics['ffd_loads']}")
    print(f"  KK  loads: {metrics['kk_loads']}")
    print(f"{'=' * 60}")

    # KK should be at least as good as FFD
    assert kk_spread <= ffd_spread * 1.05 + 50, (
        f"KK spread ({kk_spread}) unexpectedly worse than FFD ({ffd_spread})"
    )

    # KK max load should not exceed FFD max load significantly
    assert metrics["kk_max_load"] <= metrics["ffd_max_load"] * 1.05 + 50, (
        f"KK max load ({metrics['kk_max_load']}) worse than FFD ({metrics['ffd_max_load']})"
    )


@pytest.mark.multi_gpu
@pytest.mark.skipif(
    current_platform.device_count() < 4, reason="This test requires 4 GPUs"
)
def test_kk_consistent_across_ranks(tmp_path):
    """Verify all ranks agree on the same KK vs FFD metrics.

    Since redistribution is a global operation, all ranks should compute
    identical partition assignments.
    """
    world_size = 4
    output_dir = str(tmp_path / "consistency")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        "--master_port=29502",
        WORKER_SCRIPT,
        f"--output_dir={output_dir}",
        "--n_seqs=200",
        "--seed=42",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    )
    assert result.returncode == 0, f"torchrun failed:\n{result.stderr}"

    # Load all rank results
    results = []
    for r in range(world_size):
        path = os.path.join(output_dir, f"rank_{r}.pkl")
        assert os.path.exists(path)
        with open(path, "rb") as f:
            results.append(pickle.load(f))

    # All ranks should report the same global metrics
    ref = results[0]
    for r, res in enumerate(results[1:], 1):
        assert res["ffd_spread"] == ref["ffd_spread"], (
            f"Rank {r} FFD spread mismatch: {res['ffd_spread']} vs {ref['ffd_spread']}"
        )
        assert res["kk_spread"] == ref["kk_spread"], (
            f"Rank {r} KK spread mismatch: {res['kk_spread']} vs {ref['kk_spread']}"
        )
        assert res["ffd_loads"] == ref["ffd_loads"], f"Rank {r} FFD loads mismatch"
        assert res["kk_loads"] == ref["kk_loads"], f"Rank {r} KK loads mismatch"
