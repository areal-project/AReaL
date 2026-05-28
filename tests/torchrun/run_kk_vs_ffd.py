"""Torchrun worker script: compare KK vs FFD for trajectory redistribution.

Launched by test_kk_e2e.py via torchrun. Each rank:
  1. Creates synthetic trajectories with bimodal sequence lengths
  2. Runs redistribute_trajectories with FFD
  3. Runs redistribute_trajectories with KK
  4. Writes per-rank metrics (total tokens, spread) to pickle files

Usage (launched by pytest, not directly):
  torchrun --nproc_per_node=4 tests/torchrun/run_kk_vs_ffd.py \\
      --output_dir /tmp/kk_test --n_seqs 200 --seed 42
"""

import argparse
import os
import pickle
import random

import torch.distributed as dist

from areal.utils.seqpack import ffd_allocate, kk_allocate


def redistribute_trajectories_sim(seqlens, world_size, algorithm="ffd"):
    """Simulate trajectory redistribution: allocate seqlens to ranks.
    Uses the *real* kk_allocate / ffd_allocate from areal.utils.seqpack.
    This simulation mirrors the production logic where rank `i` takes group `i`.
    """
    allocate_fn = kk_allocate if algorithm == "kk" else ffd_allocate
    groups = allocate_fn(seqlens, capacity=int(1e12), min_groups=world_size)
    # Production logic maps group `i` to rank `i`. We calculate all rank loads.
    rank_indices = groups
    rank_loads = [sum(seqlens[i] for i in group) for group in groups]
    # Pad with zeros if fewer groups than ranks are returned, and truncate if more.
    final_loads = (rank_loads + [0] * world_size)[:world_size]
    return rank_indices, final_loads


# =====================================================================
# Data generator
# =====================================================================


def generate_bimodal_seqlens(
    n, short_range=(50, 200), long_range=(800, 2048), long_ratio=0.3, seed=42
):
    rng = random.Random(seed)
    seqlens = []
    for _ in range(n):
        if rng.random() < long_ratio:
            seqlens.append(rng.randint(*long_range))
        else:
            seqlens.append(rng.randint(*short_range))
    return seqlens


# =====================================================================
# Main worker
# =====================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_seqs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    os.makedirs(args.output_dir, exist_ok=True)

    # All ranks use the same data (deterministic seed)
    seqlens = generate_bimodal_seqlens(args.n_seqs, seed=args.seed)

    # Run FFD redistribution
    ffd_indices, ffd_loads = redistribute_trajectories_sim(seqlens, world_size, "ffd")

    # Run KK redistribution
    kk_indices, kk_loads = redistribute_trajectories_sim(seqlens, world_size, "kk")

    # Compute metrics
    ffd_spread = max(ffd_loads) - min(ffd_loads)
    kk_spread = max(kk_loads) - min(kk_loads)
    ffd_max = max(ffd_loads)
    kk_max = max(kk_loads)

    result = {
        "rank": rank,
        "world_size": world_size,
        "n_seqs": args.n_seqs,
        "total_tokens": sum(seqlens),
        "ffd_loads": ffd_loads,
        "kk_loads": kk_loads,
        "ffd_spread": ffd_spread,
        "kk_spread": kk_spread,
        "ffd_max_load": ffd_max,
        "kk_max_load": kk_max,
        "kk_wins": kk_spread < ffd_spread,
        "improvement_pct": (
            (ffd_spread - kk_spread) / ffd_spread * 100 if ffd_spread > 0 else 0.0
        ),
    }

    output_path = os.path.join(args.output_dir, f"rank_{rank}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
