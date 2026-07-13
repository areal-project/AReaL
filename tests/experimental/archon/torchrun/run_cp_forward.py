"""Test script for Archon Engine forward_batch with Context Parallelism (Ulysses SP).

Run with torchrun for multi-GPU testing (requires at least 2 GPUs for CP=2):
    torchrun --nproc_per_node=2 tests/experimental/archon/torchrun/run_cp_forward.py

Run with 4 GPUs (dp=2, cp=2):
    torchrun --nproc_per_node=4 tests/experimental/archon/torchrun/run_cp_forward.py --cp_size=2
"""

import argparse
import os
from typing import Any

import torch
import torch.distributed as dist

from tests.utils import get_model_path

from areal.api import FinetuneSpec, ParallelStrategy
from areal.api.cli_args import (
    MicroBatchSpec,
    TrainEngineConfig,
)
from areal.experimental.engine.archon_engine import ArchonEngine
from areal.infra.platforms import current_platform
from areal.utils.data import tensor_container_to

MODEL_PATHS = {
    "qwen3": get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
    ),
}


def setup_distributed_environment():
    """Set up distributed environment for torchrun."""
    if dist.is_initialized():
        return
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )
    current_platform.set_device(rank)


def mock_input(
    device: torch.device,
    batch_size: int = 4,
    min_seqlen: int = 256,
    max_seqlen: int = 1024,
) -> dict[str, Any]:
    """Create mock padded input data for testing."""
    pad_token_id = 0
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (batch_size,), dtype=torch.int, device=device
    )
    max_len = int(max(seqlens))
    input_ids = torch.randint(
        10000, 50000, (batch_size, max_len), dtype=torch.long, device=device
    )
    attn_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    attn_mask[
        torch.arange(0, max_len, device=device).unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1
    input_ids.masked_fill_(~attn_mask, pad_token_id)

    return dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )


def make_engine_with_cp(model_type: str, mb_spec: MicroBatchSpec, cp_size: int):
    """Create and initialize an ArchonEngine with Context Parallelism."""
    config = TrainEngineConfig(
        backend=f"archon:c{cp_size}",
        experiment_name="test_archon_cp_forward",
        trial_name="test",
        path=MODEL_PATHS[model_type],
        mb_spec=mb_spec,
        optimizer=None,  # No optimizer needed for forward-only test
    )
    print(f"config = {config}")

    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)

    engine = ArchonEngine(config)

    # Create parallel strategy with CP (Ulysses)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    dp_size = world_size // cp_size
    parallel_strategy = ParallelStrategy(
        data_parallel_size=dp_size,
        context_parallel_size=cp_size,
    )

    engine.create_process_group(parallel_strategy=parallel_strategy)
    engine.initialize(addr=None, ft_spec=ft_spec)
    return engine


def make_engine_no_cp(model_type: str, mb_spec: MicroBatchSpec):
    """Create and initialize an ArchonEngine without CP (for golden comparison)."""
    config = TrainEngineConfig(
        backend="archon:d1",
        experiment_name="test_archon_cp_forward_golden",
        trial_name="test",
        path=MODEL_PATHS[model_type],
        mb_spec=mb_spec,
        optimizer=None,
    )

    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)

    engine = ArchonEngine(config)

    # Create parallel strategy without CP
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    parallel_strategy = ParallelStrategy(data_parallel_size=world_size)

    engine.create_process_group(parallel_strategy=parallel_strategy)
    engine.initialize(addr=None, ft_spec=ft_spec)
    return engine


def test_archon_cp_forward(model_type: str, cp_size: int):
    """Test ArchonEngine forward_batch with Context Parallelism (Ulysses)."""
    setup_distributed_environment()

    torch.manual_seed(42)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    dp_size = world_size // cp_size

    print(
        f"[Rank {rank}] Testing Archon Engine CP forward with "
        f"world_size={world_size}, dp={dp_size}, cp={cp_size}"
    )

    if world_size < cp_size:
        print(f"[Rank {rank}] Skipping test: world_size < cp_size")
        return

    batch_size = 4
    mb_spec = MicroBatchSpec(n_mbs=2)

    # Create engine with CP
    engine = make_engine_with_cp(model_type, mb_spec, cp_size)

    # Verify CP is enabled
    assert engine.parallel_dims.cp_enabled, "CP should be enabled"
    assert engine.parallel_dims.cp == cp_size, f"Expected cp={cp_size}"
    assert engine.parallel_dims.dp_shard == dp_size, f"Expected dp={dp_size}"

    print(
        f"[Rank {rank}] Engine initialized with "
        f"cp={engine.parallel_dims.cp}, dp={engine.parallel_dims.dp_shard}"
    )

    # Create input (same across all ranks for comparison)
    if rank == 0:
        full_input = mock_input(
            device=torch.device(f"{current_platform.device_type}:0"),
            batch_size=batch_size,
        )
        full_input_list = [full_input]
    else:
        full_input_list = [None]

    if dist.is_initialized():
        dist.broadcast_object_list(full_input_list, src=0, group=dist.group.WORLD)

    full_input = full_input_list[0]
    full_input = tensor_container_to(
        full_input, torch.device(f"{current_platform.device_type}:{rank}")
    )

    print(f"[Rank {rank}] Input shape: {full_input['input_ids'].shape}")

    # Run forward with CP
    engine.eval()
    logprobs = engine.forward_batch(
        input_=full_input,
        aggregate_fn=lambda xs: torch.cat(xs, dim=0),
    )

    print(f"[Rank {rank}] Output logprobs shape: {logprobs.shape}")

    # Verify output shape
    assert logprobs.shape[0] == batch_size, (
        f"Expected batch_size={batch_size}, got {logprobs.shape[0]}"
    )

    # Gather logprobs from all ranks to compare
    # With CP, all ranks in the same CP group should produce identical outputs
    if world_size > 1:
        logprobs_list = [torch.empty_like(logprobs) for _ in range(world_size)]
        dist.all_gather(logprobs_list, logprobs, group=dist.group.WORLD)

        # Group ranks by their DP rank (ranks in same DP group should have same output)
        # CP dimension is internal to the model - after forward_batch, results should be identical
        for i, lp in enumerate(logprobs_list):
            if not torch.allclose(lp, logprobs_list[0], atol=1e-4):
                diff = torch.abs(lp - logprobs_list[0])
                print(
                    f"[Rank {rank}] Warning: logprobs differ between rank 0 and rank {i}"
                )
                print(f"  Max diff: {diff.max()}, Mean diff: {diff.mean()}")

    print(f"[Rank {rank}] CP forward test passed!")
    print(f"[Rank {rank}] Sample logprobs (first 5): {logprobs[0, :5]}")

    # Test is_data_parallel_head
    is_head = engine.is_data_parallel_head()
    print(f"[Rank {rank}] is_data_parallel_head={is_head}")

    # Cleanup CP engine
    engine.destroy()

    # Create golden engine (no CP) for comparison
    # All ranks must participate in engine creation (distributed initialization)
    print(f"[Rank {rank}] Creating golden engine without CP for comparison...")
    golden_engine = make_engine_no_cp(model_type, mb_spec)
    golden_engine.eval()

    golden_logprobs = golden_engine.forward_batch(
        input_=full_input,
        aggregate_fn=lambda xs: torch.cat(xs, dim=0),
    )

    print(f"[Rank {rank}] Golden logprobs shape: {golden_logprobs.shape}")

    # Compare CP and non-CP results (only on rank 0)
    if rank == 0:
        # Create loss mask for valid comparisons
        attn_mask = full_input["attention_mask"]
        loss_mask = attn_mask.clone()
        loss_mask[:, :-1] = attn_mask[:, :-1] & attn_mask[:, 1:]
        loss_mask[:, -1] = False

        logprobs_valid = logprobs[loss_mask]
        golden_valid = golden_logprobs[loss_mask]

        diff = torch.abs(logprobs_valid - golden_valid)
        print(
            f"[Rank {rank}] Diff between CP and non-CP: "
            f"max={diff.max():.6f}, mean={diff.mean():.6f}"
        )

        # Use cosine similarity for comparison (more robust to numerical errors)
        cos_sim = torch.nn.functional.cosine_similarity(
            logprobs_valid.to(torch.float32).unsqueeze(0),
            golden_valid.to(torch.float32).unsqueeze(0),
        ).item()
        print(f"[Rank {rank}] Cosine similarity: {cos_sim:.6f}")

        # Assert cosine similarity is very high (> 0.9999)
        assert cos_sim > 0.9999, (
            f"Cosine similarity {cos_sim:.6f} is too low (expected > 0.9999)"
        )
        print(f"[Rank {rank}] CP vs non-CP comparison PASSED!")

    golden_engine.destroy()

    current_platform.synchronize()
    if dist.is_initialized():
        dist.barrier()
        print(f"[Rank {rank}] All ranks completed successfully")
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Run Archon Engine CP Forward Test")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["qwen3"],
        default="qwen3",
        help="Type of model to test",
    )
    parser.add_argument(
        "--cp_size",
        type=int,
        default=2,
        help="Context Parallel size (must divide world_size)",
    )
    args = parser.parse_args()
    test_archon_cp_forward(args.model_type, args.cp_size)


if __name__ == "__main__":
    # Run with `torchrun --nproc_per_node=N` to test with multiple GPUs
    # Example: torchrun --nproc_per_node=2 run_cp_forward.py --cp_size=2
    # Example: torchrun --nproc_per_node=4 run_cp_forward.py --cp_size=2
    main()
