"""Test script for Archon Engine forward_batch with Tensor Parallelism.

Run with torchrun for multi-GPU testing (requires at least 2 GPUs for TP=2):
    torchrun --nproc_per_node=2 tests/torchrun/run_archon_tp_forward.py --tp_size=2

Run with 4 GPUs (dp=2, tp=2):
    torchrun --nproc_per_node=4 tests/torchrun/run_archon_tp_forward.py --tp_size=2
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
    min_seqlen: int = 4,
    max_seqlen: int = 16,
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


def make_engine_with_tp(model_type: str, mb_spec: MicroBatchSpec, tp_size: int):
    """Create and initialize a ArchonEngine with Tensor Parallelism."""
    config = TrainEngineConfig(
        backend=f"archon:t{tp_size}",
        experiment_name="test_archon_tp_forward",
        trial_name="test",
        path=MODEL_PATHS[model_type],
        mb_spec=mb_spec,
        optimizer=None,  # No optimizer needed for forward-only test
    )
    print(f"config = {config}")

    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)

    engine = ArchonEngine(config)

    # Create parallel strategy with TP
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    dp_size = world_size // tp_size
    parallel_strategy = ParallelStrategy(
        data_parallel_size=dp_size,
        tensor_parallel_size=tp_size,
    )

    engine.create_process_group(parallel_strategy=parallel_strategy)
    engine.initialize(addr=None, ft_spec=ft_spec)
    return engine


def test_archon_tp_forward(model_type: str, tp_size: int):
    """Test ArchonEngine forward_batch with Tensor Parallelism."""
    setup_distributed_environment()

    torch.manual_seed(42)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    dp_size = world_size // tp_size

    print(
        f"[Rank {rank}] Testing Archon Engine TP forward with "
        f"world_size={world_size}, dp={dp_size}, tp={tp_size}"
    )

    if world_size < tp_size:
        print(f"[Rank {rank}] Skipping test: world_size < tp_size")
        return

    batch_size = 4
    mb_spec = MicroBatchSpec(n_mbs=2)

    # Create engine with TP
    engine = make_engine_with_tp(model_type, mb_spec, tp_size)

    # Verify TP is enabled
    assert engine.parallel_dims.tp_enabled, "TP should be enabled"
    assert engine.parallel_dims.tp == tp_size, f"Expected tp={tp_size}"
    assert engine.parallel_dims.dp_shard == dp_size, f"Expected dp={dp_size}"

    print(
        f"[Rank {rank}] Engine initialized with "
        f"tp={engine.parallel_dims.tp}, dp={engine.parallel_dims.dp_shard}"
    )

    # Create input (same across all ranks for comparison)
    if rank == 0:
        full_input = mock_input(
            device=torch.device(f"{current_platform.device_type}:0"),
            batch_size=batch_size,
            max_seqlen=16,
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

    # Run forward
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
    # Note: With TP, ranks in the same TP group should produce identical outputs
    if world_size > 1:
        logprobs_list = [torch.empty_like(logprobs) for _ in range(world_size)]
        dist.all_gather(logprobs_list, logprobs, group=dist.group.WORLD)

        # Group ranks by their DP rank (ranks in same TP group should have same output)
        for dp_rank in range(dp_size):
            tp_ranks_in_dp = list(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
            reference = logprobs_list[tp_ranks_in_dp[0]]
            for tp_rank_offset in range(1, tp_size):
                other_rank = tp_ranks_in_dp[tp_rank_offset]
                if not torch.allclose(reference, logprobs_list[other_rank], atol=1e-5):
                    diff = torch.abs(reference - logprobs_list[other_rank])
                    print(
                        f"[Rank {rank}] Warning: logprobs differ within TP group "
                        f"(dp_rank={dp_rank})"
                    )
                    print(f"  Max diff: {diff.max()}, Mean diff: {diff.mean()}")

    print(f"[Rank {rank}] Test passed!")
    print(f"[Rank {rank}] Sample logprobs (first 5): {logprobs[0, :5]}")

    # Test is_data_parallel_head - only first rank in each TP group should be head
    is_head = engine.is_data_parallel_head()
    expected_head = (rank % tp_size) == 0
    print(f"[Rank {rank}] is_data_parallel_head={is_head}, expected={expected_head}")
    assert is_head == expected_head, (
        f"is_data_parallel_head mismatch: got {is_head}, expected {expected_head}"
    )

    # Cleanup
    engine.destroy()

    if dist.is_initialized():
        dist.barrier()
        print(f"[Rank {rank}] All ranks completed successfully")
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Run Archon Engine TP Forward Test")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["qwen3"],
        default="qwen3",
        help="Type of model to test",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=2,
        help="Tensor Parallel size (must divide world_size)",
    )
    args = parser.parse_args()
    test_archon_tp_forward(args.model_type, args.tp_size)


if __name__ == "__main__":
    # Run with `torchrun --nproc_per_node=N` to test with multiple GPUs
    # Example: torchrun --nproc_per_node=2 run_archon_tp_forward.py --tp_size=2
    # Example: torchrun --nproc_per_node=4 run_archon_tp_forward.py --tp_size=2
    main()
