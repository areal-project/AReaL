"""Test script for Archon Engine forward_batch functionality.

Run with torchrun for multi-GPU testing:
    torchrun --nproc_per_node=2 tests/torchrun/run_archon_forward.py

Run with single GPU:
    python tests/torchrun/run_archon_forward.py
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


def make_engine(model_type: str, mb_spec: MicroBatchSpec):
    """Create and initialize a ArchonEngine."""
    config = TrainEngineConfig(
        backend="archon:d1",
        experiment_name="test_archon_forward",
        trial_name="test",
        path=MODEL_PATHS[model_type],
        mb_spec=mb_spec,
        optimizer=None,  # No optimizer needed for forward-only test
    )
    print(f"config = {config}")

    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)

    engine = ArchonEngine(config)

    # Create parallel strategy
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    parallel_strategy = ParallelStrategy(data_parallel_size=world_size)

    engine.create_process_group(parallel_strategy=parallel_strategy)
    engine.initialize(addr=None, ft_spec=ft_spec)
    return engine


def test_archon_forward(model_type: str):
    """Test ArchonEngine forward_batch."""
    setup_distributed_environment()

    torch.manual_seed(42)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    print(f"[Rank {rank}] Testing Archon Engine forward with world_size={world_size}")

    batch_size = 4
    mb_spec = MicroBatchSpec(n_mbs=2)

    # Create engine
    engine = make_engine(model_type, mb_spec)

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

    # In data parallel mode, all ranks should produce the same output
    # (since they process the same input)
    if world_size > 1:
        logprobs_list = [torch.empty_like(logprobs) for _ in range(world_size)]
        dist.all_gather(logprobs_list, logprobs, group=dist.group.WORLD)

        for i, lp in enumerate(logprobs_list):
            if not torch.equal(lp, logprobs_list[0]):
                print(f"[Rank {rank}] Warning: logprobs differ between ranks")
                diff = torch.abs(lp - logprobs_list[0])
                print(f"  Max diff: {diff.max()}, Mean diff: {diff.mean()}")

    print(f"[Rank {rank}] Test passed!")
    print(f"[Rank {rank}] Sample logprobs (first 5): {logprobs[0, :5]}")

    # Cleanup
    engine.destroy()

    if dist.is_initialized():
        dist.barrier()
        print(f"[Rank {rank}] All ranks completed successfully")
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Run Archon Engine Forward Test")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["qwen3"],
        default="qwen3",
        help="Type of model to test",
    )
    args = parser.parse_args()
    test_archon_forward(args.model_type)


if __name__ == "__main__":
    # Run with `torchrun --nproc_per_node=N` to test with multiple GPUs
    # or just `python` for single GPU
    main()
