import argparse
import os
from typing import Any

import torch
import torch.distributed as dist

from tests.utils import get_model_path

from areal.api import FinetuneSpec, ParallelStrategy
from areal.api.cli_args import (
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.engine import FSDPEngine
from areal.infra.platforms import current_platform
from areal.utils.data import tensor_container_to

MODEL_PATHS = {
    "qwen3": get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
    ),
    "qwen3moe": get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-30B-A3B/", "Qwen/Qwen3-30B-A3B"
    ),
}


def setup_distributed_environment():
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
    batch_size=128,
    min_seqlen=1,
    max_seqlen=1024,
) -> dict[str, Any]:
    """Create mock padded input data (same format for huggingface) for testing.
    Returns a dict with input_ids, attention_mask, and position_ids.
    """
    pad_token_id = 0
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (batch_size,), dtype=torch.int, device=device
    )
    max_seqlen = int(max(seqlens))
    input_ids = torch.randint(
        10000, 50000, (batch_size, max_seqlen), dtype=torch.long, device=device
    )
    attn_mask = torch.zeros((batch_size, max_seqlen), dtype=torch.bool, device=device)

    attn_mask[
        torch.arange(0, max_seqlen, device=device).unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1
    input_ids.masked_fill_(~attn_mask, pad_token_id)

    return dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )


def mock_loss_fn(
    logprobs: torch.Tensor, entropy: torch.Tensor, input_data: dict, **kwargs
) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logprobs)


def make_engine(model_type, mb_spec, ulysses_sp_size=1, init_optimizer=False):
    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test",
        trial_name="test",
        path=MODEL_PATHS[model_type],
        mb_spec=mb_spec,
        optimizer=OptimizerConfig() if init_optimizer else None,
    )
    print(f"config = {config}")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine = FSDPEngine(config)
    assert dist.get_world_size() >= ulysses_sp_size
    parallel_strategy = ParallelStrategy(
        data_parallel_size=dist.get_world_size() // ulysses_sp_size,
        context_parallel_size=ulysses_sp_size,
    )
    engine.create_process_group(parallel_strategy=parallel_strategy)
    engine.initialize(addr=None, ft_spec=ft_spec)
    return engine


def test_ulysses(model_type: str):
    setup_distributed_environment()

    torch.manual_seed(42)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 1 or world_size == 2

    batch_size = 8
    batch_per_rank = 4  # for SP=2
    if rank == 0:
        full_input = mock_input(
            device=torch.device(f"{current_platform.device_type}:0"),
            batch_size=batch_size,
            max_seqlen=16,
        )
        full_input_list = [full_input]
    else:
        full_input_list = [None]
    dist.broadcast_object_list(full_input_list, src=0, group=dist.group.WORLD)
    full_input = full_input_list[0]
    full_input = tensor_container_to(
        full_input, torch.device(f"{current_platform.device_type}:{rank}")
    )

    input_chunks = []
    for i in range(batch_size // batch_per_rank):
        start_idx = i * batch_per_rank
        end_idx = (i + 1) * batch_per_rank
        chunk = dict(
            input_ids=full_input["input_ids"][start_idx:end_idx],
            attention_mask=full_input["attention_mask"][start_idx:end_idx],
        )
        input_chunks.append(chunk)

    mb_spec = MicroBatchSpec(n_mbs=4)

    if world_size > 1:
        input = input_chunks[rank]
        engine = make_engine(
            model_type, mb_spec, ulysses_sp_size=2, init_optimizer=True
        )
        engine.train()
        engine.train_batch(
            input_=input,
            loss_fn=mock_loss_fn,
            loss_weight_fn=lambda x: x["cu_seqlens"][-1],
        )
        engine.destroy()
    else:
        engine_golden = make_engine(model_type, mb_spec, init_optimizer=True)
        engine_golden.train()
        for input in input_chunks:
            engine_golden.train_batch(
                input_=input,
                loss_fn=mock_loss_fn,
                loss_weight_fn=lambda x: x["cu_seqlens"][-1],
            )
        engine_golden.destroy()


def main():
    parser = argparse.ArgumentParser(description="Run FSDP Ulysses Engine Test")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["qwen3"],
        default="qwen3",
        help="Type of model to test",
    )
    args = parser.parse_args()
    test_ulysses(args.model_type)


if __name__ == "__main__":
    # run with `torchrun` to test with multiple GPUs & multiple nodes
    main()
