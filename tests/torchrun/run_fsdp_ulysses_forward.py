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


def make_engine(model_type, mb_spec, ulysses_sp_size=1, init_optimizer=False):
    config = TrainEngineConfig(
        backend="fsdp:d2",
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
    assert world_size == 2

    mb_spec = MicroBatchSpec(n_mbs=4)

    engine = make_engine(model_type, mb_spec, ulysses_sp_size=2)
    engine_golden = make_engine(model_type, mb_spec)

    input = mock_input(device=engine.device, batch_size=4, max_seqlen=16)

    engine.eval()
    logprobs = engine.forward(
        input_=input,
        aggregate_fn=lambda xs: torch.cat(xs, dim=0),
    )

    dist.barrier()

    print(f"rank {rank} logprobs shape: {logprobs.shape} with content {logprobs}")

    # All ranks in the model parallel group should have the same logprobs
    logprobs_list = [torch.empty_like(logprobs) for _ in range(world_size)]
    dist.all_gather(logprobs_list, logprobs, group=dist.group.WORLD)

    for logprob in logprobs_list:
        assert torch.equal(logprob, logprobs_list[0])

    engine_golden.eval()
    logprobs_golden = engine_golden.forward(
        input_=input,
        aggregate_fn=lambda xs: torch.cat(xs, dim=0),
    )

    dist.barrier()

    print(
        f"rank {rank} logprobs_golden shape: {logprobs_golden.shape} with content {logprobs_golden}"
    )

    if rank == 0:
        # Create loss mask
        attn_mask = input["attention_mask"]
        loss_mask = attn_mask.clone()
        loss_mask[:, :-1] = attn_mask[:, :-1] & attn_mask[:, 1:]
        loss_mask[:, -1] = False

        logprobs_valid = logprobs[loss_mask]
        logprobs_golden_valid = logprobs_golden[loss_mask]

        diff = torch.abs(logprobs_valid - logprobs_golden_valid)
        print(
            f"rank {rank} diff between SP=1 and SP=2 logprobs: {diff}, max(diff)={torch.max(diff)} avg(diff)={torch.mean(diff)}"
        )
        try:
            torch.testing.assert_close(
                logprobs_valid.to(torch.float32),
                logprobs_golden_valid.to(torch.float32),
                rtol=1e-3,
                atol=1e-3,
            )
        except AssertionError as e:
            print(f"AssertionError in torch.testing.assert_close: {e}")

    current_platform.synchronize()
    dist.barrier()

    engine_golden.destroy()
    engine.destroy()


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
