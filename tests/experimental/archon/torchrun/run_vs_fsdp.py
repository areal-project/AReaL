"""Test script for Archon vs FSDP Engine logits comparison.

Run with torchrun:
    torchrun --nproc_per_node=1 tests/torchrun/run_archon_vs_fsdp.py

This test compares the logits output of ArchonLMEngine and FSDPLMEngine
to verify that Archon produces equivalent results to FSDP.
"""

import argparse
import os

import torch
import torch.distributed as dist
from datasets import load_dataset

from tests.experimental.archon.torchrun.dist_utils import write_result
from tests.utils import get_dataset_path, get_model_path

from areal.api import FinetuneSpec, ParallelStrategy
from areal.api.cli_args import MicroBatchSpec, OptimizerConfig, TrainEngineConfig
from areal.engine import FSDPLMEngine
from areal.experimental.engine.archon_engine import ArchonLMEngine
from areal.infra.platforms import current_platform
from areal.utils.data import pad_sequences_to_tensors
from areal.utils.hf_utils import load_hf_tokenizer

MODEL_PATHS = {
    "qwen2": get_model_path(
        "/storage/openpsi/models/Qwen__Qwen2.5-0.5B-Instruct/",
        "Qwen/Qwen2.5-0.5B-Instruct",
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


def create_batch(model_path: str, dataset_path: str) -> dict:
    """Load tokenizer and create batch from GSM8K dataset."""
    tokenizer = load_hf_tokenizer(model_path)
    dataset = load_dataset(path=dataset_path, name="main", split="train")

    samples = []
    for sample in dataset:
        if len(samples) >= 4:
            break
        seq_token = tokenizer.encode(
            sample["question"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        if len(seq_token) <= 512:
            samples.append({"input_ids": seq_token, "loss_mask": loss_mask})

    return pad_sequences_to_tensors(samples)


def create_config(engine_type: str, model_path: str) -> TrainEngineConfig:
    """Create engine configuration."""
    return TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name=f"test_{engine_type}_logits",
        trial_name="test",
        path=model_path,
        mb_spec=MicroBatchSpec(n_mbs=1),
        optimizer=OptimizerConfig(
            type="adam",
            lr=2e-5,
            weight_decay=0.0,
            warmup_steps_proportion=0.0,
            lr_scheduler_type="constant",
        ),
    )


def test_archon_vs_fsdp_logits(model_type: str, output: str | None = None):
    """Compare Archon and FSDP engine logits.

    Verify:
    - max_diff < 5.0 (SDPA vs FlashAttention may have numerical differences)
    - mean_diff < 0.2

    Args:
        model_type: Type of model to test
        output: Output file path for pytest verification (Passed/Failed)
    """
    setup_distributed_environment()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    print(f"[Rank {rank}] Testing Archon vs FSDP logits with world_size={world_size}")

    model_path = MODEL_PATHS[model_type]
    gsm8k_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    # Create batch
    batch = create_batch(model_path, gsm8k_path)

    # Initialize engines
    parallel_strategy = ParallelStrategy(data_parallel_size=world_size)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=4, train_batch_size=4)

    archon_engine = ArchonLMEngine(create_config("archon", model_path))
    archon_engine.create_process_group(parallel_strategy=parallel_strategy)
    archon_engine.initialize(addr=None, ft_spec=ft_spec)

    fsdp_engine = FSDPLMEngine(create_config("fsdp", model_path))
    fsdp_engine.create_process_group(parallel_strategy=parallel_strategy)
    fsdp_engine.initialize(addr=None, ft_spec=ft_spec)

    try:
        device = torch.device(current_platform.device_type)
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Prepare inputs
        archon_mb_list = archon_engine._prepare_mb_list(batch).to(device)
        fsdp_mb_list = fsdp_engine._prepare_mb_list(batch).to(device)

        # Get first micro-batch using iterator
        archon_mb = next(iter(archon_mb_list))
        fsdp_mb = next(iter(fsdp_mb_list))

        archon_inputs, archon_ctx = archon_engine._prepare_mb_inputs(archon_mb)
        fsdp_inputs, fsdp_ctx = fsdp_engine._prepare_mb_inputs(fsdp_mb)

        # Forward pass
        archon_engine.eval()
        fsdp_engine.eval()

        with torch.no_grad():
            cu_seqlens = archon_inputs.get("cu_seqlens")
            max_seqlen = archon_inputs.get("max_seqlen")
            if max_seqlen is not None:
                max_seqlen = int(max_seqlen)

            archon_logits = archon_engine.model(
                archon_inputs["input_ids"],
                archon_inputs.get("position_ids"),
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            if archon_logits.ndim == 3 and archon_logits.shape[0] == 1:
                archon_logits = archon_logits.squeeze(0)

            fsdp_outputs = fsdp_engine.model(**fsdp_inputs)
            fsdp_logits = fsdp_outputs.logits
            if fsdp_logits.ndim == 3 and fsdp_logits.shape[0] == 1:
                fsdp_logits = fsdp_logits.squeeze(0)

        # Compare non-padding area
        # Use original batch length instead of ctx.pad_length since
        # Archon and FSDP may have different padding strategies
        original_length = batch["input_ids"].numel()
        archon_logits_valid = archon_logits[:original_length]
        fsdp_logits_valid = fsdp_logits[:original_length]

        diff = (archon_logits_valid.float() - fsdp_logits_valid.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"[Rank {rank}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        # SDPA vs FlashAttention may have some numerical differences
        # Relaxed thresholds: raw logits diff is not a great metric
        success = True
        if max_diff >= 5.0:
            print(
                f"[Rank {rank}] FAILED: Non-padding logits max_diff too large: {max_diff}"
            )
            success = False
        if mean_diff >= 0.2:
            print(
                f"[Rank {rank}] FAILED: Non-padding logits mean_diff too large: {mean_diff}"
            )
            success = False

        if success:
            print(f"[Rank {rank}] Test passed!")

    finally:
        archon_engine.destroy()
        fsdp_engine.destroy()

    if dist.is_initialized():
        dist.barrier()
        if rank == 0 and output:
            write_result(output, success)
        if success:
            print(f"[Rank {rank}] All ranks completed successfully")
        dist.destroy_process_group()

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Run Archon vs FSDP Engine Logits Comparison Test"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["qwen2"],
        default="qwen2",
        help="Type of model to test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for test result (Passed/Failed)",
    )
    args = parser.parse_args()
    success = test_archon_vs_fsdp_logits(args.model_type, args.output)
    if not success:
        raise AssertionError("Archon vs FSDP logits comparison failed")


if __name__ == "__main__":
    main()
