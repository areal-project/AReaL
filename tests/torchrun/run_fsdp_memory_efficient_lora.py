"""Test memory_efficient_load with LoRA configuration."""

import argparse
import os

import torch
import torch.distributed as dist

from tests.utils import get_model_path

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    FSDPEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.engine import FSDPEngine
from areal.infra.platforms import current_platform

MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)


def write_result(out: str, succ: bool):
    with open(out, "w") as f:
        if succ:
            f.write("Passed")
        else:
            f.write("Failed")


def make_fsdp_engine_with_lora(
    backend: str,
    memory_efficient_load: bool,
):
    """Create FSDPEngine with LoRA and optionally memory_efficient_load."""
    config = TrainEngineConfig(
        backend=backend,
        experiment_name="test_fsdp_memory_efficient_lora",
        trial_name="test",
        mb_spec=MicroBatchSpec(max_tokens_per_mb=256),
        path=MODEL_PATH,
        optimizer=OptimizerConfig(),
        fsdp=FSDPEngineConfig(memory_efficient_load=memory_efficient_load),
        # LoRA config
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
        peft_type="lora",
    )
    alloc_mode = ModelAllocation.from_str(backend)
    engine = FSDPEngine(config)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine.create_process_group(parallel_strategy=alloc_mode.parallel)
    engine.initialize(None, ft_spec)
    return engine


def test_memory_efficient_lora(alloc_mode: str, output: str | None = None):
    """Test that memory_efficient_load works correctly with LoRA.

    This test verifies:
    1. Engine initializes successfully with memory_efficient_load=True and use_lora=True
    2. LoRA layers are properly applied
    3. Model can perform a forward pass
    """
    rank = int(os.environ["RANK"])
    print(f"Running memory_efficient_load + LoRA test on rank {rank}")

    succ = True

    # Test 1: Create engine with memory_efficient_load=True
    print(f"Rank {rank}: Creating engine with memory_efficient_load=True")
    engine = make_fsdp_engine_with_lora(alloc_mode, memory_efficient_load=True)

    # Test 2: Verify LoRA layers exist
    lora_params = [
        name for name, _ in engine.model.named_parameters() if "lora" in name.lower()
    ]
    if not lora_params:
        print(f"Rank {rank}: ERROR - No LoRA parameters found!")
        succ = False
    else:
        print(f"Rank {rank}: Found {len(lora_params)} LoRA parameters")

    # Test 3: Verify trainable params are only LoRA params
    trainable_params = [
        name for name, p in engine.model.named_parameters() if p.requires_grad
    ]
    non_lora_trainable = [p for p in trainable_params if "lora" not in p.lower()]
    if non_lora_trainable:
        print(
            f"Rank {rank}: WARNING - Found non-LoRA trainable params: {non_lora_trainable[:5]}"
        )

    # Test 4: Simple forward pass to verify model works
    print(f"Rank {rank}: Testing forward pass")
    try:
        with torch.no_grad():
            engine.eval()
            # Create simple input
            input_ids = torch.randint(
                100, 1000, (2, 32), dtype=torch.long, device=engine.device
            )
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

            # Get logits through the model
            outputs = engine.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
            logits = outputs.logits
            print(f"Rank {rank}: Forward pass successful, logits shape: {logits.shape}")
    except Exception as e:
        print(f"Rank {rank}: ERROR - Forward pass failed: {e}")
        succ = False

    current_platform.synchronize()
    dist.barrier()

    engine.destroy()

    if rank == 0 and output:
        write_result(output, succ)

    print(
        f"Rank {rank}: memory_efficient_load + LoRA test {'PASSED' if succ else 'FAILED'}"
    )


def main():
    parser = argparse.ArgumentParser(description="Test memory_efficient_load with LoRA")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the output result",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fsdp:d1t1c1",
        help="Backend allocation string for the model (e.g., 'fsdp:d1t1c1')",
    )
    args = parser.parse_args()

    test_memory_efficient_lora(alloc_mode=args.backend, output=args.output)


if __name__ == "__main__":
    main()
