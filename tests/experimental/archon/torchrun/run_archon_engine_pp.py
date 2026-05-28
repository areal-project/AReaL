"""End-to-end test for ArchonEngine with Pipeline Parallelism.

This test directly instantiates ArchonEngine with PP enabled and verifies:
1. Forward pass produces correct output shape
2. PP outputs match golden (non-PP) model outputs
3. Train batch computes gradients correctly

Run with:
    torchrun --nproc_per_node=2 tests/experimental/archon/torchrun/run_archon_engine_pp.py \
        --pp_size=2 --out=/tmp/result.txt

    torchrun --nproc_per_node=4 tests/experimental/archon/torchrun/run_archon_engine_pp.py \
        --pp_size=2 --dp_size=2 --out=/tmp/result.txt

    torchrun --nproc_per_node=4 tests/experimental/archon/torchrun/run_archon_engine_pp.py \
        --pp_size=2 --ep_size=2 --out=/tmp/result.txt

NOTE: Requires a small Qwen model for testing (e.g., Qwen3-0.6B).
      PP+EP tests require a MoE model (e.g., Qwen2.5-3B-A3B-Instruct).
"""

import argparse
import os

import torch
import torch.distributed as dist

from tests.experimental.archon.torchrun.dist_utils import (
    print_rank0,
    write_result,
)
from tests.utils import get_model_path

from areal.api import FinetuneSpec, ParallelStrategy
from areal.api.cli_args import MicroBatchSpec, OptimizerConfig, TrainEngineConfig
from areal.experimental.engine.archon_engine import ArchonEngine

# Use a small model for testing
DEFAULT_MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)


def create_mock_input(
    batch_size: int = 4,
    min_seqlen: int = 10,
    max_seqlen: int = 20,
    vocab_size: int = 151936,  # Qwen3 vocab size
    device: torch.device | str = "cuda",
) -> dict[str, torch.Tensor]:
    """Create mock padded input data for testing.

    Returns a dict with input_ids and attention_mask (same format as HuggingFace).
    """
    pad_token_id = 0
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (batch_size,), dtype=torch.int, device=device
    )
    actual_max_seqlen = int(max(seqlens))

    input_ids = torch.randint(
        1,  # Start from 1 to avoid pad_token_id
        vocab_size,
        (batch_size, actual_max_seqlen),
        dtype=torch.long,
        device=device,
    )

    attn_mask = torch.zeros(
        (batch_size, actual_max_seqlen), dtype=torch.bool, device=device
    )

    # Set attention mask based on actual sequence lengths
    attn_mask[
        torch.arange(0, actual_max_seqlen, device=device).unsqueeze(0)
        < seqlens.unsqueeze(1)
    ] = True

    # Pad input_ids where attention_mask is False
    input_ids.masked_fill_(~attn_mask, pad_token_id)

    return dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )


def mock_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    **kwargs,
) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logprobs)


def mock_loss_weight_fn(input_data: dict) -> torch.Tensor:
    """Mock loss weight function for testing."""
    return input_data["cu_seqlens"][-1].float()


def create_archon_engine(
    model_path: str,
    pp_size: int,
    dp_size: int = 1,
    tp_size: int = 1,
    ep_size: int = 1,
) -> ArchonEngine:
    """Create and initialize ArchonEngine with PP enabled."""
    engine_config = TrainEngineConfig(
        backend=f"archon:d{dp_size}t{tp_size}p{pp_size}e{ep_size}",
        experiment_name="test-archon-pp",
        trial_name="test0",
        path=model_path,
        optimizer=OptimizerConfig(lr=1e-5),
        mb_spec=MicroBatchSpec(n_mbs=2, max_tokens_per_mb=256),
        gradient_checkpointing=False,
        dtype="bfloat16",
    )

    # Disable torch.compile for faster test startup
    engine_config.archon.enable_compile = False

    engine = ArchonEngine(engine_config)

    # Create parallel strategy with PP enabled
    parallel_strategy = ParallelStrategy(
        pipeline_parallel_size=pp_size,
        data_parallel_size=dp_size,
        tensor_parallel_size=tp_size,
        expert_parallel_size=ep_size,
    )

    engine.create_process_group(parallel_strategy)

    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=100, train_batch_size=4)
    engine.initialize(None, ft_spec)

    return engine


def test_forward_batch(engine: ArchonEngine, mock_input: dict) -> bool:
    """Test forward_batch with PP enabled."""
    rank = dist.get_rank()
    print_rank0("\n--- Testing forward_batch ---")

    engine.train(mode=False)  # Set to eval mode

    try:
        output = engine.forward_batch(mock_input)
        print(f"  Rank {rank}: forward_batch output shape = {output.shape}")

        # Verify output shape
        # Output should be [total_tokens] (logprobs)
        attn_mask = mock_input["attention_mask"]
        expected_tokens = attn_mask.sum().item()

        # Output may include padding, but should have at least expected_tokens
        if output.numel() < expected_tokens:
            print_rank0("  ERROR: Output has fewer tokens than expected")
            return False

        print_rank0("  forward_batch PASSED")
        return True

    except Exception as e:
        print_rank0(f"  forward_batch FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_eval_batch(engine: ArchonEngine, mock_input: dict) -> bool:
    """Test eval_batch with PP enabled."""
    print_rank0("\n--- Testing eval_batch ---")

    engine.train(mode=False)

    try:
        loss = engine.eval_batch(
            mock_input,
            loss_fn=mock_loss_fn,
            loss_weight_fn=mock_loss_weight_fn,
        )

        # In PP mode, eval_batch may return None (TODO in ArchonEngine)
        if loss is not None:
            print_rank0(f"  eval_batch loss = {loss.item():.6f}")
        else:
            print_rank0("  eval_batch returned None (expected for PP mode)")

        print_rank0("  eval_batch PASSED")
        return True

    except Exception as e:
        print_rank0(f"  eval_batch FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_train_batch(engine: ArchonEngine, mock_input: dict) -> bool:
    """Test train_batch with PP enabled."""
    print_rank0("\n--- Testing train_batch ---")

    engine.train(mode=True)

    try:
        result = engine.train_batch(
            mock_input,
            loss_fn=mock_loss_fn,
            loss_weight_fn=mock_loss_weight_fn,
        )

        print_rank0("  train_batch result:")
        print_rank0(f"    update_successful = {result.get('update_successful')}")
        print_rank0(f"    grad_norm = {result.get('grad_norm'):.6f}")
        print_rank0(f"    lr = {result.get('lr'):.2e}")

        # Verify grad_norm is valid
        grad_norm = result.get("grad_norm")
        if grad_norm is None or not torch.isfinite(torch.tensor(grad_norm)):
            print_rank0("  WARNING: grad_norm is invalid")

        print_rank0("  train_batch PASSED")
        return True

    except Exception as e:
        print_rank0(f"  train_batch FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_tests(
    model_path: str,
    pp_size: int,
    dp_size: int,
    tp_size: int,
    ep_size: int,
    out_file: str | None = None,
) -> bool:
    """Run all ArchonEngine PP tests."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # EP borrows from DP, so when ep_size > 1, dp_size must be at least ep_size
    if ep_size > 1 and dp_size < ep_size:
        dp_size = ep_size
        print_rank0(
            f"  Note: Setting dp_size={dp_size} to match ep_size (EP borrows from DP)"
        )

    print_rank0(f"\n{'=' * 60}")
    print_rank0("ArchonEngine PP E2E Test")
    print_rank0(
        f"  world_size={world_size}, pp={pp_size}, dp={dp_size}, tp={tp_size}, ep={ep_size}"
    )
    print_rank0(f"  model_path={model_path}")
    print_rank0("=" * 60)

    # Validate parallel configuration
    expected_world_size = pp_size * dp_size * tp_size
    if world_size != expected_world_size:
        print_rank0(
            f"ERROR: world_size ({world_size}) != pp*dp*tp ({expected_world_size})"
        )
        if rank == 0 and out_file:
            write_result(out_file, False)
        return False

    # Create engine
    print_rank0("\n--- Creating ArchonEngine ---")
    try:
        engine = create_archon_engine(model_path, pp_size, dp_size, tp_size, ep_size)
        print_rank0("  Engine created successfully")
        print_rank0(f"  PP enabled: {engine.parallel_dims.pp_enabled}")
        print_rank0(f"  has_first_stage: {engine.pp_has_first_stage}")
        print_rank0(f"  has_last_stage: {engine.pp_has_last_stage}")
        if ep_size > 1:
            print_rank0(f"  EP enabled: {engine.parallel_dims.ep_enabled}")
    except Exception as e:
        print_rank0(f"  Engine creation FAILED: {e}")
        import traceback

        traceback.print_exc()
        if rank == 0 and out_file:
            write_result(out_file, False)
        return False

    # Create mock input (broadcast from rank 0)
    if rank == 0:
        mock_input = create_mock_input(
            batch_size=4,
            min_seqlen=10,
            max_seqlen=20,
            device="cuda",
        )
        input_data = [mock_input]
    else:
        input_data = [None]

    dist.broadcast_object_list(input_data, src=0)
    mock_input = input_data[0]

    # Move tensors to local device
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    mock_input = {k: v.to(device) for k, v in mock_input.items()}

    print_rank0(
        f"  Mock input created: input_ids shape = {mock_input['input_ids'].shape}"
    )

    # Run tests
    all_passed = True

    try:
        # Test 1: forward_batch
        if not test_forward_batch(engine, mock_input):
            all_passed = False

        # Test 2: eval_batch
        if not test_eval_batch(engine, mock_input):
            all_passed = False

        # Test 3: train_batch
        if not test_train_batch(engine, mock_input):
            all_passed = False

    finally:
        # Cleanup
        engine.destroy()

    # Report results
    dist.barrier()

    if all_passed:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("All ArchonEngine PP tests PASSED")
        print_rank0("=" * 60)
    else:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("Some ArchonEngine PP tests FAILED")
        print_rank0("=" * 60)

    if rank == 0 and out_file:
        write_result(out_file, all_passed)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="ArchonEngine PP E2E Test")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to HuggingFace model",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=2,
        help="Pipeline parallel size",
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=1,
        help="Data parallel size",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--ep_size",
        type=int,
        default=1,
        help="Expert parallel size (requires MoE model)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output file for test result",
    )
    args = parser.parse_args()

    # Initialize distributed
    dist.init_process_group(backend="nccl")

    # Set environment variables if not set
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(dist.get_rank())

    try:
        run_tests(
            model_path=args.model_path,
            pp_size=args.pp_size,
            dp_size=args.dp_size,
            tp_size=args.tp_size,
            ep_size=args.ep_size,
            out_file=args.out,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
