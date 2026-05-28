#!/usr/bin/env python3
"""Unified PP test entry point for low-level PP tests.

This script consolidates PP forward and backward tests into a single file,
following the pattern of run_ep_tests.py.

Run with:
    torchrun --nproc_per_node=2 tests/experimental/archon/torchrun/run_pp_tests.py \
        --test_type=forward_p2p --pp_size=2 --output=/tmp/result.out

    torchrun --nproc_per_node=2 tests/experimental/archon/torchrun/run_pp_tests.py \
        --test_type=forward_schedule --pp_size=2 --pp_schedule=ZBVZeroBubble --output=/tmp/result.out

Supported test types:
    - forward_p2p: Test PP forward via manual activation passing (1F1B only)
    - forward_schedule: Test PP forward via schedule.eval() API (all schedules)
    - backward_p2p: Test PP backward via manual gradient passing (1F1B only)
    - backward_schedule: Test PP backward via schedule.step() API (all schedules)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    ScheduleDualPipeV,
    ScheduleZBVZeroBubble,
    get_schedule_class,
)

from tests.experimental.archon.torchrun.dist_utils import (
    create_dense_model_args,
    create_golden_model,
    create_test_input,
    print_rank0,
    verify_outputs_match,
    write_result,
)

from areal.experimental.models.archon import ArchonParallelDims
from areal.experimental.models.archon.pipeline_parallel import (
    build_pipeline_schedule,
    generate_llm_fqn_per_model_part,
    pipeline_module_split,
)
from areal.experimental.models.archon.qwen3 import Qwen3Model

# =============================================================================
# Shared Test Utilities
# =============================================================================


@dataclass
class _PPTestContext:
    """Shared state for PP tests."""

    rank: int
    world_size: int
    device: torch.device
    model_args: object  # Qwen3ModelArgs
    num_stages: int
    pp_schedule: str
    pp_stages: list
    model_parts: list[nn.Module]
    pp_local_rank: int
    has_first: bool
    has_last: bool
    pp_group: object  # ProcessGroup
    pp_group_ranks: list[int]


def _setup_pp_context(pp_size: int, pp_schedule: str) -> _PPTestContext:
    """Common setup: create model, split into PP stages (weights on meta device).

    Args:
        pp_size: Pipeline parallel degree (must equal world_size)
        pp_schedule: Schedule type (e.g., "1F1B", "ZBVZeroBubble")

    Returns:
        _PPTestContext with model parts on meta device (not yet materialized)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    assert world_size == pp_size, (
        f"This test requires world_size == pp_size. "
        f"Got world_size={world_size}, pp_size={pp_size}"
    )

    schedule_class = get_schedule_class(pp_schedule)
    is_multi_stage = issubclass(schedule_class, PipelineScheduleMulti)
    num_stages = pp_size * 2 if is_multi_stage else pp_size
    n_layers = num_stages * 2  # at least 2 layers per stage

    model_args = create_dense_model_args(
        n_layers=n_layers,
        dim=64,
        hidden_dim=128,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        vocab_size=1000,
    )

    parallel_dims = ArchonParallelDims(
        pp=pp_size,
        dp_shard=1,
        tp=1,
        cp=1,
        world_size=world_size,
        device_type="cuda",
    )

    print_rank0(f"  PP enabled: {parallel_dims.pp_enabled}")
    print_rank0(f"  PP degree: {parallel_dims.pp}")

    with torch.device("meta"):
        model = Qwen3Model(model_args)

    pp_mesh = parallel_dims.get_mesh("pp")
    module_names_per_stage = generate_llm_fqn_per_model_part(
        num_stages=num_stages,
        num_layers=n_layers,
    )
    print_rank0(f"  Module names per stage: {module_names_per_stage}")

    pp_stages, model_parts = pipeline_module_split(
        whole_model=model,
        pp_mesh=pp_mesh,
        pp_schedule=pp_schedule,
        device=device,
        module_names_per_stage=module_names_per_stage,
    )

    pp_local_rank = pp_mesh.get_local_rank()
    has_first = any(s.is_first for s in pp_stages)
    has_last = any(s.is_last for s in pp_stages)
    pp_group = parallel_dims.get_group("pp")
    pp_group_ranks = dist.get_process_group_ranks(pp_group)

    print(
        f"  Rank {rank}: pp_local_rank={pp_local_rank}, "
        f"has_first={has_first}, has_last={has_last}"
    )

    return _PPTestContext(
        rank=rank,
        world_size=world_size,
        device=device,
        model_args=model_args,
        num_stages=num_stages,
        pp_schedule=pp_schedule,
        pp_stages=pp_stages,
        model_parts=model_parts,
        pp_local_rank=pp_local_rank,
        has_first=has_first,
        has_last=has_last,
        pp_group=pp_group,
        pp_group_ranks=pp_group_ranks,
    )


def _broadcast_inputs(
    rank: int,
    device: torch.device,
    vocab_size: int,
    with_labels: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor | None]:
    """Create test inputs on rank 0 and broadcast to all ranks."""
    if rank == 0:
        tokens, positions, cu_seqlens, max_seqlen = create_test_input(
            num_seqs=2,
            seq_len_per_seq=8,
            vocab_size=vocab_size,
            device=device,
            seed=123,
        )
        labels = tokens.clone() if with_labels else None
        input_data = [tokens, positions, cu_seqlens, max_seqlen, labels]
    else:
        input_data = [None, None, None, None, None]

    dist.broadcast_object_list(input_data, src=0)
    tokens, positions, cu_seqlens, max_seqlen, labels = input_data

    tokens = tokens.to(device)
    positions = positions.to(device)
    cu_seqlens = cu_seqlens.to(device)
    if labels is not None:
        labels = labels.to(device)

    return tokens, positions, cu_seqlens, max_seqlen, labels


def _materialize_from_golden(
    model_parts: list[nn.Module],
    golden_model: nn.Module,
    device: torch.device,
) -> None:
    """Copy weights from golden model to PP model parts."""
    golden_state = golden_model.state_dict()
    for part in model_parts:
        part.to_empty(device=device)
        part.init_buffers(buffer_device=device)
        part_state = part.state_dict()
        for key in part_state:
            if key in golden_state:
                part_state[key].copy_(golden_state[key])
        part.load_state_dict(part_state)


def _materialize_random(
    model_parts: list[nn.Module],
    device: torch.device,
    seed: int = 42,
) -> None:
    """Initialize model parts with random weights."""
    torch.manual_seed(seed)
    for part in model_parts:
        part.to_empty(device=device)
        part.init_weights()
        part.init_buffers(buffer_device=device)


def _get_last_stage_rank(pp_schedule: str, pp_group_ranks: list[int]) -> int:
    """Get the global rank holding the last pipeline stage."""
    schedule_class = get_schedule_class(pp_schedule)
    v_style = schedule_class in (ScheduleZBVZeroBubble, ScheduleDualPipeV)
    return pp_group_ranks[0] if v_style else pp_group_ranks[-1]


def _report_test_result(
    test_name: str, success: bool, rank: int, out_file: str | None
) -> None:
    """Print test result and optionally write to file."""
    status = "PASSED" if success else "FAILED"
    print_rank0(f"\n{'=' * 60}")
    print_rank0(f"{test_name}: {status}")
    print_rank0("=" * 60)
    if rank == 0 and out_file:
        write_result(out_file, success)


def validate_gradients_pp(model_parts: list[nn.Module]) -> tuple[bool, list[str]]:
    """Verify gradients flow through PP model parts.

    Args:
        model_parts: List of model parts from pipeline_module_split

    Returns:
        Tuple of (success, error_messages)
    """
    errors = []

    for part_idx, part in enumerate(model_parts):
        for name, param in part.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    errors.append(f"Part {part_idx} {name}: no gradient")
                elif torch.isnan(param.grad).any():
                    errors.append(f"Part {part_idx} {name}: NaN gradient")
                elif torch.isinf(param.grad).any():
                    errors.append(f"Part {part_idx} {name}: Inf gradient")
                elif param.grad.abs().sum() == 0:
                    errors.append(f"Part {part_idx} {name}: zero gradient")

    return len(errors) == 0, errors


def _validate_and_gather_gradients(
    ctx: _PPTestContext,
) -> bool:
    """Validate gradients on this rank and gather results from all ranks."""
    local_success, local_errors = validate_gradients_pp(ctx.model_parts)

    print(
        f"  Rank {ctx.rank}: gradient validation "
        f"{'PASSED' if local_success else 'FAILED'}"
    )
    if not local_success:
        for err in local_errors[:5]:
            print(f"    {err}")
    else:
        total_grad_norm = sum(
            p.grad.norm().item()
            for m in ctx.model_parts
            for p in m.parameters()
            if p.grad is not None
        )
        print(f"  Rank {ctx.rank}: Total gradient norm: {total_grad_norm:.4f}")

    # Gather results from all ranks
    all_results = [None] * ctx.world_size
    dist.all_gather_object(all_results, (local_success, local_errors))
    final_success = all(success for success, _ in all_results)

    if ctx.rank == 0:
        print_rank0("\n  Results by rank:")
        for r, (success, errors) in enumerate(all_results):
            status = "PASSED" if success else "FAILED"
            print_rank0(f"    Rank {r}: {status}")
            if not success:
                for err in errors[:3]:
                    print_rank0(f"      - {err}")

    return final_success


# =============================================================================
# Forward Tests
# =============================================================================


def test_pp_forward_p2p(pp_size: int, out_file: str | None = None) -> bool:
    """Test PP forward pass via manual activation passing (1F1B only).

    Verifies that a model split across PP stages produces the same output
    as the non-split golden model, using manual point-to-point communication
    to pass activations between stages.

    Args:
        pp_size: Pipeline parallel degree (must equal world_size)
        out_file: Optional file to write result ("Passed"/"Failed")

    Returns:
        True if test passed, False otherwise
    """
    ctx = _setup_pp_context(pp_size, "1F1B")

    print_rank0(f"\n{'=' * 60}")
    print_rank0(f"PP Forward P2P Test: pp_size={pp_size}")
    print_rank0("=" * 60)

    # Create golden model and copy weights
    golden_model = create_golden_model(ctx.model_args, ctx.device, seed=42)
    golden_model.eval()
    _materialize_from_golden(ctx.model_parts, golden_model, ctx.device)
    for part in ctx.model_parts:
        part.eval()

    # Create and broadcast inputs
    tokens, positions, cu_seqlens, max_seqlen, _ = _broadcast_inputs(
        ctx.rank,
        ctx.device,
        ctx.model_args.vocab_size,
    )
    print_rank0(f"  Input tokens shape: {tokens.shape}")

    # Run golden model forward
    with torch.no_grad():
        golden_output = golden_model(tokens, positions, cu_seqlens, max_seqlen)
    print_rank0(f"  Golden output shape: {golden_output.shape}")

    # Manual P2P forward through stages
    print_rank0(f"\n  Starting PP forward pass through {ctx.num_stages} stages...")
    pp_output = None

    with torch.no_grad():
        h = torch.zeros(
            (1, tokens.shape[1], ctx.model_args.dim),
            dtype=torch.float32,
            device=ctx.device,
        )

        for stage_idx in range(pp_size):
            src_rank = ctx.pp_group_ranks[stage_idx]
            is_my_stage = ctx.pp_local_rank == stage_idx

            if stage_idx == 0:
                if is_my_stage:
                    print(
                        f"  Rank {ctx.rank}: Stage {stage_idx} - processing input tokens"
                    )
                    h = ctx.model_parts[0](tokens, positions, cu_seqlens, max_seqlen)
                    print(
                        f"  Rank {ctx.rank}: Stage {stage_idx} - output shape: {h.shape}"
                    )
            else:
                if is_my_stage:
                    print(
                        f"  Rank {ctx.rank}: Stage {stage_idx} - processing received activation"
                    )
                    h = ctx.model_parts[0](h, positions, cu_seqlens, max_seqlen)
                    print(
                        f"  Rank {ctx.rank}: Stage {stage_idx} - output shape: {h.shape}"
                    )

            torch.cuda.synchronize()
            dist.barrier(group=ctx.pp_group)

            if stage_idx < pp_size - 1:
                dist.broadcast(h, src=src_rank, group=ctx.pp_group)
                print_rank0(
                    f"  Broadcast activation from stage {stage_idx} (rank {src_rank})"
                )

        if ctx.has_last:
            pp_output = h

    # Compare outputs
    torch.cuda.synchronize()
    dist.barrier()

    success = True
    if ctx.has_last:
        success, max_diff, mean_diff = verify_outputs_match(
            pp_output, golden_output, rtol=1e-4, atol=1e-4, max_diff_threshold=1e-2
        )
        print_rank0("\n  Comparison results:")
        print_rank0(f"    PP output shape: {pp_output.shape}")
        print_rank0(f"    Golden output shape: {golden_output.shape}")
        print_rank0(f"    Max diff: {max_diff:.6f}")
        print_rank0(f"    Mean diff: {mean_diff:.6f}")
        print_rank0(f"    Outputs match: {success}")

        if not success:
            print_rank0(
                f"    PP output stats: min={pp_output.min():.4f}, "
                f"max={pp_output.max():.4f}, mean={pp_output.mean():.4f}"
            )
            print_rank0(
                f"    Golden stats: min={golden_output.min():.4f}, "
                f"max={golden_output.max():.4f}, mean={golden_output.mean():.4f}"
            )

    # Broadcast result from last rank (always last rank for 1F1B)
    last_stage_rank = ctx.pp_group_ranks[-1]
    success_tensor = torch.tensor(
        [1 if success else 0], dtype=torch.int, device=ctx.device
    )
    dist.broadcast(success_tensor, src=last_stage_rank)
    success = success_tensor.item() == 1

    dist.barrier()
    _report_test_result("PP Forward P2P Test", success, ctx.rank, out_file)
    return success


def test_pp_forward_schedule(
    pp_size: int, out_file: str | None = None, pp_schedule: str = "1F1B"
) -> bool:
    """Test PP forward pass via schedule.eval() API (all schedules).

    Verifies that the PP model produces correct output when driven by the
    pipeline schedule API. Works with any supported schedule type.

    Args:
        pp_size: Pipeline parallel degree (must equal world_size)
        out_file: Optional file to write result ("Passed"/"Failed")
        pp_schedule: Pipeline schedule type (e.g., "1F1B", "ZBVZeroBubble")

    Returns:
        True if test passed, False otherwise
    """
    ctx = _setup_pp_context(pp_size, pp_schedule)

    print_rank0(f"\n{'=' * 60}")
    print_rank0(f"PP Forward Schedule Test: pp_size={pp_size}, schedule={pp_schedule}")
    print_rank0("=" * 60)

    # Create golden model and copy weights
    golden_model = create_golden_model(ctx.model_args, ctx.device, seed=42)
    golden_model.eval()
    _materialize_from_golden(ctx.model_parts, golden_model, ctx.device)
    for part in ctx.model_parts:
        part.eval()

    # Create and broadcast inputs
    tokens, positions, cu_seqlens, max_seqlen, _ = _broadcast_inputs(
        ctx.rank,
        ctx.device,
        ctx.model_args.vocab_size,
    )
    print_rank0(f"  Input tokens shape: {tokens.shape}")

    # Run golden model forward
    with torch.no_grad():
        golden_output = golden_model(tokens, positions, cu_seqlens, max_seqlen)
    print_rank0(f"  Golden output shape: {golden_output.shape}")

    # Build schedule and run eval
    print_rank0(f"\n  Starting PP forward pass through {ctx.num_stages} stages...")
    n_microbatches = max(ctx.num_stages, 2)
    schedule = build_pipeline_schedule(
        ctx.pp_stages,
        pp_schedule,
        n_microbatches,
        pp_degree=pp_size,
    )

    pp_output = None
    with torch.no_grad():
        batched_positions = positions.expand(n_microbatches, -1)
        batched_cu_seqlens = cu_seqlens.unsqueeze(0).expand(n_microbatches, -1)
        batched_max_seqlen = torch.tensor(
            [max_seqlen] * n_microbatches,
            device=ctx.device,
        )
        batched_kwargs = {
            "positions": batched_positions,
            "cu_seqlens": batched_cu_seqlens,
            "max_seqlen": batched_max_seqlen,
        }

        if ctx.has_first:
            batched_input = tokens.expand(n_microbatches, -1)
            outputs = schedule.eval(batched_input, **batched_kwargs)
        else:
            outputs = schedule.eval(**batched_kwargs)

        if ctx.has_last:
            assert outputs is not None, (
                "Last stage should produce outputs from schedule.eval()"
            )
            # Take first microbatch output for comparison
            pp_output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

    # Compare outputs
    torch.cuda.synchronize()
    dist.barrier()

    success = True
    if ctx.has_last:
        success, max_diff, mean_diff = verify_outputs_match(
            pp_output, golden_output, rtol=1e-4, atol=1e-4, max_diff_threshold=1e-2
        )
        print_rank0("\n  Comparison results:")
        print_rank0(f"    PP output shape: {pp_output.shape}")
        print_rank0(f"    Golden output shape: {golden_output.shape}")
        print_rank0(f"    Max diff: {max_diff:.6f}")
        print_rank0(f"    Mean diff: {mean_diff:.6f}")
        print_rank0(f"    Outputs match: {success}")

        if not success:
            print_rank0(
                f"    PP output stats: min={pp_output.min():.4f}, "
                f"max={pp_output.max():.4f}, mean={pp_output.mean():.4f}"
            )
            print_rank0(
                f"    Golden stats: min={golden_output.min():.4f}, "
                f"max={golden_output.max():.4f}, mean={golden_output.mean():.4f}"
            )

    # Broadcast result from last stage rank
    last_stage_rank = _get_last_stage_rank(pp_schedule, ctx.pp_group_ranks)
    success_tensor = torch.tensor(
        [1 if success else 0], dtype=torch.int, device=ctx.device
    )
    dist.broadcast(success_tensor, src=last_stage_rank)
    success = success_tensor.item() == 1

    dist.barrier()
    _report_test_result("PP Forward Schedule Test", success, ctx.rank, out_file)
    return success


# =============================================================================
# Backward Tests
# =============================================================================


def test_pp_backward_p2p(pp_size: int, out_file: str | None = None) -> bool:
    """Test PP backward pass via manual gradient passing (1F1B only).

    Verifies that gradients flow correctly through all PP stages using
    manual point-to-point communication for activation and gradient passing.

    Args:
        pp_size: Pipeline parallel degree (must equal world_size)
        out_file: Optional file to write result ("Passed"/"Failed")

    Returns:
        True if test passed, False otherwise
    """
    ctx = _setup_pp_context(pp_size, "1F1B")

    print_rank0(f"\n{'=' * 60}")
    print_rank0(f"PP Backward P2P Test: pp_size={pp_size}")
    print_rank0("=" * 60)

    # Random weight init
    _materialize_random(ctx.model_parts, ctx.device)
    for part in ctx.model_parts:
        part.train()

    # Create optimizer
    all_params = [p for m in ctx.model_parts for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(all_params, lr=1e-4)
    optimizer.zero_grad()
    print_rank0(f"  Trainable parameters: {len(all_params)}")

    # Create and broadcast inputs with labels
    tokens, positions, cu_seqlens, max_seqlen, labels = _broadcast_inputs(
        ctx.rank,
        ctx.device,
        ctx.model_args.vocab_size,
        with_labels=True,
    )
    print_rank0(f"  Input tokens shape: {tokens.shape}")
    print_rank0(f"  Labels shape: {labels.shape}")

    # Manual P2P forward + backward
    print_rank0(f"\n  Starting PP forward+backward through {ctx.num_stages} stages...")

    h_input = None
    h_output = None
    output = None

    h_buffer = torch.zeros(
        (1, tokens.shape[1], ctx.model_args.dim),
        dtype=torch.float32,
        device=ctx.device,
    )

    # Forward pass - sequential through all stages
    for stage_idx in range(pp_size):
        is_my_stage = ctx.pp_local_rank == stage_idx
        src_rank = ctx.pp_group_ranks[stage_idx]

        if stage_idx == 0:
            if is_my_stage:
                print(
                    f"  Rank {ctx.rank}: Stage {stage_idx} - forward with input tokens"
                )
                h_output = ctx.model_parts[0](tokens, positions, cu_seqlens, max_seqlen)
                h_output.retain_grad()
                h_buffer = h_output.detach().clone()
                print(
                    f"  Rank {ctx.rank}: Stage {stage_idx} - output shape: {h_output.shape}"
                )
        else:
            if is_my_stage:
                h_input = h_buffer.clone().requires_grad_(True)
                print(
                    f"  Rank {ctx.rank}: Stage {stage_idx} - forward with received activation"
                )
                h_output = ctx.model_parts[0](
                    h_input, positions, cu_seqlens, max_seqlen
                )
                h_output.retain_grad()
                h_buffer = h_output.detach().clone()
                print(
                    f"  Rank {ctx.rank}: Stage {stage_idx} - output shape: {h_output.shape}"
                )

                if ctx.pp_local_rank == pp_size - 1:
                    output = h_output

        torch.cuda.synchronize()
        dist.barrier(group=ctx.pp_group)

        if stage_idx < pp_size - 1:
            dist.broadcast(h_buffer, src=src_rank, group=ctx.pp_group)
            print_rank0(
                f"  Broadcast activation from stage {stage_idx} (rank {src_rank})"
            )

    # Compute loss and start backward (only on last stage)
    print_rank0("\n  Starting PP backward pass...")

    grad_buffer = torch.zeros(
        (1, tokens.shape[1], ctx.model_args.dim),
        dtype=torch.float32,
        device=ctx.device,
    )

    if ctx.pp_local_rank == pp_size - 1:
        print(f"  Rank {ctx.rank}: Computing loss on last stage")
        output_flat = output.view(-1, output.size(-1))
        labels_flat = labels.view(-1)
        loss = nn.functional.cross_entropy(output_flat, labels_flat)
        print(f"  Rank {ctx.rank}: Loss = {loss.item():.4f}")

        loss.backward()

        if h_input is not None and h_input.grad is not None:
            grad_buffer = h_input.grad.detach().clone()
            print(
                f"  Rank {ctx.rank}: Gradient norm to send: {grad_buffer.norm().item():.4f}"
            )
        else:
            print(
                f"  Rank {ctx.rank}: WARNING - h_input.grad is None "
                "(this is expected for pp_size=1)"
            )

    # Backward pass - reverse sequential through all stages
    for stage_idx in range(pp_size - 1, 0, -1):
        src_rank = ctx.pp_group_ranks[stage_idx]

        torch.cuda.synchronize()
        dist.barrier(group=ctx.pp_group)
        dist.broadcast(grad_buffer, src=src_rank, group=ctx.pp_group)
        print_rank0(f"  Broadcast gradient from stage {stage_idx} (rank {src_rank})")

        if ctx.pp_local_rank == stage_idx - 1:
            print(f"  Rank {ctx.rank}: Stage {ctx.pp_local_rank} - backward pass")
            print(
                f"  Rank {ctx.rank}: Received gradient norm: {grad_buffer.norm().item():.4f}"
            )

            h_output.backward(grad_buffer)

            if (
                ctx.pp_local_rank > 0
                and h_input is not None
                and h_input.grad is not None
            ):
                grad_buffer = h_input.grad.detach().clone()
                print(
                    f"  Rank {ctx.rank}: Gradient norm to send: {grad_buffer.norm().item():.4f}"
                )

    torch.cuda.synchronize()
    dist.barrier()

    # Validate gradients
    final_success = _validate_and_gather_gradients(ctx)

    dist.barrier()
    _report_test_result("PP Backward P2P Test", final_success, ctx.rank, out_file)
    return final_success


def test_pp_backward_schedule(
    pp_size: int, out_file: str | None = None, pp_schedule: str = "1F1B"
) -> bool:
    """Test PP backward pass via schedule.step() API (all schedules).

    Verifies that gradients flow correctly through all PP stages when
    driven by the pipeline schedule API. Works with any supported schedule type.

    Args:
        pp_size: Pipeline parallel degree (must equal world_size)
        out_file: Optional file to write result ("Passed"/"Failed")
        pp_schedule: Pipeline schedule type (e.g., "1F1B", "ZBVZeroBubble")

    Returns:
        True if test passed, False otherwise
    """
    ctx = _setup_pp_context(pp_size, pp_schedule)

    print_rank0(f"\n{'=' * 60}")
    print_rank0(f"PP Backward Schedule Test: pp_size={pp_size}, schedule={pp_schedule}")
    print_rank0("=" * 60)

    # Random weight init
    _materialize_random(ctx.model_parts, ctx.device)
    for part in ctx.model_parts:
        part.train()

    # Create optimizer
    all_params = [p for m in ctx.model_parts for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(all_params, lr=1e-4)
    optimizer.zero_grad()
    print_rank0(f"  Trainable parameters: {len(all_params)}")

    # Create and broadcast inputs with labels
    tokens, positions, cu_seqlens, max_seqlen, labels = _broadcast_inputs(
        ctx.rank,
        ctx.device,
        ctx.model_args.vocab_size,
        with_labels=True,
    )
    print_rank0(f"  Input tokens shape: {tokens.shape}")
    print_rank0(f"  Labels shape: {labels.shape}")

    # Build schedule and run step
    print_rank0(f"\n  Starting PP forward+backward through {ctx.num_stages} stages...")

    n_microbatches = max(ctx.num_stages, 2)

    def loss_fn(output, target):
        output_flat = output.view(-1, output.size(-1))
        target_flat = target.view(-1)
        return nn.functional.cross_entropy(output_flat, target_flat)

    schedule = build_pipeline_schedule(
        ctx.pp_stages,
        pp_schedule,
        n_microbatches,
        pp_degree=pp_size,
        loss_fn=loss_fn,
    )

    batched_positions = positions.expand(n_microbatches, -1)
    batched_cu_seqlens = cu_seqlens.unsqueeze(0).expand(n_microbatches, -1)
    batched_max_seqlen = torch.tensor(
        [max_seqlen] * n_microbatches,
        device=ctx.device,
    )
    batched_kwargs = {
        "positions": batched_positions,
        "cu_seqlens": batched_cu_seqlens,
        "max_seqlen": batched_max_seqlen,
    }

    # target is only used on the last stage for loss computation;
    # non-last ranks pass None and the schedule ignores it.
    target = labels.expand(n_microbatches, -1) if ctx.has_last else None

    if ctx.has_first:
        schedule.step(
            tokens.expand(n_microbatches, -1),
            target=target,
            **batched_kwargs,
        )
    else:
        schedule.step(
            target=target,
            **batched_kwargs,
        )

    torch.cuda.synchronize()
    dist.barrier()

    # Validate gradients
    final_success = _validate_and_gather_gradients(ctx)

    dist.barrier()
    _report_test_result("PP Backward Schedule Test", final_success, ctx.rank, out_file)
    return final_success


# =============================================================================
# Test Registry and Main
# =============================================================================

TEST_REGISTRY = {
    "forward_p2p": test_pp_forward_p2p,
    "forward_schedule": test_pp_forward_schedule,
    "backward_p2p": test_pp_backward_p2p,
    "backward_schedule": test_pp_backward_schedule,
}


def main():
    parser = argparse.ArgumentParser(description="Unified PP Tests")
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        choices=list(TEST_REGISTRY.keys()),
        help="Type of test to run",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=2,
        help="Pipeline parallel size (must equal world_size)",
    )
    parser.add_argument(
        "--pp_schedule",
        type=str,
        default="1F1B",
        help="Pipeline parallel schedule (for schedule tests only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for test result",
    )
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    print_rank0("=" * 60)
    print_rank0(f"Running PP Test: {args.test_type}")
    print_rank0("=" * 60)

    try:
        test_fn = TEST_REGISTRY[args.test_type]

        # p2p tests are 1F1B-only; schedule tests accept pp_schedule
        if args.test_type.endswith("_schedule"):
            success = test_fn(args.pp_size, args.output, pp_schedule=args.pp_schedule)
        else:
            success = test_fn(args.pp_size, args.output)

        dist.barrier()

        if success:
            print_rank0(f"\n{'=' * 60}")
            print_rank0(f"PP Test {args.test_type}: PASSED")
            print_rank0("=" * 60)
        else:
            print_rank0(f"\n{'=' * 60}")
            print_rank0(f"PP Test {args.test_type}: FAILED")
            print_rank0("=" * 60)

    except Exception as e:
        print(f"Rank {rank} failed with: {e}")
        import traceback

        traceback.print_exc()
        if rank == 0 and args.output:
            write_result(args.output, False)
        raise

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
