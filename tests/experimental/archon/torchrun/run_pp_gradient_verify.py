#!/usr/bin/env python3
"""PP vs non-PP gradient verification test.

This test verifies that the step() API produces identical gradients
to manual forward/backward passes without pipeline parallelism.

Usage:
    torchrun --nproc-per-node=2 run_pp_gradient_verify.py --output=/tmp/result.txt
    torchrun --nproc-per-node=2 run_pp_gradient_verify.py  # standalone debug
"""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import Schedule1F1B

from tests.experimental.archon.torchrun.dist_utils import (
    print_rank0,
    write_result,
)

# Config
VOCAB_SIZE = 1000
HIDDEN_SIZE = 256
N_LAYERS_PER_STAGE = 2
SEED = 42


class SimpleBlock(nn.Module):
    """A simple transformer-like block."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.norm(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.linear2(h)
        return x + h


class FirstStageModel(nn.Module):
    """First stage: embedding + first half of layers.

    Simulates Archon model with packed sequence inputs.
    """

    def __init__(self, vocab_size: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([SimpleBlock(hidden_size) for _ in range(n_layers)])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: [1, seq_len] or [seq_len] token ids
            positions: [1, seq_len] or [seq_len] position ids
            cu_seqlens: [1, B+1] or [B+1] cumulative sequence lengths
            max_seqlen: int or [1] tensor, max sequence length

        Returns:
            hidden states [1, seq_len, hidden_size]
        """
        # Note: positions/cu_seqlens/max_seqlen are accepted to match Archon's
        # interface but not used in this simplified test model (no flash attention)
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        return h


class LastStageModel(nn.Module):
    """Last stage: second half of layers + norm + output.

    Like Archon, this model also requires positions/cu_seqlens/max_seqlen
    for attention computation in its layers.
    """

    def __init__(self, vocab_size: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([SimpleBlock(hidden_size) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        h: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            h: hidden states from previous stage [1, seq_len, hidden_size]
            positions: [1, seq_len] position ids (for attention in layers)
            cu_seqlens: [1, B+1] cumulative sequence lengths
            max_seqlen: int or [1] tensor, max sequence length

        Returns:
            logits [1, seq_len, vocab_size]
        """
        # Note: positions/cu_seqlens/max_seqlen are accepted to match Archon's
        # interface but not used in this simplified test model (no flash attention)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.output(h)
        return logits


class FullModel(nn.Module):
    """Full model combining first and last stage (for non-PP and gradient verification)."""

    def __init__(self, vocab_size: int, hidden_size: int, n_layers_per_stage: int):
        super().__init__()
        self.first = FirstStageModel(vocab_size, hidden_size, n_layers_per_stage)
        self.last = LastStageModel(vocab_size, hidden_size, n_layers_per_stage)

    def forward(self, input_ids, positions, cu_seqlens, max_seqlen):
        h = self.first(input_ids, positions, cu_seqlens, max_seqlen)
        return self.last(h, positions, cu_seqlens, max_seqlen)


def generate_fixed_data(
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
    n_microbatches: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate fixed data for gradient verification (deterministic).

    Returns batched data for n_microbatches:
        input_ids: [n_mb, seq_len]
        positions: [n_mb, seq_len]
        cu_seqlens: [n_mb, max_cu_len] (padded)
        max_seqlen: [n_mb] CPU tensor
        labels: [n_mb, seq_len]
    """
    torch.manual_seed(seed)

    input_ids_list = []
    positions_list = []
    cu_seqlens_list = []
    max_seqlen_list = []
    labels_list = []

    for mb_idx in range(n_microbatches):
        # Different split for each microbatch
        n_seqs = 2 + mb_idx  # mb0: 2 seqs, mb1: 3 seqs
        seg_len = seq_len // n_seqs
        seq_lens = [seg_len] * (n_seqs - 1) + [seq_len - seg_len * (n_seqs - 1)]

        # cu_seqlens
        cu_seqlens = [0]
        for sl in seq_lens:
            cu_seqlens.append(cu_seqlens[-1] + sl)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

        # max_seqlen
        max_seqlen = max(seq_lens)

        # positions
        positions = []
        for sl in seq_lens:
            positions.extend(range(sl))
        positions = torch.tensor(positions, dtype=torch.long, device=device)

        # input_ids and labels (deterministic)
        input_ids = torch.randint(0, vocab_size, (seq_len,), device=device)
        labels = torch.randint(0, vocab_size, (seq_len,), device=device)

        input_ids_list.append(input_ids.unsqueeze(0))
        positions_list.append(positions.unsqueeze(0))
        cu_seqlens_list.append(cu_seqlens.unsqueeze(0))
        max_seqlen_list.append(max_seqlen)
        labels_list.append(labels.unsqueeze(0))

    # Pad cu_seqlens to same length
    max_cu_len = max(cs.shape[1] for cs in cu_seqlens_list)
    padded_cu_seqlens = []
    for cs in cu_seqlens_list:
        if cs.shape[1] < max_cu_len:
            pad_val = cs[0, -1]
            padding = pad_val.expand(1, max_cu_len - cs.shape[1])
            cs = torch.cat([cs, padding], dim=1)
        padded_cu_seqlens.append(cs)

    batched_input_ids = torch.cat(input_ids_list, dim=0)
    batched_positions = torch.cat(positions_list, dim=0)
    batched_cu_seqlens = torch.cat(padded_cu_seqlens, dim=0)
    batched_max_seqlen = torch.tensor(max_seqlen_list)  # CPU
    batched_labels = torch.cat(labels_list, dim=0)

    return (
        batched_input_ids,
        batched_positions,
        batched_cu_seqlens,
        batched_max_seqlen,
        batched_labels,
    )


def run_non_pp_forward_backward(
    model: FullModel,
    batched_input_ids: torch.Tensor,
    batched_positions: torch.Tensor,
    batched_cu_seqlens: torch.Tensor,
    batched_max_seqlen: torch.Tensor,
    batched_labels: torch.Tensor,
    vocab_size: int,
) -> tuple[dict[str, torch.Tensor], float]:
    """Run non-PP forward/backward on multiple microbatches and return accumulated gradients."""
    model.zero_grad()

    n_microbatches = batched_input_ids.shape[0]
    total_loss = 0.0

    for i in range(n_microbatches):
        input_ids = batched_input_ids[i : i + 1]  # [1, seq_len]
        positions = batched_positions[i : i + 1]  # [1, seq_len]
        cu_seqlens = batched_cu_seqlens[i]  # [cu_len] - squeeze batch dim
        max_seqlen = batched_max_seqlen[i].item()
        labels = batched_labels[i : i + 1]  # [1, seq_len]

        logits = model(input_ids, positions, cu_seqlens, max_seqlen)
        loss = nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
        )
        # Accumulate gradients (don't zero between microbatches)
        loss.backward()
        total_loss += loss.item()

    # Collect gradients
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.clone()

    return grads, total_loss


def compare_gradients(
    pp_grads: dict[str, torch.Tensor],
    non_pp_grads: dict[str, torch.Tensor],
    prefix: str,
    atol: float,
) -> tuple[bool, list[str]]:
    """Compare PP gradients against non-PP gradients.

    Args:
        pp_grads: Gradients from PP model (keys without prefix)
        non_pp_grads: Gradients from non-PP full model (keys with prefix)
        prefix: Prefix to add to pp_grads keys for matching (e.g., "first." or "last.")
        atol: Absolute tolerance for comparison

    Returns:
        Tuple of (all_match, error_messages)
    """
    errors = []
    all_match = True

    for name in pp_grads:
        non_pp_name = f"{prefix}{name}"
        if non_pp_name not in non_pp_grads:
            errors.append(f"{name}: not found in non-PP grads as {non_pp_name}")
            all_match = False
            continue

        pp_g = pp_grads[name]
        non_pp_g = non_pp_grads[non_pp_name]

        abs_diff = (pp_g - non_pp_g).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        rel_diff = (abs_diff / (non_pp_g.abs() + 1e-8)).max().item()

        if max_diff >= atol:
            errors.append(
                f"{name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, "
                f"rel_diff={rel_diff:.2e} (exceeds atol={atol})"
            )
            all_match = False

    return all_match, errors


def verify_gradients(
    n_microbatches: int = 2,
    seq_len: int = 64,
    atol: float = 1e-5,
) -> bool:
    """Verify PP gradients match non-PP gradients.

    Args:
        n_microbatches: Number of microbatches to use
        seq_len: Sequence length for test data
        atol: Absolute tolerance for gradient comparison

    Returns:
        True if all gradients match within tolerance.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    assert world_size == 2, "Gradient verification requires exactly 2 GPUs"

    print_rank0(f"\n{'=' * 60}")
    print_rank0("PP Gradient Verification Test")
    print_rank0(f"  n_microbatches={n_microbatches}, seq_len={seq_len}, atol={atol}")
    print_rank0("=" * 60)

    # 1. Create full model on rank 0, compute non-PP gradients
    torch.manual_seed(SEED)
    full_model = FullModel(VOCAB_SIZE, HIDDEN_SIZE, N_LAYERS_PER_STAGE).to(device)

    # Generate fixed data (n_microbatches samples)
    (
        batched_input_ids,
        batched_positions,
        batched_cu_seqlens,
        batched_max_seqlen,
        batched_labels,
    ) = generate_fixed_data(seq_len, VOCAB_SIZE, device, SEED + 100, n_microbatches)

    print_rank0(
        f"  Data shapes: input_ids={batched_input_ids.shape}, "
        f"cu_seqlens={batched_cu_seqlens.shape}, max_seqlen={batched_max_seqlen.shape}"
    )

    # Compute non-PP gradients (only rank 0, but both ranks need the model for state_dict)
    non_pp_grads: dict[str, torch.Tensor] = {}
    if rank == 0:
        non_pp_grads, non_pp_loss = run_non_pp_forward_backward(
            full_model,
            batched_input_ids,
            batched_positions,
            batched_cu_seqlens,
            batched_max_seqlen,
            batched_labels,
            VOCAB_SIZE,
        )
        print_rank0(f"  Non-PP loss: {non_pp_loss:.6f}")

    dist.barrier()

    # 2. Create PP stages with same initialization
    torch.manual_seed(SEED)  # Reset seed for same initialization
    if rank == 0:
        pp_model = FirstStageModel(VOCAB_SIZE, HIDDEN_SIZE, N_LAYERS_PER_STAGE).to(
            device
        )
        # Copy weights from full_model.first
        pp_model.load_state_dict(full_model.first.state_dict())
    else:
        pp_model = LastStageModel(VOCAB_SIZE, HIDDEN_SIZE, N_LAYERS_PER_STAGE).to(
            device
        )
        # Copy weights from full_model.last (need to broadcast from rank 0)

    # Broadcast last stage weights from rank 0
    if rank == 0:
        last_state = full_model.last.state_dict()
        for key in last_state:
            dist.broadcast(last_state[key], src=0)
    else:
        # Receive weights
        last_state = pp_model.state_dict()
        for key in last_state:
            dist.broadcast(last_state[key], src=0)
        pp_model.load_state_dict(last_state)

    pp_model.train()

    # Create PP stages
    pp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("pp",))
    pp_group = pp_mesh.get_group("pp")

    if rank == 0:
        stage = PipelineStage(
            pp_model, stage_index=0, num_stages=2, device=device, group=pp_group
        )
    else:
        stage = PipelineStage(
            pp_model, stage_index=1, num_stages=2, device=device, group=pp_group
        )

    # 3. Run PP forward/backward with n_microbatches
    def loss_fn(logits, labels):
        logits = logits.view(-1, VOCAB_SIZE)
        labels = labels.view(-1)
        return nn.functional.cross_entropy(logits, labels)

    pp_model.zero_grad()

    # Data is already batched with shape [n_microbatches, ...]
    # scale_grads=False to match non-PP gradient accumulation behavior
    schedule = Schedule1F1B(
        stage,
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
        scale_grads=False,
    )

    losses: list = []
    if rank == 0:
        schedule.step(
            batched_input_ids,
            positions=batched_positions,
            cu_seqlens=batched_cu_seqlens,
            max_seqlen=batched_max_seqlen,
            losses=losses,
        )
    else:
        schedule.step(
            target=batched_labels,
            losses=losses,
            positions=batched_positions,
            cu_seqlens=batched_cu_seqlens,
            max_seqlen=batched_max_seqlen,
        )

    # Collect PP gradients
    pp_grads = {}
    for name, p in pp_model.named_parameters():
        if p.grad is not None:
            pp_grads[name] = p.grad.clone()

    if rank == 1 and losses:
        pp_loss = sum(loss.item() for loss in losses) / len(losses)
        print(f"  [Rank 1] PP loss: {pp_loss:.6f}")

    # 4. Compare gradients
    # Rank 0 compares first stage gradients
    first_stage_success = True
    first_stage_errors: list[str] = []
    if rank == 0:
        first_stage_success, first_stage_errors = compare_gradients(
            pp_grads, non_pp_grads, "first.", atol
        )
        print_rank0("\n  First Stage Gradient Comparison:")
        if first_stage_success:
            print_rank0(f"    All {len(pp_grads)} parameters match within atol={atol}")
        else:
            for err in first_stage_errors[:5]:
                print_rank0(f"    MISMATCH: {err}")

    dist.barrier()

    # Rank 1 needs non-PP grads for last stage - broadcast from rank 0
    last_stage_success = True
    last_stage_errors: list[str] = []

    if rank == 0:
        # Send last stage grads to rank 1
        last_grad_names = [k for k in non_pp_grads if k.startswith("last.")]
        for name in last_grad_names:
            grad = non_pp_grads[name]
            dist.broadcast(grad, src=0)
    else:
        # Receive last stage grads from rank 0
        non_pp_last_grads: dict[str, torch.Tensor] = {}
        # We need to know the param names - use pp_model's params
        for name, p in pp_model.named_parameters():
            grad = torch.zeros_like(p)
            dist.broadcast(grad, src=0)
            non_pp_last_grads[f"last.{name}"] = grad

        last_stage_success, last_stage_errors = compare_gradients(
            pp_grads, non_pp_last_grads, "last.", atol
        )
        print("\n  [Rank 1] Last Stage Gradient Comparison:")
        if last_stage_success:
            print(f"    All {len(pp_grads)} parameters match within atol={atol}")
        else:
            for err in last_stage_errors[:5]:
                print(f"    MISMATCH: {err}")

    dist.barrier()

    # 5. Gather results from all ranks
    all_results = [None] * world_size
    local_result = (
        first_stage_success if rank == 0 else last_stage_success,
        first_stage_errors if rank == 0 else last_stage_errors,
    )
    dist.all_gather_object(all_results, local_result)

    # Check all stages passed
    success = all(r[0] for r in all_results)

    if success:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP Gradient Verification: PASSED")
        print_rank0("=" * 60)
    else:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP Gradient Verification: FAILED")
        for r, (_, errors) in enumerate(all_results):
            if errors:
                print_rank0(f"  Rank {r} errors:")
                for err in errors[:3]:
                    print_rank0(f"    - {err}")
        print_rank0("=" * 60)

    return success


def main():
    parser = argparse.ArgumentParser(
        description="PP vs non-PP gradient verification test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for test result (omit for standalone debug mode)",
    )
    parser.add_argument(
        "--n_microbatches",
        type=int,
        default=2,
        help="Number of microbatches (must be >= num_stages)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=64,
        help="Sequence length for test data",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for gradient comparison",
    )
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    try:
        success = verify_gradients(
            n_microbatches=args.n_microbatches,
            seq_len=args.seq_len,
            atol=args.atol,
        )
        if rank == 0:
            if args.output:
                write_result(args.output, success)
            else:
                # Standalone mode: print result
                print_rank0(
                    f"\nGradient verification: {'PASSED' if success else 'FAILED'}"
                )
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
