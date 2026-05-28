# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/distributed/expert_parallel.py

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_module,
    distribute_tensor,
)
from torch.distributed.tensor.parallel.style import ParallelStyle

from areal.experimental.models.archon.moe.utils import _permute, _unpermute


class BaseExpertParallel(ParallelStyle, ABC):
    """Abstract base class for Expert Parallelism styles.

    Subclasses implement specific communication patterns for
    dispatching tokens to experts and combining results.

    Subclasses must implement:
    - _partition_fn: Shard expert weights across devices
    - _token_dispatch: Dispatch tokens to devices holding their assigned experts
    - _token_combine: Combine expert outputs back to original token locations
    - _apply: Apply parallelism using distribute_module with hooks
    """

    @abstractmethod
    def _partition_fn(
        self,
        name: str,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> None:
        """Partition expert weights across devices."""

    @abstractmethod
    def _token_dispatch(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch tokens to devices holding their assigned experts."""

    @abstractmethod
    def _token_combine(
        self,
        module: nn.Module,
        output: torch.Tensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        """Combine expert outputs back to original token locations."""


class ExpertParallel(BaseExpertParallel):
    """Expert Parallelism with ETP=1 (no tensor parallelism within experts).

    This class implements EP by:
    1. Sharding expert weights on dim 0 (expert dimension) across EP ranks
    2. Using all-to-all to dispatch tokens to devices holding their experts
    3. Using all-to-all to combine results back

    Each EP rank holds num_experts // ep_size local experts.
    """

    def __init__(self) -> None:
        super().__init__()
        # State saved during dispatch for use in combine
        self.input_splits: list[int] | None = None
        self.output_splits: list[int] | None = None
        self.input_shape: tuple[int, ...] | None = None
        self.permuted_indices: torch.Tensor | None = None

    def _partition_fn(
        self,
        name: str,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> None:
        """Shard expert weights on expert dimension (dim 0).

        Args:
            name: Module name (unused).
            module: GroupedExperts module to partition.
            device_mesh: 1D mesh for EP.
        """
        for param_name, param in module.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            module.register_parameter(param_name, dist_param)

    def _token_dispatch(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch tokens to EP ranks via all-to-all.

        Args:
            module: GroupedExperts module.
            inputs: Tuple of (routed_input, num_tokens_per_expert).
                routed_input: Shape (total_tokens, dim), tokens sorted by expert.
                num_tokens_per_expert: Shape (num_experts,), token counts per expert.
            device_mesh: 1D mesh for EP.

        Returns:
            Tuple of (dispatched_tokens, local_num_tokens_per_expert).
        """
        routed_input, num_tokens_per_expert = inputs
        group = device_mesh.get_group()
        ep_degree = device_mesh.size()
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        # Step 1: Exchange token counts via all-to-all
        with torch.no_grad():
            num_tokens_per_expert_received = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=group,
            )
            # Need to wait explicitly because it is used by downstream operations
            # which don't realize that AsyncCollectiveTensor needs unwrapping.
            # This is required for torch.compile compatibility.
            num_tokens_per_expert_received = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_received
            )

            # Compute splits for variable-size all-to-all
            # input_splits: how many tokens to send to each rank
            # output_splits: how many tokens to receive from each rank
            counts_view = num_tokens_per_expert.view(ep_degree, num_local_experts)
            received_view = num_tokens_per_expert_received.view(
                ep_degree, num_local_experts
            )

            # NOTE: tolist() requires sync to CPU
            self.input_splits = counts_view.sum(dim=1).to(
                torch.device("cpu"), non_blocking=True
            )
            self.output_splits = received_view.sum(dim=1).to(
                torch.device("cpu"), non_blocking=False
            )
            self.input_splits = self.input_splits.tolist()
            self.output_splits = self.output_splits.tolist()

        # Step 2: All-to-all to dispatch tokens
        routed_input = all_to_all_single_autograd(
            routed_input,
            self.output_splits,
            self.input_splits,
            group,
        )

        # Step 3: Permute for grouped_mm alignment
        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            aligned_num_tokens,
        ) = _permute(
            routed_input,
            num_tokens_per_expert_received,
            ep_degree,
            num_local_experts,
        )

        return routed_input, aligned_num_tokens

    def _token_combine(
        self,
        module: nn.Module,
        output: torch.Tensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        """Combine expert outputs via all-to-all back to original locations.

        Args:
            module: GroupedExperts module.
            output: Expert computation output.
            device_mesh: 1D mesh for EP.

        Returns:
            Combined output in original token order.
        """
        group = device_mesh.get_group()

        # Step 1: Unpermute to pre-permutation order
        output = _unpermute(output, self.input_shape, self.permuted_indices)

        # Step 2: All-to-all to combine (reverse direction)
        output = all_to_all_single_autograd(
            output,
            self.input_splits,  # Swap for reverse
            self.output_splits,
            group,
        )

        return output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """Apply expert parallelism to the module.

        Uses distribute_module to:
        1. Shard expert weights via partition_fn
        2. Register _token_dispatch as forward pre-hook (input_fn)
        3. Register _token_combine as forward hook (output_fn)

        This way EP communication happens automatically during forward pass.

        Args:
            module: GroupedExperts module to parallelize.
            device_mesh: 1D mesh for EP.

        Returns:
            The parallelized module.
        """
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


def apply_expert_parallel(
    experts_module: nn.Module,
    ep_mesh: DeviceMesh,
) -> None:
    """Apply Expert Parallelism to a GroupedExperts module.

    This is a convenience function that applies EP with ETP=1 to the
    given experts module. It uses distribute_module to register hooks
    for automatic token dispatch/combine during forward pass.

    Args:
        experts_module: GroupedExperts module to parallelize.
        ep_mesh: 1D device mesh for EP.

    Example:
        >>> from areal.experimental.models.archon import apply_expert_parallel
        >>> apply_expert_parallel(moe.experts, ep_mesh)
    """
    ep_style = ExpertParallel()
    ep_style._apply(experts_module, ep_mesh)


class TensorParallel(ParallelStyle):
    """Tensor Parallelism for experts when EP is disabled.

    This class implements TP-only parallelism for MoE experts by:
    1. Sharding w1/w3 on dim 1 (column-wise)
    2. Sharding w2 on dim 2 (row-wise)
    3. All-reducing outputs during backward

    Weight sharding:
    - w1: [num_experts, dim, hidden_dim] -> Shard on dim 1
    - w2: [num_experts, hidden_dim, dim] -> Shard on dim 2
    - w3: [num_experts, dim, hidden_dim] -> Shard on dim 1

    Used when: EP is disabled but TP is enabled for the model.
    """

    def _prepare_input_fn(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare input for TP by marking it for Partial gradient accumulation.

        The input is replicated across TP ranks, but we need Partial placement
        so that gradients are summed during backward.
        """
        routed_input, num_tokens_per_expert = inputs
        # Create DTensor with Replicate placement for forward,
        # Partial for backward gradient accumulation
        routed_input = DTensor.from_local(
            routed_input, device_mesh, (Replicate(),)
        ).to_local(grad_placements=(Partial(),))
        return routed_input, num_tokens_per_expert

    def _partition_fn(
        self,
        name: str,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> None:
        """Shard expert weights for tensor parallelism.

        w1/w3: Column-wise (Shard on dim 1)
        w2: Row-wise (Shard on dim 2)
        """
        # w1 shape = (num_experts, dim, hidden_dim) -> Column-wise sharding
        module.register_parameter(
            "w1", nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(1)]))
        )
        # w2 shape = (num_experts, hidden_dim, dim) -> Row-wise sharding
        module.register_parameter(
            "w2", nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(2)]))
        )
        # w3 shape = (num_experts, dim, hidden_dim) -> Column-wise sharding
        module.register_parameter(
            "w3", nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(1)]))
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """Apply tensor parallelism to the experts module."""
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._prepare_input_fn,
        )


class ExpertTensorParallel(ExpertParallel):
    """Expert Parallelism with Tensor Parallelism (EP + ETP).

    This class implements combined EP + TP for experts by:
    1. Sharding expert weights on both expert dimension (dim 0) and TP dimension
       - w1/w3: [Shard(0), Shard(1)]
       - w2: [Shard(0), Shard(2)]
    2. Using all-to-all for token dispatch/combine (inherited from ExpertParallel)
    3. All-reducing outputs within TP groups during backward

    Weight sharding (2D):
    - w1: [num_experts, dim, hidden_dim] -> [Shard(0), Shard(1)]
    - w2: [num_experts, hidden_dim, dim] -> [Shard(0), Shard(2)]
    - w3: [num_experts, dim, hidden_dim] -> [Shard(0), Shard(1)]

    device_mesh is 2D with dimensions: ["ep", "tp"]

    Used when: Both EP and ETP are enabled (etp=tp).
    """

    def _token_dispatch(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch tokens with TP input preparation.

        First prepares input for TP (Replicate with Partial gradient),
        then performs EP all-to-all dispatch.
        """
        routed_input, num_tokens_per_expert = inputs

        # Prepare input for TP (replicated input, partial gradient)
        routed_input = DTensor.from_local(
            routed_input, device_mesh["tp"], (Replicate(),)
        ).to_local(grad_placements=(Partial(),))

        # Call parent's token dispatch with EP mesh
        return super()._token_dispatch(
            module, (routed_input, num_tokens_per_expert), device_mesh["ep"]
        )

    def _partition_fn(
        self,
        name: str,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> None:
        """Shard expert weights on both EP and TP dimensions.

        Uses 2D sharding:
        - First dimension (Shard(0)): Expert parallel
        - Second dimension (Shard(1/2)): Tensor parallel
        """
        # w1 shape = (num_experts, dim, hidden_dim) -> [Shard(0), Shard(1)]
        module.register_parameter(
            "w1",
            nn.Parameter(
                distribute_tensor(module.w1, device_mesh, [Shard(0), Shard(1)])
            ),
        )
        # w2 shape = (num_experts, hidden_dim, dim) -> [Shard(0), Shard(2)]
        module.register_parameter(
            "w2",
            nn.Parameter(
                distribute_tensor(module.w2, device_mesh, [Shard(0), Shard(2)])
            ),
        )
        # w3 shape = (num_experts, dim, hidden_dim) -> [Shard(0), Shard(1)]
        module.register_parameter(
            "w3",
            nn.Parameter(
                distribute_tensor(module.w3, device_mesh, [Shard(0), Shard(1)])
            ),
        )

    def _token_combine(
        self,
        module: nn.Module,
        output: torch.Tensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        """Combine expert outputs via all-to-all back to original locations."""
        return super()._token_combine(module, output, device_mesh["ep"])

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """Apply expert + tensor parallelism to the module.

        Args:
            module: GroupedExperts module to parallelize.
            device_mesh: 2D mesh with dimensions [ep, tp].

        Returns:
            The parallelized module.
        """
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


class ReordererSequenceParallel(ParallelStyle):
    """Sequence Parallel for TokenReorderer when etp=1.

    When EP>1 and etp=1 (TP borrowed by EP), this parallelism:
    1. Splits tokens across TP ranks (each rank gets 1/tp of tokens)
    2. Adjusts token indices from local to global after reordering

    This ensures each TP rank processes different tokens, so the
    subsequent EP all-to-all doesn't send duplicate data.
    """

    def _prepare_input_fn(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split tokens across TP ranks."""
        top_scores, selected_indices = inputs
        num_tokens = top_scores.shape[0]
        tp_size = device_mesh.size()
        tp_rank = device_mesh.get_local_rank()

        if num_tokens % tp_size != 0:
            raise ValueError(
                f"Number of tokens ({num_tokens}) must be divisible by "
                f"TP size ({tp_size}) for ReordererSequenceParallel"
            )

        local_num_tokens = num_tokens // tp_size
        offset = tp_rank * local_num_tokens

        return (
            top_scores[offset : offset + local_num_tokens],
            selected_indices[offset : offset + local_num_tokens],
        )

    def _prepare_output_fn(
        self,
        module: nn.Module,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Adjust token indices from local to global.

        As we shard routed tokens along bs*slen dim across the TP ranks,
        the MoE gather and scatter still require global token indices.
        """
        top_scores_sorted, token_indices_sorted, num_tokens_per_expert = outputs
        tp_rank = device_mesh.get_local_rank()

        # token_indices_sorted are indices into local (num_tokens/tp * top_k) space
        # We need to adjust to global indices by adding the offset
        # top_scores_sorted.shape[0] == local_num_tokens * top_k
        token_indices_global = (
            token_indices_sorted + top_scores_sorted.shape[0] * tp_rank
        )

        return top_scores_sorted, token_indices_global, num_tokens_per_expert

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """Apply sequence parallel to TokenReorderer."""
        return distribute_module(
            module,
            device_mesh,
            input_fn=self._prepare_input_fn,
            output_fn=self._prepare_output_fn,
        )


__all__ = [
    "ExpertParallel",
    "ExpertTensorParallel",
    "TensorParallel",
    "ReordererSequenceParallel",
    "apply_expert_parallel",
]
