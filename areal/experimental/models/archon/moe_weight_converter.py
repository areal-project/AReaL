# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/models/utils.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard, _StridedShard

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


@dataclass
class MoEConversionState:
    """Explicit state container for DTensor conversion metadata.

    This dataclass holds the metadata needed for DTensor-aware MoE expert weight
    conversion. The state is populated during to_hf (split) operations and consumed
    during from_hf (concatenate) operations.

    Important: For DTensor checkpoints, the conversion methods have a stateful dependency:
    `split_expert_weights_dtensor()` (called during to_hf) populates metadata that
    `concatenate_expert_weights_dtensor()` (called during from_hf) requires.
    Therefore, when working with DTensor expert weights, the same state instance
    must be used for both to_hf() and from_hf() to ensure the metadata is available.
    """

    grouped_expert_weight_placements: dict[str, tuple] = field(default_factory=dict)
    grouped_expert_weight_shape: dict[str, tuple[int, ...]] = field(
        default_factory=dict
    )
    local_experts_indices: dict[str, tuple[int, int]] = field(default_factory=dict)

    def clear(self) -> None:
        """Clear all stored conversion metadata."""
        self.grouped_expert_weight_placements.clear()
        self.grouped_expert_weight_shape.clear()
        self.local_experts_indices.clear()


class MoEWeightConverter:
    """DTensor-aware MoE expert weight converter.

    HF MoE models store experts as a module list each with 2D weights. In Archon, we
    store experts as a 3D param with the first dimension being num_experts. The methods
    in this class help convert 3D param into list of 2D params so that the checkpoint
    can be loaded without incurring local memory overhead, and then concatenate
    the results back to 3D param.

    This class provides:
    - DTensor-aware methods for distributed checkpoint support (EP, EP+TP, EP+ETP)
    - Offline conversion methods for non-distributed scenarios

    Note: This class is designed to be composed into StateDictAdapter classes rather
    than inherited from. The conversion state is managed externally via MoEConversionState.
    """

    @staticmethod
    def calculate_strided_shard_indices(
        strided_shard_dim_degree: int,
        strided_shard_dim_rank: int,
        shard_dim_degree: int,
        shard_dim_rank: int,
        dim_size_to_split: int,
    ) -> tuple[int, int]:
        """Calculate start/end indices for [StridedShard(dim=i), Shard(dim=i)] placement.

        GPU Layout (strided_shard_rank, shard_rank):

        StridedShard Rank                  Shard rank
                        ┌─────────────────┐
                    0   │    GPU(0, 0)    │  0
                    ────┼─────────────────┤
                    1   │    GPU(1, 0)    │
                    ────┼─────────────────┤
                    2   │    GPU(2, 0)    │
                  ──────┼─────────────────┼────
                    0   │    GPU(0, 1)    │  1
                    ────┼─────────────────┤
                    1   │    GPU(1, 1)    │
                    ────┼─────────────────┤
                    2   │    GPU(2, 1)    │
                        └─────────────────┘

        Calculates start_index from inner dimension (Shard) to outer dimension (StridedShard).

        Args:
            strided_shard_dim_degree: Degree of the StridedShard mesh dimension
            strided_shard_dim_rank: Rank in the StridedShard mesh dimension
            shard_dim_degree: Degree of the Shard mesh dimension
            shard_dim_rank: Rank in the Shard mesh dimension
            dim_size_to_split: Total size of the dimension being split

        Returns:
            Tuple of (start_index, end_index) for the local GPU

        Raises:
            ValueError: If dimension cannot be evenly split
        """
        block_size = dim_size_to_split // (strided_shard_dim_degree * shard_dim_degree)

        # Error out if can not evenly divide
        if (
            block_size * (strided_shard_dim_degree * shard_dim_degree)
            != dim_size_to_split
        ):
            raise ValueError(
                f"Cannot evenly split dim_size {dim_size_to_split} with "
                f"strided_shard_degree={strided_shard_dim_degree}, shard_degree={shard_dim_degree}"
            )

        start_index = block_size * (
            strided_shard_dim_degree * shard_dim_rank + strided_shard_dim_rank
        )
        end_index = start_index + block_size

        return start_index, end_index

    @staticmethod
    def calculate_indices_from_placements(
        dim: int,
        dim_size: int,
        dtensor_placements: tuple,
        device_mesh: DeviceMesh,
    ) -> tuple[int | None, int | None]:
        """Calculate local indices for a given dimension from DTensor placements.

        Handles various sharding strategies:
        - 0 placements on dim: No split needed, returns (None, None)
        - 1 placement (Shard): Simple division by shard degree
        - 2 placements (StridedShard + Shard): Complex EP+ETP sharding

        Args:
            dim: The dimension to calculate indices for
            dim_size: Total size of the dimension
            dtensor_placements: Tuple of DTensor placements
            device_mesh: DeviceMesh for the DTensor

        Returns:
            Tuple of (start_index, end_index) or (None, None) if no split on this dim
        """
        mesh_names: list[str] = []
        dim_i_placements: list = []

        # Find all device mesh dimensions that shard on dim-i
        for i, name in enumerate(device_mesh.mesh_dim_names):
            placement = dtensor_placements[i]
            if isinstance(placement, (Shard, _StridedShard)):
                if placement.dim == dim:
                    mesh_names.append(name)
                    dim_i_placements.append(placement)
            elif isinstance(placement, Replicate):
                # Replicate does not shard on any dimension, skip
                pass
            else:
                raise ValueError(
                    f"Unexpected placement type: {type(placement).__name__} on "
                    f"mesh dim '{name}'. Expected Shard, _StridedShard, or Replicate."
                )

        # Calculate local indices based on sharding strategy
        start_index, end_index = None, None

        if len(dim_i_placements) == 2:
            # Handle StridedShard(i) + Shard(i) case (e.g., EP + ETP)
            assert isinstance(dim_i_placements[0], _StridedShard), (
                "Expected StridedShard as first placement"
            )

            strided_shard_mesh = device_mesh[mesh_names[0]]
            shard_mesh = device_mesh[mesh_names[1]]

            strided_degree = strided_shard_mesh.size()
            strided_rank = strided_shard_mesh.get_local_rank()
            shard_degree = shard_mesh.size()
            shard_rank = shard_mesh.get_local_rank()

            start_index, end_index = MoEWeightConverter.calculate_strided_shard_indices(
                strided_degree, strided_rank, shard_degree, shard_rank, dim_size
            )

        elif len(dim_i_placements) == 1:
            # Handle single Shard(i) case (e.g., EP only)
            assert not isinstance(dim_i_placements[0], _StridedShard), (
                "Expected regular Shard, not StridedShard"
            )

            shard_mesh = device_mesh[mesh_names[0]]
            shard_degree = shard_mesh.size()
            shard_rank = shard_mesh.get_local_rank()

            block_size = dim_size // shard_degree
            if block_size * shard_degree != dim_size:
                raise ValueError(
                    f"Dim {dim} size ({dim_size}) cannot be evenly divided by "
                    f"shard degree ({shard_degree})"
                )

            start_index = block_size * shard_rank
            end_index = start_index + block_size

        elif len(dim_i_placements) == 0:
            # No split on this dimension
            return start_index, end_index

        else:
            raise NotImplementedError(
                f"Unsupported DTensor placements for GroupedExperts: "
                f"{dtensor_placements} {dim_i_placements} {mesh_names}"
            )

        return start_index, end_index

    def split_expert_weights_dtensor(
        self,
        hf_abstract_key: str,
        archon_abstract_key: str,
        layer_id: str,
        grouped_expert_weight: DTensor,
        state: MoEConversionState,
    ) -> dict[str, DTensor]:
        """Split GroupedExperts weight into individual expert weights for distributed save.

        This method handles various sharding strategies for expert weights:
        - FSDP + EP: StridedShard(0)Shard(0) or Shard(0)
        - FSDP + ETP + EP: StridedShard(0)Shard(0)Shard(1/2) or StridedShard(1)Shard(0)Shard(1/2)

        Args:
            hf_abstract_key: HuggingFace template key with {} placeholders for layer and expert IDs
            archon_abstract_key: Archon template key with {} placeholder for layer ID
            layer_id: Layer identifier (string)
            grouped_expert_weight: DTensor containing all experts' weights (3D)
            state: MoEConversionState to store metadata for from_hf reconstruction

        Returns:
            Dictionary mapping individual expert HF keys to their DTensor weights (2D)
        """
        device_mesh = grouped_expert_weight.device_mesh
        dtensor_placements = grouped_expert_weight.placements

        # Use concrete key (with layer_id) to avoid collision between layers
        archon_key = archon_abstract_key.format(layer_id)

        # Step 1: Store metadata for from_hf reconstruction
        state.grouped_expert_weight_placements[archon_key] = dtensor_placements
        state.grouped_expert_weight_shape[archon_key] = tuple(
            grouped_expert_weight.shape
        )

        # Step 2: Calculate local expert indices from placements
        num_experts = grouped_expert_weight.shape[0]
        start_index, end_index = self.calculate_indices_from_placements(
            dim=0,
            dim_size=num_experts,
            dtensor_placements=dtensor_placements,
            device_mesh=device_mesh,
        )
        assert start_index is not None and end_index is not None, (
            "Start index and end index cannot be None on dim-0!"
        )

        # Step 3: Store indices for from_hf reconstruction
        state.local_experts_indices[archon_key] = (start_index, end_index)

        # Step 4: Create new placements for individual expert weights (2D)
        # Expert dimension (dim-0) becomes Replicate, other dims shift down by 1
        new_placements = []
        for i, _name in enumerate(device_mesh.mesh_dim_names):
            placement = dtensor_placements[i]
            if hasattr(placement, "dim") and placement.dim == 0:
                # Expert dimension is removed, convert to Replicate
                new_placements.append(Replicate())
            elif isinstance(placement, Shard):
                # Other Shard dimensions keep same dim (2D expert weight)
                new_placements.append(Shard(placement.dim))
            elif isinstance(placement, _StridedShard):
                # Keep strided shard with same parameters
                new_placements.append(
                    _StridedShard(placement.dim, placement.split_factor)
                )
            else:
                # Replicate stays as Replicate
                new_placements.append(placement)

        # Step 5: Extract local tensor and split into individual experts
        local_grouped_weights = grouped_expert_weight.to_local()
        expected_local_experts = end_index - start_index

        if local_grouped_weights.shape[0] != expected_local_experts:
            raise ValueError(
                f"Local tensor shape mismatch: expected {expected_local_experts} experts, "
                f"got {local_grouped_weights.shape[0]}"
            )

        local_expert_tensors: dict[str, DTensor] = {}
        for expert_id in range(start_index, end_index):
            hf_key = hf_abstract_key.format(layer_id, expert_id)
            local_expert_index = expert_id - start_index

            # Extract individual expert weight and add temporary batch dimension
            expert_weight = local_grouped_weights[local_expert_index, :, :].unsqueeze(0)

            # Create DTensor and remove batch dimension
            expert_dtensor = DTensor.from_local(
                expert_weight, device_mesh, tuple(new_placements), run_check=False
            ).squeeze(0)

            local_expert_tensors[hf_key] = expert_dtensor

        return local_expert_tensors

    def concatenate_expert_weights_dtensor(
        self,
        expert_weights_by_layer: dict[str, dict[str, dict[int, Any]]],
        archon_abstract_key: str,
        layer_id: str,
        device_mesh: DeviceMesh,
        state: MoEConversionState,
    ) -> DTensor | None:
        """Concatenate individual expert weights back into GroupedExperts DTensor.

        Args:
            expert_weights_by_layer: Dictionary tracking expert weights by layer, abstract key, and expert ID.
                Structure: {layer_id: {abstract_key: {expert_id: tensor_weight}}}
            archon_abstract_key: Archon template key with {} placeholder for layer ID
            layer_id: Layer identifier (string)
            device_mesh: DeviceMesh for the target GroupedExperts weight DTensor
            state: MoEConversionState containing metadata from split operation

        Returns:
            Concatenated GroupedExperts weight DTensor if all experts are available, otherwise None
        """
        if layer_id not in expert_weights_by_layer:
            return None
        if archon_abstract_key not in expert_weights_by_layer[layer_id]:
            return None

        # Use concrete key (with layer_id) to lookup metadata
        archon_key = archon_abstract_key.format(layer_id)

        experts = expert_weights_by_layer[layer_id][archon_abstract_key]
        expected_n_experts = (
            state.local_experts_indices[archon_key][1]
            - state.local_experts_indices[archon_key][0]
        )
        if len(experts) < expected_n_experts:
            return None

        # Sort and stack expert weights
        sorted_expert_ids = sorted(experts.keys())
        sorted_experts = [experts[i] for i in sorted_expert_ids]

        # Stack and get local tensor (input experts are DTensors)
        stacked = torch.stack(sorted_experts, dim=0)
        if isinstance(stacked, DTensor):
            local_tensor = stacked.to_local()
        else:
            local_tensor = stacked

        # Verify we have stored placements
        assert (
            archon_key in state.grouped_expert_weight_placements
            and archon_key in state.grouped_expert_weight_shape
        ), (
            f"GroupedExperts weight metadata (placements, shape) for {archon_key} cannot be None!"
        )

        # Reconstruct DTensor with original 3D placements
        stacked_dtensor = DTensor.from_local(
            local_tensor,
            device_mesh,
            state.grouped_expert_weight_placements[archon_key],
            run_check=False,
        )

        # Cleanup
        del expert_weights_by_layer[layer_id][archon_abstract_key]
        if not expert_weights_by_layer[layer_id]:
            del expert_weights_by_layer[layer_id]

        return stacked_dtensor

    @staticmethod
    def split_expert_weights_offline(
        weight: torch.Tensor, n_experts: int
    ) -> tuple[torch.Tensor, ...]:
        """Split 3D expert weight into tuple of 2D weights. Used for offline conversion.

        NOTE: If used for online conversion with DTensors, torch.split() might incur
        communication to gather the weight, causing OOM.

        Args:
            weight: 3D tensor of shape (n_experts, out_dim, in_dim)
            n_experts: Number of experts

        Returns:
            Tuple of 2D tensors, one per expert
        """
        split_weight = torch.split(weight, weight.shape[0] // n_experts, dim=0)
        return split_weight

    @staticmethod
    def concatenate_expert_weights_offline(
        expert_weights_by_layer: dict[str, dict[str, dict[int, torch.Tensor]]],
        abstract_key: str,
        layer_num: str,
        n_experts: int,
    ) -> torch.Tensor | None:
        """Concatenate 2D expert weights into 3D using torch.stack(). Used for offline conversion.

        Args:
            expert_weights_by_layer: Dictionary tracking expert weights by layer, abstract key, and expert ID.
                Structure: {layer_id: {abstract_key: {expert_id: tensor_weight}}}
            abstract_key: Archon template key with {} placeholder for layer ID
            layer_num: Layer identifier (string)
            n_experts: Number of experts in the GroupedExperts module

        Returns:
            Concatenated GroupedExperts weight if all experts are available, otherwise None
        """
        if layer_num not in expert_weights_by_layer:
            return None
        if abstract_key not in expert_weights_by_layer[layer_num]:
            return None

        experts = expert_weights_by_layer[layer_num][abstract_key]
        if len(experts) < n_experts:
            return None

        sorted_expert_ids = sorted(experts.keys())
        sorted_experts = [experts[i] for i in sorted_expert_ids]
        stacked_tensor = torch.stack(sorted_experts, dim=0)

        del expert_weights_by_layer[layer_num][abstract_key]
        if not expert_weights_by_layer[layer_num]:
            del expert_weights_by_layer[layer_num]

        return stacked_tensor


__all__ = ["MoEConversionState", "MoEWeightConverter"]
