# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/models/moe/moe.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from transformers import PretrainedConfig


@dataclass
class MoEArgs:
    """Arguments for Mixture of Experts (MoE) configuration.

    Attributes:
        num_experts: Total number of experts in each MoE layer.
        num_shared_experts: Number of shared experts (always activated).
            Qwen3-MoE uses 0 (no shared experts).
        top_k: Number of experts each token is routed to.

        score_func: Scoring function for routing.
            - "softmax": Standard softmax normalization (used by Qwen3-MoE)
            - "sigmoid": Sigmoid activation (used by DeepSeek-V3)
        route_norm: Whether to normalize routing scores.
        route_scale: Scale factor for routing scores.
        score_before_experts: Whether to apply scores before or after expert computation.
        router_dtype: Data type for router gate GEMM computation.
            If None, the model's default dtype is used (no override).

        num_expert_groups: Number of expert groups for node-limited routing.
            If None, standard top-k routing is used.
        num_limited_groups: Number of groups to select in node-limited routing.

        use_grouped_mm: Whether to use grouped matrix multiplication.
            If True, uses torch._grouped_mm for efficient batched computation.
            If False, uses a for-loop over experts.

        load_balance_coeff: Coefficient for auxiliary-loss-free load balancing.
            If None, load balancing is disabled.

        _debug_force_load_balance: Force uniform token distribution across experts.
            Only for testing purposes.
    """

    # Basic configuration
    num_experts: int = 8
    num_shared_experts: int = 0
    top_k: int = 1

    # Shared expert configuration
    shared_expert_intermediate_size: int = 0
    use_shared_expert_gate: bool = False

    # Router configuration
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = False
    router_dtype: torch.dtype | None = None

    # Node-limited routing (optional)
    num_expert_groups: int | None = None
    num_limited_groups: int | None = None

    # Computation mode
    use_grouped_mm: bool = True

    # Load balancing (for DeepSeek MoE style auxiliary-loss-free load balancing)
    load_balance_coeff: float | None = None

    # Debug
    _debug_force_load_balance: bool = False

    @classmethod
    def from_hf_config(cls, hf_config: PretrainedConfig) -> MoEArgs:
        """Create MoEArgs from a HuggingFace config.

        Expected HuggingFace config fields:
            - num_experts or num_local_experts: Number of experts
            - num_experts_per_tok: Top-k value
            - norm_topk_prob: Whether to normalize routing scores

        Args:
            hf_config: HuggingFace model configuration.

        Returns:
            MoEArgs instance.
        """
        num_experts = getattr(
            hf_config, "num_experts", getattr(hf_config, "num_local_experts", 8)
        )
        top_k = getattr(hf_config, "num_experts_per_tok", 1)
        route_norm = getattr(hf_config, "norm_topk_prob", False)

        return cls(
            num_experts=num_experts,
            top_k=top_k,
            route_norm=route_norm,
            # Qwen3-MoE uses softmax scoring
            score_func="softmax",
        )


__all__ = ["MoEArgs"]
