# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/models/moe/moe.py

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from areal.experimental.models.archon.moe.args import MoEArgs
from areal.experimental.models.archon.moe.grouped_experts import GroupedExperts
from areal.experimental.models.archon.moe.router import TokenChoiceTopKRouter
from areal.experimental.models.archon.moe.token_reorderer import TokenReorderer


class FeedForward(nn.Module):
    """Standard SwiGLU feedforward module (for shared experts).

    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden dimension.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3.weight, mean=0.0, std=init_std)


class MoE(nn.Module):
    """Mixture of Experts layer.

    This module routes tokens to a subset of experts and combines their outputs.
    It supports:
    - Top-k routing (each token goes to k experts)
    - Optional shared experts (always activated for all tokens)
    - Auxiliary-loss-free load balancing via expert_bias
    - Score application before or after expert computation

    Args:
        moe_args: MoE configuration.
        dim: Input/output dimension.
        hidden_dim: Hidden dimension for expert FFN.

    Attributes:
        router: Token-choice top-k router.
        experts: GroupedExperts with 3D weight tensors.
        shared_experts: Optional shared experts (always active).
        expert_bias: Buffer for load balancing (updated externally).
        tokens_per_expert: Buffer tracking expert usage (for load balancing).
    """

    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__()
        self.moe_args = moe_args
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = moe_args.num_experts
        self.top_k = moe_args.top_k
        self.score_before_experts = moe_args.score_before_experts
        self.load_balance_coeff = moe_args.load_balance_coeff

        # Router for token-to-expert assignment
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=moe_args.num_experts,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
            router_dtype=moe_args.router_dtype,
            num_expert_groups=moe_args.num_expert_groups,
            num_limited_groups=moe_args.num_limited_groups,
            _debug_force_load_balance=moe_args._debug_force_load_balance,
        )

        # Grouped experts with 3D weights
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=moe_args.num_experts,
            use_grouped_mm=moe_args.use_grouped_mm,
        )

        # Token reorderer (separate module for EP sequence parallel)
        self.reorderer = TokenReorderer(
            num_experts=moe_args.num_experts, top_k=moe_args.top_k
        )

        # Optional shared experts (always activated)
        # Priority: shared_expert_intermediate_size (Qwen3.5 MoE) > num_shared_experts * hidden_dim
        if moe_args.shared_expert_intermediate_size > 0:
            self.shared_experts = FeedForward(
                dim=dim, hidden_dim=moe_args.shared_expert_intermediate_size
            )
        elif moe_args.num_shared_experts > 0:
            shared_hidden_dim = hidden_dim * moe_args.num_shared_experts
            self.shared_experts = FeedForward(dim=dim, hidden_dim=shared_hidden_dim)
        else:
            self.shared_experts = None

        # Optional sigmoid gate for shared expert output (Qwen3.5 MoE)
        if moe_args.use_shared_expert_gate and self.shared_experts is not None:
            self.shared_expert_gate = nn.Linear(dim, 1, bias=False)
        else:
            self.shared_expert_gate = None

        # Buffers for auxiliary-loss-free load balancing
        # expert_bias is used to adjust routing probabilities
        # tokens_per_expert tracks usage for bias updates
        self.expert_bias: torch.Tensor | None
        self.tokens_per_expert: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE layer.

        Args:
            x: Input tensor, shape (batch_size, seq_len, dim).

        Returns:
            Output tensor, shape (batch_size, seq_len, dim).
        """
        bs, slen, dim = x.shape
        x_flat = x.view(-1, dim)  # (bs * slen, dim)

        # Route tokens to experts
        # top_scores: (bs * slen, top_k)
        # selected_indices: (bs * slen, top_k)
        # num_tokens_per_expert: (num_experts,)
        top_scores, selected_indices, num_tokens_per_expert = self.router(
            x_flat, self.expert_bias
        )

        # Track expert usage for load balancing
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert.float())

        # Reorder tokens by expert (separate module for EP sequence parallel)
        # When ReordererSequenceParallel is applied, each rank processes
        # different tokens and token_indices_experts_sorted contains global indices
        # NOTE: the reason we need to compute num_tokens_per_expert again is:
        #       1st computation in router is to update self.tokens_per_expert
        #       which would be the same across all TP ranks.
        #       2nd computation in reorderer is for the actual routing and experts computation
        #       which would be sharded over TP ranks if expert_tensor_parallel_degree==1.
        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_per_expert,
        ) = self.reorderer(top_scores, selected_indices)

        # Gather tokens using sorted indices
        # Divide by top_k to get the original token index
        routed_input = x_flat[token_indices_experts_sorted // self.top_k]

        # Apply scores before expert computation (optional)
        if self.score_before_experts:
            routed_input = (
                routed_input.float() * top_scores_experts_sorted.unsqueeze(-1)
            ).to(x.dtype)

        # Expert computation
        # If EP is enabled, dispatch/combine happens automatically via hooks
        # registered by distribute_module in ExpertParallel._apply
        routed_output = self.experts(routed_input, num_per_expert)

        # Shared expert (executed before unsorting to overlap with token combine)
        if self.shared_experts is not None:
            shared_out = self.shared_experts(x_flat)
            if self.shared_expert_gate is not None:
                shared_out = F.sigmoid(self.shared_expert_gate(x_flat)) * shared_out
        else:
            shared_out = None

        # Unsort routed outputs back to original positions
        # When ReordererSequenceParallel is applied, token_indices_experts_sorted
        # contains global indices, so each rank fills different positions
        routed_output_unsorted = torch.zeros(
            bs * slen * self.top_k,
            dim,
            device=routed_output.device,
            dtype=routed_output.dtype,
        )
        routed_output_unsorted[token_indices_experts_sorted] = routed_output
        routed_output_unsorted = routed_output_unsorted.view(bs * slen, self.top_k, dim)

        # Combine expert outputs
        if self.score_before_experts:
            # Scores already applied, just sum
            out_experts = routed_output_unsorted.sum(dim=1)
        else:
            # Apply scores as weighted sum
            # top_scores: (bs * slen, top_k) -> (bs * slen, 1, top_k)
            out_experts = (
                torch.bmm(
                    top_scores.unsqueeze(1).float(),
                    routed_output_unsorted.float(),
                )
                .squeeze(1)
                .to(x.dtype)
            )

        # Add shared experts output if present
        if shared_out is not None:
            out = shared_out + out_experts
        else:
            out = out_experts

        return out.view(bs, slen, dim)

    def init_weights(self, init_std: float):
        """Initialize MoE parameters.

        Args:
            init_std: Standard deviation for output projections.
        """
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)

        if self.shared_experts is not None:
            self.shared_experts.init_weights(init_std)

        if self.shared_expert_gate is not None:
            nn.init.trunc_normal_(
                self.shared_expert_gate.weight, mean=0.0, std=init_std
            )

    def init_buffers(self, buffer_device: torch.device | str):
        """Initialize MoE buffers (tokens_per_expert, expert_bias).

        Args:
            buffer_device: Device for buffers.
        """
        with torch.device(buffer_device):
            self.register_buffer(
                "tokens_per_expert",
                torch.zeros(self.num_experts, dtype=torch.float32),
                persistent=False,
            )
            if self.load_balance_coeff is not None:
                assert self.load_balance_coeff > 0.0
                self.register_buffer(
                    "expert_bias",
                    torch.zeros(self.num_experts, dtype=torch.float32),
                    persistent=True,
                )
            else:
                self.expert_bias = None


__all__ = ["MoE", "FeedForward"]
