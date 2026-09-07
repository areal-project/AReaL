# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from areal.api.cli_args import PPOActorConfig
    from areal.api.io_struct import ModelRequest
    from areal.utils.data import TrajBatchMeta


class GenerationRequestPlugin(ABC):
    """Modify one backend request before dispatch; implementations must be stateless per request."""

    @abstractmethod
    def build_request(self, request: "ModelRequest", payload: dict[str, Any]) -> None:
        """Modify the backend payload using explicit request metadata."""


class PolicyDistribution(ABC):
    """Compute differentiable policy statistics from unpadded logits.

    The engine supplies original token-aligned metadata, causal labels and an
    explicit TP group. Plugins own any causal alignment of their metadata.
    Every TP rank must execute the same collectives, including zero-loss rows.
    Implementations must preserve gradients for the local vocabulary shard.
    """

    @abstractmethod
    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        inputs: dict[str, Any],
        temperature: float,
        tp_group: dist.ProcessGroup | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return log-probabilities and entropy, both shaped like labels."""


class PPOObjective(ABC):
    """Compose a PPO objective without replacing the engine or trainer.

    Advantage preparation sees complete trajectory groups and unshifted rollout
    tensors. It returns the standard PPO batch, with loss_mask/logprobs aligned
    to prediction positions. Group statistics and reduction weights must be
    fixed here, before any microbatch split. Trajectory groups are preserved
    by the existing dispatcher; plugins must not change row order or count.
    """

    @abstractmethod
    def compute_advantages(
        self,
        data: dict[str, Any],
        meta: "TrajBatchMeta | None",
        config: "PPOActorConfig",
    ) -> dict[str, Any]:
        """Return rewards, advantages, returns, KL statistics and aligned masks."""

    def loss(
        self,
        logprobs: torch.Tensor,
        entropy: torch.Tensor,
        input_data: dict[str, Any],
        default_loss: Callable[..., torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Reuse the configured PPO surrogate unless the plugin overrides reduction."""
        return default_loss(logprobs, entropy, input_data, **kwargs)

    def loss_weight(self, data: dict[str, Any]) -> torch.Tensor:
        """Return the pre-rejection normalization mass of one microbatch."""
        return data["loss_mask"].count_nonzero()
