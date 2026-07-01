# SPDX-License-Identifier: Apache-2.0
"""Experimental diffusion RL post-training (Phase 1: SD1.5 + GRPO PoC).

This package adds a diffusion-model RL post-training path on top of AReaL's
``Workflow -> Engine -> Reward -> Training`` abstractions without touching the
existing LLM code path. See ``docs/exec-plans/active/`` for the design rationale.
"""

from areal.experimental.diffusion.diffusion_api import (
    DiffusionModelRequest,
    DiffusionModelResponse,
)
from areal.experimental.diffusion.diffusion_engine import DiffusionInferenceEngine
from areal.experimental.diffusion.diffusion_loss import (
    compute_group_advantages,
    diffusion_grpo_loss_fn,
)
from areal.experimental.diffusion.diffusion_workflow import DiffusionRolloutWorkflow

__all__ = [
    "DiffusionModelRequest",
    "DiffusionModelResponse",
    "DiffusionInferenceEngine",
    "DiffusionRolloutWorkflow",
    "compute_group_advantages",
    "diffusion_grpo_loss_fn",
]
