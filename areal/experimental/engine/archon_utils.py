# SPDX-License-Identifier: Apache-2.0

"""Utility functions extracted from ArchonEngine for reuse and testability."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch.distributed.pipelining.schedules import (
    ScheduleDualPipeV,
    ScheduleInterleavedZeroBubble,
    ScheduleZBVZeroBubble,
    get_schedule_class,
)
from transformers import (
    PretrainedConfig,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from areal.engine.fsdp_utils import get_cosine_schedule_with_warmup
from areal.experimental.models.archon.activation_checkpoint import (
    ActivationCheckpointConfig,
)
from areal.experimental.models.archon.utils import is_moe_model_config

if TYPE_CHECKING:
    from areal.api.cli_args import (
        OptimizerConfig,
        TrainEngineConfig,
    )
    from areal.experimental.models.archon import ArchonParallelDims
    from areal.utils import logging


# =========================================================================
# Optimizer & LR Scheduler
# =========================================================================


def create_optimizer(
    params: list[torch.nn.Parameter],
    optimizer_config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    lr = optimizer_config.lr
    weight_decay = optimizer_config.weight_decay
    beta1 = optimizer_config.beta1
    beta2 = optimizer_config.beta2
    eps = optimizer_config.eps

    if optimizer_config.type == "adam":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            eps=eps,
            fused=True,
        )
    elif optimizer_config.type == "muon":
        raise NotImplementedError(
            "Muon optimizer is not yet supported under ArchonEngine. "
        )
    elif optimizer_config.type == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_config.type}")


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    optimizer_config: OptimizerConfig,
    total_train_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create LR scheduler from config."""
    num_warmup_steps = int(optimizer_config.warmup_steps_proportion * total_train_steps)

    if optimizer_config.lr_scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            total_train_steps,
            min_lr_ratio=optimizer_config.min_lr_ratio,
        )
    elif optimizer_config.lr_scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            total_train_steps,
        )
    elif optimizer_config.lr_scheduler_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
        )
    else:
        raise ValueError(
            f"Unknown lr scheduler type: {optimizer_config.lr_scheduler_type}"
        )


# =========================================================================
# Activation Checkpoint Config
# =========================================================================


def build_ac_config(
    config: TrainEngineConfig,
    logger: logging.Logger,
) -> ActivationCheckpointConfig | None:
    """Build ActivationCheckpointConfig from engine config.

    Returns None if gradient checkpointing is disabled.
    """
    if not config.gradient_checkpointing:
        return None

    archon_config = config.archon
    mode = archon_config.ac_mode

    if mode == "none":
        return None

    ac_config = ActivationCheckpointConfig(
        mode=mode,
        selective_ac_option=archon_config.selective_ac_option,
        memory_budget=archon_config.ac_memory_budget,
        preserve_rng_state=archon_config.ac_preserve_rng_state,
        debug=archon_config.ac_debug,
    )

    logger.info(
        f"Activation checkpointing: mode={ac_config.mode}, "
        f"selective_option={ac_config.selective_ac_option}, "
        f"memory_budget={ac_config.memory_budget}, "
        f"preserve_rng={ac_config.preserve_rng_state}, debug={ac_config.debug}"
    )

    return ac_config


# =========================================================================
# Zero-Bubble Schedule Compatibility
# =========================================================================

# Schedule classes that use split backward (I/W steps) with retain_graph=True.
_ZERO_BUBBLE_SCHEDULES = (
    ScheduleInterleavedZeroBubble,
    ScheduleZBVZeroBubble,
    ScheduleDualPipeV,
)


def validate_zero_bubble_compatibility(
    pp_schedule: str,
    enable_compile: bool,
    model_config: PretrainedConfig,
    ac_config: ActivationCheckpointConfig | None,
    logger: logging.Logger,
) -> bool:
    """Fix zero-bubble schedule incompatibilities.

    Zero-bubble schedules (split backward with retain_graph=True) conflict with
    torch.compile, donated_buffer (MoE), op-level selective AC, and memory_budget AC.

    Returns updated ``enable_compile`` flag.
    """
    if get_schedule_class(pp_schedule) not in _ZERO_BUBBLE_SCHEDULES:
        return enable_compile

    # 1. Disable torch.compile
    if enable_compile:
        logger.warning(
            f"{pp_schedule} is incompatible with torch.compile. "
            "Disabling torch.compile."
        )
        enable_compile = False

    # 2. Disable donated_buffer for MoE models
    if is_moe_model_config(model_config):
        import torch._functorch.config as functorch_config

        if getattr(functorch_config, "donated_buffer", False):
            logger.info(
                f"{pp_schedule} requires donated_buffer=False "
                "for MoE models (internally compiled ops conflict "
                "with retain_graph=True in split backward). Disabling."
            )
            functorch_config.donated_buffer = False

    # 3. Fall back from op-level selective AC / memory_budget AC to full AC
    if ac_config is not None and (
        (ac_config.mode == "selective" and ac_config.selective_ac_option == "op")
        or ac_config.mode == "memory_budget"
    ):
        logger.warning(
            f"{pp_schedule} is incompatible with {ac_config.mode} AC. "
            "Falling back to full AC."
        )
        ac_config.mode = "full"

    return enable_compile


# =========================================================================
# Deterministic Mode
# =========================================================================


def setup_deterministic_mode(
    ac_config: ActivationCheckpointConfig | None,
    enable_compile: bool,
    logger: logging.Logger,
) -> None:
    """Set env vars and PyTorch flags for deterministic training."""
    logger.info("Deterministic mode enabled. May reduce performance.")

    if ac_config is not None:
        ac_config.preserve_rng_state = True
        logger.info("Deterministic mode: overriding ac_config.preserve_rng_state=True.")

    # Enable PyTorch deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)

    # cuBLAS workspace config for deterministic matmuls
    cublas_valid = (":4096:8", ":16:8")
    if os.getenv("CUBLAS_WORKSPACE_CONFIG") not in cublas_valid:
        logger.info(
            "For deterministic mode, env [CUBLAS_WORKSPACE_CONFIG] "
            "will be set to ':4096:8'."
        )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # NCCL algorithm for deterministic collective operations
    nccl_valid = ("Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS")
    if os.getenv("NCCL_ALGO") not in nccl_valid:
        logger.info("For deterministic mode, env [NCCL_ALGO] will be set to 'Ring'.")
        os.environ["NCCL_ALGO"] = "Ring"

    # Torch compile deterministic (only when compile is active)
    if enable_compile and os.getenv("TORCH_COMPILE_DETERMINISTIC") != "1":
        logger.info(
            "For deterministic mode, env [TORCH_COMPILE_DETERMINISTIC] "
            "will be set to '1'."
        )
        os.environ["TORCH_COMPILE_DETERMINISTIC"] = "1"


# =========================================================================
# pad_to_maximum
# =========================================================================


def force_pad_to_maximum(
    config: TrainEngineConfig,
    parallel_dims: ArchonParallelDims,
    enable_compile: bool,
    enable_tree_training: bool,
    logger: logging.Logger,
) -> None:
    """Force ``config.pad_to_maximum = True`` when compile, PP, or tree training
    requires it. Also validates tree training constraints.
    """
    # Force pad_to_maximum when compile is enabled to avoid dynamic shape issues
    if enable_compile and not config.pad_to_maximum:
        logger.info(
            "torch.compile is enabled: forcing pad_to_maximum=True to avoid "
            "dynamic shape issues with Inductor. Original pad_to_maximum=False."
        )
        config.pad_to_maximum = True

    # Force pad_to_maximum when PP is enabled to avoid shape mismatch
    if parallel_dims.pp_enabled and not config.pad_to_maximum:
        logger.info(
            "Pipeline Parallelism is enabled: forcing pad_to_maximum=True to avoid "
            "shape mismatch across microbatches. Original pad_to_maximum=False."
        )
        config.pad_to_maximum = True

    # Tree training constraints
    if enable_tree_training:
        if config.is_critic:
            raise NotImplementedError(
                "Tree training with critic model is not supported yet."
            )
        if parallel_dims.pp_enabled or parallel_dims.cp_enabled:
            raise ValueError(
                "Tree training with pipeline parallelism (pp > 1) or "
                "context parallelism (cp > 1) is currently not supported."
            )
        # Force pad_to_maximum for tree training
        if not config.pad_to_maximum:
            logger.info(
                "Tree training enabled: forcing pad_to_maximum=True for "
                "block mask alignment. Original pad_to_maximum=False."
            )
            config.pad_to_maximum = True


# =========================================================================
# Combined Config Preparation
# =========================================================================


def prepare_training_config(
    config: TrainEngineConfig,
    parallel_dims: ArchonParallelDims,
    model_config: PretrainedConfig,
    enable_tree_training: bool,
    logger: logging.Logger,
) -> tuple[ActivationCheckpointConfig | None, bool]:
    """Build and validate all training configs before parallelism setup.

    Returns (ac_config, enable_compile). May mutate ``config.pad_to_maximum``
    and set deterministic env vars.

    Note: the returned ``enable_compile`` may differ from
    ``config.archon.enable_compile`` (zero-bubble or FP8 can disable it).
    ``config.archon.enable_compile`` is **not** written back — callers
    must use the returned value.
    """
    ac_config = build_ac_config(config, logger)
    enable_compile = config.archon.enable_compile

    enable_compile = validate_zero_bubble_compatibility(
        pp_schedule=config.archon.pp_schedule,
        enable_compile=enable_compile,
        model_config=model_config,
        ac_config=ac_config,
        logger=logger,
    )
    if config.archon.fp8_config.enabled and enable_compile:
        logger.warning(
            "FP8 blockwise training is incompatible with torch.compile. "
            "Disabling torch.compile."
        )
        enable_compile = False

    if config.archon.use_deterministic_algorithms:
        setup_deterministic_mode(ac_config, enable_compile, logger)
    force_pad_to_maximum(
        config=config,
        parallel_dims=parallel_dims,
        enable_compile=enable_compile,
        enable_tree_training=enable_tree_training,
        logger=logger,
    )

    # PP weight tying constraint (independent of pad_to_maximum)
    if parallel_dims.pp_enabled:
        if getattr(model_config, "tie_word_embeddings", False):
            raise ValueError(
                f"Pipeline Parallelism (PP={parallel_dims.pp}) is not supported "
                f"with weight tying (tie_word_embeddings=True). "
                f"When PP > 1, tok_embeddings and output layers are on different GPUs "
                f"and cannot share the same weight tensor. "
                f"Please either disable PP (set pipeline_parallel_size=1) or use a model "
                f"without weight tying."
            )

    return ac_config, enable_compile
