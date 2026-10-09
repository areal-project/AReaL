# SPDX-License-Identifier: Apache-2.0

import os
import sys

import torch

from areal.utils import logging

logger = logging.getLogger("MCoreDeterm")


def set_deterministic_algorithms(model_config, prebuild: bool = False):
    """Enable deterministic execution on a Megatron transformer config.

    ``deterministic_mode`` has two kinds of consumers in Megatron-Core:
    modules that copy config flags at construction time (tensor-parallel
    linear layers, TransformerEngine extensions) and code that reads the
    config at runtime (loss fusions, the pipeline schedules). Setting the
    flag only on a built model's config engages just the runtime consumers
    and silently leaves the layer-level kernels nondeterministic.

    Call this with ``prebuild=True`` on the ``TransformerConfig`` used to
    build the model, and again without ``prebuild`` on the built model's
    config to cover runtime consumers.
    """
    model_config.deterministic_mode = True
    model_config.cross_entropy_loss_fusion = False
    model_config.bias_dropout_fusion = False

    if prebuild and hasattr(model_config, "attention_backend"):
        from megatron.core.transformer.enums import AttnBackend

        # Megatron-Core owns the NVTE_*_ATTN selection env vars and asserts
        # that they match TransformerConfig.attention_backend, so the backend
        # must be chosen here rather than exported externally. Select flash:
        # the cuDNN fused-attention deterministic backward requires workspaces
        # that grow prohibitively with context length, while flash-attention
        # provides a deterministic backward without that cost.
        model_config.attention_backend = AttnBackend.flash
        logger.info("Deterministic prebuild: attention_backend set to flash.")

    # Set env variables about deterministic mode
    if os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1") != "0":
        if "transformer_engine" in sys.modules:
            logger.warning(
                "transformer_engine was imported before "
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO was set to '0'. Some TE "
                "versions snapshot it at import, so attention kernels may "
                "remain nondeterministic. AReaL launchers export it "
                "automatically when use_deterministic_algorithms is enabled; "
                "with a custom launcher, export it before the training "
                "process starts."
            )
        logger.info(
            "For deterministic algo, env [NVTE_ALLOW_NONDETERMINISTIC_ALGO] will be set to '0'."
        )
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"

    all_reduce_choices = ["Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS"]
    if os.getenv("NCCL_ALGO") not in all_reduce_choices:
        logger.info("For deterministic algo, env [NCCL_ALGO] will be set to 'Ring'.")
        os.environ["NCCL_ALGO"] = "Ring"

    cublas_workspace_config_choices = [":4096:8", ":16:8"]
    if os.getenv("CUBLAS_WORKSPACE_CONFIG") not in cublas_workspace_config_choices:
        logger.info(
            "For deterministic algo, env [CUBLAS_WORKSPACE_CONFIG] will be set to ':4096:8'."
        )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.use_deterministic_algorithms(True, warn_only=True)
