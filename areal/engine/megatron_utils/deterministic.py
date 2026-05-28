# SPDX-License-Identifier: Apache-2.0

import os

import torch

from areal.utils import logging

logger = logging.getLogger("MCoreDeterm")


def set_deterministic_algorithms(model_config):
    """
    args: Megatron args, acquired by get_args()
    """
    model_config.deterministic_mode = True
    model_config.cross_entropy_loss_fusion = False
    model_config.bias_dropout_fusion = False

    # Set env variables about deterministic mode
    if os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1") != "0":
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
