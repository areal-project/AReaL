# SPDX-License-Identifier: Apache-2.0

"""Shared constants for tree attention functionality."""

import os

from areal.utils.logging import getLogger

logger = getLogger("TreeAttentionConstants")

BLOCK_SIZE = int(os.environ.get("AREAL_FLEX_ATTENTION_BLOCK_SIZE", "128"))
USE_TRITON_TREE_ATTN = int(os.environ.get("AREAL_USE_TRITON_TREE_ATTN", "0")) == 1

if USE_TRITON_TREE_ATTN:
    logger.warning(
        "Triton tree attention kernel is only an experimental feature "
        "that requires further practical RL experiment testing."
    )
