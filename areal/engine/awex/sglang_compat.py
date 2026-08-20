# SPDX-License-Identifier: Apache-2.0

"""Compatibility re-exports for legacy SGLang AWEX integrations."""

from areal.engine.weight_update.awex.sglang import (
    SingleInstanceMetaResolver,
    ensure_awex_models_registered,
    patch_tms_hook_mode,
)

__all__ = [
    "SingleInstanceMetaResolver",
    "ensure_awex_models_registered",
    "patch_tms_hook_mode",
]
