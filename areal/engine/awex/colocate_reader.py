# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for the shared SGLang AWEX adapter."""

from areal.engine.awex.sglang_adapter import (
    AwexSGLangAdapter,
    _get_router_dtype,
    _PhysicalDeviceMetaServerClient,
)
from areal.engine.awex.sglang_adapter import (
    _get_legacy_awex_hf_config as _get_awex_infer_hf_config,
)

AwexColocateReader = AwexSGLangAdapter

__all__ = [
    "AwexColocateReader",
    "_PhysicalDeviceMetaServerClient",
    "_get_awex_infer_hf_config",
    "_get_router_dtype",
]
