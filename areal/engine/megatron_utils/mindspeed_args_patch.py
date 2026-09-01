# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from typing import Any


def sanitize_get_full_args(
    get_full_args: Callable[[], Any],
) -> Callable[[], Any]:
    """Remove invalid field names from MegatronAdaptor's argument namespace."""

    @functools.wraps(get_full_args)
    def wrapper():
        result = get_full_args()
        values = vars(result)
        invalid_keys = [
            key
            for key in values
            if not isinstance(key, str) or not key or not key.isidentifier()
        ]
        for key in invalid_keys:
            del values[key]
        return result

    setattr(wrapper, "_areal_sanitized_get_full_args", True)
    return wrapper


def ensure_mindspeed_args_sanitized() -> bool:
    """Patch the shared MA argument accessor before MindSpeed imports it."""
    try:
        import megatron_adaptor.utils.args_utils as args_utils
    except ImportError:
        return False

    get_full_args = args_utils.get_full_args
    if getattr(get_full_args, "_areal_sanitized_get_full_args", False):
        return True

    sanitized_get_full_args = sanitize_get_full_args(get_full_args)
    args_utils.get_full_args = sanitized_get_full_args
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith(("megatron_adaptor.", "mindspeed.")):
            continue
        if getattr(module, "get_full_args", None) is get_full_args:
            module.get_full_args = sanitized_get_full_args
    return True
