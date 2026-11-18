# SPDX-License-Identifier: Apache-2.0

"""Load worker-side data hooks from configuration."""

from __future__ import annotations

from areal.api.data_hook import HookConfig
from areal.utils.dynamic_import import import_from_string


def load_data_hook(config: HookConfig | None) -> object | None:
    """Construct a configured hook; callers check the capability they use."""
    if config is None:
        return None
    hook_class = import_from_string(config.class_path)
    if not isinstance(hook_class, type):
        raise TypeError(f"Data hook {config.class_path!r} must name a class")
    return hook_class(**config.kwargs)
