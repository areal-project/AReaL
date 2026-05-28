# SPDX-License-Identifier: Apache-2.0

import os

from areal.utils import logging

logger = logging.getLogger("EnvironUtils")

_warned_bool_env_var_keys = set()


def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if (value not in truthy_values) and (value not in falsy_values):
        if value not in _warned_bool_env_var_keys:
            logger.warning(
                f"get_bool_env_var({name}) see non-understandable value={value} and treat as false"
            )
        _warned_bool_env_var_keys.add(value)

    return value in truthy_values


def is_in_ci():
    return get_bool_env_var("AREAL_IS_IN_CI")


def is_single_controller():
    return not get_bool_env_var("AREAL_SPMD_MODE")
