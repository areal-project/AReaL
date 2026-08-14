# SPDX-License-Identifier: Apache-2.0

import os

from areal.utils import logging

logger = logging.getLogger("EnvironUtils")

_warned_bool_env_var_keys = set()
_warned_rank_env_var_values = set()


def get_env_var(
    name: str,
    default: str | None = None,
    *,
    fallback_names: tuple[str, ...] = (),
    allow_empty: bool = False,
) -> str | None:
    """Read an environment variable with ordered fallback names.

    Empty values are skipped by default so a legacy name can supply the value.
    Set ``allow_empty=True`` when an explicitly empty value has domain meaning.
    """
    for candidate in (name, *fallback_names):
        value = os.getenv(candidate)
        if value is None:
            continue
        if allow_empty or value.strip() != "":
            return value
    return default


def get_bool_env_var(
    name: str,
    default: str = "false",
    *,
    fallback_names: tuple[str, ...] = (),
    truthy_values: tuple[str, ...] = ("true", "1"),
    falsy_values: tuple[str, ...] = ("false", "0"),
    strip_value: bool = False,
) -> bool:
    """Read a boolean environment variable.

    The default accepted values remain backward compatible. Callers that
    historically accepted additional spellings can opt in through
    ``truthy_values`` and ``falsy_values``.
    """
    value = get_env_var(
        name,
        default,
        fallback_names=fallback_names,
        # Preserve the original single-name behavior: an explicitly empty
        # boolean is invalid rather than silently replaced by the default.
        allow_empty=not fallback_names,
    )
    assert value is not None
    value = (value.strip() if strip_value else value).lower()

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


def rank_in_env_filter(name: str, rank: int) -> bool:
    """Return whether rank is selected by a comma/range env filter.

    Empty or unset values mean all ranks. Accepted examples: ``0``, ``0,2``,
    ``0-3``, and ``all``.
    """

    value = os.getenv(name)
    if value is None or value.strip() == "":
        return True

    value = value.strip().lower()
    if value == "all":
        return True

    selected: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start_raw, end_raw = part.split("-", 1)
                start = int(start_raw)
                end = int(end_raw)
                if end < start:
                    raise ValueError
                selected.update(range(start, end + 1))
            else:
                selected.add(int(part))
        except ValueError:
            warn_key = (name, value)
            if warn_key not in _warned_rank_env_var_values:
                logger.warning(
                    "rank_in_env_filter(%s) got invalid value=%s and will ignore part=%s",
                    name,
                    value,
                    part,
                )
                _warned_rank_env_var_values.add(warn_key)

    return rank in selected
