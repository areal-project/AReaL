# SPDX-License-Identifier: Apache-2.0

"""Small Archon utility helpers shared by experimental test runners."""


def strip_wrapper_prefixes(name: str) -> str:
    """Drop wrapper-generated path segments from parameter names."""
    return name.replace("._checkpoint_wrapped_module", "").replace("._orig_mod", "")
