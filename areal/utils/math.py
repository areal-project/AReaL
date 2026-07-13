# SPDX-License-Identifier: Apache-2.0

# Common math utilities


def ceil_div(x: int, y: int) -> int:
    """Ceiling division."""
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    """Align x to the next multiple of y."""
    return ceil_div(x, y) * y
