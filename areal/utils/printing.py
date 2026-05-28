# SPDX-License-Identifier: Apache-2.0

from tabulate import tabulate


def tabulate_stats(data: dict[str, float], col=4, floatfmt=".4e") -> str:
    items = list(data.items())
    # Calculate how many rows we'll need
    row_count = (len(items) + col - 1) // col

    # Reorganize items in column-major order
    column_major = []
    for i in range(row_count):
        row = []
        for j in range(col):
            index = i + j * row_count
            if index < len(items):
                row.extend(items[index])
        column_major.append(row)

    return tabulate(column_major, floatfmt=floatfmt, tablefmt="fancy_grid")
