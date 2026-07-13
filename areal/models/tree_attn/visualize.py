# SPDX-License-Identifier: Apache-2.0

import torch

from areal.utils import logging

logger = logging.getLogger("TreeAttentionViz")


# Helper function for visualizing attention masks
# Used for debugging and performance optimization
def visualize_attention_mask(mask_tensor: torch.Tensor, granularity: int = 128) -> None:
    """Visualize an attention mask as a text grid with configurable granularity.

    Parameters
    ----------
    mask_tensor : torch.Tensor
        A 2D boolean or numeric tensor representing the attention mask.
    granularity : int, default=128
        Maximum number of cells to display in each dimension.
        If the mask is larger than this, it will be downsampled by aggregating
        blocks. Each cell shows the density of attention in that block.
    """
    mask = mask_tensor.bool().cpu().float()
    n = mask.shape[0]

    if n == 0:
        logger.info("Attention Mask Visualization: empty mask (0x0)")
        return

    # Determine block size for downsampling
    if n <= granularity:
        # No downsampling needed, show full resolution
        display_size = n
        block_size = 1
        downsampled = mask
    else:
        # Downsample by averaging blocks
        display_size = granularity
        block_size = (n + granularity - 1) // granularity  # ceiling division

        # Pad mask to be evenly divisible by block_size
        padded_size = block_size * display_size
        if padded_size > n:
            padded_mask = torch.zeros((padded_size, padded_size), dtype=mask.dtype)
            padded_mask[:n, :n] = mask
            mask = padded_mask

        # Reshape and compute mean for each block
        mask_reshaped = mask[: block_size * display_size, : block_size * display_size]
        mask_reshaped = mask_reshaped.view(
            display_size, block_size, display_size, block_size
        )
        downsampled = mask_reshaped.mean(dim=(1, 3))

    # Define density characters (from empty to full)
    density_chars = " ·░▒▓█"

    # Build visualization
    lines = []
    lines.append(
        f"Attention Mask ({n}x{n}), displayed at {display_size}x{display_size} (block size: {block_size}x{block_size})"
    )

    # Header with column indices (every 10 columns)
    header_line1 = "     "
    header_line2 = "     "
    for i in range(display_size):
        if i % 10 == 0:
            header_line1 += (
                f"{(i * block_size) // 1000 % 10}" if i * block_size >= 1000 else " "
            )
            header_line2 += (
                f"{(i * block_size) // 100 % 10}" if i * block_size >= 100 else " "
            )
        else:
            header_line1 += " "
            header_line2 += " "
    lines.append(header_line1)
    lines.append(header_line2)

    header_line3 = "     " + "".join(
        f"{(i * block_size) // 10 % 10}" for i in range(display_size)
    )
    header_line4 = "     " + "".join(
        f"{(i * block_size) % 10}" for i in range(display_size)
    )
    lines.append(header_line3)
    lines.append(header_line4)
    lines.append("     " + "-" * display_size)

    for row_idx in range(display_size):
        row_chars = []
        for col_idx in range(display_size):
            density = downsampled[row_idx, col_idx].item()
            # Map density [0, 1] to character index [0, 5]
            char_idx = min(int(density * len(density_chars)), len(density_chars) - 1)
            if density > 0 and char_idx == 0:
                char_idx = (
                    1  # Ensure non-zero density shows at least the lightest character
                )
            row_chars.append(density_chars[char_idx])
        row_str = "".join(row_chars)
        actual_row = row_idx * block_size
        lines.append(f"{actual_row:4d}|{row_str}")

    visualization = "\n".join(lines)
    logger.info("Attention Mask Visualization:\n%s", visualization)
