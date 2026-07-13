# SPDX-License-Identifier: Apache-2.0


def get_block_size_from_config(
    quantization_config: dict[str, int | str | list[str]] | None,
    default: int = 128,
    strict: bool = False,
) -> int:
    """Extract block size from quantization configuration.

    Args:
        quantization_config: Quantization configuration dict, may contain "weight_block_size"
        default: Default block size to return if not specified in config or if format is invalid
        strict: If True, enforce that weight_block_size must be a square matrix [N, N].
                If False, only check that it's a list/tuple of length 2 and use the first element.
                If format is invalid and strict=False, returns default instead of raising.

    Returns:
        Block size as an integer.

    Raises:
        ValueError: If strict=True and weight_block_size format is invalid or not a square matrix.

    Examples:
        >>> config = {"weight_block_size": [128, 128]}
        >>> get_block_size_from_config(config)
        128
        >>> get_block_size_from_config(config, strict=True)
        128
        >>> get_block_size_from_config(None)
        128
    """
    if quantization_config is None:
        return default

    weight_block_size = quantization_config.get("weight_block_size", None)
    if weight_block_size is None:
        return default

    if not isinstance(weight_block_size, (list, tuple)) or len(weight_block_size) != 2:
        if strict:
            raise ValueError(
                f"weight_block_size must be a list/tuple of length 2, got {weight_block_size}"
            )
        return default

    if strict:
        if weight_block_size[0] != weight_block_size[1]:
            raise ValueError(
                f"weight_block_size must be a square matrix for hardware efficiency and simplicity, "
                f"got {weight_block_size}."
            )

    return weight_block_size[0]
