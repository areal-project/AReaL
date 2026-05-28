"""Comparison utilities for FP8/BF16 comparison tests.

This module contains reusable functions for comparing tensors, activations,
gradients, and other model outputs between FP8 and BF16 models.
"""

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from tests.fp8.model_hooks import categorize_op_name

from areal.utils import logging

logger = logging.getLogger("FP8 BF16 Comparison Utils")


def compare_tensors(
    tensor_bf16: torch.Tensor,
    tensor_fp8: torch.Tensor,
    name: str = "tensor",
    check_nan_inf: bool = False,
    check_zero_norm: bool = False,
) -> dict[str, Any]:
    """Compare two tensors and return statistics.

    Args:
        tensor_bf16: BF16 tensor
        tensor_fp8: FP8 tensor
        name: Name identifier for logging
        check_nan_inf: Whether to check for NaN/Inf values
        check_zero_norm: Whether to check for zero norm

    Returns:
        Dictionary with comparison statistics:
        - max_diff: Maximum absolute difference
        - mean_diff: Mean absolute difference
        - cos_sim: Cosine similarity
        - bf16_norm: Norm of BF16 tensor
        - fp8_norm: Norm of FP8 tensor
        - has_nan: Whether any tensor has NaN
        - has_inf: Whether any tensor has Inf
        - zero_norm: Whether any tensor has zero norm
    """
    result = {
        "name": name,
        "shape_match": tensor_bf16.shape == tensor_fp8.shape,
    }

    if not result["shape_match"]:
        logger.warning(
            f"{name} shapes don't match: BF16={tensor_bf16.shape}, FP8={tensor_fp8.shape}"
        )
        return result

    # Calculate differences
    diff = (tensor_bf16 - tensor_fp8).abs()
    result["max_diff"] = diff.max().item()
    result["mean_diff"] = diff.mean().item()

    # Calculate norms
    bf16_norm = tensor_bf16.norm().item()
    fp8_norm = tensor_fp8.norm().item()
    result["bf16_norm"] = bf16_norm
    result["fp8_norm"] = fp8_norm

    # Check for NaN/Inf
    if check_nan_inf:
        bf16_has_nan = torch.isnan(tensor_bf16).any().item()
        bf16_has_inf = torch.isinf(tensor_bf16).any().item()
        fp8_has_nan = torch.isnan(tensor_fp8).any().item()
        fp8_has_inf = torch.isinf(tensor_fp8).any().item()
        result["has_nan"] = bf16_has_nan or fp8_has_nan
        result["has_inf"] = bf16_has_inf or fp8_has_inf

        if result["has_nan"] or result["has_inf"]:
            logger.warning(
                f"{name} has NaN/Inf: "
                f"BF16 NaN={bf16_has_nan}, Inf={bf16_has_inf}, "
                f"FP8 NaN={fp8_has_nan}, Inf={fp8_has_inf}"
            )

    # Check for zero norm
    if check_zero_norm:
        result["zero_norm"] = bf16_norm == 0.0 or fp8_norm == 0.0
        if result["zero_norm"]:
            logger.warning(
                f"{name} has zero norm: BF16 norm={bf16_norm:.6e}, FP8 norm={fp8_norm:.6e}"
            )

    # Calculate cosine similarity
    if check_zero_norm and result.get("zero_norm", False):
        result["cos_sim"] = 0.0
    else:
        tensor_bf16_flat = tensor_bf16.flatten()
        tensor_fp8_flat = tensor_fp8.flatten()
        cos_sim = F.cosine_similarity(
            tensor_bf16_flat.unsqueeze(0), tensor_fp8_flat.unsqueeze(0), dim=1
        ).item()

        if torch.isnan(torch.tensor(cos_sim)):
            logger.warning(f"{name} cosine similarity is NaN, setting to 0.0")
            cos_sim = 0.0

        result["cos_sim"] = cos_sim

    return result


def compare_tensors_dict(
    dict_bf16: dict[str, torch.Tensor],
    dict_fp8: dict[str, torch.Tensor],
    title: str = "Comparison",
    check_nan_inf: bool = False,
    check_zero_norm: bool = False,
    group_by_op_type: bool = True,
    name_width: int = 50,
) -> dict[str, Any]:
    """Compare two dictionaries of tensors and return statistics grouped by operation type.

    Args:
        dict_bf16: Dictionary of BF16 tensors
        dict_fp8: Dictionary of FP8 tensors
        title: Title for logging
        check_nan_inf: Whether to check for NaN/Inf values
        check_zero_norm: Whether to check for zero norm
        group_by_op_type: Whether to group statistics by operation type
        name_width: Width for name formatting in logs

    Returns:
        Dictionary with comparison statistics:
        - stats_by_type: Statistics grouped by operation type
        - individual_stats: Individual tensor statistics
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"{title} by Operation Type")
    logger.info("=" * 80)

    stats_by_type = defaultdict(
        lambda: {"max_diffs": [], "mean_diffs": [], "cos_sims": [], "names": []}
    )
    individual_stats = {}

    common_names = set(dict_bf16.keys()) & set(dict_fp8.keys())
    for name in sorted(common_names):
        tensor_bf16 = dict_bf16[name]
        tensor_fp8 = dict_fp8[name]

        # Skip None values
        if tensor_bf16 is None or tensor_fp8 is None:
            continue

        # Compare tensors
        comparison = compare_tensors(
            tensor_bf16,
            tensor_fp8,
            name=name,
            check_nan_inf=check_nan_inf,
            check_zero_norm=check_zero_norm,
        )

        if not comparison["shape_match"]:
            continue

        individual_stats[name] = comparison

        # Group by operation type if requested
        if group_by_op_type:
            op_type = categorize_op_name(name)
            stats_by_type[op_type]["max_diffs"].append(comparison["max_diff"])
            stats_by_type[op_type]["mean_diffs"].append(comparison["mean_diff"])
            stats_by_type[op_type]["cos_sims"].append(comparison["cos_sim"])
            stats_by_type[op_type]["names"].append(name)

            # Format with fixed width for alignment
            name_str = f"{name} ({op_type})"
            logger.info(
                f"{name_str:<{name_width}} "
                f"max_diff={comparison['max_diff']:>12.6f}, "
                f"mean_diff={comparison['mean_diff']:>12.6f}, "
                f"cos_sim={comparison['cos_sim']:>10.6f}"
            )
        else:
            logger.info(
                f"{name:<{name_width}} "
                f"max_diff={comparison['max_diff']:>12.6f}, "
                f"mean_diff={comparison['mean_diff']:>12.6f}, "
                f"cos_sim={comparison['cos_sim']:>10.6f}"
            )

    # Summary by op type
    if group_by_op_type and stats_by_type:
        logger.info("\n" + "-" * 80)
        logger.info(f"{title} Summary by Operation Type")
        logger.info("-" * 80)
        for op_type in sorted(stats_by_type.keys()):
            stats = stats_by_type[op_type]
            if stats["max_diffs"]:
                max_diff_val = max(stats["max_diffs"])
                mean_diff_val = sum(stats["mean_diffs"]) / len(stats["mean_diffs"])
                cos_sim_val = sum(stats["cos_sims"]) / len(stats["cos_sims"])
                logger.info(
                    f"{op_type:<50} "
                    f"max_diff={max_diff_val:>12.6f}, "
                    f"mean_diff={mean_diff_val:>12.6f}, "
                    f"cos_sim={cos_sim_val:>10.6f}, "
                    f"n_ops={len(stats['names']):>4}"
                )

    return {
        "stats_by_type": dict(stats_by_type),
        "individual_stats": individual_stats,
    }


def compare_logits(
    logits_bf16: torch.Tensor,
    logits_fp8: torch.Tensor,
) -> dict[str, Any]:
    """Compare logits between BF16 and FP8 models.

    Args:
        logits_bf16: BF16 logits tensor
        logits_fp8: FP8 logits tensor

    Returns:
        Dictionary with comparison statistics
    """
    logger.info("\n" + "=" * 80)
    logger.info("Logits Comparison")
    logger.info("=" * 80)

    comparison = compare_tensors(logits_bf16, logits_fp8, name="logits")

    if comparison["shape_match"]:
        logger.info(f"Logits max diff: {comparison['max_diff']:.6f}")
        logger.info(f"Logits mean diff: {comparison['mean_diff']:.6f}")
        logger.info(f"Logits cosine similarity: {comparison['cos_sim']:.6f}")
    else:
        logger.warning(
            f"Logits shapes don't match: BF16={logits_bf16.shape}, FP8={logits_fp8.shape}"
        )

    return comparison


def find_problematic_operations(
    stats_by_type: dict[str, dict[str, list]],
    threshold: float = 0.95,
) -> list[tuple[str, str, float, float]]:
    """Find operations with cosine similarity below threshold.

    Args:
        stats_by_type: Statistics grouped by operation type
        threshold: Cosine similarity threshold

    Returns:
        List of tuples: (op_type, name, cos_sim, max_diff)
    """
    problematic = []
    for op_type, stats in stats_by_type.items():
        for i, (name, cos_sim) in enumerate(zip(stats["names"], stats["cos_sims"])):
            if cos_sim < threshold:
                problematic.append((op_type, name, cos_sim, stats["max_diffs"][i]))
    return sorted(problematic, key=lambda x: x[2])  # Sort by cos_sim


def log_problematic_operations(
    stats_by_type: dict[str, dict[str, list]],
    threshold: float = 0.95,
    title: str = "Problematic Operations",
):
    """Log operations with cosine similarity below threshold.

    Args:
        stats_by_type: Statistics grouped by operation type
        threshold: Cosine similarity threshold
        title: Title for logging
    """
    problematic = find_problematic_operations(stats_by_type, threshold)

    logger.info("\n" + "=" * 80)
    logger.info(f"{title} (low cosine similarity, threshold={threshold})")
    logger.info("=" * 80)

    if problematic:
        for op_type, name, cos_sim, max_diff in problematic:
            logger.info(
                f"  {name} ({op_type}): cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}"
            )
    else:
        logger.info(f"No problematic operations found (all cos_sim >= {threshold})")

    logger.info("=" * 80)
