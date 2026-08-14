#!/usr/bin/env python3
"""
Extract per-subtask Q values from a region checkpoint.

Since per-subtask Q is maintained in RegionManager but may not be saved in snapshots,
this script reconstructs it by:
1. Loading the memory pool from checkpoint
2. Re-running a short evaluation phase with Q tracking enabled
3. Extracting and saving the per-subtask Q data

Usage:
    python extract_per_subtask_q.py \
        --checkpoint results/deepseek_region_local_embed_b32/.../epoch10/snapshot/10 \
        --output per_subtask_q.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_from_region_manager_file(region_manager_path: str) -> Dict:
    """
    Extract per-subtask Q from a saved region_manager.json file.

    Args:
        region_manager_path: Path to region_manager.json

    Returns:
        Dict[mem_id, Dict[subtask, q_value]]
    """
    logger.info(f"Loading region_manager from {region_manager_path}")

    with open(region_manager_path) as f:
        state = json.load(f)

    subtask_q = state.get("subtask_q", {})
    logger.info(f"Extracted per-subtask Q for {len(subtask_q)} memories")

    # Print sample
    if subtask_q:
        sample_id = list(subtask_q.keys())[0]
        logger.info(f"Sample: {sample_id[:40]}... -> {subtask_q[sample_id]}")

    return subtask_q


def reconstruct_from_checkpoint(checkpoint_dir: str, num_eval_samples: int = 200) -> Dict:
    """
    Reconstruct per-subtask Q by re-running evaluation on the checkpoint.

    This is needed if region_manager state was not saved.

    Args:
        checkpoint_dir: Path to checkpoint directory
        num_eval_samples: Number of samples to evaluate for Q estimation

    Returns:
        Dict[mem_id, Dict[subtask, q_value]]
    """
    logger.info(f"Reconstructing per-subtask Q from checkpoint: {checkpoint_dir}")
    logger.info(f"Will evaluate {num_eval_samples} samples to estimate Q values")

    # TODO: Implement reconstruction logic
    # Steps:
    # 1. Load memory pool from checkpoint
    # 2. Initialize RegionManager with the loaded memories
    # 3. Run evaluation on BCB val set (or a subset)
    # 4. Track which memories are retrieved for which subtasks and their rewards
    # 5. Build per-subtask Q from the tracked data

    raise NotImplementedError(
        "Reconstruction from checkpoint not yet implemented. "
        "Please ensure your region checkpoint includes region_manager.json"
    )


def compute_utility_statistics(subtask_q: Dict) -> Dict:
    """Compute statistics about utility patterns."""
    stats = {
        'n_memories': len(subtask_q),
        'subtasks': set(),
        'variance_distribution': [],
        'mean_distribution': [],
        'generalist_memories': [],  # Low variance
        'specialist_memories': [],  # High variance
    }

    import numpy as np

    for mem_id, q_dict in subtask_q.items():
        if not q_dict:
            continue

        stats['subtasks'].update(q_dict.keys())
        q_values = list(q_dict.values())

        variance = np.var(q_values)
        mean = np.mean(q_values)

        stats['variance_distribution'].append(variance)
        stats['mean_distribution'].append(mean)

        # Classify memory type
        if variance < 0.01:  # Very low variance = generalist
            stats['generalist_memories'].append({
                'mem_id': mem_id,
                'variance': variance,
                'mean': mean,
                'q_values': q_dict
            })
        elif variance > 0.05:  # High variance = specialist
            stats['specialist_memories'].append({
                'mem_id': mem_id,
                'variance': variance,
                'mean': mean,
                'q_values': q_dict
            })

    stats['subtasks'] = sorted(list(stats['subtasks']))
    stats['n_subtasks'] = len(stats['subtasks'])

    # Sort by variance
    stats['generalist_memories'].sort(key=lambda x: x['variance'])
    stats['specialist_memories'].sort(key=lambda x: x['variance'], reverse=True)

    # Keep only top-10 for each
    stats['generalist_memories'] = stats['generalist_memories'][:10]
    stats['specialist_memories'] = stats['specialist_memories'][:10]

    logger.info(f"Statistics:")
    logger.info(f"  Total memories: {stats['n_memories']}")
    logger.info(f"  Subtasks: {stats['n_subtasks']} - {stats['subtasks']}")
    logger.info(f"  Variance range: [{min(stats['variance_distribution']):.4f}, {max(stats['variance_distribution']):.4f}]")
    logger.info(f"  Mean range: [{min(stats['mean_distribution']):.4f}, {max(stats['mean_distribution']):.4f}]")
    logger.info(f"  Generalist memories (variance < 0.01): {len([v for v in stats['variance_distribution'] if v < 0.01])}")
    logger.info(f"  Specialist memories (variance > 0.05): {len([v for v in stats['variance_distribution'] if v > 0.05])}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract per-subtask Q from region checkpoint")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint directory or region_manager.json file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='per_subtask_q.json',
        help='Output JSON file for per-subtask Q data'
    )
    parser.add_argument(
        '--stats_output',
        type=str,
        default='per_subtask_q_stats.json',
        help='Output JSON file for statistics'
    )
    parser.add_argument(
        '--reconstruct',
        action='store_true',
        help='Reconstruct from checkpoint if region_manager.json not found'
    )
    parser.add_argument(
        '--num_eval_samples',
        type=int,
        default=200,
        help='Number of samples for reconstruction (if --reconstruct)'
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    # Try to find region_manager.json
    if checkpoint_path.is_file() and checkpoint_path.name == 'region_manager.json':
        region_manager_file = checkpoint_path
    else:
        # Search in checkpoint directory
        region_manager_file = checkpoint_path / 'region_manager.json'
        if not region_manager_file.exists():
            # Try parent directories
            for parent in [checkpoint_path, checkpoint_path.parent, checkpoint_path.parent.parent]:
                candidate = parent / 'region_manager.json'
                if candidate.exists():
                    region_manager_file = candidate
                    break

    # Extract per-subtask Q
    if region_manager_file.exists():
        logger.info(f"Found region_manager.json at {region_manager_file}")
        subtask_q = extract_from_region_manager_file(str(region_manager_file))
    elif args.reconstruct:
        logger.warning("region_manager.json not found, attempting reconstruction...")
        subtask_q = reconstruct_from_checkpoint(str(checkpoint_path), args.num_eval_samples)
    else:
        logger.error(f"region_manager.json not found in {checkpoint_path}")
        logger.error("Use --reconstruct to rebuild from checkpoint (not yet implemented)")
        sys.exit(1)

    # Save per-subtask Q
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(subtask_q, f, indent=2)
    logger.info(f"Saved per-subtask Q to {output_path}")

    # Compute and save statistics
    stats = compute_utility_statistics(subtask_q)

    # Convert numpy types to native Python for JSON serialization
    stats['variance_distribution'] = [float(v) for v in stats['variance_distribution']]
    stats['mean_distribution'] = [float(v) for v in stats['mean_distribution']]

    stats_path = Path(args.stats_output)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")

    logger.info("Extraction complete!")


if __name__ == '__main__':
    main()
