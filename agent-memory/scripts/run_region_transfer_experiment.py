#!/usr/bin/env python3
"""
Region-Based Memory Transfer Experiment

Evaluates region-based memory transfer method on:
1. Intra-transfer: same benchmark, cross-subtask (e.g., BCB Computation → Network)
2. Inter-transfer: cross-benchmark (e.g., HLE → BCB)

Usage:
    # Train BCB with region tracking
    python run_region_transfer_experiment.py --mode train --benchmark bcb

    # Evaluate intra-transfer on BCB
    python run_region_transfer_experiment.py --mode eval_intra --benchmark bcb --checkpoint results/bcb_train/final

    # Evaluate inter-transfer: HLE → BCB
    python run_region_transfer_experiment.py --mode eval_inter --source hle --target bcb --checkpoint results/hle_train/final
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from memrl.configs.task_hierarchy import (
    TASK_HIERARCHY,
    BCB_INTRA_SPLITS,
    HLE_INTRA_SPLITS,
    get_primary_subtask,
)
from memrl.service.region_manager import RegionManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_with_region_tracking(benchmark: str, config_path: str, output_dir: str):
    """
    Train on benchmark with region utility tracking.

    Args:
        benchmark: "bcb" or "hle"
        config_path: Path to training config YAML
        output_dir: Where to save checkpoints
    """
    logger.info(f"Training {benchmark} with region tracking...")

    if benchmark == "bcb":
        from memrl.run.bcb_runner import BCBRunner
        runner = BCBRunner.from_config(config_path)
    elif benchmark == "hle":
        from memrl.run.hle_runner import HLERunner
        runner = HLERunner.from_config(config_path)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # Initialize region manager
    region_manager = RegionManager(
        memory_service=runner.mem,
        task_hierarchy=TASK_HIERARCHY,
        K_global=30,
        K_local=10,
    )

    # Attach to memory service for utility tracking during training
    runner.mem.region_manager = region_manager

    # Run training (region utility tracked automatically via update hooks)
    if benchmark == "bcb":
        runner.run(num_epochs=3)
    else:  # hle
        runner.run(num_sections=10)

    # After training, cluster memories into regions
    logger.info("Clustering memories into regions...")
    region_manager.cluster_memories()

    # Classify transfer patterns
    logger.info("Classifying transfer patterns...")
    region_manager.classify_transfer_patterns()

    # Save checkpoint with region info
    checkpoint_dir = Path(output_dir) / "final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    runner.mem.save_checkpoint_snapshot(str(checkpoint_dir), ckpt_id="final")
    region_manager.save(str(checkpoint_dir / "region_manager.json"))

    logger.info(f"Training complete. Checkpoint saved to {checkpoint_dir}")

    # Print region statistics
    logger.info(f"Global regions: {len(region_manager.global_regions)}")
    for benchmark_name, local_regions in region_manager.local_regions.items():
        logger.info(f"Local regions ({benchmark_name}): {len(local_regions)}")


def evaluate_intra_transfer(
    benchmark: str,
    checkpoint_dir: str,
    split_name: str = "split_1",
    use_region_gating: bool = True,
):
    """
    Evaluate intra-transfer: same benchmark, cross-subtask.

    Args:
        benchmark: "bcb" or "hle"
        checkpoint_dir: Path to trained checkpoint
        split_name: Which split to use (e.g., "split_1")
        use_region_gating: Whether to use region gating (False = MemRL baseline)
    """
    logger.info(f"Evaluating intra-transfer on {benchmark} (split={split_name}, region_gating={use_region_gating})")

    # Load checkpoint
    if benchmark == "bcb":
        from memrl.run.bcb_runner import BCBRunner
        runner = BCBRunner.from_checkpoint(checkpoint_dir)
        splits = BCB_INTRA_SPLITS
    elif benchmark == "hle":
        from memrl.run.hle_runner import HLERunner
        runner = HLERunner.from_checkpoint(checkpoint_dir)
        splits = HLE_INTRA_SPLITS
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # Load region manager if using region gating
    region_manager = None
    if use_region_gating:
        region_path = Path(checkpoint_dir) / "region_manager.json"
        if region_path.exists():
            region_manager = RegionManager.load(str(region_path), runner.mem)
            logger.info(f"Loaded region manager from {region_path}")
        else:
            logger.warning(f"Region manager not found at {region_path}, proceeding without region gating")

    # Get source and target subtasks
    if split_name not in splits:
        raise ValueError(f"Unknown split: {split_name}. Available: {list(splits.keys())}")

    source_subtasks = splits[split_name]["source"]
    target_subtasks = splits[split_name]["target"]

    logger.info(f"Source subtasks: {source_subtasks}")
    logger.info(f"Target subtasks: {target_subtasks}")

    # Run evaluation on target subtasks with source filtering
    results = run_eval_with_filtering(
        runner=runner,
        target_subtasks=target_subtasks,
        filter_source_subtasks=source_subtasks,
        region_manager=region_manager,
        eval_mode="intra",
    )

    # Save results
    output_file = Path(checkpoint_dir) / f"intra_eval_{split_name}_{'region' if use_region_gating else 'baseline'}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")
    logger.info(f"Success rate: {results['success_rate']:.2%}")
    logger.info(f"Negative transfer ratio: {results.get('negative_transfer_ratio', 0):.2%}")

    return results


def evaluate_inter_transfer(
    source_benchmark: str,
    target_benchmark: str,
    checkpoint_dir: str,
    use_region_gating: bool = True,
):
    """
    Evaluate inter-transfer: cross-benchmark.

    Args:
        source_benchmark: "bcb" or "hle" (where memory was trained)
        target_benchmark: "bcb" or "hle" (where to evaluate)
        checkpoint_dir: Path to source checkpoint
        use_region_gating: Whether to use region gating
    """
    logger.info(f"Evaluating inter-transfer: {source_benchmark} → {target_benchmark} (region_gating={use_region_gating})")

    # Load source checkpoint into target runner
    if target_benchmark == "bcb":
        from memrl.run.bcb_runner import BCBRunner
        runner = BCBRunner.from_checkpoint(checkpoint_dir)
    elif target_benchmark == "hle":
        from memrl.run.hle_runner import HLERunner
        runner = HLERunner.from_checkpoint(checkpoint_dir)
    else:
        raise ValueError(f"Unknown target benchmark: {target_benchmark}")

    # Load region manager
    region_manager = None
    if use_region_gating:
        region_path = Path(checkpoint_dir) / "region_manager.json"
        if region_path.exists():
            region_manager = RegionManager.load(str(region_path), runner.mem)
            logger.info(f"Loaded region manager from {region_path}")
        else:
            logger.warning(f"Region manager not found, proceeding without region gating")

    # Run evaluation on all target tasks with source benchmark filtering
    target_subtasks = TASK_HIERARCHY[target_benchmark]["subtasks"]

    results = run_eval_with_filtering(
        runner=runner,
        target_subtasks=target_subtasks,
        filter_source_benchmark=source_benchmark,
        region_manager=region_manager,
        eval_mode="inter",
    )

    # Save results
    output_file = Path(checkpoint_dir) / f"inter_eval_{source_benchmark}_to_{target_benchmark}_{'region' if use_region_gating else 'baseline'}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")
    logger.info(f"Success rate: {results['success_rate']:.2%}")

    return results


def run_eval_with_filtering(
    runner,
    target_subtasks: List[str],
    filter_source_subtasks: Optional[List[str]] = None,
    filter_source_benchmark: Optional[str] = None,
    region_manager: Optional[RegionManager] = None,
    eval_mode: str = "intra",
) -> Dict[str, Any]:
    """
    Run evaluation with source filtering and optional region gating.

    Returns:
        Dict with success_rate, per_subtask_results, etc.
    """
    # Get tasks filtered by target subtasks
    tasks = get_tasks_by_subtypes(runner, target_subtasks)

    results = []
    for task in tasks:
        # Get task description and metadata
        task_desc, task_meta = extract_task_info(runner, task)
        target_subtask = task_meta.get("source_subtask")

        # Retrieve with filtering
        retrieved = runner.mem.retrieve_query(
            task_desc,
            k=runner.retrieve_k if hasattr(runner, 'retrieve_k') else 5,
            threshold=get_threshold(runner),
            filter_source_subtasks=filter_source_subtasks,
            filter_source_benchmark=filter_source_benchmark,
            region_manager=region_manager,
            target_subtask=target_subtask,
            eval_mode=eval_mode,
        )

        # Generate solution
        solution = generate_solution(runner, task, retrieved)

        # Evaluate
        success = evaluate_solution(runner, task, solution)

        results.append({
            "task_id": task_meta.get("task_id"),
            "target_subtask": target_subtask,
            "success": success,
            "num_retrieved": len(retrieved.get("selected", [])),
        })

    # Compute metrics
    total = len(results)
    successes = sum(1 for r in results if r["success"])
    success_rate = successes / total if total > 0 else 0.0

    # Per-subtask breakdown
    per_subtask = {}
    for subtask in target_subtasks:
        subtask_results = [r for r in results if r["target_subtask"] == subtask]
        if subtask_results:
            subtask_successes = sum(1 for r in subtask_results if r["success"])
            per_subtask[subtask] = {
                "success_rate": subtask_successes / len(subtask_results),
                "count": len(subtask_results),
            }

    return {
        "success_rate": success_rate,
        "total_tasks": total,
        "successes": successes,
        "per_subtask": per_subtask,
        "detailed_results": results,
    }


def get_tasks_by_subtypes(runner, subtypes: List[str]) -> List:
    """Get tasks filtered by subtask types."""
    # Implementation depends on runner type
    if hasattr(runner, '_problems'):  # BCB
        all_tasks = list(runner._problems.values())
        filtered = []
        for task in all_tasks:
            domains = runner._get_task_domains(task)
            primary_subtask = get_primary_subtask("bigcodebench", {"domains": domains})
            if primary_subtask in subtypes:
                filtered.append(task)
        return filtered
    else:  # HLE
        # Load HLE dataset and filter by category
        # This is a placeholder - actual implementation depends on HLE data structure
        raise NotImplementedError("HLE task filtering not yet implemented")


def extract_task_info(runner, task) -> tuple:
    """Extract task description and metadata."""
    if hasattr(runner, '_problems'):  # BCB
        from memrl.run.bcb_runner import get_prompt
        task_desc = get_prompt(task, split=runner.sel.split)
        domains = runner._get_task_domains(task)
        task_meta = {
            "task_id": task.get("task_id"),
            "source_subtask": get_primary_subtask("bigcodebench", {"domains": domains}),
        }
        return task_desc, task_meta
    else:  # HLE
        raise NotImplementedError("HLE task info extraction not yet implemented")


def generate_solution(runner, task, retrieved):
    """Generate solution using runner's generation method."""
    # Placeholder - actual implementation depends on runner
    raise NotImplementedError("Solution generation not yet implemented")


def evaluate_solution(runner, task, solution) -> bool:
    """Evaluate solution correctness."""
    # Placeholder - actual implementation depends on runner
    raise NotImplementedError("Solution evaluation not yet implemented")


def get_threshold(runner) -> float:
    """Get retrieval threshold from runner."""
    if hasattr(runner, '_get_retrieve_threshold'):
        return runner._get_retrieve_threshold()
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Region-Based Memory Transfer Experiment")
    parser.add_argument("--mode", required=True, choices=["train", "eval_intra", "eval_inter"],
                        help="Experiment mode")
    parser.add_argument("--benchmark", help="Benchmark name (for train/eval_intra)")
    parser.add_argument("--source", help="Source benchmark (for eval_inter)")
    parser.add_argument("--target", help="Target benchmark (for eval_inter)")
    parser.add_argument("--config", help="Path to training config YAML")
    parser.add_argument("--checkpoint", help="Path to checkpoint directory")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--split", default="split_1", help="Intra-transfer split name")
    parser.add_argument("--no-region-gating", action="store_true",
                        help="Disable region gating (MemRL baseline)")

    args = parser.parse_args()

    if args.mode == "train":
        if not args.benchmark or not args.config:
            parser.error("--benchmark and --config required for train mode")
        train_with_region_tracking(args.benchmark, args.config, args.output)

    elif args.mode == "eval_intra":
        if not args.benchmark or not args.checkpoint:
            parser.error("--benchmark and --checkpoint required for eval_intra mode")
        evaluate_intra_transfer(
            args.benchmark,
            args.checkpoint,
            args.split,
            use_region_gating=not args.no_region_gating,
        )

    elif args.mode == "eval_inter":
        if not args.source or not args.target or not args.checkpoint:
            parser.error("--source, --target, and --checkpoint required for eval_inter mode")
        evaluate_inter_transfer(
            args.source,
            args.target,
            args.checkpoint,
            use_region_gating=not args.no_region_gating,
        )


if __name__ == "__main__":
    main()
