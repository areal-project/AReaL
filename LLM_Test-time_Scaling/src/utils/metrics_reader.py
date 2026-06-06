"""Utility functions for reading and displaying metrics from evaluation results."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_metrics_from_file(result_file: Path) -> Dict[str, Any]:
    """Load metrics from a result JSON file.

    Args:
        result_file: Path to the result JSON file

    Returns:
        Dictionary containing metrics and metadata

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file doesn't contain valid metrics
    """
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")

    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check if metrics are already computed
    if "metrics" in data:
        metrics = data["metrics"]
        metadata = {
            "config": data.get("config", {}),
            "benchmark": data.get("benchmark", "unknown"),
            "start_time": data.get("start_time"),
            "end_time": data.get("end_time"),
            "experiment_name": data.get("config", {}).get("experiment_name", "unknown"),
        }
        return {
            "metrics": metrics,
            "metadata": metadata,
            "source": "precomputed",
        }

    # If no precomputed metrics, compute from results
    results = data.get("results", [])
    if not results:
        raise ValueError(f"No results found in file: {result_file}")

    metrics = compute_metrics_from_results(results)
    metadata = {
        "config": data.get("config", {}),
        "benchmark": data.get("benchmark", "unknown"),
        "start_time": data.get("start_time"),
        "end_time": data.get("end_time"),
        "experiment_name": data.get("config", {}).get("experiment_name", "unknown"),
    }

    return {
        "metrics": metrics,
        "metadata": metadata,
        "source": "computed",
    }


def compute_metrics_from_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute aggregate metrics from a list of problem results.

    Args:
        results: List of problem result dictionaries

    Returns:
        Dictionary with computed metrics
    """
    total = len(results)
    if total == 0:
        return {}

    successful_results = [r for r in results if "error" not in r]
    if not successful_results:
        return {
            "total_problems": total,
            "successful_problems": 0,
            "failed_problems": total,
        }

    num_successful = len(successful_results)

    # Check if results have pass@k metrics
    if "metrics" in successful_results[0] and "pass@1" in successful_results[0]["metrics"]:
        # Results have pass@k metrics
        pass_at_1 = sum(r["metrics"]["pass@1"] for r in successful_results) / num_successful
        pass_at_k = sum(r["metrics"]["pass@k"] for r in successful_results) / num_successful
        avg_final_score = sum(r.get("final_solution", {}).get("score", 0.0) for r in successful_results) / num_successful
        avg_best_score = sum(r["metrics"].get("best_score_all", 0.0) for r in successful_results) / num_successful
        total_solutions = sum(r["metrics"]["num_total"] for r in successful_results)
        total_correct = sum(r["metrics"]["num_correct"] for r in successful_results)
    else:
        # Legacy format: only final solution evaluated
        pass_at_1 = sum(1 for r in successful_results if r.get("is_correct", False)) / num_successful
        pass_at_k = pass_at_1  # Same as pass@1 if we only have final solution
        avg_final_score = sum(r.get("score", 0.0) for r in successful_results) / num_successful
        avg_best_score = avg_final_score
        total_solutions = num_successful
        total_correct = sum(1 for r in successful_results if r.get("is_correct", False))

    total_tokens = sum(r.get("total_tokens", 0) for r in successful_results)

    return {
        "pass@1": pass_at_1,
        "pass@k": pass_at_k,
        "avg_final_score": avg_final_score,
        "avg_best_score": avg_best_score,
        "total_problems": total,
        "successful_problems": num_successful,
        "failed_problems": total - num_successful,
        "total_solutions_generated": total_solutions,
        "total_correct_solutions": total_correct,
        "total_tokens": total_tokens,
        "avg_tokens_per_problem": total_tokens / num_successful if num_successful > 0 else 0,
    }


def print_metrics_summary(metrics_data: Dict[str, Any], show_metadata: bool = True) -> None:
    """Print a formatted summary of metrics.

    Args:
        metrics_data: Dictionary returned by load_metrics_from_file
        show_metadata: Whether to show metadata (experiment name, benchmark, etc.)
    """
    metrics = metrics_data["metrics"]
    metadata = metrics_data.get("metadata", {})

    if show_metadata:
        print("\n" + "=" * 80)
        print("METRICS SUMMARY")
        print("=" * 80)
        print(f"Experiment: {metadata.get('experiment_name', 'unknown')}")
        print(f"Benchmark: {metadata.get('benchmark', 'unknown')}")
        if metadata.get("start_time") and metadata.get("end_time"):
            print(f"Time Range: {metadata['start_time']} to {metadata['end_time']}")
        print("=" * 80)

    print(f"\nTotal Problems: {metrics.get('total_problems', 0)}")
    print(f"Successful: {metrics.get('successful_problems', 0)}")
    print(f"Failed: {metrics.get('failed_problems', 0)}")

    if "pass@1" in metrics:
        print(f"\nPass@1: {metrics['pass@1']:.4f} ({metrics['pass@1']:.2%})")
    if "pass@k" in metrics:
        print(f"Pass@k: {metrics['pass@k']:.4f} ({metrics['pass@k']:.2%})")
    if "avg_final_score" in metrics:
        print(f"Avg Final Score: {metrics['avg_final_score']:.3f}")
    if "avg_best_score" in metrics:
        print(f"Avg Best Score: {metrics['avg_best_score']:.3f}")

    if "total_solutions_generated" in metrics:
        print(f"\nTotal Solutions Generated: {metrics['total_solutions_generated']}")
    if "total_correct_solutions" in metrics:
        print(f"Total Correct Solutions: {metrics['total_correct_solutions']}")
        if metrics.get('total_solutions_generated', 0) > 0:
            correct_rate = metrics['total_correct_solutions'] / metrics['total_solutions_generated']
            print(f"Solution Correctness Rate: {correct_rate:.4f} ({correct_rate:.2%})")

    if "total_tokens" in metrics:
        print(f"\nTotal Tokens: {metrics['total_tokens']:,}")
    if "avg_tokens_per_problem" in metrics:
        print(f"Avg Tokens per Problem: {metrics['avg_tokens_per_problem']:.0f}")

    if "duration_seconds" in metrics:
        print(f"\nDuration: {metrics['duration_seconds']:.1f}s")

    print("=" * 80)


def compare_metrics(
    result_files: List[Path],
    show_metadata: bool = False,
) -> None:
    """Compare metrics from multiple result files.

    Args:
        result_files: List of paths to result JSON files
        show_metadata: Whether to show metadata for each experiment
    """
    print("\n" + "=" * 80)
    print("METRICS COMPARISON")
    print("=" * 80)

    all_metrics = []
    for result_file in result_files:
        try:
            metrics_data = load_metrics_from_file(result_file)
            all_metrics.append((result_file, metrics_data))
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            continue

    if not all_metrics:
        print("No valid result files found.")
        return

    # Print header
    print(f"\n{'Experiment':<30} {'Pass@1':<10} {'Pass@k':<10} {'Avg Score':<12} {'Problems':<10}")
    print("-" * 80)

    # Print metrics for each file
    for result_file, metrics_data in all_metrics:
        metrics = metrics_data["metrics"]
        metadata = metrics_data.get("metadata", {})
        exp_name = metadata.get("experiment_name", result_file.stem)

        pass_at_1 = metrics.get("pass@1", 0.0)
        pass_at_k = metrics.get("pass@k", 0.0)
        avg_score = metrics.get("avg_final_score", 0.0)
        total_problems = metrics.get("total_problems", 0)

        print(
            f"{exp_name:<30} "
            f"{pass_at_1:<10.4f} "
            f"{pass_at_k:<10.4f} "
            f"{avg_score:<12.3f} "
            f"{total_problems:<10}"
        )

    print("=" * 80)

    # Print detailed summaries if requested
    if show_metadata:
        for result_file, metrics_data in all_metrics:
            print(f"\n{'='*80}")
            print(f"Details for: {result_file.name}")
            print_metrics_summary(metrics_data, show_metadata=True)


if __name__ == "__main__":
    """Command-line interface for reading metrics."""
    import argparse

    parser = argparse.ArgumentParser(description="Read and display metrics from evaluation results")
    parser.add_argument(
        "result_file",
        type=Path,
        help="Path to result JSON file",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        type=Path,
        help="Compare metrics from multiple files",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't show metadata",
    )

    args = parser.parse_args()

    if args.compare:
        compare_metrics(args.compare, show_metadata=not args.no_metadata)
    else:
        metrics_data = load_metrics_from_file(args.result_file)
        print_metrics_summary(metrics_data, show_metadata=not args.no_metadata)

