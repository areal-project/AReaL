"""GPQA Diamond experiment script using ExperimentRunner for generating rollouts.

This script runs test-time scaling experiments on GPQA Diamond and generates rollouts
(all intermediate solutions) for analysis of pass@k metrics on graduate-level science questions.

Usage:
    # Run all experiments from scratch (default: gpt-oss-120b)
    python scripts/run_gpqa_experiment.py

    # Run with specific model
    python scripts/run_gpqa_experiment.py --model qwen3-235b

    # Control concurrency (default: 128)
    python scripts/run_gpqa_experiment.py --model gpt-oss-20b --max-concurrent 64

    # Auto-resume from latest results for each experiment
    python scripts/run_gpqa_experiment.py --resume --model gpt-oss-20b

    # Resume from specific results file (applies to all experiments)
    python scripts/run_gpqa_experiment.py --resume-from results/gpqa_rollouts_gpt-oss-120b/baseline_20240115_103000.json

Resume Functionality:
    The script supports resuming interrupted experiments by skipping already completed problems:
    - Loads existing results file
    - Extracts problem IDs that were successfully completed
    - Skips those problems and only runs remaining ones
    - Merges new results with existing results
    - Saves combined results to new file

    Use --resume to automatically detect and resume from the latest results file for each experiment.
    Use --resume-from to specify a specific results file to resume from.
"""

import asyncio
import os
from pathlib import Path

from src.experiment_runner import ExperimentRunner
from src.utils.config import (
    AggregationConfig,
    Config,
    EvaluationConfig,
    LLMConfig,
    ReflectionConfig,
)


def create_experiment_config(
    experiment_name: str,
    model_name: str,
    reflection_strategy: str,
    aggregation_strategy: str,
    n_samples: int,
    n_iterations: int,
    apply_agg_each_turn: bool,
    output_dir: str,
    api_key: str = None,
    api_base: str = None,
    max_concurrent_problems: int = 128,
    reasoning_effort: str = "auto",
    initial_effort: str = "auto",
) -> Config:
    """Create experiment configuration.

    Args:
        experiment_name: Name of the experiment
        model_name: Model name (e.g., "gpt-oss-120b")
        reflection_strategy: Reflection strategy name
        aggregation_strategy: Aggregation strategy name
        n_samples: Number of samples per iteration
        n_iterations: Number of iterations
        apply_agg_each_turn: Whether to aggregate at each turn
        output_dir: Output directory for results
        api_key: API key (defaults to OPENAI_API_KEY env var)
        api_base: API base URL (defaults to OPENAI_API_BASE env var)
        max_concurrent_problems: Maximum number of problems to process concurrently
        reasoning_effort: Reasoning effort level for extended thinking

    Returns:
        Config object
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY", "None")
    api_base = api_base or os.getenv("OPENAI_API_BASE")

    return Config(
        experiment_name=experiment_name,
        llm=LLMConfig(
            provider="litellm",
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=0.7,
            reasoning_effort=initial_effort,  # Use initial_effort for initial solution generation
        ),
        evaluation=EvaluationConfig(
            provider="litellm",
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            benchmark="gpqa_diamond",
            evaluator_type="llm_judge",  # Use rule-based GPQA evaluator
        ),
        reflection=ReflectionConfig(
            strategy=reflection_strategy,
            n_iterations=n_iterations,
            n_samples_per_iteration=n_samples,
            reasoning_effort=reasoning_effort  # Use reasoning_effort (refine_effort) for reflection
        ),
        aggregation=AggregationConfig(
            strategy=aggregation_strategy,
            apply_at_each_turn=apply_agg_each_turn,
            reasoning_effort=reasoning_effort
        ),
        output_dir=output_dir,
        max_concurrent_problems=max_concurrent_problems,
    )


async def run_baseline(
    model_name: str,
    output_dir: str,
    api_key: str = None,
    api_base: str = None,
    resume_from: str = None,
    max_concurrent_problems: int = 128,
    reasoning_effort: str = "auto",
    experiment_name: str = "baseline",
    initial_effort: str = "auto",
) -> None:
    """Run baseline experiment (single generation, no reflection/aggregation).

    Args:
        model_name: Model name
        output_dir: Output directory
        api_key: API key
        api_base: API base URL
        resume_from: Path to existing results file to resume from
        max_concurrent_problems: Maximum number of problems to process concurrently
        reasoning_effort: Reasoning effort level
    """
    print("\n" + "=" * 80)
    print("BASELINE EXPERIMENT")
    print("=" * 80)

    config = create_experiment_config(
        experiment_name=experiment_name,
        model_name=model_name,
        reflection_strategy="none",
        aggregation_strategy="none",
        n_samples=8,
        n_iterations=0,
        apply_agg_each_turn=False,
        output_dir=output_dir,
        api_key=api_key,
        api_base=api_base,
        max_concurrent_problems=max_concurrent_problems,
        reasoning_effort=reasoning_effort,
        initial_effort=initial_effort,
    )

    runner = ExperimentRunner(config, resume_from=resume_from)
    await runner.run()


async def run_scaling_experiment(
    experiment_name: str,
    model_name: str,
    reflection_strategy: str,
    aggregation_strategy: str,
    n_samples: int,
    n_iterations: int,
    apply_agg_each_turn: bool,
    output_dir: str,
    api_key: str = None,
    api_base: str = None,
    resume_from: str = None,
    max_concurrent_problems: int = 128,
    reasoning_effort: str = "auto",
    initial_effort: str = "auto",
) -> None:
    """Run a test-time scaling experiment.

    Args:
        experiment_name: Name of the experiment
        model_name: Model name
        reflection_strategy: Reflection strategy
        aggregation_strategy: Aggregation strategy
        n_samples: Number of samples per iteration
        n_iterations: Number of iterations
        apply_agg_each_turn: Whether to aggregate at each turn
        output_dir: Output directory
        api_key: API key
        api_base: API base URL
        resume_from: Path to existing results file to resume from
        max_concurrent_problems: Maximum number of problems to process concurrently
        reasoning_effort: Reasoning effort level
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Reflection: {reflection_strategy}")
    print(f"Aggregation: {aggregation_strategy}")
    print(f"Samples: {n_samples}")
    print(f"Iterations: {n_iterations}")
    print(f"Architecture: {'agg_each_turn' if apply_agg_each_turn else 'reflect_then_agg'}")
    print(f"Initial Effort: {initial_effort}")
    print(f"Refine Effort: {reasoning_effort}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print("=" * 80)

    config = create_experiment_config(
        experiment_name=experiment_name,
        model_name=model_name,
        reflection_strategy=reflection_strategy,
        aggregation_strategy=aggregation_strategy,
        n_samples=n_samples,
        n_iterations=n_iterations,
        apply_agg_each_turn=apply_agg_each_turn,
        output_dir=output_dir,
        api_key=api_key,
        api_base=api_base,
        max_concurrent_problems=max_concurrent_problems,
        reasoning_effort=reasoning_effort,
        initial_effort=initial_effort,
    )

    runner = ExperimentRunner(config, resume_from=resume_from)
    await runner.run()


async def main():
    """Main function to run all GPQA Diamond experiments."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run GPQA Diamond experiments with test-time scaling"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-resume from latest results files for each experiment",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from specific results file (applies to all experiments)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-120b",
        choices=["gpt-oss-120b", "gpt-oss-20b", "qwen3-235b", "qwen3-30b", "qwen3-4b", "qwen3-4b-newenc", "qwen3-30b-a3b", "qwen3-30b-a3b-newenc"],
        help="Model to use for experiments (default: gpt-oss-120b)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=128,
        help="Maximum number of problems to process concurrently (default: 128)",
    )
    args = parser.parse_args()

    # ========== Configuration ==========
    # Map model names to full paths
    MODEL_PATHS = {
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "qwen3-235b": "openai/Qwen__Qwen3-235B-A22B",
        "qwen3-30b": "openai/Qwen__Qwen3-30B-A3B",
        "qwen3-30b-a3b":"openai/qwen3-30b-a3b",
        "qwen3-4b": "openai/qwen3-4b",
        "qwen3-4b-newenc": "openai/qwen3-4b-newenc",
        "qwen3-30b-a3b-newenc": "openai/qwen3-30b-a3b-newenc"
    }

    MODEL = MODEL_PATHS[args.model]
    OUTPUT_DIR = f"results/test_time_compute/gpqa_rollouts_{args.model}"
    API_KEY = os.getenv("OPENAI_API_KEY")
    API_BASE = os.getenv("OPENAI_API_BASE")  # Optional: Custom API endpoint

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ========== Experiment Definitions ==========
    experiments = [
        # Baseline: Single generation, no scaling
        # {
        #     "type": "baseline",
        #     "name": "baselinex8"
        # },

        # Self-Evaluation: Test different effort combinations for initial solution and refinement
        # Test with iteration 1 to see the effect of different effort combinations
        # {
        #     "name": "init_low_refine_low",
        #     "reflection": "self_evaluation",
        #     "aggregation": "none",
        #     "n_samples": 4,
        #     "n_iterations": 1,
        #     "agg_each_turn": False,
        #     "initial_effort": "low",
        #     "refine_effort": "low",
        # },
        # {
        #     "name": "init_low_refine_high",
        #     "reflection": "self_evaluation",
        #     "aggregation": "none",
        #     "n_samples": 4,
        #     "n_iterations": 1,
        #     "agg_each_turn": False,
        #     "initial_effort": "low",
        #     "refine_effort": "high",
        # },
        # {
        #     "name": "init_high_refine_low",
        #     "reflection": "self_evaluation",
        #     "aggregation": "none",
        #     "n_samples": 4,
        #     "n_iterations": 1,
        #     "agg_each_turn": False,
        #     "initial_effort": "high",
        #     "refine_effort": "low",
        # },
        # {
        #     "name": "init_high_refine_high",
        #     "reflection": "self_evaluation",
        #     "aggregation": "none",
        #     "n_samples": 4,
        #     "n_iterations": 1,
        #     "agg_each_turn": False,
        #     "initial_effort": "high",
        #     "refine_effort": "high",
        # },

        
        # No-Feedback: Generate → Refine (Sequential)
        # {
        #     "name": "no_feedback_sequential",
        #     "reflection": "no_feedback",
        #     "aggregation": "none",
        #     "n_samples": 1,
        #     "n_iterations": 8,
        #     "agg_each_turn": False,
        #     "reasoning_effort": "auto"
        # },
        {
            "name": "ground_truth_correctness_sequential_simple_feedback_4*8_debug",
            "reflection": "ground_truth_simple",
            "aggregation": "none",
            "n_samples": 4,
            "n_iterations": 8,
            "agg_each_turn": False,
        },
    ]

    # ========== Run Experiments ==========
    print("\n" + "=" * 80)
    print("GPQA DIAMOND ROLLOUT GENERATION")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"API Base: {API_BASE or 'Default'}")
    print(f"Max Concurrent Problems: {args.max_concurrent}")
    print(f"Total Experiments: {len(experiments)}")
    if args.resume:
        print("Mode: Auto-resume from latest results")
    elif args.resume_from:
        print(f"Mode: Resume from {args.resume_from}")
    print("=" * 80)

    for idx, exp_config in enumerate(experiments, 1):
        print(f"\n[{idx}/{len(experiments)}] Starting experiment...")

        # Determine resume file for this experiment
        resume_from = args.resume_from

        if args.resume and not resume_from:
            # Auto-detect latest results file for this experiment
            exp_name = exp_config.get("name", exp_config.get("type", "unknown"))
            output_path = Path(OUTPUT_DIR)
            if output_path.exists():
                pattern = f"{exp_name}_gpqa_diamond_*.json"
                result_files = sorted(
                    output_path.glob(pattern), key=lambda p: p.stat().st_mtime
                )
                if result_files:
                    resume_from = str(result_files[-1])
                    print(f"  Auto-detected resume file: {resume_from}")

        try:
            if exp_config.get("type") == "baseline":
                await run_baseline(
                    MODEL, OUTPUT_DIR, API_KEY, API_BASE,
                    resume_from=resume_from,
                    max_concurrent_problems=args.max_concurrent,
                    reasoning_effort=exp_config.get("reasoning_effort", "auto"),
                    experiment_name=exp_config["name"],
                    initial_effort=exp_config.get("initial_effort", "auto"),
                )
            else:
                await run_scaling_experiment(
                    experiment_name=exp_config["name"],
                    model_name=MODEL,
                    reflection_strategy=exp_config["reflection"],
                    aggregation_strategy=exp_config["aggregation"],
                    n_samples=exp_config["n_samples"],
                    n_iterations=exp_config["n_iterations"],
                    apply_agg_each_turn=exp_config["agg_each_turn"],
                    output_dir=OUTPUT_DIR,
                    api_key=API_KEY,
                    api_base=API_BASE,
                    resume_from=resume_from,
                    max_concurrent_problems=args.max_concurrent,
                    reasoning_effort=exp_config.get("refine_effort", exp_config.get("reasoning_effort", "auto")),  # refine_effort maps to reflection_effort
                    initial_effort=exp_config.get("initial_effort", "auto"),
                )

        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Small delay between experiments
        await asyncio.sleep(2)

    # ========== Final Summary ==========
    print("\n" + "=" * 80)
    print("✅ ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print(f"Results saved in: {OUTPUT_DIR}")
    print("\nGenerated rollouts include:")
    print("  • All intermediate solutions (n_samples × n_iterations)")
    print("  • Rule-based evaluation results for each solution (A/B/C/D answer matching)")
    print("  • Pass@1 and Pass@k metrics")
    print("  • Token usage and timing information")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
