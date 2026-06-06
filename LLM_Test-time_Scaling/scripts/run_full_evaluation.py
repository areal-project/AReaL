#!/usr/bin/env python3
"""Complete evaluation pipeline: Start LLM services and run benchmarks.

This script:
1. Launches LLM services on SLURM nodes
2. Waits for services to be ready
3. Runs evaluations on IMOBench and LiveCodeBench-Pro
4. Optionally cleans up services

Usage:
    python scripts/run_full_evaluation.py --model-path <MODEL_PATH> --num-services <N>
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.slurm.service_manager import ServiceRegistry
from scripts.slurm.wait_for_services import wait_for_services


def launch_services(
    num_services: int,
    model_path: str,
    start_port: int = 8000,
    registry_file: str = "services.txt",
) -> list:
    """Launch LLM services on SLURM nodes.

    Args:
        num_services: Number of services to launch
        model_path: Path to model
        start_port: Starting port number
        registry_file: Service registry file path

    Returns:
        List of job IDs
    """
    print("=" * 80)
    print("LAUNCHING LLM SERVICES")
    print("=" * 80)
    print(f"Number of services: {num_services}")
    print(f"Model path: {model_path}")
    print(f"Start port: {start_port}")
    print(f"Registry: {registry_file}")
    print("=" * 80)

    # Initialize registry
    registry_path = project_root / registry_file
    if registry_path.exists():
        registry_path.unlink()  # Clear old registry

    # Launch services using the launch script
    launch_script = project_root / "scripts" / "slurm" / "launch_multiple_services.sh"

    cmd = [
        "bash",
        str(launch_script),
        str(num_services),
        model_path,
        str(start_port),
        registry_file,
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Failed to launch services:")
        print(result.stderr)
        return []

    # Extract job IDs from output
    job_ids = []
    for line in result.stdout.split("\n"):
        if "Submitted: Job ID" in line:
            job_id = line.split()[-1]
            job_ids.append(job_id)

    print(f"\n✅ Launched {len(job_ids)} services")
    print(f"Job IDs: {job_ids}")

    return job_ids


async def run_benchmark_evaluation(
    benchmark: str,
    model_name: str,
    api_base: str,
    output_dir: Path,
    experiment_config: dict = None,
) -> bool:
    """Run evaluation on a benchmark.

    Args:
        benchmark: Benchmark name ("imobench" or "lcb_pro")
        model_name: Model name for API
        api_base: API base URL(s)
        output_dir: Output directory
        experiment_config: Experiment configuration dict

    Returns:
        True if successful
    """
    print("\n" + "=" * 80)
    print(f"RUNNING {benchmark.upper()} EVALUATION")
    print("=" * 80)

    # Set environment variables
    os.environ["OPENAI_API_BASE"] = api_base
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "None"

    # Import and run the appropriate experiment script
    if benchmark == "imobench":
        try:
            # Import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "run_imobench_experiment",
                project_root / "scripts" / "run_imobench_experiment.py"
            )
            imobench_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(imobench_module)
            
            # Run the async main function
            await imobench_module.main()
            return True
        except Exception as e:
            print(f"❌ IMOBench evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    elif benchmark == "lcb_pro":
        try:
            # Import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "run_lcb_pro_experiment",
                project_root / "scripts" / "run_lcb_pro_experiment.py"
            )
            lcb_pro_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lcb_pro_module)
            
            # Run the async main function
            await lcb_pro_module.main()
            return True
        except Exception as e:
            print(f"❌ LiveCodeBench-Pro evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    else:
        print(f"❌ Unknown benchmark: {benchmark}")
        return False


def cleanup_services(registry_file: str = "services.txt", force: bool = False) -> None:
    """Clean up running services.

    Args:
        registry_file: Service registry file
        force: Force cleanup without confirmation
    """
    print("\n" + "=" * 80)
    print("CLEANING UP SERVICES")
    print("=" * 80)

    registry = ServiceRegistry(registry_file)
    services = registry.get_running_services()

    if not services:
        print("No running services to clean up.")
        return

    if not force:
        response = input(f"Cancel {len(services)} running services? [y/N]: ")
        if response.lower() != "y":
            print("Skipping cleanup.")
            return

    # Use service_manager to cancel
    try:
        from scripts.slurm.service_manager import cmd_cancel
        
        class Args:
            registry = registry_file
            force = force

        cmd_cancel(Args())
    except Exception as e:
        print(f"Warning: Could not use service_manager to cancel: {e}")
        print("Please cancel services manually:")
        for service in services:
            print(f"  scancel {service.job_id}")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Complete evaluation pipeline with LLM services"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model (e.g., gpt-oss-120b)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for API (default: derived from model-path)",
    )
    parser.add_argument(
        "--num-services",
        type=int,
        default=4,
        help="Number of service instances to launch (default: 4)",
    )
    parser.add_argument(
        "--start-port",
        type=int,
        default=8000,
        help="Starting port number (default: 8000)",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="services.txt",
        help="Service registry file (default: services.txt)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["imobench", "lcb_pro", "all"],
        default=["all"],
        help="Benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--skip-launch",
        action="store_true",
        help="Skip launching services (use existing services)",
    )
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Skip waiting for services (assume they're ready)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up services after evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout for waiting for services in seconds (default: 600)",
    )

    args = parser.parse_args()

    # Determine model name
    model_name = args.model_name
    if model_name is None:
        # Extract model name from path
        model_path_parts = Path(args.model_path).parts
        if "models" in model_path_parts:
            idx = model_path_parts.index("models")
            model_name = f"openai//{Path(*model_path_parts[idx:])}"
        else:
            model_name = f"openai//{args.model_path}"

    print("=" * 80)
    print("FULL EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Model name: {model_name}")
    print(f"Number of services: {args.num_services}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Step 1: Launch services
    job_ids = []
    if not args.skip_launch:
        job_ids = launch_services(
            num_services=args.num_services,
            model_path=args.model_path,
            start_port=args.start_port,
            registry_file=args.registry,
        )

        if not job_ids:
            print("❌ Failed to launch services. Exiting.")
            return

        # Give services a moment to register
        print("\nWaiting for services to register...")
        time.sleep(10)

    # Step 2: Wait for services to be ready
    if not args.skip_wait:
        print("\n" + "=" * 80)
        print("WAITING FOR SERVICES")
        print("=" * 80)

        success = wait_for_services(
            registry_file=args.registry,
            timeout=args.timeout,
            min_healthy=args.num_services if not args.skip_launch else 1,
        )

        if not success:
            print("❌ Services did not become ready. Exiting.")
            return

    # Step 3: Get service URLs
    registry = ServiceRegistry(args.registry)
    api_bases = registry.get_api_bases()
    if not api_bases:
        print("❌ No running services found. Exiting.")
        return

    print(f"\n✅ Using services: {api_bases}")

    # Step 4: Run benchmarks
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks_to_run = []
    if "all" in args.benchmarks:
        benchmarks_to_run = ["imobench", "lcb_pro"]
    else:
        benchmarks_to_run = args.benchmarks

    results = {}
    for benchmark in benchmarks_to_run:
        success = await run_benchmark_evaluation(
            benchmark=benchmark,
            model_name=model_name,
            api_base=api_bases,
            output_dir=output_dir,
        )
        results[benchmark] = success

    # Step 5: Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    for benchmark, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{benchmark}: {status}")
    print("=" * 80)

    # Step 6: Cleanup (optional)
    if args.cleanup:
        cleanup_services(registry_file=args.registry, force=True)
    else:
        print(f"\nServices are still running. Job IDs: {job_ids}")
        print("To clean up later, run:")
        print(f"  python scripts/slurm/service_manager.py cancel --force")


if __name__ == "__main__":
    asyncio.run(main())

