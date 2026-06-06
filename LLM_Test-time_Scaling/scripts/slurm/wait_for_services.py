#!/usr/bin/env python3
"""Wait for LLM services to be ready before running experiments.

This script checks service health and waits until all services are ready.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import service manager
try:
    from scripts.slurm.service_manager import ServiceRegistry
except ImportError:
    # Fallback: import directly if in same directory
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "service_manager",
        Path(__file__).parent / "service_manager.py"
    )
    service_manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service_manager)
    ServiceRegistry = service_manager.ServiceRegistry


def wait_for_services(
    registry_file: str = "services.txt",
    timeout: int = 600,
    check_interval: int = 10,
    min_healthy: int = 1,
) -> bool:
    """Wait for services to become healthy.

    Args:
        registry_file: Path to service registry file
        timeout: Maximum time to wait in seconds
        check_interval: Time between health checks in seconds
        min_healthy: Minimum number of healthy services required

    Returns:
        True if services are ready, False if timeout
    """
    registry = ServiceRegistry(registry_file)
    start_time = time.time()

    print(f"Waiting for services to be ready...")
    print(f"  Registry: {registry_file}")
    print(f"  Timeout: {timeout}s")
    print(f"  Check interval: {check_interval}s")
    print(f"  Min healthy services: {min_healthy}")
    print()

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"\n❌ Timeout after {timeout}s")
            return False

        services = registry.get_running_services()
        if not services:
            print(f"[{int(elapsed)}s] No running services found, waiting...")
            time.sleep(check_interval)
            continue

        # Check health of all services
        healthy_count = 0
        for service in services:
            if registry.check_service_health(service, timeout=5):
                healthy_count += 1

        print(
            f"[{int(elapsed)}s] {healthy_count}/{len(services)} services healthy",
            end="\r",
        )

        if healthy_count >= min_healthy:
            print(f"\n✅ {healthy_count} service(s) are healthy and ready!")
            return True

        time.sleep(check_interval)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Wait for LLM services to be ready"
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="services.txt",
        help="Path to service registry file",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Maximum time to wait in seconds (default: 600)",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=10,
        help="Time between health checks in seconds (default: 10)",
    )
    parser.add_argument(
        "--min-healthy",
        type=int,
        default=1,
        help="Minimum number of healthy services required (default: 1)",
    )

    args = parser.parse_args()

    success = wait_for_services(
        registry_file=args.registry,
        timeout=args.timeout,
        check_interval=args.check_interval,
        min_healthy=args.min_healthy,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

