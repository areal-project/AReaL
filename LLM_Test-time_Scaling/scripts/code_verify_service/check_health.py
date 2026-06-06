#!/usr/bin/env python3
"""Check health of code verify services.

This script reads service registry and checks health of all registered services.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ServiceInfo:
    """Information about a running service."""

    job_id: str
    hostname: str
    host_ip: str
    port: int
    data_dir: str
    start_time: str
    status: str
    end_time: Optional[str] = None
    exit_code: Optional[int] = None

    @property
    def url(self) -> str:
        """Get the service URL."""
        return f"http://{self.host_ip}:{self.port}"

    @property
    def health_url(self) -> str:
        """Get the health check URL."""
        return f"{self.url}/health"

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self.status == "running"


def parse_service_registry(registry_file: str) -> List[ServiceInfo]:
    """Parse service registry file.

    Args:
        registry_file: Path to service registry file

    Returns:
        List of ServiceInfo objects
    """
    registry_path = Path(registry_file)
    if not registry_path.exists():
        return []

    services = []
    with open(registry_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            if len(parts) < 7:
                continue

            service = ServiceInfo(
                job_id=parts[0],
                hostname=parts[1],
                host_ip=parts[2],
                port=int(parts[3]),
                data_dir=parts[4],
                start_time=parts[5],
                status=parts[6],
                end_time=parts[7] if len(parts) > 7 else None,
                exit_code=int(parts[8]) if len(parts) > 8 and parts[8] else None,
            )
            services.append(service)

    return services


def check_service_health(service: ServiceInfo, timeout: int = 5) -> bool:
    """Check if a service is healthy by making a test request.

    Args:
        service: ServiceInfo to check
        timeout: Request timeout in seconds

    Returns:
        True if service is healthy, False otherwise
    """
    try:
        import requests

        response = requests.get(service.health_url, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def check_all_services(
    registry_file: str = "services.txt",
    timeout: int = 5,
    only_running: bool = True,
) -> tuple[List[ServiceInfo], List[ServiceInfo]]:
    """Check health of all services in registry.

    Args:
        registry_file: Path to service registry file
        timeout: Request timeout in seconds
        only_running: Only check services with status "running"

    Returns:
        Tuple of (healthy_services, unhealthy_services)
    """
    services = parse_service_registry(registry_file)

    if only_running:
        services = [s for s in services if s.is_running]

    healthy_services = []
    unhealthy_services = []

    for service in services:
        if check_service_health(service, timeout=timeout):
            healthy_services.append(service)
        else:
            unhealthy_services.append(service)

    return healthy_services, unhealthy_services


def wait_for_services(
    registry_file: str = "services.txt",
    timeout: int = 600,
    check_interval: int = 10,
    min_healthy: int = 1,
    health_check_timeout: int = 5,
) -> bool:
    """Wait for services to become healthy.

    Args:
        registry_file: Path to service registry file
        timeout: Maximum time to wait in seconds
        check_interval: Time between health checks in seconds
        min_healthy: Minimum number of healthy services required
        health_check_timeout: Timeout for individual health checks

    Returns:
        True if services are ready, False if timeout
    """
    start_time = time.time()

    print(f"Waiting for code verify services to be ready...")
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

        services = parse_service_registry(registry_file)
        running_services = [s for s in services if s.is_running]

        if not running_services:
            print(f"[{int(elapsed)}s] No running services found, waiting...")
            time.sleep(check_interval)
            continue

        # Check health of all services
        healthy_count = 0
        for service in running_services:
            if check_service_health(service, timeout=health_check_timeout):
                healthy_count += 1

        print(
            f"[{int(elapsed)}s] {healthy_count}/{len(running_services)} services healthy",
            end="\r",
        )

        if healthy_count >= min_healthy:
            print(f"\n✅ {healthy_count} service(s) are healthy and ready!")
            return True

        time.sleep(check_interval)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check health of code verify services"
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
        default=5,
        help="Request timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for services to become healthy",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=600,
        help="Maximum time to wait in seconds when using --wait (default: 600)",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=10,
        help="Time between health checks in seconds when using --wait (default: 10)",
    )
    parser.add_argument(
        "--min-healthy",
        type=int,
        default=1,
        help="Minimum number of healthy services required when using --wait (default: 1)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all services, not just running ones",
    )
    parser.add_argument(
        "--urls-only",
        action="store_true",
        help="Only print healthy service URLs (one per line)",
    )

    args = parser.parse_args()

    if args.wait:
        success = wait_for_services(
            registry_file=args.registry,
            timeout=args.wait_timeout,
            check_interval=args.check_interval,
            min_healthy=args.min_healthy,
            health_check_timeout=args.timeout,
        )
        sys.exit(0 if success else 1)

    # Check all services
    healthy_services, unhealthy_services = check_all_services(
        registry_file=args.registry,
        timeout=args.timeout,
        only_running=not args.all,
    )

    if args.urls_only:
        # Only print URLs
        for service in healthy_services:
            print(service.url)
        sys.exit(0)

    # Print detailed results
    print(f"\n{'='*80}")
    print(f"Code Verify Service Health Check")
    print(f"{'='*80}\n")
    print(f"Registry file: {args.registry}")
    print(f"Timeout: {args.timeout}s")
    print()

    if not healthy_services and not unhealthy_services:
        print("No services found in registry.")
        sys.exit(1)

    print(f"Healthy Services: {len(healthy_services)}")
    print(f"{'─'*80}")
    for service in healthy_services:
        print(f"  ✓ {service.hostname}:{service.port} - {service.url}")
        print(f"    Health: {service.health_url}")

    if unhealthy_services:
        print(f"\nUnhealthy Services: {len(unhealthy_services)}")
        print(f"{'─'*80}")
        for service in unhealthy_services:
            print(f"  ✗ {service.hostname}:{service.port} - {service.url}")
            if not service.is_running:
                print(f"    Status: {service.status}")

    print(f"\n{'='*80}\n")

    # Exit with error if any services are unhealthy
    sys.exit(0 if not unhealthy_services else 1)


if __name__ == "__main__":
    main()
