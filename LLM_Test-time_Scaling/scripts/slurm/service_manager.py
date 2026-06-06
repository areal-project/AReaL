#!/usr/bin/env python3
"""Service manager for SGlang services on SLURM cluster.

This script helps manage and monitor SGlang services launched on SLURM nodes.
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class ServiceInfo:
    """Information about a running service."""

    job_id: str
    hostname: str
    host_ip: str
    port: int
    model_path: str
    start_time: str
    status: str
    end_time: Optional[str] = None
    exit_code: Optional[int] = None

    @property
    def url(self) -> str:
        """Get the service URL."""
        return f"http://{self.host_ip}:{self.port}/v1"

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self.status == "running"

    def __str__(self) -> str:
        """String representation."""
        status_icon = "✓" if self.is_running else "✗"
        return (
            f"[{status_icon}] Job {self.job_id} @ {self.hostname} - "
            f"{self.url} (started: {self.start_time})"
        )


class ServiceRegistry:
    """Manager for service registry file."""

    def __init__(self, registry_file: str = "services.txt"):
        """Initialize service registry.

        Args:
            registry_file: Path to service registry file
        """
        self.registry_file = Path(registry_file)

    def load_services(self) -> List[ServiceInfo]:
        """Load all services from registry.

        Returns:
            List of ServiceInfo objects
        """
        if not self.registry_file.exists():
            return []

        services = []
        with open(self.registry_file, "r") as f:
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
                    model_path=parts[4],
                    start_time=parts[5],
                    status=parts[6],
                    end_time=parts[7] if len(parts) > 7 else None,
                    exit_code=int(parts[8]) if len(parts) > 8 and parts[8] else None,
                )
                services.append(service)

        return services

    def get_running_services(self) -> List[ServiceInfo]:
        """Get only running services.

        Returns:
            List of running ServiceInfo objects
        """
        return [s for s in self.load_services() if s.is_running]

    def get_api_bases(self, separator: str = ",") -> str:
        """Get all running service URLs as comma-separated string.

        Args:
            separator: Separator between URLs (default: comma)

        Returns:
            Comma-separated string of service URLs
        """
        urls = [s.url for s in self.get_running_services()]
        return separator.join(urls)

    def export_env_vars(self, output_file: Optional[str] = None) -> str:
        """Export environment variables for running services.

        Args:
            output_file: Optional file to write exports to

        Returns:
            String containing export commands
        """
        api_bases = self.get_api_bases()
        exports = f'export OPENAI_API_BASE="{api_bases}"\n'
        exports += f'export SGLANG_API_BASES="{api_bases}"\n'

        if output_file:
            with open(output_file, "w") as f:
                f.write(exports)
            print(f"Environment variables exported to: {output_file}")

        return exports

    def check_service_health(self, service: ServiceInfo, timeout: int = 5) -> bool:
        """Check if a service is healthy by making a test request.

        Args:
            service: ServiceInfo to check
            timeout: Request timeout in seconds

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            import requests

            response = requests.get(
                f"http://{service.host_ip}:{service.port}/health",
                timeout=timeout,
            )
            return response.status_code == 200
        except Exception:
            return False

    @staticmethod
    def check_url_health(url: str, timeout: int = 5) -> bool:
        """Check if a service URL is healthy by making a test request.

        Args:
            url: Service URL
            timeout: Request timeout in seconds

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            import requests
            from urllib.parse import urlparse

            # Extract base URL (remove /v1 if present)
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            
            # Try health endpoint
            health_url = f"{base_url}/health"
            response = requests.get(health_url, timeout=timeout)
            return response.status_code == 200
        except Exception as e:
            return False


def cmd_list(args: argparse.Namespace) -> None:
    """List all services."""
    registry = ServiceRegistry(args.registry)
    services = registry.load_services()

    if not services:
        print("No services found in registry.")
        return

    print(f"\n{'='*80}")
    print(f"Service Registry: {args.registry}")
    print(f"{'='*80}\n")

    running = [s for s in services if s.is_running]
    stopped = [s for s in services if not s.is_running]

    print(f"Running Services: {len(running)}")
    print(f"{'─'*80}")
    for service in running:
        print(f"  {service}")

    if stopped:
        print(f"\nStopped Services: {len(stopped)}")
        print(f"{'─'*80}")
        for service in stopped[:5]:  # Show only first 5 stopped
            print(f"  {service}")
        if len(stopped) > 5:
            print(f"  ... and {len(stopped) - 5} more")

    print(f"\n{'='*80}\n")


def cmd_urls(args: argparse.Namespace) -> None:
    """Print service URLs."""
    registry = ServiceRegistry(args.registry)
    services = registry.get_running_services()

    if not services:
        print("No running services found.")
        return

    print("\nRunning Service URLs:")
    print("─" * 80)
    for service in services:
        print(service.url)

    print()


def cmd_export(args: argparse.Namespace) -> None:
    """Export environment variables."""
    registry = ServiceRegistry(args.registry)
    services = registry.get_running_services()

    if not services:
        print("No running services found.")
        return

    exports = registry.export_env_vars(args.output)

    if not args.output:
        print("\nExport these environment variables:")
        print("─" * 80)
        print(exports)
        print("Or source the output file:")
        print("  source env_vars.sh")


def cmd_config(args: argparse.Namespace) -> None:
    """Generate experiment config with service URLs."""
    registry = ServiceRegistry(args.registry)
    api_bases = registry.get_api_bases()

    if not api_bases:
        print("No running services found.")
        return

    print(f"\nAdd this to your experiment configuration:")
    print("─" * 80)
    print(f'API_BASE = "{api_bases}"')
    print("\nOr set as environment variable:")
    print(f'export OPENAI_API_BASE="{api_bases}"')
    print()


def cmd_health(args: argparse.Namespace) -> None:
    """Check health of all services."""
    # If URLs are provided directly, check those instead
    if args.urls:
        urls = [url.strip() for url in args.urls.split(",") if url.strip()]
        print("\nChecking service health from provided URLs...")
        print("─" * 80)

        healthy_count = 0
        for url in urls:
            print(f"Checking {url}... ", end="", flush=True)
            is_healthy = ServiceRegistry.check_url_health(url, timeout=args.timeout)

            if is_healthy:
                print("✓ Healthy")
                healthy_count += 1
            else:
                print("✗ Unhealthy")

        print(f"\n{healthy_count}/{len(urls)} services are healthy\n")
        return

    # Otherwise, check services from registry
    registry = ServiceRegistry(args.registry)
    services = registry.get_running_services()

    if not services:
        print("No running services found.")
        return

    print("\nChecking service health...")
    print("─" * 80)

    healthy_count = 0
    for service in services:
        print(f"Checking {service.url}... ", end="", flush=True)
        is_healthy = registry.check_service_health(service, timeout=args.timeout)

        if is_healthy:
            print("✓ Healthy")
            healthy_count += 1
        else:
            print("✗ Unhealthy")

    print(f"\n{healthy_count}/{len(services)} services are healthy\n")


def cmd_cancel(args: argparse.Namespace) -> None:
    """Cancel running services."""
    registry = ServiceRegistry(args.registry)
    services = registry.get_running_services()

    if not services:
        print("No running services found.")
        return

    job_ids = [s.job_id for s in services]

    print(f"\nCanceling {len(job_ids)} jobs...")
    print("─" * 80)

    if not args.force:
        response = input(f"Cancel jobs {job_ids}? [y/N]: ")
        if response.lower() != "y":
            print("Cancelled.")
            return

    for job_id in job_ids:
        try:
            subprocess.run(["scancel", job_id], check=True)
            print(f"✓ Cancelled job {job_id}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to cancel job {job_id}: {e}")

    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manage SGlang services on SLURM cluster"
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="services.txt",
        help="Path to service registry file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    subparsers.add_parser("list", help="List all services")

    # URLs command
    subparsers.add_parser("urls", help="Print service URLs")

    # Export command
    parser_export = subparsers.add_parser(
        "export", help="Export environment variables"
    )
    parser_export.add_argument(
        "-o", "--output", type=str, help="Output file for export commands"
    )

    # Config command
    subparsers.add_parser("config", help="Generate experiment config")

    # Health command
    parser_health = subparsers.add_parser("health", help="Check service health")
    parser_health.add_argument(
        "--timeout", type=int, default=5, help="Request timeout in seconds"
    )
    parser_health.add_argument(
        "--urls",
        type=str,
        default=None,
        help="Comma-separated list of service URLs to check (e.g., 'http://host1:port1/v1,http://host2:port2/v1')",
    )

    # Cancel command
    parser_cancel = subparsers.add_parser("cancel", help="Cancel running services")
    parser_cancel.add_argument(
        "-f", "--force", action="store_true", help="Force cancel without confirmation"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Route to command handlers
    commands = {
        "list": cmd_list,
        "urls": cmd_urls,
        "export": cmd_export,
        "config": cmd_config,
        "health": cmd_health,
        "cancel": cmd_cancel,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
