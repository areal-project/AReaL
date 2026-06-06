#!/usr/bin/env python3
"""URL health checker.

This script checks the health status of multiple URLs by making HTTP requests
to their health endpoints.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse


@dataclass
class HealthResult:
    """Result of a health check for a URL."""

    url: str
    is_healthy: bool
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    error: Optional[str] = None

    def __str__(self) -> str:
        """String representation."""
        status_icon = "✓" if self.is_healthy else "✗"
        if self.is_healthy:
            return (
                f"[{status_icon}] {self.url} - Healthy "
                f"(Status: {self.status_code}, Time: {self.response_time:.2f}s)"
            )
        else:
            error_msg = f" - {self.error}" if self.error else ""
            return f"[{status_icon}] {self.url} - Unhealthy{error_msg}"


def check_url_health(url: str, timeout: int = 5) -> HealthResult:
    """Check if a service URL is healthy by making a test request.

    Args:
        url: Service URL
        timeout: Request timeout in seconds

    Returns:
        HealthResult object with health status and details
    """
    try:
        import requests
    except ImportError:
        return HealthResult(
            url=url,
            is_healthy=False,
            error="requests library not installed. Install with: pip install requests",
        )

    start_time = time.time()
    try:
        # Parse URL to extract base URL
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Try health endpoint
        health_url = f"{base_url}/health"
        response = requests.get(health_url, timeout=timeout)
        response_time = time.time() - start_time

        is_healthy = response.status_code == 200
        return HealthResult(
            url=url,
            is_healthy=is_healthy,
            status_code=response.status_code,
            response_time=response_time,
            error=None if is_healthy else f"HTTP {response.status_code}",
        )
    except requests.exceptions.Timeout:
        response_time = time.time() - start_time
        return HealthResult(
            url=url,
            is_healthy=False,
            response_time=response_time,
            error=f"Timeout after {timeout}s",
        )
    except requests.exceptions.ConnectionError as e:
        response_time = time.time() - start_time
        return HealthResult(
            url=url,
            is_healthy=False,
            response_time=response_time,
            error=f"Connection error: {str(e)}",
        )
    except Exception as e:
        response_time = time.time() - start_time
        return HealthResult(
            url=url,
            is_healthy=False,
            response_time=response_time,
            error=f"Error: {str(e)}",
        )


def check_urls_sequential(urls: List[str], timeout: int = 5) -> List[HealthResult]:
    """Check health of URLs sequentially.

    Args:
        urls: List of URLs to check
        timeout: Request timeout in seconds

    Returns:
        List of HealthResult objects
    """
    results = []
    for url in urls:
        result = check_url_health(url, timeout=timeout)
        results.append(result)
    return results


def check_urls_parallel(urls: List[str], timeout: int = 5, max_workers: int = 10) -> List[HealthResult]:
    """Check health of URLs in parallel.

    Args:
        urls: List of URLs to check
        timeout: Request timeout in seconds
        max_workers: Maximum number of concurrent workers

    Returns:
        List of HealthResult objects (in order of completion)
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {executor.submit(check_url_health, url, timeout): url for url in urls}

        # Collect results as they complete
        for future in as_completed(future_to_url):
            result = future.result()
            results.append(result)

    # Sort results to match input order
    url_to_result = {r.url: r for r in results}
    return [url_to_result[url] for url in urls]


def load_urls_from_file(file_path: str) -> List[str]:
    """Load URLs from a file (one per line).

    Args:
        file_path: Path to file containing URLs

    Returns:
        List of URLs
    """
    urls = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    return urls


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check health status of multiple URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check URLs from command line arguments
  python check_url_health.py --urls "service_addr1,service_addr2"

  # Check URLs from a file
  python check_url_health.py --file urls.txt

  # Check URLs with custom timeout and parallel execution
  python check_url_health.py --file urls.txt --timeout 10 --parallel --workers 20""",
    )

    # URL input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--urls",
        type=str,
        help="Comma-separated list of URLs to check",
    )
    input_group.add_argument(
        "--file",
        type=str,
        help="Path to file containing URLs (one per line)",
    )
    input_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read URLs from stdin (one per line)",
    )

    # Optional arguments
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Request timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Check URLs in parallel (faster for many URLs)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Maximum number of concurrent workers for parallel execution (default: 10)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show only summary statistics",
    )

    args = parser.parse_args()

    # Load URLs based on input method
    if args.urls:
        urls = [url.strip() for url in args.urls.split(",") if url.strip()]
    elif args.file:
        urls = load_urls_from_file(args.file)
    elif args.stdin:
        urls = []
        for line in sys.stdin:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    if not urls:
        print("Error: No URLs provided", file=sys.stderr)
        sys.exit(1)

    # Check health
    if args.parallel:
        results = check_urls_parallel(urls, timeout=args.timeout, max_workers=args.workers)
    else:
        results = check_urls_sequential(urls, timeout=args.timeout)

    # Display results
    if not args.summary:
        print("\n" + "=" * 80)
        print("URL Health Check Results")
        print("=" * 80 + "\n")

        for result in results:
            print(result)

    # Summary
    healthy_count = sum(1 for r in results if r.is_healthy)
    total_count = len(results)

    print("\n" + "─" * 80)
    print(f"Summary: {healthy_count}/{total_count} URLs are healthy")
    print("─" * 80)

    if healthy_count < total_count:
        print("\nUnhealthy URLs:")
        for result in results:
            if not result.is_healthy:
                print(f"  ✗ {result.url}")

    # Exit with error code if any URL is unhealthy
    sys.exit(0 if healthy_count == total_count else 1)


if __name__ == "__main__":
    main()
