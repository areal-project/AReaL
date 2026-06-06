#!/usr/bin/env python3
"""Check health of code verify services.

This script can check code verify services in multiple ways:
1. Direct URL check (using /health endpoint)
2. From service registry file
3. Test verification endpoint with a simple test case
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    import requests
except ImportError:
    print("Error: requests library not installed. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)


@dataclass
class HealthResult:
    """Result of a health check for a code verify service."""

    url: str
    is_healthy: bool
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    error: Optional[str] = None
    verify_test_passed: Optional[bool] = None

    def __str__(self) -> str:
        """String representation."""
        status_icon = "✓" if self.is_healthy else "✗"
        if self.is_healthy:
            result = f"[{status_icon}] {self.url} - Healthy (Status: {self.status_code}, Time: {self.response_time:.2f}s)"
            if self.verify_test_passed is not None:
                verify_status = "✓" if self.verify_test_passed else "✗"
                result += f" [Verify Test: {verify_status}]"
            return result
        else:
            error_msg = f" - {self.error}" if self.error else ""
            return f"[{status_icon}] {self.url} - Unhealthy{error_msg}"


def check_health_endpoint(url: str, timeout: int = 5) -> HealthResult:
    """Check /health endpoint of a code verify service.

    Args:
        url: Service base URL 
        timeout: Request timeout in seconds

    Returns:
        HealthResult object
    """
    start_time = time.time()
    health_url = f"{url.rstrip('/')}/health"

    try:
        response = requests.get(health_url, timeout=timeout)
        response_time = time.time() - start_time

        is_healthy = response.status_code == 200
        if is_healthy:
            # Check response content
            try:
                data = response.json()
                if data.get("status") == "healthy":
                    return HealthResult(
                        url=url,
                        is_healthy=True,
                        status_code=response.status_code,
                        response_time=response_time,
                    )
            except:
                pass

        return HealthResult(
            url=url,
            is_healthy=False,
            status_code=response.status_code,
            response_time=response_time,
            error=f"HTTP {response.status_code}",
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


def test_verify_endpoint(url: str, timeout: int = 30) -> bool:
    """Test /verify endpoint with a simple test case.

    Args:
        url: Service base URL
        timeout: Request timeout in seconds

    Returns:
        True if verification test passed, False otherwise
    """
    verify_url = f"{url.rstrip('/')}/verify"

    # Simple test case: Hello World program
    test_code = """
#include <iostream>
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
"""
    test_payload = {
        "code": test_code,
        "problem_id": "test_health_check",
        "data_dir": None,
    }

    try:
        response = requests.post(verify_url, json=test_payload, timeout=timeout)
        if response.status_code == 200:
            return True
        return False
    except Exception:
        return False


def check_service(url: str, timeout: int = 5, test_verify: bool = False) -> HealthResult:
    """Check a code verify service.

    Args:
        url: Service base URL
        timeout: Request timeout in seconds
        test_verify: Whether to also test the /verify endpoint

    Returns:
        HealthResult object
    """
    result = check_health_endpoint(url, timeout=timeout)

    if result.is_healthy and test_verify:
        verify_passed = test_verify_endpoint(url, timeout=timeout)
        result.verify_test_passed = verify_passed

    return result


def check_services_parallel(
    urls: List[str],
    timeout: int = 5,
    test_verify: bool = False,
    max_workers: int = 10,
) -> List[HealthResult]:
    """Check multiple services in parallel.

    Args:
        urls: List of service URLs
        timeout: Request timeout in seconds
        test_verify: Whether to also test the /verify endpoint
        max_workers: Maximum number of concurrent workers

    Returns:
        List of HealthResult objects
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(check_service, url, timeout, test_verify): url
            for url in urls
        }

        for future in as_completed(future_to_url):
            result = future.result()
            results.append(result)

    # Sort results to match input order
    url_to_result = {r.url: r for r in results}
    return [url_to_result[url] for url in urls]


def load_urls_from_registry(registry_file: str) -> List[str]:
    """Load service URLs from registry file.

    Args:
        registry_file: Path to service registry file

    Returns:
        List of service URLs
    """
    registry_path = Path(registry_file)
    if not registry_path.exists():
        return []

    urls = []
    with open(registry_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            if len(parts) >= 3:
                host_ip = parts[2]
                port = parts[3]
                url = f"http://{host_ip}:{port}"
                urls.append(url)

    return urls


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check health of code verify services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check single service
  python check_service_health.py --url service_addr

  # Check multiple services (comma-separated)
  python check_service_health.py --urls "service_addr1,service_addr2,service_addr3"

  # Check multiple services (space-separated)
  python check_service_health.py --urls service_addr1,service_addr2

  # Check services from registry
  python check_service_health.py --registry services.txt

  # Test verify endpoint as well
  python check_service_health.py --url service_addr --test-verify

  # Check from file (one URL per line)
  python check_service_health.py --file urls.txt
        """,
    )

    # URL input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--url",
        type=str,
        help="Single service URL to check",
    )
    input_group.add_argument(
        "--urls",
        type=str,
        help="Multiple service URLs to check (comma-separated or space-separated, e.g., 'url1,url2,url3' or 'url1 url2 url3')",
    )
    input_group.add_argument(
        "--registry",
        type=str,
        help="Path to service registry file",
    )
    input_group.add_argument(
        "--file",
        type=str,
        help="Path to file containing URLs (one per line)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Request timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--test-verify",
        action="store_true",
        help="Also test the /verify endpoint with a simple test case",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Check services in parallel (faster for many services)",
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
    if args.url:
        urls = [args.url]
    elif args.urls:
        # Parse URLs: support both comma-separated and space-separated
        urls_str = args.urls.strip()
        if ',' in urls_str:
            # Comma-separated
            urls = [url.strip() for url in urls_str.split(',') if url.strip()]
        else:
            # Space-separated (original behavior)
            urls = urls_str.split()
    elif args.registry:
        urls = load_urls_from_registry(args.registry)
        if not urls:
            print(f"Error: No URLs found in registry file: {args.registry}", file=sys.stderr)
            sys.exit(1)
    elif args.file:
        urls = []
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        urls.append(line)
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)

    if not urls:
        print("Error: No URLs provided", file=sys.stderr)
        sys.exit(1)

    # Check services
    if args.parallel and len(urls) > 1:
        results = check_services_parallel(
            urls, timeout=args.timeout, test_verify=args.test_verify, max_workers=args.workers
        )
    else:
        results = [check_service(url, timeout=args.timeout, test_verify=args.test_verify) for url in urls]

    # Display results
    if not args.summary:
        print("\n" + "=" * 80)
        print("Code Verify Service Health Check")
        print("=" * 80 + "\n")

        for result in results:
            print(result)

    # Summary
    healthy_count = sum(1 for r in results if r.is_healthy)
    total_count = len(results)

    print("\n" + "─" * 80)
    print(f"Summary: {healthy_count}/{total_count} services are healthy")
    print("─" * 80)

    if args.test_verify:
        verify_passed_count = sum(1 for r in results if r.verify_test_passed is True)
        print(f"Verify Test: {verify_passed_count}/{total_count} services passed verification test")

    if healthy_count < total_count:
        print("\nUnhealthy Services:")
        for result in results:
            if not result.is_healthy:
                print(f"  ✗ {result.url}")

    # Exit with error code if any service is unhealthy
    sys.exit(0 if healthy_count == total_count else 1)


if __name__ == "__main__":
    main()
