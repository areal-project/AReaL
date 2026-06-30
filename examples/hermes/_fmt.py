"""Shared formatting helpers for online RL CLI scripts."""

from __future__ import annotations

import json
import sys

# =============================================================================
# ANSI color codes
# =============================================================================
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[0;32m"
CYAN = "\033[0;36m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
MAGENTA = "\033[0;35m"
BLUE = "\033[0;34m"


# =============================================================================
# Output helpers
# =============================================================================


def info(msg: str) -> None:
    print(f"  {CYAN}ℹ{RESET}  {msg}")


def success(msg: str) -> None:
    print(f"  {GREEN}✔{RESET}  {msg}")


def error(msg: str) -> None:
    print(f"  {RED}✘{RESET}  {msg}")


def dim(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")


def arrow(msg: str) -> None:
    print(f"  {YELLOW}→{RESET} {msg}")


def die(msg: str) -> None:
    error(msg)
    sys.exit(1)


def show_request(method: str, path: str, auth_label: str, gateway_url: str) -> None:
    print(f"  {DIM}{method} {gateway_url}/{path}{RESET}")
    print(f"  {DIM}Auth: {auth_label}{RESET}")


def show_response(status_code: int, body: str) -> None:
    if 200 <= status_code < 300:
        print(f"  {GREEN}HTTP {status_code}{RESET}")
    else:
        print(f"  {RED}HTTP {status_code}{RESET}")
    if body:
        try:
            formatted = json.dumps(json.loads(body), indent=2)
            for line in formatted.split("\n"):
                print(f"  {DIM}{line}{RESET}")
        except (json.JSONDecodeError, ValueError):
            for line in body.split("\n"):
                print(f"  {DIM}{line}{RESET}")


def header(title: str) -> None:
    """Print a boxed header."""
    print()
    print(
        f"{BOLD}{BLUE}══════════════════════════════════════════════════════════════{RESET}"
    )
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(
        f"{BOLD}{BLUE}══════════════════════════════════════════════════════════════{RESET}"
    )
