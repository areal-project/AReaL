# SPDX-License-Identifier: Apache-2.0

# CUDA device detection utilities

import torch


def get_device_sm() -> int:
    """Get device SM version (e.g., 90 for Hopper, 100 for Blackwell)."""
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def is_hopper() -> bool:
    """Check if device is Hopper (SM90-99)."""
    try:
        sm = get_device_sm()
        return 90 <= sm < 100
    except Exception:
        return False


def is_blackwell() -> bool:
    """Check if device is Blackwell (SM100+)."""
    try:
        return get_device_sm() >= 100
    except Exception:
        return False


def is_sm90_or_above() -> bool:
    """Check if device is SM90 or above (Hopper+)."""
    try:
        return get_device_sm() >= 90
    except Exception:
        return False
