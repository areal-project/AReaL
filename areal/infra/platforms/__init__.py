# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from transformers.utils.import_utils import is_torch_npu_available

import areal.utils.logging as logging

from .cpu import CpuPlatform
from .cuda import CudaPlatform
from .npu import NPUPlatform
from .platform import Platform
from .unknown import UnknownPlatform

logger = logging.getLogger("PlatformInit")


is_npu_available = is_torch_npu_available()


def _init_platform() -> Platform:
    """
    Detect and initialize the appropriate platform based on available devices.
    Priority:
    1. CUDA (NVIDIA)
    2. TODO: NPU (if torch_npu is installed)
    3. CPU (fallback)
    Returns:
        An instance of a subclass of Platform corresponding to the detected hardware.
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name().upper()
        logger.info(f"Detected CUDA device: {device_name}")
        if "NVIDIA" in device_name:
            logger.info("Initializing CUDA platform (NVIDIA).")
            return CudaPlatform()
        logger.warning("Unrecognized CUDA device. Falling back to UnknownPlatform.")
        return UnknownPlatform()
    elif is_npu_available:
        logger.info("Initializing NPU platform (NPU).")
        return NPUPlatform()
    else:
        logger.info("No supported accelerator detected. Initializing CPU platform.")
        return CpuPlatform()


class _LazyPlatform:
    """
    Lazy initialization wrapper for platform detection.

    This class defers platform detection and initialization until the first
    attribute access, avoiding CUDA initialization during import.
    """

    def __init__(self):
        self._platform: Platform | None = None
        self._initialized = False

    def _ensure_initialized(self) -> Platform:
        """Ensure the platform is initialized and return it."""
        if not self._initialized:
            self._platform = _init_platform()
            self._initialized = True
        assert self._platform is not None, "Platform should be initialized here."
        return self._platform

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying platform."""
        platform = self._ensure_initialized()
        return getattr(platform, name)

    def __setattr__(self, name: str, value):
        """Handle setting attributes on the wrapper."""
        if name.startswith("_"):
            # Allow setting private attributes on the wrapper itself
            super().__setattr__(name, value)
        else:
            # Delegate to the underlying platform
            platform = self._ensure_initialized()
            setattr(platform, name, value)

    def __repr__(self) -> str:
        """Return string representation."""
        if self._initialized:
            return f"LazyPlatform({self._platform!r})"
        else:
            return "LazyPlatform(uninitialized)"


# Global singleton representing the current platform in use.
# Platform detection and initialization is deferred until first access.
current_platform: Platform | _LazyPlatform = (
    _LazyPlatform()
)  # NOTE: This is a proxy, not a subclass of Platform.


__all__ = ["Platform", "current_platform", "is_npu_available"]
