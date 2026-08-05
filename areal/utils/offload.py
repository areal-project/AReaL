# SPDX-License-Identifier: Apache-2.0

"""Utilities for torch_memory_saver (TMS) configuration and setup.

This module handles the environment variable setup required for TMS to work
properly with LD_PRELOAD hooks.
"""

import os
from contextlib import nullcontext

try:
    from torch_memory_saver import torch_memory_saver
except ImportError:

    class MockTorchMemorySaver:
        def disable(self):
            return nullcontext()

        def pause(self):
            pass

        def resume(self):
            pass

    torch_memory_saver = MockTorchMemorySaver()


def get_tms_env_vars() -> dict[str, str]:
    """Get environment variables for torch_memory_saver (TMS)."""
    import torch_memory_saver as tms_pkg

    # Locate the LD_PRELOAD shared library
    dynlib_path = os.path.join(
        os.path.dirname(os.path.dirname(tms_pkg.__file__)),
        "torch_memory_saver_hook_mode_preload.abi3.so",
    )

    if not os.path.exists(dynlib_path):
        raise RuntimeError(f"LD_PRELOAD so file {dynlib_path} does not exist.")

    env_vars = {
        "LD_PRELOAD": dynlib_path,
        "TMS_INIT_ENABLE": "1",
        "TMS_INIT_ENABLE_CPU_BACKUP": "1",
    }
    return env_vars


def apply_tms_env_vars(env_vars: dict[str, str]) -> None:
    """Add default TMS env vars without overriding explicit user settings."""
    tms_init = env_vars.get("TMS_INIT_ENABLE")
    ld_preload = env_vars.get("LD_PRELOAD")

    if tms_init is not None and tms_init != "1":
        return
    if ld_preload == "":
        return

    for key, value in get_tms_env_vars().items():
        env_vars.setdefault(key, value)


def is_tms_enabled() -> bool:
    return os.environ.get("TMS_INIT_ENABLE", "0") == "1"


def normalize_tms_ld_preload() -> None:
    """Keep only the TMS preload library in LD_PRELOAD before TMS initializes."""
    ld_preload = os.environ.get("LD_PRELOAD")
    if not ld_preload or ":" not in ld_preload:
        return

    for path in ld_preload.split(":"):
        if os.path.basename(path) == "torch_memory_saver_hook_mode_preload.abi3.so":
            # torch_memory_saver loads this value with ctypes.CDLL(), which
            # cannot parse a colon-separated LD_PRELOAD list.
            os.environ["LD_PRELOAD"] = path
            return
