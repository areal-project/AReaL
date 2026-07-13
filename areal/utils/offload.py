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


def is_tms_enabled() -> bool:
    return os.environ.get("TMS_INIT_ENABLE", "0") == "1"
