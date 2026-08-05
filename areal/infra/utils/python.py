# SPDX-License-Identifier: Apache-2.0

import os
import sys


def get_python_executable() -> str:
    return os.environ.get("AREAL_PYTHON_EXECUTABLE") or sys.executable
