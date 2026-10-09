# SPDX-License-Identifier: Apache-2.0

import sys

from areal.api.cli_args import get_py_cmd


def test_get_py_cmd_uses_current_python_interpreter():
    cmd = get_py_cmd("example.module", {"flag": "value"})

    assert cmd == [sys.executable, "-m", "example.module", "--flag", "value"]
