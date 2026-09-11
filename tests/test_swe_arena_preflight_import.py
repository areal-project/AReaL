# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_arena_preflight_import_does_not_load_training_stack():
    """Login-node preflight must not initialize the AReaL training package."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import examples.swe.arena_preflight; "
                "assert 'examples.swe.utils' not in sys.modules; "
                "assert 'areal' not in sys.modules"
            ),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
