# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples/weftmoe/run.sh"


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    """Run the real shell wrapper without starting training or allocating GPUs."""
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1] == '-c':\n"
        "    if os.environ.get('FAIL_KEY_GENERATION'):\n"
        "        sys.exit(23)\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "assert sys.argv[1] == 'examples/cpu_staged_offload/dapo-math_rl_cpu_staged.py'\n"
        "Path(os.environ['LAUNCH_CAPTURE']).write_text(json.dumps({\n"
        "    'key': os.environ['AREAL_PROXY_ADMIN_API_KEY'], 'argv': sys.argv[1:]}))\n"
    )
    python.chmod(0o755)
    capture = tmp_path / "capture.json"
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("LAUNCH_CAPTURE", str(capture))
    monkeypatch.delenv("AREAL_PROXY_ADMIN_API_KEY", raising=False)
    monkeypatch.delenv("FAIL_KEY_GENERATION", raising=False)

    def run():
        result = subprocess.run(
            ["bash", "-x", str(SCRIPT), "baseline", "total_train_steps=1"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        recorded = json.loads(capture.read_text()) if capture.exists() else None
        return result, recorded

    return run


@pytest.mark.parametrize("configured", [None, "", "areal-admin-key", "custom-test-key"])
def test_launcher_exports_private_key_without_logging(
    launcher, monkeypatch, configured
):
    """Replace insecure defaults, preserve explicit keys, and suppress shell tracing."""
    if configured is not None:
        monkeypatch.setenv("AREAL_PROXY_ADMIN_API_KEY", configured)
    result, recorded = launcher()
    assert result.returncode == 0, result.stderr
    key = recorded["key"]
    if configured == "custom-test-key":
        assert key == configured
    else:
        assert len(key) >= 43 and key != "areal-admin-key"
    assert key not in result.stdout + result.stderr
    assert all(key not in arg for arg in recorded["argv"])
    assert recorded["argv"][-1] == "total_train_steps=1"


def test_launcher_generates_independent_keys(launcher):
    """Independent launches should not reuse the public default or a fixed secret."""
    first_result, first = launcher()
    second_result, second = launcher()
    assert first_result.returncode == second_result.returncode == 0
    assert first["key"] != second["key"]


def test_launcher_does_not_start_when_key_generation_fails(launcher, monkeypatch):
    """A failed generator must not be hidden by a successful shell export builtin."""
    monkeypatch.setenv("FAIL_KEY_GENERATION", "1")
    result, recorded = launcher()
    assert result.returncode == 23
    assert recorded is None
