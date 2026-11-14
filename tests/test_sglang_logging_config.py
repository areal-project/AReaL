# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path


def test_swe_sglang_logging_config_uses_available_standard_handlers(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "examples/swe/rl/qwen38_27b/configure_sglang_logging.py"
    )
    config_path = tmp_path / "sglang-0.json"
    subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ, "SGLANG_LOGGING_CONFIG_PATH": str(config_path)},
        check=True,
        capture_output=True,
        text=True,
    )

    config = json.loads(config_path.read_text())
    assert (
        config["handlers"]["sglang"]["class"] == "logging.handlers.RotatingFileHandler"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, logging, logging.config, sys\n"
            "with open(sys.argv[1]) as file: config = json.load(file)\n"
            "logging.config.dictConfig(config)\n"
            "logging.getLogger('sglang').info('logging smoke test')\n",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "logging smoke test" in config_path.with_suffix(".log").read_text()
