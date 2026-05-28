# SPDX-License-Identifier: Apache-2.0

"""Guard test: ``areal.experimental.cli`` must stay import-light.

The ``areal`` console-script is intended to be installable on a login
node (``pip install areal[cli]``) without dragging in the full training
stack (torch, transformers, sglang/vllm, ray, megatron, ...). This test
spawns a fresh subprocess, imports the CLI entrypoint, and asserts that
no module from the known-heavy list ends up in ``sys.modules``.

Run in a fresh subprocess (not via ``importlib`` in the test process) so
that accidental imports done elsewhere in the pytest session do not mask
leaks. If you add a verb that needs a heavy dep, do the import inside
``_handle`` — never at module top level.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Top-level package names that the CLI scaffold must NOT cause to be
# imported. Picked from pyproject.toml's heavy deps: training / inference
# backends, web servers, experiment trackers, and large transformer
# libraries. Also blocks AReaL's own heavy subpackages — the scaffold
# must not transitively load them either.
FORBIDDEN_TOP_LEVEL = {
    # Deep-learning runtimes
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    # Inference backends
    "sglang",
    "vllm",
    # Distributed runtime / training-stack hangers-on
    "ray",
    "megatron",
    "mbridge",
    "flash_attn",
    "kernels",
    "tilelang",
    "modelopt",
    # Web frameworks / async HTTP
    "aiohttp",
    "fastapi",
    "uvicorn",
    # Experiment trackers
    "wandb",
    "tensorboardx",
    "swanlab",
    "swanboard",
    "trackio",
    # Data / numerical
    "datasets",
    "peft",
    "pandas",
    "matplotlib",
    "seaborn",
    "numba",
    "h5py",
    "blosc",
    "huggingface_hub",
    # External LLM SDKs
    "openai",
    "anthropic",
    "litellm",
    "qwen_agent",
    "openai_agents",
    "claude_agent_sdk",
    "openhands",
    "langchain",
    # CUDA / GPU stacks
    "nvidia",
    "cupy",
    "triton",
    # AReaL's own heavy subpackages — CLI must not transitively load them.
    "areal.infra",
    "areal.engine",
    "areal.trainer",
    "areal.workflow",
    "areal.dataset",
    "areal.reward",
    "areal.api",
}


def _modules_after(import_stmt: str) -> set[str]:
    """Spawn a fresh interpreter, run *import_stmt*, return ``sys.modules`` keys."""
    code = (
        "import sys, json\n"
        f"{import_stmt}\n"
        "print(json.dumps(sorted(sys.modules.keys())))\n"
    )
    out = subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
    )
    last_line = out.decode().strip().splitlines()[-1]
    return set(json.loads(last_line))


def _leaks(modules: set[str]) -> set[str]:
    leaked: set[str] = set()
    for m in modules:
        for f in FORBIDDEN_TOP_LEVEL:
            # Exact match on a forbidden package (e.g. "areal.infra"), or
            # any descendant (e.g. "areal.infra.launcher").
            if m == f or m.startswith(f + "."):
                leaked.add(m)
                break
    return leaked


def test_cli_main_module_is_light():
    """Importing the CLI entry point must not load any heavy backend."""
    mods = _modules_after("import areal.experimental.cli.main")
    leaked = _leaks(mods)
    assert not leaked, (
        f"`import areal.experimental.cli.main` leaked heavy modules: "
        f"{sorted(leaked)}"
    )


def test_build_parser_is_light():
    """Building the argparse tree must not load any heavy backend either."""
    mods = _modules_after(
        "from areal.experimental.cli.main import build_parser\n"
        "build_parser()"
    )
    leaked = _leaks(mods)
    assert not leaked, (
        f"`build_parser()` leaked heavy modules: {sorted(leaked)}"
    )


def test_each_namespace_help_is_light():
    """Triggering each namespace's --help path must stay light."""
    for ns in ("inf", "agent", "train", "weight-update"):
        mods = _modules_after(
            "from areal.experimental.cli.main import cli\n"
            f"try:\n"
            f"    cli(['{ns}', '--help'])\n"
            f"except SystemExit:\n"
            f"    pass\n"
        )
        leaked = _leaks(mods)
        assert not leaked, (
            f"`areal {ns} --help` leaked heavy modules: {sorted(leaked)}"
        )
