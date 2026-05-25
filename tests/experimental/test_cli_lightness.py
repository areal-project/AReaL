# SPDX-License-Identifier: Apache-2.0

"""Guard test: ``areal.experimental.cli`` must stay lightweight.

The ``areal`` console-script is intended to be installable on a login node
(``pip install areal[cli]``) without dragging in the full training stack
(torch, transformers, sglang/vllm, ray, megatron, ...). This test spawns a
fresh subprocess, imports the CLI entrypoint, and asserts that no module
from a known-heavy list ends up in ``sys.modules``.

Run in a fresh subprocess (not via ``importlib``) so accidental imports done
elsewhere in the pytest session don't mask leaks.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Top-level package names that the CLI must NOT cause to be imported. Picked
# from pyproject.toml's heavy deps: training/inference backends, web servers,
# experiment trackers, and large transformer libraries.
FORBIDDEN_TOP_LEVEL = {
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "sglang",
    "vllm",
    "ray",
    "megatron",
    "mbridge",
    "aiohttp",
    "fastapi",
    "uvicorn",
    "wandb",
    "tensorboardx",
    "swanlab",
    "swanboard",
    "trackio",
    "datasets",
    "peft",
    "openai",
    "anthropic",
    "litellm",
    "qwen_agent",
    "openai_agents",
    "claude_agent_sdk",
    "openhands",
    "langchain",
    "flash_attn",
    "kernels",
    "tilelang",
    "modelopt",
    "huggingface_hub",
    "pandas",
    "matplotlib",
    "seaborn",
    "numba",
    "h5py",
    "blosc",
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
    """Spawn a fresh interpreter, run ``import_stmt``, return sys.modules keys."""
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
        # Exact match on a forbidden subpackage (e.g. "areal.infra"), or any
        # descendant (e.g. "areal.infra.launcher").
        for f in FORBIDDEN_TOP_LEVEL:
            if m == f or m.startswith(f + "."):
                leaked.add(m)
                break
    return leaked


def test_cli_main_module_is_light():
    """Importing the CLI entrypoint must not load any heavy backend."""
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


def test_areal_top_level_lazy_loads_infra():
    """`import areal` itself must stay light — infra is lazy-loaded via __getattr__."""
    mods = _modules_after("import areal")
    leaked = _leaks(mods)
    assert not leaked, (
        f"`import areal` eagerly loaded heavy modules: {sorted(leaked)}"
    )
