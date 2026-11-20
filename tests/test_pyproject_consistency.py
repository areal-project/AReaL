# SPDX-License-Identifier: Apache-2.0

import runpy
from copy import deepcopy
from pathlib import Path

# This metadata checker does not need AReaL's optional training dependencies.
_Checker = runpy.run_path(
    str(Path(__file__).parents[1] / "areal/tools/check_pyproject_consistency.py")
)["_Checker"]


def test_scoped_overrides_allow_backend_differences_but_check_shared_packages():
    shared = {
        "package": {"name": "litellm", "version": "1.83.7"},
        "dependencies": ["tokenizers>=0.22.2,<0.24"],
    }
    backend = {
        "package": {"name": "sglang"},
        "dependencies": ["transformers<=5.16.1"],
    }
    checker = _Checker("sglang", "vllm")
    checker.check_override_deps([shared, backend], [deepcopy(shared)])
    assert not checker.errors

    changed = deepcopy(shared)
    changed["dependencies"] = ["tokenizers==0.22.2"]
    checker.check_override_deps([shared], [changed])
    assert checker.errors


def test_scoped_override_version_scope_must_match():
    override = {
        "package": {"name": "litellm", "version": "1.83.7"},
        "dependencies": ["tokenizers>=0.22.2,<0.24"],
    }
    changed = deepcopy(override)
    changed["package"]["version"] = "1.83.14"
    checker = _Checker("sglang", "vllm")
    checker.check_override_deps([override], [changed])
    assert checker.errors


def test_default_transformers_can_differ_but_qwen_group_must_match():
    a = {
        "dependency-groups": {
            "transformers-default": ["transformers==5.3.0"],
            "qwen-flash-next": ["transformers==5.16.1"],
        }
    }
    b = deepcopy(a)
    b["dependency-groups"]["transformers-default"] = ["transformers==5.7.0"]
    assert _Checker("sglang", "vllm").run(a, b) == 0

    b["dependency-groups"]["qwen-flash-next"] = ["transformers==5.3.0"]
    assert _Checker("sglang", "vllm").run(a, b) == 1


def _torch_uv(cuda: str) -> dict:
    index = f"pytorch-{cuda}"
    return {
        "tool": {
            "uv": {
                "index": [
                    {
                        "name": index,
                        "url": f"https://download.pytorch.org/whl/{cuda}",
                        "explicit": True,
                    }
                ],
                "sources": {
                    pkg: [{"index": index, "extra": "cuda"}]
                    for pkg in ("torch", "torchaudio", "torchvision")
                },
            }
        }
    }


def test_torch_index_can_differ_between_backends():
    assert _Checker("sglang", "vllm").run(_torch_uv("cu129"), _torch_uv("cu130")) == 0


def test_shared_index_must_match_even_when_torch_uses_it():
    a, b = _torch_uv("cu129"), _torch_uv("cu130")
    # A non-escapable package on the torch index makes that index shared.
    a["tool"]["uv"]["sources"]["triton"] = [{"index": "pytorch-cu129"}]
    b["tool"]["uv"]["sources"]["triton"] = [{"index": "pytorch-cu130"}]
    assert _Checker("sglang", "vllm").run(a, b) == 1


def test_non_torch_index_must_match():
    a, b = _torch_uv("cu129"), _torch_uv("cu129")
    b["tool"]["uv"]["index"].append({"name": "extra", "url": "https://example.invalid"})
    assert _Checker("sglang", "vllm").run(a, b) == 1
