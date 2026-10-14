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


def test_cuda_backend_dependencies_and_indexes_can_differ_but_shared_deps_match():
    a = {
        "project": {
            "optional-dependencies": {
                "kernels": ["kernels>=0.14.1,<0.15"],
                "tms": ["torch_memory_saver>=0.0.9.post1"],
                "cuda-train": ["areal[kernels]", "shared-package==1"],
            }
        },
        "tool": {"uv": {"index": [{"name": "pytorch-cu130"}], "prerelease": "allow"}},
    }
    b = deepcopy(a)
    b["project"]["optional-dependencies"]["kernels"] = ["kernels==0.12.2"]
    b["project"]["optional-dependencies"]["tms"] = ["torch_memory_saver==0.0.9"]
    b["tool"]["uv"] = {"index": [{"name": "pytorch-cu129"}]}
    assert _Checker("sglang", "vllm").run(a, b) == 0

    b["project"]["optional-dependencies"]["cuda-train"] = [
        "areal[kernels]",
        "shared-package==2",
    ]
    assert _Checker("sglang", "vllm").run(a, b) == 1

    b = deepcopy(a)
    b["tool"]["uv"]["index"].append({"name": "shared-index"})
    assert _Checker("sglang", "vllm").run(a, b) == 1
