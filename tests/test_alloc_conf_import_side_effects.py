# SPDX-License-Identifier: Apache-2.0
"""Guards that the CUDA allocator config is not rewritten at import time."""

import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code: str, env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        env.pop(name, None)
    env.update({"PYTHONPATH": REPO_ROOT, **env_extra})
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )


class TestNoImportTimeAllocConfRewrite:
    @pytest.mark.parametrize("role", ["actor", "ref", "rollout"])
    def test_importing_areal_leaves_the_allocator_config_alone(self, role):
        code = (
            "import sys, os, json;"
            f" sys.argv = ['prog', '--role', {role!r}];"
            " import areal;"
            " print(json.dumps(os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')))"
        )
        result = _run(
            code,
            {
                "PYTORCH_CUDA_ALLOC_CONF": "",
                "AWEX_ACTOR_ALLOC_CONF": "expandable_segments:True",
            },
        )

        assert result.returncode == 0, result.stderr[-1500:]
        assert result.stdout.strip().splitlines()[-1] == '""', (
            "importing areal rewrote PYTORCH_CUDA_ALLOC_CONF; the per-role "
            "allocator config belongs in each role's scheduling_spec.env_vars"
        )

    def test_awex_actor_alloc_conf_is_gone(self):
        hits = subprocess.run(
            ["grep", "-rIl", "AWEX_ACTOR_ALLOC_CONF", "areal"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert hits.stdout.strip() == "", (
            f"AWEX_ACTOR_ALLOC_CONF still referenced in: {hits.stdout.strip()}"
        )


class TestSglangPluginRejectsExpandableSegments:
    @pytest.mark.parametrize("name", ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"])
    def test_import_fails_loudly_when_expandable_segments_is_enabled(self, name):
        result = _run(
            "import areal.engine.awex.plugins.sglang_plugin",
            {
                "PYTORCH_ALLOC_CONF": "",
                "PYTORCH_CUDA_ALLOC_CONF": "",
                name: "expandable_segments:True",
            },
        )

        assert result.returncode != 0
        assert "expandable segments" in result.stderr.lower(), result.stderr[-1500:]

    def test_import_is_allowed_without_expandable_segments(self):
        result = _run(
            "import areal.engine.awex.plugins.sglang_plugin",
            {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"},
        )

        assert "expandable segments" not in result.stderr.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.parametrize(
    "env,rejected",
    [
        ({}, False),
        ({"PYTORCH_ALLOC_CONF": "max_split_size_mb:128"}, False),
        ({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}, False),
        (
            {
                "PYTORCH_ALLOC_CONF": "expandable_segments:False",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            },
            False,
        ),
        (
            {"PYTORCH_ALLOC_CONF": "max_split_size_mb:128, expandable_segments: True"},
            True,
        ),
        ({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, True),
        (
            {
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            },
            True,
        ),
        (
            {
                "PYTORCH_ALLOC_CONF": "expandable_segments:False",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
            True,
        ),
    ],
)
def test_memory_saver_allocator_validation_never_rewrites_environment(env, rejected):
    from areal.engine.awex.utils import validate_memory_saver_allocator

    original = env.copy()
    if rejected:
        with pytest.raises(RuntimeError, match="rollout role"):
            validate_memory_saver_allocator(env)
    else:
        validate_memory_saver_allocator(env)
    assert env == original
