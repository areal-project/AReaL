# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

from areal.api import cli_args
from areal.api.cli_args import SGLangConfig


@pytest.fixture
def build_command(monkeypatch, tmp_path):
    """Exercise native CLI generation without requiring an installed server."""
    monkeypatch.setattr(
        cli_args.pkg_version, "is_version_greater_or_equal", lambda *args: True
    )

    def build(**settings):
        config = OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(SGLangConfig),
                {"model_path": str(tmp_path), **settings},
            )
        )
        return SGLangConfig.build_cmd(config, tp_size=8, base_gpu_id=0)

    return build


def test_prefill_graph_flag_default_is_omitted(build_command):
    """Older SGLang runtimes do not receive a new flag by default."""
    command = build_command()
    assert "--disable-prefill-cuda-graph" not in command


def test_prefill_disabled_with_memory_saver_keeps_decode_enabled(build_command):
    """The phase-specific switch does not emit the global graph-off flag."""
    command = build_command(
        disable_cuda_graph=False,
        disable_prefill_cuda_graph=True,
        enable_memory_saver=True,
    )
    assert "--disable-prefill-cuda-graph" in command
    assert "--enable-memory-saver" in command
    assert "--disable-cuda-graph" not in command


def test_global_graph_disable_still_emits_existing_flag(build_command):
    """The existing global switch remains available independently."""
    command = build_command(disable_cuda_graph=True)
    assert "--disable-cuda-graph" in command
    assert "--disable-prefill-cuda-graph" not in command
