# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf
from packaging.version import Version

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


@pytest.mark.parametrize(
    ("installed_version", "suffix"),
    [("0.5.19", ""), ("0.5.20", "-decode"), ("0.5.21", "-decode")],
)
def test_graph_batch_sizes_use_runtime_cli_names(
    build_command, monkeypatch, installed_version, suffix
):
    """Preserve decode graph settings across SGLang's phase-specific rename."""
    monkeypatch.setattr(
        cli_args.pkg_version,
        "is_version_greater_or_equal",
        lambda package, version: Version(installed_version) >= Version(version),
    )

    command = build_command(cuda_graph_max_bs=48, cuda_graph_bs=[1, 2, 4])

    max_flag = f"--cuda-graph-max-bs{suffix}"
    sizes_flag = f"--cuda-graph-bs{suffix}"
    assert command[command.index(max_flag) + 1] == "48"
    sizes_start = command.index(sizes_flag) + 1
    assert command[sizes_start : sizes_start + 3] == ["1", "2", "4"]
    unused_suffix = "" if suffix else "-decode"
    assert f"--cuda-graph-max-bs{unused_suffix}" not in command
    assert f"--cuda-graph-bs{unused_suffix}" not in command
    assert "--cuda-graph-max-bs-prefill" not in command
    assert "--cuda-graph-bs-prefill" not in command


@pytest.mark.parametrize("installed_version", ["0.5.19", "0.5.20"])
def test_graph_batch_size_defaults_emit_no_cli_options(
    build_command, monkeypatch, installed_version
):
    monkeypatch.setattr(
        cli_args.pkg_version,
        "is_version_greater_or_equal",
        lambda package, version: Version(installed_version) >= Version(version),
    )

    command = build_command()

    assert not any(token.startswith("--cuda-graph-max-bs") for token in command)
    assert not any(token.startswith("--cuda-graph-bs") for token in command)


@pytest.mark.parametrize("installed_version", ["0.5.19", "0.5.20"])
def test_graph_cli_conversion_preserves_input_args(monkeypatch, installed_version):
    monkeypatch.setattr(
        cli_args.pkg_version,
        "is_version_greater_or_equal",
        lambda package, version: Version(installed_version) >= Version(version),
    )
    args = {"cuda_graph_max_bs": 48, "cuda_graph_bs": [1, 2, 4]}

    SGLangConfig.build_cmd_from_args(args)

    assert args == {"cuda_graph_max_bs": 48, "cuda_graph_bs": [1, 2, 4]}
