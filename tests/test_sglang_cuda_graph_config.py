# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

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
    assert "--mamba-scheduler-strategy" not in command
    assert "--mamba-radix-cache-strategy" not in command


@pytest.mark.parametrize(
    ("installed_version", "expected_flag", "legacy_flag"),
    [
        ("0.5.19", "--mamba-scheduler-strategy", "--mamba-radix-cache-strategy"),
        ("0.5.20", "--mamba-radix-cache-strategy", "--mamba-scheduler-strategy"),
        ("0.5.21", "--mamba-radix-cache-strategy", "--mamba-scheduler-strategy"),
    ],
)
def test_mamba_cache_strategy_uses_runtime_cli_name(
    build_command, monkeypatch, installed_version, expected_flag, legacy_flag
):
    """Translate the renamed hybrid-Mamba radix cache strategy flag on 0.5.20+."""
    monkeypatch.setattr(
        cli_args.pkg_version,
        "is_version_greater_or_equal",
        lambda package, version: Version(installed_version) >= Version(version),
    )

    command = build_command(mamba_scheduler_strategy="extra_buffer")

    assert command[command.index(expected_flag) + 1] == "extra_buffer"
    assert legacy_flag not in command


@pytest.mark.parametrize("installed_version", ["0.5.19", "0.5.20"])
def test_graph_cli_conversion_preserves_input_args(monkeypatch, installed_version):
    monkeypatch.setattr(
        cli_args.pkg_version,
        "is_version_greater_or_equal",
        lambda package, version: Version(installed_version) >= Version(version),
    )
    args = {
        "cuda_graph_max_bs": 48,
        "cuda_graph_bs": [1, 2, 4],
        "mamba_scheduler_strategy": "extra_buffer",
    }

    SGLangConfig.build_cmd_from_args(args)

    assert args == {
        "cuda_graph_max_bs": 48,
        "cuda_graph_bs": [1, 2, 4],
        "mamba_scheduler_strategy": "extra_buffer",
    }


def test_qwen3_5_colocate_recipe_sglang_args_translate_for_0520(monkeypatch, tmp_path):
    """Verify the Qwen3.5 colocate recipe translates flags cleanly for SGLang 0.5.20."""
    recipe_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "math"
        / "qwen3_5_27b_grpo_awex_colocate.yaml"
    )
    recipe = OmegaConf.load(recipe_path)
    config = OmegaConf.to_object(
        OmegaConf.merge(
            OmegaConf.structured(SGLangConfig),
            recipe.sglang,
            {"model_path": str(tmp_path)},
        )
    )
    monkeypatch.setattr(
        cli_args.pkg_version,
        "is_version_greater_or_equal",
        lambda package, version: Version("0.5.20") >= Version(version),
    )

    command = SGLangConfig.build_cmd(config, tp_size=2, base_gpu_id=0)

    assert command[command.index("--mamba-radix-cache-strategy") + 1] == "extra_buffer"
    assert "--mamba-scheduler-strategy" not in command


def test_sglang_0520_parser_accepts_translated_cli_args(tmp_path):
    """Parse generated flags with SGLang 0.5.20's real CLI parser when installed."""
    sglang_server_args = pytest.importorskip("sglang.srt.server_args")
    if not cli_args.pkg_version.is_version_greater_or_equal("sglang", "0.5.20"):
        pytest.skip("Requires SGLang >= 0.5.20")

    recipe_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "math"
        / "qwen3_5_27b_grpo_awex_colocate.yaml"
    )
    recipe = OmegaConf.load(recipe_path)
    config = OmegaConf.to_object(
        OmegaConf.merge(
            OmegaConf.structured(SGLangConfig),
            recipe.sglang,
            {
                "model_path": str(tmp_path),
                "cuda_graph_max_bs": 48,
                "cuda_graph_bs": [1, 2, 4],
            },
        )
    )

    command = SGLangConfig.build_cmd(config, tp_size=2, base_gpu_id=0)
    parser = argparse.ArgumentParser()
    sglang_server_args.ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(command[3:])

    assert parsed.mamba_radix_cache_strategy == "extra_buffer"
    assert parsed.cuda_graph_max_bs_decode == 48
    assert parsed.cuda_graph_bs_decode == [1, 2, 4]
