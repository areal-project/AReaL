# SPDX-License-Identifier: Apache-2.0

import json

from transformers import AutoConfig, PretrainedConfig

from areal.utils.hf_utils import finalize_hf_export, save_hf_config


class RuntimeModelTypeConfig(PretrainedConfig):
    """Config that deliberately inherits the empty class-level model_type."""


def test_save_hf_config_preserves_instance_model_type_without_source(tmp_path):
    """Runtime model_type survives even when no local base model is available."""
    config = RuntimeModelTypeConfig(architectures=["RuntimeModel"])
    config.model_type = "runtime_only"

    save_path = tmp_path / "saved"
    save_hf_config(config, save_path)

    with (save_path / "config.json").open() as f:
        saved_config = json.load(f)
    assert saved_config["model_type"] == "runtime_only"
    assert saved_config["architectures"] == ["RuntimeModel"]


def test_save_hf_config_supports_remote_config_round_trip(tmp_path):
    """A config with no class model_type remains loadable through AutoConfig."""
    source_path = tmp_path / "source"
    source_path.mkdir()
    module_name = "configuration_runtime_only.py"
    (source_path / module_name).write_text(
        "from transformers import PretrainedConfig\n\n"
        "class RuntimeOnlyConfig(PretrainedConfig):\n"
        "    pass\n"
    )
    with (source_path / "config.json").open("w") as f:
        json.dump(
            {
                "architectures": ["RuntimeOnlyModel"],
                "auto_map": {
                    "AutoConfig": "configuration_runtime_only.RuntimeOnlyConfig"
                },
                "model_type": "runtime_only",
                "torch_dtype": "bfloat16",
            },
            f,
        )

    config = PretrainedConfig.from_pretrained(source_path)
    save_path = tmp_path / "saved"
    (source_path / "generation_config.json").write_text('{"max_new_tokens": 1}')
    (source_path / "chat_template.jinja").write_text("{{ messages }}")
    finalize_hf_export(config, save_path, source_model_path=source_path)

    loaded = AutoConfig.from_pretrained(save_path, trust_remote_code=True)

    assert type(loaded).__name__ == "RuntimeOnlyConfig"
    assert loaded.model_type == "runtime_only"
    assert (save_path / module_name).is_file()
    assert (save_path / "generation_config.json").is_file()
    assert (save_path / "chat_template.jinja").is_file()
    with (save_path / "config.json").open() as f:
        saved_config = json.load(f)
    assert saved_config["torch_dtype"] == "bfloat16"


def test_finalize_hf_export_snapshots_same_directory_config(tmp_path):
    """In-place finalization preserves fields before save_pretrained overwrites them."""
    config = RuntimeModelTypeConfig(architectures=["RuntimeModel"])
    config.model_type = "runtime_only"
    with (tmp_path / "config.json").open("w") as f:
        json.dump(
            {"model_type": "runtime_only", "torch_dtype": "bfloat16"},
            f,
        )

    finalize_hf_export(config, tmp_path, source_model_path=tmp_path)

    with (tmp_path / "config.json").open() as f:
        saved_config = json.load(f)
    assert saved_config["model_type"] == "runtime_only"
    assert saved_config["torch_dtype"] == "bfloat16"
