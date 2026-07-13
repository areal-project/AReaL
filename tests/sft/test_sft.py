import json
import os
import subprocess
import sys
from dataclasses import asdict

import pytest
import yaml

from tests.utils import get_dataset_path, get_model_path

from areal.api.cli_args import SFTConfig, load_expr_config


@pytest.mark.parametrize(
    ("backend", "v2"),
    [
        ("fsdp", False),
        ("megatron", False),
        ("archon", False),
        ("fsdp", True),
    ],
    ids=[
        "fsdp-v1",
        "megatron-v1",
        "archon-v1",
        "fsdp-v2",
    ],
)
def test_sft(tmp_path: str, backend: str, v2: bool) -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, f"config_{backend}.yaml")
    ref_losses_path = os.path.join(base_dir, f"ref_losses_{backend}.json")

    # Wrap over the original config to use local models/datasets if possible
    config, _ = load_expr_config(["--config", config_path], SFTConfig)

    # Use get_model_path to check local or download from HuggingFace
    local_model_path = config.actor.path.replace("/", "__")
    model_path = get_model_path(
        os.path.join("/storage/openpsi/models", local_model_path),
        config.actor.path,
    )
    config.actor.path = model_path
    config.tokenizer_path = model_path

    # Use get_dataset_path to check local or download from HuggingFace
    local_dataset_path = config.train_dataset.path.replace("/", "__")
    dataset_path = get_dataset_path(
        os.path.join("/storage/openpsi/data", local_dataset_path),
        config.train_dataset.path,
    )
    config.train_dataset.path = dataset_path

    # save new config
    os.makedirs(os.path.join(tmp_path, "config"), exist_ok=True)
    with open(os.path.join(tmp_path, "config", "config.yaml"), "w") as f:
        yaml.dump(
            asdict(config),
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    cmd = [
        sys.executable,
        os.path.join(base_dir, "entrypoint.py"),
        "--config",
        os.path.join(tmp_path, "config", "config.yaml"),
        f"cluster.fileroot={tmp_path}",
    ]
    if v2:
        cmd.append("actor._version=v2")

    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, env=os.environ)
    assert result.returncode == 0, (
        f"SFT subprocess failed with exit code {result.returncode}"
    )

    with open(os.path.join(tmp_path, "losses.json")) as f:
        losses: list[float] = json.load(f)

    with open(ref_losses_path) as f:
        ref_losses: list[float] = json.load(f)

    # Refer to https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close
    assert all(
        loss == pytest.approx(ref_loss, rel=1.6e-2, abs=1e-5)
        for loss, ref_loss in zip(losses, ref_losses)
    )
