import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from tests.utils import get_dataset_path, get_model_path

from areal.api.cli_args import GRPOConfig, load_expr_config

# Each (training_backend, inference_backend) pair and its pytest mark.
_SGLANG_CASES = [
    pytest.param("fsdp", "sglang", id="fsdp-sglang", marks=pytest.mark.sglang),
    pytest.param("megatron", "sglang", id="megatron-sglang", marks=pytest.mark.sglang),
    pytest.param("archon", "sglang", id="archon-sglang", marks=pytest.mark.sglang),
]
_VLLM_CASES = [
    pytest.param("fsdp", "vllm", id="fsdp-vllm", marks=pytest.mark.vllm),
    pytest.param("megatron", "vllm", id="megatron-vllm", marks=pytest.mark.vllm),
    pytest.param("archon", "vllm", id="archon-vllm", marks=pytest.mark.vllm),
]

# Megatron uses a smaller learning rate and weight decay than FSDP/Archon.
_MEGATRON_OVERRIDES = [
    "actor.optimizer.lr=3e-6",
    "actor.optimizer.weight_decay=0.003",
]


@pytest.mark.parametrize(("backend", "inference"), _SGLANG_CASES + _VLLM_CASES)
def test_grpo(tmp_path: Path, backend: str, inference: str) -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")

    # Load unified config and resolve local model/dataset paths
    config, _ = load_expr_config(["--config", config_path], GRPOConfig)

    local_model_path = config.actor.path.replace("/", "__")
    model_path = get_model_path(
        os.path.join("/storage/openpsi/models", local_model_path),
        config.actor.path,
    )
    config.actor.path = model_path
    config.ref.path = model_path
    config.tokenizer_path = model_path
    config.sglang.model_path = model_path
    config.vllm.model = model_path

    local_dataset_path = config.train_dataset.path.replace("/", "__")
    dataset_path = get_dataset_path(
        os.path.join("/storage/openpsi/data", local_dataset_path),
        config.train_dataset.path,
    )
    config.train_dataset.path = dataset_path

    # Save resolved config (model/dataset paths baked in)
    os.makedirs(os.path.join(tmp_path, "config"), exist_ok=True)
    with open(os.path.join(tmp_path, "config", "config.yaml"), "w") as f:
        yaml.dump(
            asdict(config),
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    # Build CLI overrides for backend-specific values
    cli_overrides = [
        f"rollout.backend='{inference}:d1'",
        f"actor.backend='{backend}:d1'",
    ]
    if backend == "megatron":
        cli_overrides.extend(_MEGATRON_OVERRIDES)

    cmd = [
        sys.executable,
        os.path.join(base_dir, "entrypoint.py"),
        "--config",
        os.path.join(tmp_path, "config", "config.yaml"),
        f"cluster.fileroot={tmp_path}",
        *cli_overrides,
    ]

    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, env=os.environ)
    assert result.returncode == 0, (
        f"GRPO subprocess failed with exit code {result.returncode}"
    )

    with open(os.path.join(tmp_path, "rewards.json")) as f:
        rewards: list[float] = json.load(f)

    assert rewards[-1] > 0.6
