# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.trainer.rl_trainer import PPOTrainer


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_awex_colocation_accepts_megatron_and_sglang(version):
    PPOTrainer._validate_awex_colocate_backends(
        "megatron", "sglang", enable_memory_saver=True, version=version
    )


def test_v2_awex_colocation_rejects_fsdp_actor():
    with pytest.raises(ValueError, match="requires Megatron actor"):
        PPOTrainer._validate_awex_colocate_backends(
            "fsdp", "sglang", enable_memory_saver=True, version="v2"
        )


def test_v2_awex_colocation_rejects_vllm_rollout():
    with pytest.raises(ValueError, match="requires SGLang rollout"):
        PPOTrainer._validate_awex_colocate_backends(
            "megatron", "vllm", enable_memory_saver=True, version="v2"
        )


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_awex_colocation_requires_sglang_memory_saver(version):
    with pytest.raises(ValueError, match="enable_memory_saver=True"):
        PPOTrainer._validate_awex_colocate_backends(
            "megatron", "sglang", enable_memory_saver=False, version=version
        )
