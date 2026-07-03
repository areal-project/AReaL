# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from areal.trainer.rl_trainer import PPOTrainer, _build_v2_weight_update_meta


def _make_config(*, use_lora: bool, weight_update_mode: str):
    return SimpleNamespace(
        experiment_name="test-experiment",
        trial_name="test-trial",
        cluster=SimpleNamespace(fileroot="/tmp/areal-tests"),
        actor=SimpleNamespace(
            _version="v2",
            use_lora=use_lora,
            weight_update_mode=weight_update_mode,
            path="test-model",
            scheduling_strategy=SimpleNamespace(type="separation", target=None),
        ),
        gconfig=SimpleNamespace(lora_name="test-adapter"),
        rollout=SimpleNamespace(
            _version="v2",
            api_url=None,
            backend="sglang:d1",
            max_head_offpolicyness=2,
            return_routed_experts=False,
            scheduling_strategy=SimpleNamespace(type="separation", target=None),
        ),
        enable_offload=True,
    )


def test_v2_full_parameter_explicit_disk_mode_is_honored():
    config = _make_config(use_lora=False, weight_update_mode="disk")

    meta = _build_v2_weight_update_meta(config)

    assert meta.type == "disk"
    assert meta.use_lora is False


def test_v2_full_parameter_xccl_mode_uses_awex():
    config = _make_config(use_lora=False, weight_update_mode="xccl")

    meta = _build_v2_weight_update_meta(config)

    assert meta.type == "awex"


def test_v2_lora_uses_disk_regardless_of_configured_mode():
    config = _make_config(use_lora=True, weight_update_mode="xccl")

    meta = _build_v2_weight_update_meta(config)

    assert meta.type == "disk"
    assert meta.use_lora is True
    assert meta.lora_name == "test-adapter"
    assert meta.lora_keep_versions == 4


def test_v2_disk_mode_rejects_vllm_backend():
    config = _make_config(use_lora=False, weight_update_mode="disk")
    config.rollout.backend = "vllm:d1"

    with pytest.raises(ValueError, match="local SGLang"):
        _build_v2_weight_update_meta(config)


def test_v2_disk_mode_rejects_external_model():
    config = _make_config(use_lora=False, weight_update_mode="disk")
    config.rollout.backend = None
    config.rollout.api_url = "https://example.com/v1"

    with pytest.raises(ValueError, match="local SGLang"):
        _build_v2_weight_update_meta(config)


def test_v2_megatron_lora_error_recommends_supported_backend_pair():
    config = _make_config(use_lora=True, weight_update_mode="xccl")
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = config
    trainer.actor_alloc = SimpleNamespace(backend="megatron")
    trainer.rollout_alloc = SimpleNamespace(backend="sglang")
    trainer._should_offload_rollout = False
    trainer._should_offload_actor = False
    trainer._should_offload_critic = False
    trainer._should_offload_ref = False
    trainer._should_offload_teacher = False

    with pytest.raises(ValueError, match="FSDP actor with local SGLang"):
        trainer._validate_cfg()
