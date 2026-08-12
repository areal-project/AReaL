# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from omegaconf import OmegaConf

from examples.mopd.gsm8k_qwen3_14b_to_0_6b import (
    MOPD_ROUTE,
    GSM8KRewardDistillationAgent,
    add_mopd_route,
    dynamic_filter,
    load_routed_gsm8k_dataset,
    validate_heterogeneous_models,
)
from examples.mopd.gsm8k_qwen3_14b_to_0_6b import (
    model_fingerprint as _model_fingerprint,
)

from areal.api.cli_args import GRPOConfig, to_structured_cfg
from areal.reward import gsm8k_reward_fn


class _Tokenizer:
    def encode(self, text):
        return text.split()


def _write_model_metadata(
    path,
    *,
    intermediate_size: int = 6144,
    chat_template: str = "actor template",
) -> None:
    path.mkdir()
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": intermediate_size,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "vocab_size": 151936,
        "max_position_embeddings": 262144,
        "rope_theta": 10000000,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
    }
    tokenizer_config = {
        "added_tokens_decoder": {
            "151643": {"content": "<|endoftext|>", "special": True},
            "151645": {"content": "<|im_end|>", "special": True},
        },
        "bos_token": None,
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "unk_token": None,
        "chat_template": chat_template,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "vocab.json").write_text(
        json.dumps({"token-a": 0, "token-b": 1}), encoding="utf-8"
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config), encoding="utf-8"
    )


def test_model_fingerprint_ignores_chat_template_for_token_compatible_teacher(
    tmp_path,
):
    """Different prompting templates do not reject identical token mappings."""
    actor_path = tmp_path / "actor"
    teacher_path = tmp_path / "teacher"
    _write_model_metadata(actor_path)
    _write_model_metadata(teacher_path, chat_template="teacher-specific template")

    actor = _model_fingerprint(actor_path)
    teacher = _model_fingerprint(teacher_path)

    assert actor["architecture_sha256"] == teacher["architecture_sha256"]
    assert actor["tokenizer_sha256"] == teacher["tokenizer_sha256"]


def test_model_fingerprint_detects_incompatible_ffn_shape(tmp_path):
    """A checkpoint with a different dense FFN shape fails compatibility."""
    actor_path = tmp_path / "actor"
    teacher_path = tmp_path / "teacher"
    _write_model_metadata(actor_path)
    _write_model_metadata(teacher_path, intermediate_size=5472)

    actor = _model_fingerprint(actor_path)
    teacher = _model_fingerprint(teacher_path)

    assert actor["architecture_sha256"] != teacher["architecture_sha256"]


def test_heterogeneous_models_accept_different_architecture_with_same_tokenizer(
    tmp_path,
):
    """MOPD permits different model sizes when their token mappings match."""
    actor_path = tmp_path / "actor"
    teacher_path = tmp_path / "teacher"
    _write_model_metadata(actor_path, intermediate_size=3072)
    _write_model_metadata(teacher_path, intermediate_size=17408)

    fingerprints = validate_heterogeneous_models(
        actor_path, {"qwen3_14b": teacher_path}
    )

    assert (
        fingerprints["actor"]["architecture_sha256"]
        != fingerprints["qwen3_14b"]["architecture_sha256"]
    )
    assert (
        fingerprints["actor"]["tokenizer_sha256"]
        == fingerprints["qwen3_14b"]["tokenizer_sha256"]
    )


def test_heterogeneous_models_reject_different_tokenizer(tmp_path):
    """Teacher scoring fails fast when token IDs do not match the student."""
    actor_path = tmp_path / "actor"
    teacher_path = tmp_path / "teacher"
    _write_model_metadata(actor_path)
    _write_model_metadata(teacher_path)
    (teacher_path / "vocab.json").write_text(
        json.dumps({"different-token": 0}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="compatible token-ID mappings"):
        validate_heterogeneous_models(actor_path, {"qwen3_14b": teacher_path})


def test_qwen3_heterogeneous_local_example_has_expected_topology(monkeypatch):
    """The checked-in example resolves to one local node and eight GPUs."""
    monkeypatch.setenv("MOPD_STUDENT_MODEL_PATH", "/models/Qwen3-0.6B")
    monkeypatch.setenv("MOPD_TEACHER_MODEL_PATH", "/models/Qwen3-14B")
    monkeypatch.setenv("MOPD_GSM8K_PATH", "/datasets/gsm8k")
    monkeypatch.setenv("AREAL_ADMIN_API_KEY", "test-only-non-default-key")
    monkeypatch.setenv("AREAL_IMAGE", "areal:test")
    raw = OmegaConf.load("examples/mopd/gsm8k_qwen3_14b_to_0_6b_local.yaml")

    config = OmegaConf.to_object(to_structured_cfg(raw, GRPOConfig))

    assert isinstance(config, GRPOConfig) and config.mopd is not None
    assert config.enable_offload is False
    assert config.scheduler.type == "local"
    assert (config.cluster.n_nodes, config.cluster.n_gpus_per_node) == (1, 8)
    assert config.actor.backend == "megatron:d1p1t8"
    assert config.mopd.teacher_engine.backend == config.actor.backend
    assert config.rollout.backend == "sglang:d8t1p1"
    assert config.mopd.routes == {MOPD_ROUTE: {"qwen3_14b": 1.0}}
    assert config.mopd.loss.rl_coefficient == 0.0
    assert config.mopd.loss.distillation_coefficient == 1.0
    assert config.total_train_epochs == 10
    assert config.total_train_steps is None
    assert config.train_dataset.batch_size == 256
    assert config.train_dataset.max_length is None
    assert config.valid_dataset is not None
    assert config.valid_dataset.batch_size == 256
    assert config.valid_dataset.max_length is None
    assert config.rollout.max_concurrent_rollouts == 256
    assert config.sglang.max_running_requests is None
    assert config.actor.mb_spec.max_tokens_per_mb == 10240
    assert config.mopd.teacher_engine.mb_spec.max_tokens_per_mb == 10240
    assert config.actor.recompute_logprob is False
    assert config.actor.use_decoupled_loss is True
    assert config.actor.prox_logp_method == "reuse_train_logp"
    assert config.actor.should_compute_prox_logp() is False
    assert config.actor.reward_norm is None
    assert config.actor.adv_norm is None
    assert config.actor.rejection_sampling is None
    assert add_mopd_route({"question": "1+1?"}) == {"task_type": MOPD_ROUTE}


def test_add_mopd_route_appends_no_think_suffix_once():
    """Both dataset loaders use the same reference no-think prompt format."""
    sample = {"messages": [{"role": "user", "content": "What is 1 + 1?"}]}

    routed = add_mopd_route(sample)
    routed_twice = add_mopd_route(routed)

    assert sample["messages"][0]["content"] == "What is 1 + 1?"
    assert routed["messages"][0]["content"] == "What is 1 + 1? /no_think"
    assert routed_twice["messages"] == routed["messages"]


def test_dynamic_filter_matches_reference_all_correct_threshold():
    """Keep mixed groups and reject groups whose mean reward exceeds 0.95."""
    assert dynamic_filter({"rewards": torch.tensor([1.0, 1.0, 1.0, 0.0])})
    assert not dynamic_filter({"rewards": torch.ones(4)})


def test_load_routed_gsm8k_dataset_accepts_local_parquet_mirror(tmp_path):
    """The example directly consumes the parquet layout from the reference run."""
    from datasets import Dataset

    main_path = tmp_path / "main"
    main_path.mkdir()
    Dataset.from_dict(
        {"question": ["What is 1 + 1?"], "answer": ["#### 2"]}
    ).to_parquet(main_path / "train-00000-of-00001.parquet")
    config = type("DatasetConfig", (), {"path": str(tmp_path), "max_length": 32})()

    dataset = load_routed_gsm8k_dataset(
        config,
        split="train",
        tokenizer=_Tokenizer(),
    )

    assert len(dataset) == 1
    assert dataset[0]["answer"] == "#### 2"
    assert dataset[0]["task_type"] == MOPD_ROUTE
    assert dataset[0]["messages"][0]["content"].startswith("What is 1 + 1?")
    assert dataset[0]["messages"][0]["content"].endswith(" /no_think")


@pytest.mark.asyncio
async def test_gsm8k_distillation_agent_returns_verifier_reward(monkeypatch):
    """Pure distillation reports task quality without using it in the loss."""
    calls = []
    reward_calls = []

    class _Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            message = type("Message", (), {"content": "The answer is \\boxed{2}."})()
            choice = type("Choice", (), {"message": message})()
            return type(
                "Response",
                (),
                {"id": "completion-1", "choices": [choice]},
            )()

    class _Client:
        def __init__(self, **kwargs):
            del kwargs
            self.chat = type("Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr("openai.AsyncOpenAI", _Client)
    agent = GSM8KRewardDistillationAgent(temperature=1.0)
    assert agent._reward.reward_fn is gsm8k_reward_fn

    async def _reward(**kwargs):
        reward_calls.append(kwargs)
        return 1.0

    agent._reward = _reward

    reward = await agent.run(
        {
            "messages": [{"role": "user", "content": "What is 1 + 1?"}],
            "answer": "#### 2",
        },
        base_url="http://localhost:30000/v1",
        api_key="test-key",
    )

    assert reward == {"completion-1": 1.0}
    assert reward_calls == [
        {
            "prompt": "[{'role': 'user', 'content': 'What is 1 + 1?'}]",
            "completions": "The answer is \\boxed{2}.",
            "prompt_ids": [],
            "completion_ids": [],
            "answer": "#### 2",
        }
    ]
    assert calls == [
        {
            "messages": [{"role": "user", "content": "What is 1 + 1?"}],
            "model": "default",
            "temperature": 1.0,
        }
    ]
