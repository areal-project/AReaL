# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from omegaconf import OmegaConf

from examples.mopd.gsm8k_qwen3_14b_to_0_6b import (
    MOPD_TEACHER_GROUP,
    GSM8KRewardDistillationAgent,
    add_no_think_suffix,
    dynamic_filter,
    load_routed_gsm8k_dataset,
)

from areal.api.cli_args import (
    DatasetSourceConfig,
    GRPOConfig,
    TrainDatasetConfig,
    to_structured_cfg,
)
from areal.dataset.mopd import MOPD_ROUTE_METADATA_KEY
from areal.reward import gsm8k_reward_fn


class _Tokenizer:
    def encode(self, text):
        return text.split()


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
    assert config.mopd.teacher_groups == {MOPD_TEACHER_GROUP: {"qwen3_14b": 1.0}}
    assert config.mopd.loss.rl_coefficient == 0.0
    assert config.mopd.loss.distillation_coefficient == 1.0
    assert config.total_train_epochs == 10
    assert config.total_train_steps is None
    assert config.train_dataset.batch_size == 256
    assert config.train_dataset.path is None
    assert config.train_dataset.sources[0].teacher_group == MOPD_TEACHER_GROUP
    assert config.train_dataset.max_length is None
    assert config.valid_dataset is not None
    assert config.valid_dataset.batch_size == 256
    assert config.valid_dataset.sources[0].teacher_group == MOPD_TEACHER_GROUP
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
    assert add_no_think_suffix({"question": "1+1?"}) == {}


def test_add_no_think_suffix_appends_once():
    """Both dataset loaders use the same reference no-think prompt format."""
    sample = {"messages": [{"role": "user", "content": "What is 1 + 1?"}]}

    routed = add_no_think_suffix(sample)
    routed_twice = add_no_think_suffix(routed)

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
    config = TrainDatasetConfig(
        sources=[
            DatasetSourceConfig(
                path=str(tmp_path),
                type="rl",
                teacher_group=MOPD_TEACHER_GROUP,
                max_length=32,
            )
        ],
        scheduling_spec=None,
    )

    dataset = load_routed_gsm8k_dataset(
        config,
        tokenizer=_Tokenizer(),
    )

    assert len(dataset) == 1
    assert dataset[0]["answer"] == "#### 2"
    assert dataset[0][MOPD_ROUTE_METADATA_KEY].route == MOPD_TEACHER_GROUP
    assert "task_type" not in dataset[0]
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
