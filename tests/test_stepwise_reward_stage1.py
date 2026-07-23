# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import types

import pytest
import torch

from areal.api.io_struct import ModelRequest
from areal.api.reward_api import RewardResult, normalize_reward_result
from areal.trainer.ppo.actor import PPOActor
from areal.utils.data import concat_batch
from areal.workflow.rlvr import RLVRWorkflow

CUDA_AVAILABLE = torch.cuda.is_available()


class _DummyTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def decode(self, token_ids):
        return "|".join(str(x) for x in token_ids)


class _DummyGConfig:
    def __init__(self, max_new_tokens=8, n_samples=1):
        self.max_new_tokens = max_new_tokens
        self.n_samples = n_samples

    def new_with_stop_and_pad_token_ids(self, _tokenizer):
        return self

    def new(self, n_samples=1):
        return _DummyGConfig(
            max_new_tokens=self.max_new_tokens,
            n_samples=n_samples,
        )


class _DummyModelResponse:
    def __init__(
        self,
        input_tokens,
        output_tokens,
        output_logprobs,
        output_versions,
        stop_reason="stop",
        tokenizer=None,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.output_logprobs = output_logprobs
        self.output_versions = output_versions
        self.stop_reason = stop_reason
        self.tokenizer = tokenizer

    @property
    def input_len(self):
        return len(self.input_tokens)

    @property
    def output_len(self):
        return len(self.output_tokens)


class _DummyEngine:
    async def agenerate(self, req):
        assert isinstance(req, ModelRequest)
        copied_req = req.copy()
        return _DummyModelResponse(
            input_tokens=copied_req.input_ids,
            output_tokens=[11, 12, 13, 14],
            output_logprobs=[-0.1, -0.2, -0.3, -0.4],
            output_versions=[1, 1, 1, 1],
            stop_reason="stop",
            tokenizer=copied_req.tokenizer,
        )


def _dummy_get_input_ids_fn(_data, _tokenizer, _enable_thinking):
    return [101, 102]


def _scalar_reward_fn(prompt, completions, prompt_ids, completion_ids, **kwargs):
    del prompt, completions, prompt_ids, completion_ids, kwargs
    return 1.25


def _stepwise_reward_fn(prompt, completions, prompt_ids, completion_ids, **kwargs):
    del prompt, completions, prompt_ids, completion_ids, kwargs
    return RewardResult(
        final_reward=0.5,
        step_rewards=[0.2, -0.1],
        step_ends=[2, 4],
        metadata={"num_steps": 2},
    )


def _duplicate_step_reward_fn(
    prompt, completions, prompt_ids, completion_ids, **kwargs
):
    del prompt, completions, prompt_ids, completion_ids, kwargs
    return RewardResult(
        final_reward=0.0,
        step_rewards=[0.8, 0.8],
        step_ends=[1, 1],
    )


def _empty_step_reward_fn(prompt, completions, prompt_ids, completion_ids, **kwargs):
    del prompt, completions, prompt_ids, completion_ids, kwargs
    return RewardResult(final_reward=0.25, step_rewards=[], step_ends=[])


def _bad_reward_fn(prompt, completions, prompt_ids, completion_ids, **kwargs):
    del prompt, completions, prompt_ids, completion_ids, kwargs
    return RewardResult(
        final_reward=0.0,
        step_rewards=[1.0],
        step_ends=[5],
    )


class TestRewardResultNormalization:
    def test_normalize_scalar_reward(self):
        reward = normalize_reward_result(1.5)
        assert reward == RewardResult(final_reward=1.5)

    def test_normalize_reward_result_passthrough(self):
        reward = RewardResult(final_reward=2.0, step_rewards=[0.1], step_ends=[1])
        assert normalize_reward_result(reward) is reward


class TestRLVRWorkflowStepwiseReward:
    @pytest.mark.asyncio
    async def test_scalar_reward_keeps_zero_stepwise_tensors(self):
        workflow = RLVRWorkflow(
            reward_fn=_scalar_reward_fn,
            gconfig=_DummyGConfig(max_new_tokens=8),
            tokenizer=_DummyTokenizer(),
            enable_thinking=False,
            get_input_ids_fn=_dummy_get_input_ids_fn,
        )

        result = await workflow.arun_episode(_DummyEngine(), {"messages": []})

        torch.testing.assert_close(
            result["rewards"],
            torch.tensor([1.25], dtype=torch.float32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result["step_rewards"],
            torch.zeros((1, 6), dtype=torch.float32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result["step_reward_mask"],
            torch.zeros((1, 6), dtype=torch.bool),
            rtol=0,
            atol=0,
        )

    @pytest.mark.asyncio
    async def test_stepwise_reward_aligns_to_completion_end_positions(self):
        workflow = RLVRWorkflow(
            reward_fn=_stepwise_reward_fn,
            gconfig=_DummyGConfig(max_new_tokens=8),
            tokenizer=_DummyTokenizer(),
            enable_thinking=False,
            get_input_ids_fn=_dummy_get_input_ids_fn,
        )

        result = await workflow.arun_episode(_DummyEngine(), {"messages": []})

        torch.testing.assert_close(
            result["rewards"],
            torch.tensor([0.5], dtype=torch.float32),
            rtol=0,
            atol=0,
        )
        expected_step_rewards = torch.tensor(
            [[0.0, 0.0, 0.2, 0.0, -0.1, 0.0]], dtype=torch.float32
        )
        expected_mask = torch.tensor(
            [[False, False, True, False, True, False]], dtype=torch.bool
        )
        torch.testing.assert_close(
            result["step_rewards"], expected_step_rewards, rtol=0, atol=0
        )
        torch.testing.assert_close(
            result["step_reward_mask"], expected_mask, rtol=0, atol=0
        )

    @pytest.mark.asyncio
    async def test_invalid_stepwise_reward_raises(self):
        workflow = RLVRWorkflow(
            reward_fn=_bad_reward_fn,
            gconfig=_DummyGConfig(max_new_tokens=8),
            tokenizer=_DummyTokenizer(),
            enable_thinking=False,
            get_input_ids_fn=_dummy_get_input_ids_fn,
        )

        with pytest.raises(ValueError, match="Invalid step_end"):
            await workflow.arun_episode(_DummyEngine(), {"messages": []})

    @pytest.mark.asyncio
    async def test_duplicate_first_step_rewards_aggregate_before_actor_clip(self):
        """Duplicate boundaries aggregate before per-timestep clipping."""
        workflow = RLVRWorkflow(
            reward_fn=_duplicate_step_reward_fn,
            gconfig=_DummyGConfig(max_new_tokens=8),
            tokenizer=_DummyTokenizer(),
            enable_thinking=False,
            get_input_ids_fn=_dummy_get_input_ids_fn,
        )
        result = await workflow.arun_episode(_DummyEngine(), {"messages": []})
        actor = _build_actor_for_test(reward_clip=1.0)

        result = actor._compute_advantages(result)

        expected_tot_rewards = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
        torch.testing.assert_close(
            result["tot_rewards"], expected_tot_rewards, rtol=0, atol=0
        )

    @pytest.mark.parametrize(
        "non_stepwise_reward_fn",
        [_scalar_reward_fn, _empty_step_reward_fn],
        ids=["scalar", "empty-step-list"],
    )
    @pytest.mark.asyncio
    async def test_non_stepwise_and_stepwise_samples_batch_together(
        self, non_stepwise_reward_fn
    ):
        """Optional process rewards retain a batch-compatible tensor schema."""
        non_stepwise_workflow = RLVRWorkflow(
            reward_fn=non_stepwise_reward_fn,
            gconfig=_DummyGConfig(max_new_tokens=8),
            tokenizer=_DummyTokenizer(),
            enable_thinking=False,
            get_input_ids_fn=_dummy_get_input_ids_fn,
        )
        stepwise_workflow = RLVRWorkflow(
            reward_fn=_stepwise_reward_fn,
            gconfig=_DummyGConfig(max_new_tokens=8),
            tokenizer=_DummyTokenizer(),
            enable_thinking=False,
            get_input_ids_fn=_dummy_get_input_ids_fn,
        )
        non_stepwise_result = await non_stepwise_workflow.arun_episode(
            _DummyEngine(), {"messages": []}
        )
        stepwise_result = await stepwise_workflow.arun_episode(
            _DummyEngine(), {"messages": []}
        )

        batched, _ = concat_batch([non_stepwise_result, stepwise_result])

        assert batched["step_rewards"].shape == (2, 6)
        assert batched["step_reward_mask"].shape == (2, 6)
        assert not batched["step_reward_mask"][0].any()
        assert batched["step_reward_mask"][1].sum() == 2


class _DummyStatsTracker:
    def __init__(self):
        self.denominators = []
        self.stats = []
        self.scalars = []

    def denominator(self, **kwargs):
        self.denominators.append(kwargs)

    def stat(self, **kwargs):
        self.stats.append(kwargs)

    def scalar(self, **kwargs):
        self.scalars.append(kwargs)

    @contextlib.contextmanager
    def scope(self, _name):
        yield


class _DummyEngineForPPO:
    def train(self):
        return None

    def get_version(self):
        return 0

    def train_batch(self, *args, **kwargs):
        return {}


class _DummyRewardNorm:
    def __call__(self, x, group_sizes=None):
        del group_sizes
        return x


class _ShiftRewardNorm:
    def __call__(self, x, group_sizes=None):
        del group_sizes
        return x + 1.0


class _DummyKL:
    def __call__(self, old_logp, ref_logp):
        del ref_logp
        return torch.zeros_like(old_logp)


def _build_actor_for_test(
    *,
    reward_bias: float = 0.0,
    reward_scaling: float = 1.0,
    reward_clip: float = 10.0,
    reward_norm: _DummyRewardNorm | _ShiftRewardNorm | None = None,
    mask_no_eos_with_zero: bool = False,
) -> PPOActor:
    actor = PPOActor.__new__(PPOActor)
    actor.reward_bias = reward_bias
    actor.reward_scaling = reward_scaling
    actor.reward_clip = reward_clip
    actor.reward_norm = reward_norm
    actor.adv_norm = None
    actor.kl_ctl = 0.0
    actor.kl_estimator = _DummyKL()
    actor.discount = 1.0
    actor.gae_lambda = 1.0
    actor.mask_no_eos_with_zero = mask_no_eos_with_zero
    actor.m2_threshold = None
    actor.engine = _DummyEngineForPPO()
    actor.config = types.SimpleNamespace(
        overlong_reward_penalty=False,
        overlong_tokens=None,
        overlong_penalty_factor=None,
        use_decoupled_loss=False,
        recompute_logprob=False,
        log_agent_stats=False,
        mask_no_eos_with_zero=mask_no_eos_with_zero,
        ppo_n_minibatches=1,
        eps_clip=0.2,
        eps_clip_higher=None,
        c_clip=None,
        rejection_sampling=None,
        importance_sampling_level="token",
        prox_logp_method="recompute",
        use_sapo_loss=False,
        sapo_tau_pos=1.0,
        sapo_tau_neg=1.05,
        use_cispo_loss=False,
    )
    return actor


class TestPPOActorStepwiseReward:
    def test_compute_advantages_keeps_final_step_reward(self):
        actor = _build_actor_for_test()
        data = {
            "input_ids": torch.tensor([[101, 102, 11, 12, 13, 14]], dtype=torch.int32),
            "attention_mask": torch.ones((1, 6), dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.int32),
            "logprobs": torch.zeros((1, 6), dtype=torch.float32),
            "rewards": torch.tensor([0.5], dtype=torch.float32),
            "step_rewards": torch.tensor(
                [[0.0, 0.0, 0.2, 0.0, -0.1, 0.0]], dtype=torch.float32
            ),
            "step_reward_mask": torch.tensor(
                [[False, False, True, False, True, False]], dtype=torch.bool
            ),
        }

        result = actor._compute_advantages(data)

        expected_tot_rewards = torch.tensor(
            [[0.0, 0.0, 0.2, 0.0, 0.4, 0.0]], dtype=torch.float32
        )
        torch.testing.assert_close(
            result["tot_rewards"], expected_tot_rewards, rtol=0, atol=0
        )

    def test_compute_advantages_scales_and_clips_step_rewards(self):
        """Step deltas share scaling and clipping, but not terminal bias."""
        actor = _build_actor_for_test(
            reward_bias=-0.5,
            reward_scaling=10.0,
            reward_clip=1.0,
        )
        data = {
            "input_ids": torch.tensor([[101, 102, 11, 12, 13, 14]], dtype=torch.int32),
            "attention_mask": torch.ones((1, 6), dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.int32),
            "logprobs": torch.zeros((1, 6), dtype=torch.float32),
            "rewards": torch.tensor([0.5], dtype=torch.float32),
            "step_rewards": torch.tensor(
                [[0.0, 0.0, 0.2, float("nan"), -0.2, 0.0]], dtype=torch.float32
            ),
            "step_reward_mask": torch.tensor(
                [[False, False, True, False, True, False]], dtype=torch.bool
            ),
        }

        result = actor._compute_advantages(data)

        expected_tot_rewards = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, -1.0, 0.0]], dtype=torch.float32
        )
        torch.testing.assert_close(
            result["tot_rewards"], expected_tot_rewards, rtol=0, atol=0
        )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
    def test_compute_advantages_moves_step_rewards_to_actor_device(self):
        """CPU process rewards are moved to the actor's CUDA device."""
        actor = _build_actor_for_test()
        data = {
            "input_ids": torch.tensor(
                [[101, 102, 11, 12]], dtype=torch.int32, device="cuda"
            ),
            "attention_mask": torch.ones((1, 4), dtype=torch.bool, device="cuda"),
            "loss_mask": torch.tensor([[0, 0, 1, 1]], dtype=torch.int32, device="cuda"),
            "logprobs": torch.zeros((1, 4), dtype=torch.float32, device="cuda"),
            "rewards": torch.tensor([0.5], dtype=torch.float32, device="cuda"),
            "step_rewards": torch.tensor([[0.0, 0.2, -0.1, 0.0]], dtype=torch.float32),
            "step_reward_mask": torch.tensor(
                [[False, True, True, False]], dtype=torch.bool
            ),
        }

        result = actor._compute_advantages(data)

        expected_tot_rewards = torch.tensor(
            [[0.0, 0.2, 0.4, 0.0]], dtype=torch.float32, device="cuda"
        )
        torch.testing.assert_close(
            result["tot_rewards"], expected_tot_rewards, rtol=0, atol=0
        )
        assert result["step_rewards"].device.type == "cuda"
        assert result["step_reward_mask"].device.type == "cuda"

    def test_compute_advantages_applies_reward_norm_only_to_terminal_reward(self):
        """Sequence normalization does not alter sparse process increments."""
        actor = _build_actor_for_test(reward_norm=_ShiftRewardNorm())
        data = {
            "input_ids": torch.tensor([[101, 102, 11, 12]], dtype=torch.int32),
            "attention_mask": torch.ones((1, 4), dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 0, 1, 1]], dtype=torch.int32),
            "logprobs": torch.zeros((1, 4), dtype=torch.float32),
            "rewards": torch.tensor([0.5], dtype=torch.float32),
            "step_rewards": torch.tensor([[0.0, 0.2, -0.1, 0.0]], dtype=torch.float32),
            "step_reward_mask": torch.tensor(
                [[False, True, True, False]], dtype=torch.bool
            ),
        }

        result = actor._compute_advantages(data)

        expected_tot_rewards = torch.tensor([[0.0, 0.2, 1.4, 0.0]], dtype=torch.float32)
        torch.testing.assert_close(
            result["tot_rewards"], expected_tot_rewards, rtol=0, atol=0
        )

    def test_compute_advantages_keeps_scalar_reward_norm_path(self):
        """Scalar rewards retain the existing bias, scale, clip, and norm path."""
        actor = _build_actor_for_test(
            reward_bias=-0.5,
            reward_scaling=10.0,
            reward_clip=2.0,
            reward_norm=_DummyRewardNorm(),
        )
        data = {
            "input_ids": torch.tensor([[101, 102, 11, 12]], dtype=torch.int32),
            "attention_mask": torch.ones((1, 4), dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 0, 1, 1]], dtype=torch.int32),
            "logprobs": torch.zeros((1, 4), dtype=torch.float32),
            "rewards": torch.tensor([0.6], dtype=torch.float32),
        }

        result = actor._compute_advantages(data)

        expected_tot_rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
        torch.testing.assert_close(
            result["tot_rewards"], expected_tot_rewards, rtol=1e-6, atol=1e-6
        )

    def test_compute_advantages_masks_step_rewards_for_truncated_sequence(self):
        """No-EOS masking removes terminal and process task rewards."""
        actor = _build_actor_for_test(mask_no_eos_with_zero=True)
        actor.kl_ctl = 0.1
        actor.kl_estimator = lambda old_logp, ref_logp: old_logp - ref_logp
        actor.config.overlong_reward_penalty = True
        actor.config.overlong_tokens = 1
        actor.config.overlong_penalty_factor = 1.0
        actor.config.max_new_tokens = 2
        data = {
            "input_ids": torch.tensor([[101, 102, 11, 12]], dtype=torch.int32),
            "attention_mask": torch.ones((1, 4), dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 0, 1, 1]], dtype=torch.int32),
            "logprobs": torch.tensor([[0.0, 0.0, 0.5, 0.5]], dtype=torch.float32),
            "rewards": torch.tensor([0.5], dtype=torch.float32),
            "step_rewards": torch.tensor([[0.0, 0.2, -0.1, 0.0]], dtype=torch.float32),
            "step_reward_mask": torch.tensor(
                [[False, True, True, False]], dtype=torch.bool
            ),
        }

        result = actor._compute_advantages(data)

        expected_tot_rewards = torch.tensor(
            [[0.0, -0.05, -0.05, 0.0]], dtype=torch.float32
        )
        torch.testing.assert_close(
            result["tot_rewards"], expected_tot_rewards, rtol=0, atol=0
        )

    def test_ppo_update_reads_stepwise_rewards_from_batch(self, monkeypatch):
        actor = _build_actor_for_test()
        tracker = _DummyStatsTracker()
        monkeypatch.setattr("areal.trainer.ppo.actor.stats_tracker", tracker)
        monkeypatch.setattr(
            "areal.trainer.ppo.actor.split_padded_tensor_dict_into_mb_list",
            lambda data, mb_spec: types.SimpleNamespace(mbs=[data]),
        )

        data = {
            "input_ids": torch.tensor([[101, 102, 11, 12]], dtype=torch.int32),
            "attention_mask": torch.ones((1, 4), dtype=torch.bool),
            "loss_mask": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            "rewards": torch.tensor([0.5], dtype=torch.float32),
            "advantages": torch.zeros((1, 4), dtype=torch.float32),
            "kl_rewards": torch.zeros((1, 4), dtype=torch.float32),
            "tot_rewards": torch.zeros((1, 4), dtype=torch.float32),
            "step_rewards": torch.tensor(
                [[float("nan"), 0.2, 0.0, 0.0]], dtype=torch.float32
            ),
            "step_reward_mask": torch.tensor(
                [[False, True, False, False]], dtype=torch.bool
            ),
            "logprobs": torch.zeros((1, 4), dtype=torch.float32),
            "versions": torch.zeros((1, 4), dtype=torch.int32),
        }

        actor._ppo_update(data)

        step_reward_stat = next(stat for stat in tracker.stats if "step_reward" in stat)
        assert step_reward_stat["denominator"] == "step_reward_events"
        torch.testing.assert_close(
            step_reward_stat["step_reward"],
            torch.tensor([[0.0, 0.2, 0.0, 0.0]], dtype=torch.float32),
            rtol=0,
            atol=0,
        )
        assert tracker.denominators[0]["step_reward_events"].sum() == 1
        assert "step_rewards" not in data
        assert "step_reward_mask" not in data
