# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import sys
import types

import pytest
import torch

from areal.api.reward_api import normalize_reward_result
from areal.api.reward_api import RewardResult
from areal.api.io_struct import ModelRequest
from areal.trainer.ppo.actor import PPOActor


@contextlib.asynccontextmanager
async def _dummy_atrace_session_phase(_name):
    yield


def _dummy_trace_session(_name):
    def decorator(fn):
        return fn

    return decorator


def _dummy_session_context():
    def decorator(fn):
        return fn

    return decorator


_perf_tracer_stub = types.ModuleType("areal.utils.perf_tracer")
_perf_tracer_stub.atrace_session_phase = _dummy_atrace_session_phase
_perf_tracer_stub.trace_session = _dummy_trace_session
_perf_tracer_stub.session_context = _dummy_session_context
sys.modules.setdefault("areal.utils.perf_tracer", _perf_tracer_stub)


class _DummyTracker:
    def scalar(self, **kwargs):
        return None


_stats_tracker_stub = types.ModuleType("areal.utils.stats_tracker")
_stats_tracker_stub.get = lambda _scope: _DummyTracker()
sys.modules.setdefault("areal.utils.stats_tracker", _stats_tracker_stub)

_workflow_context_stub = types.ModuleType("areal.infra.workflow_context")
_workflow_context_stub.stat_scope = lambda: "rollout"
sys.modules.setdefault("areal.infra.workflow_context", _workflow_context_stub)

_infra_stub = types.ModuleType("areal.infra")
_infra_stub.workflow_context = _workflow_context_stub
sys.modules.setdefault("areal.infra", _infra_stub)

from areal.workflow.rlvr import RLVRWorkflow


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

        assert torch.equal(result["rewards"], torch.tensor([1.25], dtype=torch.float32))
        assert torch.equal(
            result["step_rewards"], torch.zeros((1, 6), dtype=torch.float32)
        )
        assert torch.equal(
            result["step_reward_mask"], torch.zeros((1, 6), dtype=torch.bool)
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

        assert torch.equal(result["rewards"], torch.tensor([0.5], dtype=torch.float32))
        expected_step_rewards = torch.tensor(
            [[0.0, 0.0, 0.2, 0.0, -0.1, 0.0]], dtype=torch.float32
        )
        expected_mask = torch.tensor(
            [[False, False, True, False, True, False]], dtype=torch.bool
        )
        assert torch.equal(result["step_rewards"], expected_step_rewards)
        assert torch.equal(result["step_reward_mask"], expected_mask)

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


class _DummyKL:
    def __call__(self, old_logp, ref_logp):
        del old_logp, ref_logp
        return torch.zeros_like(old_logp)


def _build_actor_for_test():
    actor = PPOActor.__new__(PPOActor)
    actor.reward_bias = 0.0
    actor.reward_scaling = 1.0
    actor.reward_clip = 10.0
    actor.reward_norm = _DummyRewardNorm()
    actor.adv_norm = None
    actor.kl_ctl = 0.0
    actor.kl_estimator = _DummyKL()
    actor.discount = 1.0
    actor.gae_lambda = 1.0
    actor.mask_no_eos_with_zero = False
    actor.engine = _DummyEngineForPPO()
    actor.config = types.SimpleNamespace(
        overlong_reward_penalty=False,
        overlong_tokens=None,
        overlong_penalty_factor=None,
        use_decoupled_loss=False,
        recompute_logprob=False,
        log_agent_stats=False,
        ppo_n_minibatches=1,
        eps_clip=0.2,
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
    def test_compute_advantages_keeps_final_step_reward(self, monkeypatch):
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
        assert torch.equal(result["tot_rewards"], expected_tot_rewards)

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
            "step_rewards": torch.tensor([[0.0, 0.2, 0.0, 0.0]], dtype=torch.float32),
            "step_reward_mask": torch.tensor(
                [[False, True, False, False]], dtype=torch.bool
            ),
            "logprobs": torch.zeros((1, 4), dtype=torch.float32),
            "versions": torch.zeros((1, 4), dtype=torch.int32),
        }

        actor._ppo_update(data)

        assert any("step_reward" in stat for stat in tracker.stats)
        assert any(
            scalar.get("step_reward_events") == 1.0 for scalar in tracker.scalars
        )
