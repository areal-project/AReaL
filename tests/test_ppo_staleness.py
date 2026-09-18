# SPDX-License-Identifier: Apache-2.0

import copy
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.api.cli_args import PPOActorConfig, PPOCriticConfig
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.critic import PPOCritic
from areal.trainer.ppo.staleness import (
    apply_staleness_mask,
    has_global_trainable_tokens,
)
from areal.utils.data import TRANSPORT_DUMMY_KEY


@pytest.fixture(autouse=True)
def isolate_stats(monkeypatch):
    for module in ("actor", "critic", "staleness"):
        monkeypatch.setattr(f"areal.trainer.ppo.{module}.stats_tracker", MagicMock())


def _batch(versions=(0, 1, 3)):
    """One prompt, three response tokens, then padding; mask already shifted."""
    return {
        "input_ids": torch.tensor([[10, 11, 12, 13, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool),
        "loss_mask": torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32),
        "versions": torch.tensor([[-1, *versions, -1]], dtype=torch.int32),
        "logprobs": torch.zeros(1, 5),
        "prox_logp": torch.zeros(1, 5),
        "advantages": torch.ones(1, 5),
        "returns": torch.ones(1, 5),
        "values": torch.zeros(1, 5),
        "rewards": torch.ones(1),
        "tot_rewards": torch.ones(1, 5),
        "kl_rewards": torch.zeros(1, 5),
        "is_truncated": torch.zeros(1, dtype=torch.bool),
    }


def _engine(version=3):
    return SimpleNamespace(
        get_version=lambda: version,
        train=MagicMock(),
        train_batch=MagicMock(return_value={}),
        data_parallel_group=None,
        stream_microbatches_from_cpu=False,
    )


@pytest.mark.parametrize(
    "version,limit,expected",
    [(2, 2, [1, 1, 1, 0, 0]), (3, 2, [0, 1, 1, 0, 0]), (3, 0, [0, 0, 0, 0, 0])],
)
def test_mask_versions_align_with_targets_and_include_boundary(
    version, limit, expected
):
    data = _batch((0, 1, 2))
    original_advantages = data["advantages"].clone()

    apply_staleness_mask(data, current_version=version, max_staleness=limit)

    torch.testing.assert_close(
        data["loss_mask"], torch.tensor([expected], dtype=torch.float32), rtol=0, atol=0
    )
    torch.testing.assert_close(
        data["versions"],
        torch.tensor([[0, 1, 2, -1, -1]], dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(data["advantages"], original_advantages, rtol=0, atol=0)


@pytest.mark.parametrize("bad_version", [-1, 4])
def test_mask_unknown_or_future_active_version_fails(bad_version):
    with pytest.raises(RuntimeError, match="known rollout versions"):
        apply_staleness_mask(
            _batch((bad_version, 1, 3)), current_version=3, max_staleness=2
        )


@pytest.mark.parametrize(
    "bad_versions", [None, torch.zeros(1, 5), torch.zeros(1, 4, dtype=torch.long)]
)
def test_mask_missing_or_malformed_versions_fails(bad_versions):
    data = _batch()
    data["versions"] = bad_versions
    with pytest.raises(ValueError, match="versions"):
        apply_staleness_mask(data, current_version=3, max_staleness=2)


@pytest.mark.parametrize("actor_kind", ["actor", "critic"])
@pytest.mark.parametrize("enabled", [False, True])
def test_update_uses_masked_weight_and_preserves_caller_data(actor_kind, enabled):
    engine = _engine()
    updater = (
        PPOActor(PPOActorConfig(backend="fsdp:d1", ppo_n_minibatches=1), engine)
        if actor_kind == "actor"
        else PPOCritic(PPOCriticConfig(backend="fsdp:d1", ppo_n_minibatches=1), engine)
    )
    data = _batch()
    original = copy.deepcopy(data)

    updater.ppo_update([data], max_token_staleness=2 if enabled else None)

    engine.train_batch.assert_called_once()
    call = engine.train_batch.call_args
    mb = call.args[0]
    assert int(call.kwargs["loss_weight_fn"](mb)) == (2 if enabled else 3)
    for key in original:
        torch.testing.assert_close(data[key], original[key], rtol=0, atol=0)
    if actor_kind == "actor":
        logp = torch.zeros_like(mb["logprobs"], requires_grad=True)
        loss = call.kwargs["loss_fn"](logp, torch.zeros_like(logp), mb)
        loss.backward()
        expected = (
            torch.tensor([[0, -0.5, -0.5, 0, 0]])
            if enabled
            else torch.tensor([[-1 / 3, -1 / 3, -1 / 3, 0, 0]])
        )
        torch.testing.assert_close(logp.grad, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("actor_kind", ["actor", "critic"])
@pytest.mark.parametrize("all_empty", [False, True])
def test_globally_empty_optimizer_minibatch_skips_train_batch(actor_kind, all_empty):
    engine = _engine()
    updater = (
        PPOActor(PPOActorConfig(backend="fsdp:d1", ppo_n_minibatches=2), engine)
        if actor_kind == "actor"
        else PPOCritic(PPOCriticConfig(backend="fsdp:d1", ppo_n_minibatches=2), engine)
    )
    batches = [_batch((0, 0, 0)), _batch((0, 0, 0) if all_empty else (3, 3, 3))]

    updater.ppo_update(batches, max_token_staleness=2)

    assert engine.train_batch.call_count == (0 if all_empty else 1)
    if not all_empty:
        call = engine.train_batch.call_args
        assert int(call.kwargs["loss_weight_fn"](call.args[0])) == 3


def test_mask_after_advantages_preserves_group_reward_baseline():
    from areal.api.cli_args import NormConfig

    actor = PPOActor(
        PPOActorConfig(
            backend="fsdp:d1",
            ppo_n_minibatches=1,
            reward_norm=NormConfig(mean_level="group", std_level=None, group_size=2),
            adv_norm=None,
            recompute_logprob=False,
            kl_ctl=0.0,
        ),
        _engine(),
    )
    data = {
        key: torch.cat([value, value.clone()], dim=0) for key, value in _batch().items()
    }
    data["loss_mask"] = torch.roll(data["loss_mask"], shifts=1, dims=-1)
    data["rewards"] = torch.tensor([1.0, 0.0])
    data["versions"][0, 1:4] = 0
    data["versions"][1, 1:4] = 3
    prepared = actor._compute_advantages(data)
    advantages = prepared["advantages"].clone()
    rewards = prepared["rewards"].clone()

    apply_staleness_mask(prepared, current_version=3, max_staleness=2)

    assert not prepared["loss_mask"][0].any()
    assert prepared["loss_mask"][1].count_nonzero() == 3
    torch.testing.assert_close(prepared["advantages"], advantages, rtol=0, atol=0)
    torch.testing.assert_close(prepared["rewards"], rewards, rtol=0, atol=0)
    assert (
        advantages[1, :3] < 0
    ).all()  # The stale positive sibling stays in the baseline.


@pytest.mark.parametrize("surrogate", ["ppo", "gspo", "sapo", "cispo"])
def test_excluded_extreme_probability_ratios_have_finite_zero_gradients(surrogate):
    from areal.trainer.ppo.actor import grpo_loss_fn

    inputs = _batch()
    apply_staleness_mask(inputs, current_version=3, max_staleness=2)
    inputs["logprobs"][0, 0] = -1000
    inputs["prox_logp"][0, 0] = -1000
    logp = torch.zeros(1, 5, requires_grad=True)
    loss = grpo_loss_fn(
        logp,
        torch.zeros_like(logp),
        inputs,
        eps_clip=0.2,
        eps_clip_higher=0.2,
        c_clip=None,
        importance_sampling_level="sequence" if surrogate == "gspo" else "token",
        use_sapo_loss=surrogate == "sapo",
        use_cispo_loss=surrogate == "cispo",
        sanitize_masked_tokens=True,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logp.grad).all()
    torch.testing.assert_close(
        logp.grad, torch.tensor([[0, -0.5, -0.5, 0, 0]]), rtol=1e-6, atol=1e-6
    )


def test_pure_distillation_uses_filtered_targets():
    from areal.api.cli_args import MOPDLossConfig

    engine = _engine()
    actor = PPOActor(PPOActorConfig(backend="fsdp:d1", ppo_n_minibatches=1), engine)
    actor.configure_mopd_loss(MOPDLossConfig(rl_coefficient=0))
    data = _batch()
    data["mopd_behavior_logprobs"] = torch.zeros(1, 5)
    data["mopd_teacher_logp_sum"] = -torch.ones(1, 5)
    data["mopd_teacher_weight_sum"] = torch.ones(1, 5)

    actor.ppo_update([data], max_token_staleness=2)

    call = engine.train_batch.call_args
    mb = call.args[0]
    assert call.kwargs["loss_weight_fn"](mb) == 2
    logp = torch.zeros(1, 5, requires_grad=True)
    call.kwargs["loss_fn"](logp, torch.zeros_like(logp), mb).backward()
    assert logp.grad[0, 0] == 0
    assert torch.isfinite(logp.grad).all()
    assert torch.count_nonzero(logp.grad) == 2


@pytest.mark.parametrize("controller_version", [1, 2])
def test_controller_transports_mask_threshold_without_materializing_trajectories(
    controller_version,
):
    from areal.infra.rpc.serialization import deserialize_value
    from areal.trainer.ppo.actor import PPOActorController, PPOActorControllerV2

    cls = PPOActorController if controller_version == 1 else PPOActorControllerV2
    controller = object.__new__(cls)
    data = [{"input_ids": torch.tensor([[1, 2]])}]
    if controller_version == 1:
        controller._custom_function_call = MagicMock()
        controller.ppo_update(data, max_token_staleness=2)
        controller._custom_function_call.assert_called_once_with(
            "ppo_update", data, max_token_staleness=2, rpc_meta={"broadcast": True}
        )
    else:
        controller._gateway_post = MagicMock()
        controller.ppo_update(data, max_token_staleness=2)
        route, payload = controller._gateway_post.call_args.args
        assert route == "/ppo/actor/update"
        assert deserialize_value(payload["kwargs"]) == {"max_token_staleness": 2}


def _distributed_empty_check(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        data = {"loss_mask": torch.tensor([[rank]], dtype=torch.bool)}
        assert has_global_trainable_tokens(data, dist.group.WORLD)
        data["loss_mask"].zero_()
        assert not has_global_trainable_tokens(data, dist.group.WORLD)
        data["loss_mask"].fill_(True)
        data[TRANSPORT_DUMMY_KEY] = True
        assert not has_global_trainable_tokens(data, dist.group.WORLD)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
def test_dp_empty_rank_participates_and_all_empty_ranks_skip(tmp_path):
    """Use real CPU collectives to verify identical rank decisions."""
    mp.spawn(
        _distributed_empty_check,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        join=True,
    )
