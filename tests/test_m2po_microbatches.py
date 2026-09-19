# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from areal.api.cli_args import (
    MicroBatchSpec,
    MOPDLossConfig,
    PPOActorConfig,
    RejectionSamplingConfig,
)
from areal.engine.core import compute_microbatch_loss_weight
from areal.trainer.ppo.actor import (
    PPOActor,
    _apply_m2po_masking,
    _prepare_m2po_minibatch,
    _validate_m2po_batch,
    grpo_loss_fn,
)
from areal.utils.data import (
    MicroBatchItem,
    make_transport_microbatch,
    pack_tensor_dict,
    pad_packed_tensor_dict,
    split_padded_tensor_dict_into_mb_list,
)


def _batch():
    prox = torch.tensor(
        [[3.0, 3.0, 3.0], [1.1, 1.3, 0.0], [0.1, 0.3, 0.4], [0.5, 0.0, 0.0]],
        dtype=torch.float64,
    )
    mask = torch.arange(3)[None, :] < torch.tensor([3, 2, 3, 1])[:, None]
    return {
        "input_ids": torch.arange(12).reshape(4, 3),
        "attention_mask": mask,
        "loss_mask": mask.clone(),
        "logprobs": torch.zeros_like(prox),
        "prox_logp": prox,
        "advantages": torch.tensor([1.0, -0.5, 0.7, -1.0])[:, None].expand_as(prox),
        "features": torch.linspace(-0.7, 0.9, 12, dtype=torch.float64).reshape(4, 3),
    }


def _loss_and_gradient(data, sizes, mode, *, prepared):
    data = dict(data)
    kwargs = dict(eps_clip=0.2, eps_clip_higher=None, c_clip=None, m2_threshold=0.35)
    if "rs" in mode:
        kwargs["rejection_sampling"] = RejectionSamplingConfig(upper=1.5, action="mask")
    if "mopd" in mode:
        kwargs["mopd_loss_config"] = MOPDLossConfig(
            rl_coefficient=0.0 if mode.startswith("pure") else 0.7,
            distillation_coefficient=0.3,
        )
        data["mopd_behavior_logprobs"] = data["logprobs"]
        data["mopd_teacher_logp_sum"] = torch.full_like(data["logprobs"], -0.8)
        data["mopd_teacher_weight_sum"] = torch.ones_like(data["logprobs"])
    elif mode == "single_teacher":
        data["teacher_logp"] = torch.full_like(data["logprobs"], -0.8)
    if prepared:
        _prepare_m2po_minibatch(data, 0.35)
        total_weight = data["m2_loss_mask"].count_nonzero()
    weight = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    total = weight * 0.0
    start = 0
    for size in sizes:
        mb = {key: value[start : start + size] for key, value in data.items()}
        start += size
        logp = mb["prox_logp"] + weight * mb["features"]
        if prepared:
            count = compute_microbatch_loss_weight(
                mb, lambda x: x["m2_loss_mask"].count_nonzero()
            )
            if count == 0:
                # Match the engine's differentiable zero-weight path.
                total = total + logp.sum() * 0.0
                continue
        with patch("areal.trainer.ppo.actor.stats_tracker", MagicMock()):
            loss = grpo_loss_fn(logp, torch.zeros_like(logp), mb, **kwargs)
        # Use float64 weights for this algebraic oracle; engine accumulation
        # in model precision has the usual floating-point rounding error.
        total = total + loss * (
            count.to(logp.dtype) / total_weight if prepared else 1.0
        )
    return total.detach(), torch.autograd.grad(total, weight)[0]


@pytest.mark.parametrize("sizes", [(4,), (1, 1, 1, 1), (1, 3), (3, 1), (2, 2)])
@pytest.mark.parametrize(
    "mode",
    [
        "ppo",
        "rs",
        "pure_mopd",
        "pure_mopd_rs",
        "joint_mopd",
        "joint_mopd_rs",
        "single_teacher",
    ],
)
def test_m2po_accumulation_matches_native_unsplit_loss_and_gradient(sizes, mode):
    data = _batch()
    original = data["loss_mask"].clone()
    expected_loss, expected_grad = _loss_and_gradient(data, (4,), mode, prepared=False)
    loss, grad = _loss_and_gradient(data, sizes, mode, prepared=True)
    torch.testing.assert_close(loss, expected_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad, expected_grad, rtol=1e-12, atol=1e-12)
    assert torch.equal(original, data["loss_mask"])
    _prepare_m2po_minibatch(data, 0.35)
    assert not data["m2_loss_mask"][0].any()  # One complete microbatch is rejected.


def test_m2po_mask_survives_split_pack_and_padding():
    data = _batch()
    original = data["loss_mask"].clone()
    _prepare_m2po_minibatch(data, 0.35)
    expected = dict(
        zip(
            data["input_ids"].flatten().tolist(),
            data["m2_loss_mask"].flatten().tolist(),
            strict=True,
        )
    )
    mb_list = split_padded_tensor_dict_into_mb_list(
        data, MicroBatchSpec(n_mbs=3), sync_mbs=False
    )
    seen = []
    for mb in mb_list.mbs:
        packed = pack_tensor_dict(mb)
        token_ids = packed["input_ids"].tolist()
        assert packed["m2_loss_mask"].tolist() == [
            expected[token] for token in token_ids
        ]
        padded, _, _, _ = pad_packed_tensor_dict(packed, pad_to_length=16)
        assert torch.equal(
            padded["m2_loss_mask"][: len(token_ids)], packed["m2_loss_mask"]
        )
        assert not padded["m2_loss_mask"][len(token_ids) :].any()
        seen.extend(token_ids)
    assert sorted(seen) == sorted(data["input_ids"][data["attention_mask"]].tolist())
    assert torch.equal(data["loss_mask"], original)


def test_m2po_single_token_and_transport_dummy():
    data = {key: value[2:3, :1] for key, value in _batch().items()}
    _prepare_m2po_minibatch(data, 0.35)
    assert data["m2_loss_mask"].shape == (1, 1)
    assert data["m2_loss_mask"].all()
    dummy = make_transport_microbatch(data)

    def unexpected_callback(_):
        raise AssertionError("transport dummies must bypass the loss callback")

    assert compute_microbatch_loss_weight(dummy, unexpected_callback) == 0
    assert not dummy["m2_loss_mask"].any()


def _actor(threshold=0.35, method="recompute"):
    actor = object.__new__(PPOActor)
    actor.config = PPOActorConfig(ppo_n_minibatches=2, prox_logp_method=method)
    actor.m2_threshold = threshold
    actor._mopd_loss_config = None
    actor.engine = SimpleNamespace(
        data_parallel_group=None, train=lambda: None, get_version=lambda: 0, calls=[]
    )

    def train_batch(mb, loss_fn, loss_weight_fn):
        actor.engine.calls.append((dict(mb), loss_weight_fn(mb)))
        return {}

    actor.engine.train_batch = train_batch
    return actor


@pytest.mark.parametrize("enabled", [False, True])
def test_actor_selects_each_optimizer_minibatch_before_engine_split(enabled):
    actor = _actor(threshold=0.35 if enabled else None)
    data = _batch()
    data.update(
        rewards=torch.ones(4),
        kl_rewards=torch.zeros(4, 3),
        tot_rewards=torch.zeros(4, 3),
    )
    with patch("areal.trainer.ppo.actor.stats_tracker", MagicMock()):
        actor._ppo_update(data)
    assert len(actor.engine.calls) == 2
    for mb, weight in actor.engine.calls:
        if enabled:
            expected = _apply_m2po_masking(
                mb["logprobs"], mb["prox_logp"], mb["loss_mask"].bool(), 0.35
            )
            assert torch.equal(mb["m2_loss_mask"], expected)
            assert weight == expected.count_nonzero()
        else:
            assert "m2_loss_mask" not in mb
            assert weight == mb["loss_mask"].count_nonzero()


@pytest.mark.parametrize("method", ["loglinear", "reuse_train_logp"])
def test_dynamic_proximal_modes_fail_before_training(method):
    actor = _actor(method=method)
    with pytest.raises(ValueError, match="cached proximal"):
        actor._ppo_update(_batch())
    assert not actor.engine.calls


def test_missing_cache_fails_before_training():
    actor = _actor()
    data = _batch()
    del data["prox_logp"]
    with pytest.raises(ValueError, match="requires prox_logp"):
        actor._ppo_update(data)
    assert not actor.engine.calls


def test_m2po_rejects_multiple_dp_ranks_and_misaligned_cache():
    with pytest.raises(ValueError, match="data parallel size 1"):
        _validate_m2po_batch(_batch(), "recompute", 2)
    data = _batch()
    data["prox_logp"] = data["prox_logp"][:, :2]
    with pytest.raises(ValueError, match="matching shape and device"):
        _validate_m2po_batch(data, "recompute", 1)


def test_fsdp_keeps_m2po_mask_in_loss_context_only():
    from areal.engine.fsdp_engine import FSDPEngine

    data = _batch()
    _prepare_m2po_minibatch(data, 0.35)
    packed = pack_tensor_dict(data)
    padded, padding_length, old_cu_seqlens, _ = pad_packed_tensor_dict(
        packed, pad_to_length=16
    )
    item = MicroBatchItem(packed, padded, padding_length, old_cu_seqlens)
    engine = object.__new__(FSDPEngine)
    engine.parallel_helper = SimpleNamespace(sp_size=1)
    inputs, context = engine._prepare_mb_inputs(item)
    assert "m2_loss_mask" not in inputs
    assert context.mb_input is packed
    assert torch.equal(context.mb_input["m2_loss_mask"], packed["m2_loss_mask"])
