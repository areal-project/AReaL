# SPDX-License-Identifier: Apache-2.0

import math
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo.actor import _group_training_metrics, grpo_loss_fn
from areal.trainer.ppo.loss_reduction import (
    PG_TOKEN_WEIGHTS,
    PreparedLossStep,
    prepare_policy_gradient_batch,
)
from areal.utils.functional.loss_aggregation import (
    ConstantLength,
    PolicyGradientReduction,
    SequenceMean,
    TokenMean,
)

LOSS = torch.tensor([[1.0, 3.0, 0.0, 0.0], [2.0, 4.0, 6.0, 0.0], [10.0, 0.0, 0.0, 0.0]])
MASK = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)


def _prepare(data, mode, group_sizes):
    prepared = prepare_policy_gradient_batch(
        data,
        mode=mode,
        group_sizes=group_sizes,
        divisor=4.0 if mode == "constant" else None,
    )
    return prepared.for_steps([data])[0]


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("token_mean", 13 / 3),
        ("seq_mean", 16 / 3),
        ("prompt_mean", 33 / 5),
        ("constant", 13 / 6),
    ],
)
def test_policy_gradient_reduction_matches_definition(mode, expected):
    data = {"loss_mask": MASK}
    reduction = _prepare(data, mode, [2, 1]).bind(data)
    torch.testing.assert_close(
        reduction.aggregate(LOSS, MASK), torch.tensor(expected), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("mode", ["token_mean", "seq_mean", "prompt_mean", "constant"])
def test_packed_reduction_matches_padded(mode):
    data = {"loss_mask": MASK}
    step = _prepare(data, mode, [2, 1])
    packed = {
        "loss_mask": MASK[MASK],
        "cu_seqlens": torch.tensor([0, 2, 5, 6], dtype=torch.int32),
    }
    if PG_TOKEN_WEIGHTS in data:
        packed[PG_TOKEN_WEIGHTS] = data[PG_TOKEN_WEIGHTS][MASK]
    padded_loss = step.bind(data).aggregate(LOSS, MASK)
    packed_loss = step.bind(packed).aggregate(LOSS[MASK], packed["loss_mask"])
    torch.testing.assert_close(packed_loss, padded_loss, rtol=1e-5, atol=1e-6)


def test_token_mean_preserves_existing_dtype_and_reduction_path():
    loss = LOSS.to(torch.bfloat16)
    expected = torch.where(MASK, loss, 0).sum() / MASK.count_nonzero()
    actual = TokenMean(MASK).aggregate(loss, MASK)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_constant_length_preserves_default_dtype_promotion():
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        expected = torch.where(MASK, LOSS, 0).float().sum() / (
            MASK.sum(-1).count_nonzero() * math.pi
        )
        actual = ConstantLength(math.pi, MASK).aggregate(LOSS, MASK)
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("mode", ["seq_mean", "prompt_mean"])
def test_bound_denominator_keeps_filtered_units_in_mean(mode):
    loss = torch.tensor([[2.0, 8.0], [6.0, 4.0]])
    retained = torch.tensor([[1, 0], [0, 0]], dtype=torch.bool)
    data = {"loss_mask": torch.ones_like(retained)}
    reduction = _prepare(data, mode, [1, 1]).bind(data)
    torch.testing.assert_close(
        reduction.aggregate(loss, retained), torch.tensor(0.5), rtol=0, atol=0
    )


@pytest.mark.parametrize("mode", ["token_mean", "seq_mean", "prompt_mean", "constant"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("empty_batch", [False, True])
def test_callback_pair_preserves_loss_and_gradient_across_partitions(
    mode, packed, empty_batch
):
    loss = torch.tensor(
        [[2.0, 8.0, 1.0], [6.0, 4.0, 3.0], [9.0, 2.0, 7.0], [5.0, 3.0, 6.0]],
        requires_grad=True,
    )
    denominator = torch.tensor(
        [[1, 1, 0], [1, 1, 1], [0, 0, 0], [1, 0, 0]], dtype=torch.bool
    )
    numerator = torch.tensor(
        [[1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=torch.bool
    )
    data = {"loss_mask": denominator}
    if empty_batch and mode != "prompt_mean":
        denominator.zero_()
    step = _prepare(data, mode, [2, 1, 1])
    if empty_batch:
        # An entirely filtered fragment remains valid; G=0 is tested separately.
        numerator.zero_()

    def evaluate(rows):
        local = {key: value[rows] for key, value in data.items()}
        values, retained = loss[rows], numerator[rows]
        if packed:
            local = {key: value.flatten() for key, value in local.items()}
            local["cu_seqlens"] = torch.arange(
                0, values.numel() + 1, 3, dtype=torch.int32
            )
            values, retained = values.flatten(), retained.flatten()
        reduction = step.bind(local)
        return reduction.aggregate(values, retained), reduction.normalizer()

    full, full_weight = evaluate(slice(None))
    partitions = [evaluate(slice(0, 1)), evaluate(slice(1, 3)), evaluate(slice(3, 4))]
    total_weight = sum(weight for _, weight in partitions)
    combined = sum(
        value * weight for value, weight in partitions
    ) / total_weight.clamp_min(1)
    full_gradient = torch.autograd.grad(full, loss, retain_graph=True)[0]
    split_gradient = torch.autograd.grad(combined, loss)[0]
    torch.testing.assert_close(total_weight, full_weight, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(combined, full, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(split_gradient, full_gradient, rtol=1e-6, atol=1e-6)
    if empty_batch:
        torch.testing.assert_close(full, torch.zeros_like(full), rtol=0, atol=0)
        torch.testing.assert_close(
            full_gradient, torch.zeros_like(loss), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "mode,divisor",
    [
        ("bogus", None),
        ("constant", None),
        ("constant", 0),
        ("constant", -1),
        ("constant", float("inf")),
        ("constant", float("nan")),
        ("token_mean", 4),
        ("seq_mean", 4),
        ("prompt_mean", 4),
    ],
)
def test_preparation_rejects_invalid_configuration(mode, divisor):
    with pytest.raises(ValueError):
        prepare_policy_gradient_batch(
            {"loss_mask": MASK},
            mode=mode,
            divisor=divisor,
            group_sizes=[2, 1],
        )


@pytest.mark.parametrize("divisor", [0, -1, float("inf"), float("nan")])
def test_constant_length_rejects_invalid_divisor(divisor):
    with pytest.raises(ValueError, match="positive finite"):
        ConstantLength(divisor, MASK)


@pytest.mark.parametrize("packed", [False, True])
def test_loss_accepts_mode_free_bound_reduction(packed):
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    if packed:
        mask = mask.flatten()

    class DoubleTokenMean:
        def __bool__(self):
            return False

        def normalizer(self):
            return TokenMean(mask).normalizer()

        def aggregate(self, loss, numerator_mask):
            return 2 * TokenMean(mask).aggregate(loss, numerator_mask)

    reduction: PolicyGradientReduction = DoubleTokenMean()
    step = PreparedLossStep(lambda _: reduction)
    logprobs = torch.zeros_like(mask, dtype=torch.float32, requires_grad=True)
    data = {
        "logprobs": torch.zeros_like(logprobs),
        "advantages": torch.ones_like(logprobs),
        "loss_mask": mask,
        "prox_logp": torch.zeros_like(logprobs),
    }
    with patch("areal.trainer.ppo.actor.stats_tracker"):
        loss = grpo_loss_fn(
            logprobs=logprobs,
            entropy=torch.zeros_like(logprobs),
            input_data=data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            loss_step=step,
        )
    torch.testing.assert_close(step.loss_weight(data), torch.tensor(3), rtol=0, atol=0)
    torch.testing.assert_close(loss, torch.tensor(-2.0), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(loss, logprobs)[0],
        -2 * mask.float() / 3,
        rtol=1e-6,
        atol=1e-6,
    )


def test_normalizer_counts_original_active_units():
    mask = torch.tensor([[1, 0], [0, 0], [1, 1]], dtype=torch.bool)
    assert TokenMean(mask).normalizer().item() == 3
    assert SequenceMean(mask).normalizer().item() == 2
    assert ConstantLength(4.0, mask).normalizer().item() == 2
    data = {"loss_mask": mask}
    assert _prepare(data, "prompt_mean", [2, 1]).loss_weight(data).item() == 1


def test_prompt_preparation_rejects_globally_empty_objective():
    data = {"loss_mask": torch.zeros_like(MASK)}
    prepared = prepare_policy_gradient_batch(
        data, mode="prompt_mean", group_sizes=[2, 1]
    )
    with pytest.raises(ValueError, match="active prompt groups"):
        prepared.for_steps([data])


@pytest.mark.parametrize("mode", ["seq_mean", "constant"])
def test_sequence_reduction_requires_packed_sequence_boundaries(mode):
    with pytest.raises(ValueError, match="requires cu_seqlens"):
        _prepare({"loss_mask": torch.ones(2, dtype=torch.bool)}, mode, None).bind(
            {"loss_mask": torch.ones(2, dtype=torch.bool)}
        ).aggregate(torch.ones(2), torch.ones(2, dtype=torch.bool))


@pytest.mark.parametrize(
    "mode,weights",
    [
        ("token_mean", [5.0, 1.0]),
        ("seq_mean", [2.0, 1.0]),
        ("prompt_mean", [1.0, 1.0]),
        ("constant", [2.0, 1.0]),
    ],
)
def test_group_weight_metrics_follow_loss_aggregation(mode, weights):
    starts, sizes, actual = _group_training_metrics(MASK, [2, 1], [1, 1], mode)
    torch.testing.assert_close(actual[starts], torch.tensor(weights), rtol=0, atol=0)
    torch.testing.assert_close(sizes[starts], torch.ones(2), rtol=0, atol=0)


def test_m2_mask_narrows_numerator_but_preserves_original_denominator():
    input_data = {
        "input_ids": torch.tensor([[11, 12]]),
        "logprobs": torch.zeros(1, 2),
        "advantages": torch.ones(1, 2),
        "loss_mask": torch.ones(1, 2, dtype=torch.bool),
        "prox_logp": torch.zeros(1, 2),
    }
    filtered_mask = torch.tensor([[1, 0]], dtype=torch.bool)

    with (
        patch(
            "areal.trainer.ppo.actor._apply_m2po_masking",
            return_value=filtered_mask,
        ),
        patch("areal.trainer.ppo.actor.stats_tracker"),
    ):
        loss = grpo_loss_fn(
            logprobs=torch.zeros(1, 2),
            entropy=torch.zeros(1, 2),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            m2_threshold=0.1,
        )

    torch.testing.assert_close(loss, torch.tensor(-0.5), rtol=0, atol=0)


def test_loss_aggregation_config_is_omegaconf_compatible():
    config = OmegaConf.structured(PPOActorConfig)

    assert config.loss_aggregation == "token_mean"


def test_loss_aggregation_config_validation():
    with pytest.raises(ValueError, match="loss_aggregation must be"):
        PPOActorConfig(loss_aggregation="bogus")
    with pytest.raises(ValueError, match="loss_aggregation_divisor"):
        PPOActorConfig(loss_aggregation="constant")
    with pytest.raises(ValueError, match="loss_aggregation_divisor"):
        PPOActorConfig(loss_aggregation="constant", loss_aggregation_divisor=0)
    with pytest.raises(ValueError, match="only used"):
        PPOActorConfig(loss_aggregation="seq_mean", loss_aggregation_divisor=10)
    with pytest.raises(ValueError, match="tree"):
        PPOActorConfig(loss_aggregation="seq_mean", enable_tree_training=True)

    PPOActorConfig(loss_aggregation="constant", loss_aggregation_divisor=10)
