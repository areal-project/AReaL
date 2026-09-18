# SPDX-License-Identifier: Apache-2.0

import math
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from areal.api.cli_args import MicroBatchSpec, PPOActorConfig
from areal.trainer.ppo.actor import (
    PPOActor,
    _group_training_metrics,
    _policy_gradient_loss_weight,
    grpo_loss_fn,
)
from areal.utils.data import (
    RolloutGroup,
    make_transport_microbatch,
    split_padded_tensor_dict_into_mb_list,
    split_training_batch_into_microbatches,
)
from areal.utils.functional.loss_aggregation import (
    ConstantLength,
    PolicyGradientReduction,
    TokenMean,
    make_policy_gradient_reduction,
    prepare_prompt_token_weights,
)

LOSS = torch.tensor(
    [
        [1.0, 3.0, 0.0, 0.0],
        [2.0, 4.0, 6.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],
    ]
)
MASK = torch.tensor(
    [
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 0, 0, 0],
    ],
    dtype=torch.bool,
)
GROUP_SIZES = [2, 1]


def _reduction(mode: str) -> PolicyGradientReduction:
    return make_policy_gradient_reduction(
        mode=mode,
        divisor=4.0 if mode == "constant" else None,
    )


def _aggregate(
    mode: str,
    loss: torch.Tensor = LOSS,
    loss_mask: torch.Tensor = MASK,
    *,
    denominator_mask: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    prompt_token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return _reduction(mode).aggregate(
        loss,
        loss_mask,
        denominator_mask=denominator_mask,
        cu_seqlens=cu_seqlens,
        prompt_token_weights=prompt_token_weights,
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("token_mean", 13 / 3),
        ("seq_mean", 16 / 3),
        ("prompt_mean", 33 / 5),
        ("constant", 13 / 6),
    ],
)
def test_policy_gradient_reduction_matches_definition(mode, expected):
    weights = prepare_prompt_token_weights(MASK, GROUP_SIZES)

    actual = _aggregate(mode, prompt_token_weights=weights)

    torch.testing.assert_close(actual, torch.tensor(expected), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", ["seq_mean", "prompt_mean", "constant"])
def test_packed_reduction_matches_padded(mode):
    packed_loss = torch.cat([LOSS[0, :2], LOSS[1, :3], LOSS[2, :1]])
    packed_mask = torch.ones_like(packed_loss, dtype=torch.bool)
    cu_seqlens = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    prompt_token_weights = prepare_prompt_token_weights(MASK, GROUP_SIZES)

    padded = _aggregate(mode, prompt_token_weights=prompt_token_weights)
    packed = _aggregate(
        mode,
        packed_loss,
        packed_mask,
        cu_seqlens=cu_seqlens,
        prompt_token_weights=prompt_token_weights[MASK],
    )

    torch.testing.assert_close(packed, padded, rtol=1e-5, atol=1e-6)


def test_token_mean_preserves_existing_dtype_and_reduction_path():
    loss = LOSS.to(torch.bfloat16)
    expected = torch.where(MASK, loss, 0).sum() / MASK.count_nonzero()

    actual = _aggregate("token_mean", loss)

    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_constant_length_preserves_default_dtype_promotion():
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        expected = torch.where(MASK, LOSS, 0).float().sum() / (
            MASK.sum(-1).count_nonzero() * math.pi
        )

        actual = ConstantLength(math.pi).aggregate(LOSS, MASK)

        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("mode", ["seq_mean", "prompt_mean"])
def test_denominator_mask_keeps_filtered_units_in_mean(mode):
    loss = torch.tensor([[2.0, 8.0], [6.0, 4.0]])
    numerator_mask = torch.tensor([[1, 0], [0, 0]], dtype=torch.bool)
    denominator_mask = torch.ones_like(numerator_mask)
    prompt_token_weights = prepare_prompt_token_weights(denominator_mask, [1, 1])

    actual = _aggregate(
        mode,
        loss,
        numerator_mask,
        denominator_mask=denominator_mask,
        prompt_token_weights=prompt_token_weights,
    )

    torch.testing.assert_close(actual, torch.tensor(0.5), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["token_mean", "seq_mean", "prompt_mean", "constant"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("empty_batch", [False, True])
def test_callback_pair_preserves_loss_and_gradient_across_partitions(
    mode, packed, empty_batch
):
    """Engine weighting preserves original denominators despite rejected tokens."""
    reduction = _reduction(mode)
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
    if empty_batch:
        denominator.zero_()
        numerator.zero_()

    token_weights = prepare_prompt_token_weights(denominator, [2, 1, 1])

    def evaluate(rows):
        local_loss = loss[rows]
        local_numerator = numerator[rows]
        local_denominator = denominator[rows]
        local_weights = token_weights[rows]
        cu_seqlens = None
        if packed:
            # Keep every physical token; loss masks carry the training boundary.
            cu_seqlens = torch.arange(0, local_loss.numel() + 1, 3, dtype=torch.int32)
            local_loss = local_loss.flatten()
            local_numerator = local_numerator.flatten()
            local_denominator = local_denominator.flatten()
            local_weights = local_weights.flatten()
        metadata = dict(cu_seqlens=cu_seqlens, prompt_token_weights=local_weights)
        weight = reduction.normalizer(local_denominator, **metadata)
        result = reduction.aggregate(
            local_loss,
            local_numerator,
            denominator_mask=local_denominator,
            **metadata,
        )
        return result, weight

    full, full_weight = evaluate(slice(None))
    partitions = [
        evaluate(slice(0, 1)),
        evaluate(slice(1, 3)),
        evaluate(slice(3, 4)),
    ]
    split_weight = sum(weight for _, weight in partitions)
    combined = sum(
        value * weight for value, weight in partitions
    ) / split_weight.clamp_min(1)
    full_gradient = torch.autograd.grad(full, loss, retain_graph=True)[0]
    split_gradient = torch.autograd.grad(combined, loss)[0]

    # Prompt fragments sum fractional float32 weights in different orders.
    torch.testing.assert_close(split_weight, full_weight, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(combined, full, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(split_gradient, full_gradient, rtol=1e-6, atol=1e-6)
    if empty_batch:
        torch.testing.assert_close(full, torch.zeros_like(full), rtol=0, atol=0)
        torch.testing.assert_close(
            full_gradient, torch.zeros_like(loss), rtol=0, atol=0
        )


@pytest.mark.parametrize("packed", [False, True])
def test_prompt_fragments_preserve_fractional_weights_and_gradient_oracle(packed):
    loss = torch.tensor(
        [[2.0, 8.0, 1.0], [6.0, 4.0, 3.0], [9.0, 2.0, 7.0], [5.0, 3.0, 6.0]],
        requires_grad=True,
    )
    original = torch.tensor(
        [[1, 1, 0], [1, 1, 1], [0, 0, 0], [1, 0, 0]], dtype=torch.bool
    )
    retained = torch.tensor(
        [[1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=torch.bool
    )
    weights = prepare_prompt_token_weights(original, [2, 1, 1])
    expected_weights = torch.tensor(
        [[0.2, 0.2, 0.0], [0.2, 0.2, 0.2], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
    reduction = _reduction("prompt_mean")

    def evaluate(rows):
        values, mask, token_weights = loss[rows], retained[rows], weights[rows]
        if packed:
            values, mask, token_weights = (
                value.flatten() for value in (values, mask, token_weights)
            )
        # The original weights own normalization even when a fragment loses
        # every numerator token. Packed fragments need no group reconstruction.
        value = reduction.aggregate(values, mask, prompt_token_weights=token_weights)
        weight = reduction.normalizer(mask, prompt_token_weights=token_weights)
        return value, weight

    full, full_weight = evaluate([0, 1, 2, 3])
    mixed, mixed_weight = evaluate([0, 3])
    filtered, filtered_weight = evaluate([1, 2])
    combined = (mixed * mixed_weight + filtered * filtered_weight) / full_weight
    expected_loss = (loss[0, 0] / 5 + loss[3, 0]) / 2
    expected_gradient = torch.zeros_like(loss)
    expected_gradient[0, 0] = 0.1
    expected_gradient[3, 0] = 0.5

    torch.testing.assert_close(filtered_weight, torch.tensor(0.6), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(filtered, torch.tensor(0.0), rtol=0, atol=0)
    torch.testing.assert_close(full, expected_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(combined, expected_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        torch.autograd.grad(combined, loss)[0], expected_gradient, rtol=1e-6, atol=1e-6
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
def test_factory_rejects_invalid_configuration(mode, divisor):
    with pytest.raises(ValueError):
        make_policy_gradient_reduction(mode, divisor=divisor)


@pytest.mark.parametrize("divisor", [0, -1, float("inf"), float("nan")])
def test_constant_length_rejects_invalid_divisor(divisor):
    with pytest.raises(ValueError, match="positive finite"):
        ConstantLength(divisor)


@pytest.mark.parametrize("packed", [False, True])
def test_loss_and_actor_adapter_accept_mode_free_reduction(packed):
    """Consumers honor a structural implementation without a mode discriminator."""
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32) if packed else None
    if packed:
        mask = mask.flatten()
    weights = torch.ones_like(mask, dtype=torch.float32)

    class DoubleTokenMean:
        def __bool__(self):
            return False

        def normalizer(self, loss_mask, *, cu_seqlens=None, prompt_token_weights=None):
            assert loss_mask is mask
            assert cu_seqlens is data.get("cu_seqlens")
            assert prompt_token_weights is weights
            return TokenMean().normalizer(loss_mask)

        def aggregate(
            self,
            loss,
            loss_mask,
            *,
            denominator_mask=None,
            cu_seqlens=None,
            prompt_token_weights=None,
        ):
            assert cu_seqlens is data.get("cu_seqlens")
            assert prompt_token_weights is weights
            return 2 * TokenMean().aggregate(
                loss, loss_mask, denominator_mask=denominator_mask
            )

    reduction: PolicyGradientReduction = DoubleTokenMean()
    logprobs = torch.zeros_like(mask, dtype=torch.float32, requires_grad=True)
    data = {
        "logprobs": torch.zeros_like(logprobs),
        "advantages": torch.ones_like(logprobs),
        "loss_mask": mask,
        "prox_logp": torch.zeros_like(logprobs),
        "prompt_token_weights": weights,
    }
    if packed:
        data["cu_seqlens"] = cu_seqlens

    weight = _policy_gradient_loss_weight(data, reduction=reduction)
    with patch("areal.trainer.ppo.actor.stats_tracker"):
        loss = grpo_loss_fn(
            logprobs=logprobs,
            entropy=torch.zeros_like(logprobs),
            input_data=data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            pg_reduction=reduction,
        )
    gradient = torch.autograd.grad(loss, logprobs)[0]

    torch.testing.assert_close(weight, torch.tensor(3), rtol=0, atol=0)
    torch.testing.assert_close(loss, torch.tensor(-2.0), rtol=0, atol=0)
    torch.testing.assert_close(gradient, -2 * mask.float() / 3, rtol=1e-6, atol=1e-6)


def test_normalizer_counts_only_active_units():
    mask = torch.tensor([[1, 0], [0, 0], [1, 1]], dtype=torch.bool)

    assert _reduction("token_mean").normalizer(mask).item() == 3
    assert _reduction("seq_mean").normalizer(mask).item() == 2
    assert _reduction("constant").normalizer(mask).item() == 2
    weights = prepare_prompt_token_weights(mask, [2, 1])
    assert (
        _reduction("prompt_mean").normalizer(mask, prompt_token_weights=weights).item()
        == 2
    )


def test_prompt_mean_requires_precomputed_matching_token_weights():
    reduction = _reduction("prompt_mean")

    with pytest.raises(ValueError, match="prompt_token_weights are required"):
        reduction.aggregate(LOSS, MASK)
    with pytest.raises(ValueError, match="prompt_token_weights are required"):
        reduction.normalizer(MASK)
    with pytest.raises(ValueError, match="shape must match"):
        reduction.aggregate(LOSS, MASK, prompt_token_weights=torch.ones(3))


def test_prepare_prompt_weights_requires_padded_physical_groups():
    with pytest.raises(TypeError, match="sequence of ints"):
        prepare_prompt_token_weights(MASK, torch.tensor(GROUP_SIZES))
    with pytest.raises(ValueError, match="2D loss_mask"):
        prepare_prompt_token_weights(MASK.flatten(), GROUP_SIZES)
    with pytest.raises(ValueError, match="sequence count"):
        prepare_prompt_token_weights(MASK, [2])


@pytest.mark.parametrize("mode", ["seq_mean", "constant"])
def test_sequence_reduction_requires_packed_sequence_boundaries(mode):
    with pytest.raises(ValueError, match="requires cu_seqlens"):
        _aggregate(mode, torch.ones(2), torch.ones(2, dtype=torch.bool))


def test_split_padded_batch_keeps_ragged_prompt_groups_atomic():
    data = {
        "attention_mask": torch.tensor(
            [[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool
        ),
        "input_ids": torch.arange(9).view(3, 3),
        "loss_mask": torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        "group_sizes": GROUP_SIZES,
    }

    mb_list = split_padded_tensor_dict_into_mb_list(
        data, MicroBatchSpec(n_mbs=2, granularity=1)
    )

    actual_group_sizes = [tuple(mb["group_sizes"]) for mb in mb_list.mbs]
    assert sorted(actual_group_sizes) == [(1,), (2,)]
    assert sorted(mb["attention_mask"].shape[0] for mb in mb_list.mbs) == [1, 2]


def test_nested_split_keeps_groups_in_optimizer_step_and_splits_inner_fragments():
    mask = torch.ones(4, 3, dtype=torch.bool)
    data = {
        "attention_mask": mask,
        "input_ids": torch.arange(12).view(4, 3),
        "loss_mask": mask,
        "group_sizes": [2, 2],
        "prompt_token_weights": prepare_prompt_token_weights(mask, [2, 2]),
    }

    ppo_mbs = split_training_batch_into_microbatches(data, n_mbs=2)
    assert sorted(tuple(mb["group_sizes"]) for mb in ppo_mbs) == [(2,), (2,)]

    for mb in ppo_mbs:
        # Actor removes scheduling metadata before handing data to the engine.
        mb.pop("group_sizes")
        engine_mbs = split_padded_tensor_dict_into_mb_list(
            mb, MicroBatchSpec(n_mbs=1, max_tokens_per_mb=3)
        )
        assert len(engine_mbs.mbs) == 2
        weights = []
        for fragment in engine_mbs.mbs:
            assert "group_sizes" not in fragment
            assert fragment["attention_mask"].shape[0] == 1
            weights.append(fragment["prompt_token_weights"].sum())
        torch.testing.assert_close(
            sum(weights), torch.tensor(1.0), rtol=1e-6, atol=1e-6
        )

    assert len(split_training_batch_into_microbatches(data, n_mbs=3)) == 2
    dummy = make_transport_microbatch(data)
    assert "group_sizes" not in dummy
    assert torch.count_nonzero(dummy["prompt_token_weights"]) == 0


def test_prompt_mean_uses_trajectory_group_metadata():
    actor = object.__new__(PPOActor)
    actor.config = PPOActorConfig(loss_aggregation="prompt_mean")
    actor._ppo_update = MagicMock()
    data = [
        {
            "attention_mask": torch.ones(2, 2),
            "loss_mask": torch.ones(2, 2),
            "rollout_group": RolloutGroup((2,)),
        },
        {"attention_mask": torch.ones(1, 2), "loss_mask": torch.ones(1, 2)},
    ]

    actor.ppo_update(data)

    batched = actor._ppo_update.call_args.args[0]
    assert batched["group_sizes"] == [2, 1]
    torch.testing.assert_close(
        batched["prompt_token_weights"],
        torch.tensor([[0.25, 0.25], [0.25, 0.25], [0.5, 0.5]]),
        rtol=0,
        atol=0,
    )
    meta = actor._ppo_update.call_args.args[1]
    assert meta.logical_group_sizes == [1, 1]


def test_transport_padding_accepts_explicitly_absent_group_sizes():
    data = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2),
        "group_sizes": None,
    }
    result = split_padded_tensor_dict_into_mb_list(
        data, MicroBatchSpec(n_mbs=2), allow_transport_padding=True, sync_mbs=False
    )
    assert result.transport_dummy_count == 1
    assert len(result) == 2


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
