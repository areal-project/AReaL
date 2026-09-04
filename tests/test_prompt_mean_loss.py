# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from areal.api.cli_args import MicroBatchSpec, PPOActorConfig
from areal.trainer.ppo.actor import PPOActor, grpo_loss_fn
from areal.utils.data import split_padded_tensor_dict_into_mb_list
from areal.utils.functional.loss_aggregation import PolicyGradientReduction

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
    return PolicyGradientReduction(
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
    group_sizes: list[int] | None = None,
) -> torch.Tensor:
    return _reduction(mode).aggregate(
        loss,
        loss_mask,
        denominator_mask=denominator_mask,
        cu_seqlens=cu_seqlens,
        group_sizes=group_sizes,
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
    group_sizes = GROUP_SIZES if mode == "prompt_mean" else None

    actual = _aggregate(mode, group_sizes=group_sizes)

    torch.testing.assert_close(actual, torch.tensor(expected), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", ["seq_mean", "prompt_mean", "constant"])
def test_packed_reduction_matches_padded(mode):
    packed_loss = torch.cat([LOSS[0, :2], LOSS[1, :3], LOSS[2, :1]])
    packed_mask = torch.ones_like(packed_loss, dtype=torch.bool)
    cu_seqlens = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    group_sizes = GROUP_SIZES if mode == "prompt_mean" else None

    padded = _aggregate(mode, group_sizes=group_sizes)
    packed = _aggregate(
        mode,
        packed_loss,
        packed_mask,
        cu_seqlens=cu_seqlens,
        group_sizes=group_sizes,
    )

    torch.testing.assert_close(packed, padded, rtol=1e-5, atol=1e-6)


def test_token_mean_preserves_existing_dtype_and_reduction_path():
    loss = LOSS.to(torch.bfloat16)
    expected = torch.where(MASK, loss, 0).sum() / MASK.count_nonzero()

    actual = _aggregate("token_mean", loss)

    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["seq_mean", "prompt_mean"])
def test_denominator_mask_keeps_filtered_units_in_mean(mode):
    loss = torch.tensor([[2.0, 8.0], [6.0, 4.0]])
    numerator_mask = torch.tensor([[1, 0], [0, 0]], dtype=torch.bool)
    denominator_mask = torch.ones_like(numerator_mask)
    group_sizes = [1, 1] if mode == "prompt_mean" else None

    actual = _aggregate(
        mode,
        loss,
        numerator_mask,
        denominator_mask=denominator_mask,
        group_sizes=group_sizes,
    )

    torch.testing.assert_close(actual, torch.tensor(0.5), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["token_mean", "seq_mean", "prompt_mean", "constant"])
def test_callback_pair_is_invariant_to_unit_aligned_partitions(mode):
    reduction = _reduction(mode)
    full_data = {"loss_mask": MASK}
    if mode == "prompt_mean":
        full_data["group_sizes"] = GROUP_SIZES
    full = reduction.aggregate(
        LOSS,
        MASK,
        group_sizes=full_data.get("group_sizes"),
    )

    weighted_losses = []
    weights = []
    for seq_slice, group_sizes in ((slice(0, 2), [2]), (slice(2, 3), [1])):
        data = {"loss_mask": MASK[seq_slice]}
        if mode == "prompt_mean":
            data["group_sizes"] = group_sizes
        weight = reduction.normalizer_fn(data)
        local = reduction.aggregate(
            LOSS[seq_slice],
            MASK[seq_slice],
            group_sizes=data.get("group_sizes"),
        )
        weighted_losses.append(local * weight)
        weights.append(weight)

    combined = torch.stack(weighted_losses).sum() / torch.stack(weights).sum()
    torch.testing.assert_close(combined, full, rtol=1e-5, atol=1e-6)


def test_normalizer_counts_only_active_units():
    mask = torch.tensor([[1, 0], [0, 0], [1, 1]], dtype=torch.bool)

    assert _reduction("token_mean").normalizer_fn({"loss_mask": mask}).item() == 3
    assert _reduction("seq_mean").normalizer_fn({"loss_mask": mask}).item() == 2
    assert _reduction("constant").normalizer_fn({"loss_mask": mask}).item() == 2
    assert (
        _reduction("prompt_mean")
        .normalizer_fn({"loss_mask": mask, "group_sizes": [2, 1]})
        .item()
        == 2
    )


def test_prompt_mean_requires_explicit_group_sizes():
    reduction = _reduction("prompt_mean")

    with pytest.raises(ValueError, match="group_sizes are required"):
        reduction.aggregate(LOSS, MASK)
    with pytest.raises(ValueError, match="group_sizes are required"):
        reduction.normalizer_fn({"loss_mask": MASK})
    with pytest.raises(TypeError, match="sequence of ints"):
        reduction.aggregate(LOSS, MASK, group_sizes=torch.tensor(GROUP_SIZES))


@pytest.mark.parametrize("mode", ["seq_mean", "prompt_mean", "constant"])
def test_non_token_packed_reduction_requires_sequence_boundaries(mode):
    group_sizes = [1] if mode == "prompt_mean" else None

    with pytest.raises(ValueError, match="requires cu_seqlens"):
        _aggregate(
            mode,
            torch.ones(2),
            torch.ones(2, dtype=torch.bool),
            group_sizes=group_sizes,
        )


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


def test_atomic_prompt_groups_reject_oversize_token_cap():
    data = {
        "attention_mask": torch.ones(2, 4, dtype=torch.bool),
        "input_ids": torch.arange(8).view(2, 4),
        "loss_mask": torch.ones(2, 4, dtype=torch.bool),
        "group_sizes": [2],
    }

    with pytest.raises(RuntimeError, match="max_tokens_per_mb"):
        split_padded_tensor_dict_into_mb_list(
            data, MicroBatchSpec(n_mbs=1, max_tokens_per_mb=4)
        )


def test_atomic_prompt_groups_reject_more_microbatches_than_groups():
    data = {
        "attention_mask": torch.ones(2, 3, dtype=torch.bool),
        "input_ids": torch.arange(6).view(2, 3),
        "loss_mask": torch.ones(2, 3, dtype=torch.bool),
        "group_sizes": [1, 1],
    }

    with pytest.raises(RuntimeError, match="at least 4 groups"):
        split_padded_tensor_dict_into_mb_list(data, MicroBatchSpec(n_mbs=4))


def test_token_mean_split_allows_groups_that_exceed_token_cap():
    data = {
        "attention_mask": torch.ones(2, 4, dtype=torch.bool),
        "input_ids": torch.arange(8).view(2, 4),
        "loss_mask": torch.ones(2, 4, dtype=torch.bool),
    }

    mb_list = split_padded_tensor_dict_into_mb_list(
        data, MicroBatchSpec(n_mbs=1, max_tokens_per_mb=4, granularity=1)
    )

    assert len(mb_list.mbs) == 2
    assert all("group_sizes" not in mb for mb in mb_list.mbs)


def test_nested_split_preserves_prompt_group_sizes():
    data = {
        "attention_mask": torch.ones(4, 3, dtype=torch.bool),
        "input_ids": torch.arange(12).view(4, 3),
        "loss_mask": torch.ones(4, 3, dtype=torch.bool),
        "group_sizes": [2, 2],
    }

    ppo_mbs = split_padded_tensor_dict_into_mb_list(data, MicroBatchSpec(n_mbs=2))
    assert sorted(tuple(mb["group_sizes"]) for mb in ppo_mbs.mbs) == [(2,), (2,)]

    for mb in ppo_mbs.mbs:
        engine_mbs = split_padded_tensor_dict_into_mb_list(mb, MicroBatchSpec(n_mbs=1))
        assert len(engine_mbs.mbs) == 1
        nested = engine_mbs.mbs[0]
        assert nested["group_sizes"] == [2]
        assert nested["attention_mask"].shape[0] == 2


def test_prompt_mean_uses_trajectory_group_metadata():
    actor = object.__new__(PPOActor)
    actor.config = PPOActorConfig(loss_aggregation="prompt_mean")
    actor._ppo_update = MagicMock()
    data = [
        {"attention_mask": torch.ones(2, 2), "loss_mask": torch.ones(2, 2)},
        {"attention_mask": torch.ones(1, 2), "loss_mask": torch.ones(1, 2)},
    ]

    actor.ppo_update(data)

    batched = actor._ppo_update.call_args.args[0]
    assert batched["group_sizes"] == [2, 1]


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
