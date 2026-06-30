# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    InferenceEngineConfig,
    MicroBatchSpec,
    PPOActorConfig,
)
from areal.trainer.ppo.actor import _make_loss_weight_fn
from areal.utils.functional import aggregate_pg_loss

PG = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
MASK = torch.tensor(
    [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
)
PROMPT_MEAN = 2.0
TOKEN_MEAN = 2.2
CONSTANT_DIVISOR = 10.0
CONSTANT_MEAN = 0.55

SEQ_PG = torch.tensor([[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
SEQ_MASK = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])


def test_token_mean_is_global_token_average():
    loss = aggregate_pg_loss(PG, MASK, loss_aggregation="token_mean")
    torch.testing.assert_close(loss, torch.tensor(TOKEN_MEAN))


def test_seq_mean_weights_each_sequence_equally():
    loss = aggregate_pg_loss(SEQ_PG, SEQ_MASK, loss_aggregation="seq_mean")
    torch.testing.assert_close(loss, torch.tensor(1.5))
    token = aggregate_pg_loss(SEQ_PG, SEQ_MASK, loss_aggregation="token_mean")
    torch.testing.assert_close(token, torch.tensor(8.0 / 6.0))
    assert not torch.allclose(loss, token)


def test_seq_mean_packed_matches_padded():
    pg = torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    mask = torch.ones(6)
    cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int32)
    loss = aggregate_pg_loss(
        pg, mask, loss_aggregation="seq_mean", cu_seqlens=cu_seqlens
    )
    torch.testing.assert_close(loss, torch.tensor(1.5))


def test_prompt_mean_weights_each_group_equally_2d():
    loss = aggregate_pg_loss(PG, MASK, loss_aggregation="prompt_mean", group_size=2)
    torch.testing.assert_close(loss, torch.tensor(PROMPT_MEAN))
    assert not torch.allclose(loss, torch.tensor(TOKEN_MEAN))


def test_prompt_mean_packed_matches_padded():
    pg = PG.reshape(-1)
    mask = MASK.reshape(-1)
    cu_seqlens = torch.tensor([0, 3, 6, 9, 12], dtype=torch.int32)
    loss = aggregate_pg_loss(
        pg, mask, loss_aggregation="prompt_mean", group_size=2, cu_seqlens=cu_seqlens
    )
    torch.testing.assert_close(loss, torch.tensor(PROMPT_MEAN))


def test_constant_normalizes_token_sum_by_fixed_sequence_divisor():
    loss = aggregate_pg_loss(
        PG,
        MASK,
        loss_aggregation="constant",
        loss_aggregation_divisor=CONSTANT_DIVISOR,
    )
    torch.testing.assert_close(loss, torch.tensor(CONSTANT_MEAN))


def test_constant_packed_matches_padded():
    pg = PG.reshape(-1)
    mask = MASK.reshape(-1)
    cu_seqlens = torch.tensor([0, 3, 6, 9, 12], dtype=torch.int32)
    loss = aggregate_pg_loss(
        pg,
        mask,
        loss_aggregation="constant",
        loss_aggregation_divisor=CONSTANT_DIVISOR,
        cu_seqlens=cu_seqlens,
    )
    torch.testing.assert_close(loss, torch.tensor(CONSTANT_MEAN))


@pytest.mark.parametrize(
    "aggregation", ["token_mean", "seq_mean", "prompt_mean", "constant"]
)
def test_loss_weight_pairing_realizes_global_mean(aggregation):
    group_size = 2 if aggregation == "prompt_mean" else 1
    divisor = CONSTANT_DIVISOR if aggregation == "constant" else None
    weight_fn = _make_loss_weight_fn(aggregation, group_size)

    full = aggregate_pg_loss(
        PG,
        MASK,
        loss_aggregation=aggregation,
        group_size=group_size,
        loss_aggregation_divisor=divisor,
    )

    num = torch.tensor(0.0)
    den = torch.tensor(0.0)
    for s in (slice(0, 2), slice(2, 4)):
        mb_pg, mb_mask = PG[s], MASK[s]
        loss_mb = aggregate_pg_loss(
            mb_pg,
            mb_mask,
            loss_aggregation=aggregation,
            group_size=group_size,
            loss_aggregation_divisor=divisor,
        )
        w = weight_fn({"loss_mask": mb_mask})
        num = num + loss_mb * w
        den = den + w
    torch.testing.assert_close(num / den, full)


@pytest.mark.parametrize(
    "aggregation", ["token_mean", "seq_mean", "prompt_mean", "constant"]
)
def test_loss_weight_pairing_realizes_global_mean_for_packed_inputs(aggregation):
    group_size = 2 if aggregation == "prompt_mean" else 1
    divisor = CONSTANT_DIVISOR if aggregation == "constant" else None
    weight_fn = _make_loss_weight_fn(aggregation, group_size)

    pg = PG.reshape(-1)
    mask = MASK.reshape(-1)
    cu_seqlens = torch.tensor([0, 3, 6, 9, 12], dtype=torch.int32)
    full = aggregate_pg_loss(
        pg,
        mask,
        loss_aggregation=aggregation,
        group_size=group_size,
        loss_aggregation_divisor=divisor,
        cu_seqlens=cu_seqlens,
    )

    num = torch.tensor(0.0)
    den = torch.tensor(0.0)
    for s in (slice(0, 6), slice(6, 12)):
        mb_pg, mb_mask = pg[s], mask[s]
        mb_cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
        loss_mb = aggregate_pg_loss(
            mb_pg,
            mb_mask,
            loss_aggregation=aggregation,
            group_size=group_size,
            loss_aggregation_divisor=divisor,
            cu_seqlens=mb_cu_seqlens,
        )
        w = weight_fn({"loss_mask": mb_mask, "cu_seqlens": mb_cu_seqlens})
        num = num + loss_mb * w
        den = den + w
    torch.testing.assert_close(num / den, full)


def test_denom_mask_uses_pre_rejection_count():
    pg = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    denom_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    loss = aggregate_pg_loss(
        pg, loss_mask, loss_aggregation="token_mean", denom_mask=denom_mask
    )
    torch.testing.assert_close(loss, torch.tensor(1.0))
    without = aggregate_pg_loss(pg, loss_mask, loss_aggregation="token_mean")
    torch.testing.assert_close(without, torch.tensor(2.0))


@pytest.mark.parametrize(
    ("aggregation", "group_size"), [("seq_mean", 1), ("prompt_mean", 2)]
)
def test_denom_mask_applies_to_unit_mean_denominator(aggregation, group_size):
    pg = torch.tensor([[2.0, 2.0, 2.0, 2.0], [4.0, 4.0, 4.0, 4.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    denom_mask = torch.ones_like(loss_mask)

    loss = aggregate_pg_loss(
        pg,
        loss_mask,
        loss_aggregation=aggregation,
        group_size=group_size,
        denom_mask=denom_mask,
    )
    torch.testing.assert_close(loss, torch.tensor(1.5))

    without = aggregate_pg_loss(
        pg, loss_mask, loss_aggregation=aggregation, group_size=group_size
    )
    torch.testing.assert_close(without, torch.tensor(3.0))


def test_prompt_mean_group_size_one_equals_seq_mean():
    a = aggregate_pg_loss(PG, MASK, loss_aggregation="prompt_mean", group_size=1)
    b = aggregate_pg_loss(PG, MASK, loss_aggregation="seq_mean")
    torch.testing.assert_close(a, b)


def test_prompt_mean_rejects_ragged_group_count():
    pg = torch.ones(3, 2)
    mask = torch.ones(3, 2)
    with pytest.raises(ValueError, match="not divisible by group_size"):
        aggregate_pg_loss(pg, mask, loss_aggregation="prompt_mean", group_size=2)


def test_config_derives_group_size_from_n_samples():
    cfg = GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=4),
        actor=PPOActorConfig(
            loss_aggregation="prompt_mean", mb_spec=MicroBatchSpec(granularity=4)
        ),
    )
    assert cfg.actor.group_size == 4


def test_config_hand_set_group_size_cannot_silently_take_effect():
    cfg = GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=4),
        actor=PPOActorConfig(
            loss_aggregation="prompt_mean",
            group_size=99,
            mb_spec=MicroBatchSpec(granularity=4),
        ),
    )
    assert cfg.actor.group_size == 4


def test_config_granularity_is_auto_bumped_for_prompt_mean():
    cfg = GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=4),
        actor=PPOActorConfig(
            loss_aggregation="prompt_mean", mb_spec=MicroBatchSpec(granularity=2)
        ),
    )
    assert cfg.actor.mb_spec.granularity == 4
    cfg = GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=8),
        actor=PPOActorConfig(loss_aggregation="prompt_mean"),
    )
    assert cfg.actor.mb_spec.granularity == 8


def test_prompt_mean_drops_under_filled_groups():
    cfg = GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=4),
        actor=PPOActorConfig(
            loss_aggregation="prompt_mean", mb_spec=MicroBatchSpec(granularity=4)
        ),
    )
    assert cfg.rollout.min_valid_group_size == 4


def test_min_valid_group_size_cannot_exceed_n_samples():
    with pytest.raises(ValueError, match="cannot exceed gconfig.n_samples"):
        GRPOConfig(
            gconfig=GenerationHyperparameters(n_samples=4),
            rollout=InferenceEngineConfig(min_valid_group_size=5),
        )


def test_config_validation():
    with pytest.raises(ValueError, match="n_samples >= 2"):
        GRPOConfig(
            gconfig=GenerationHyperparameters(n_samples=1),
            actor=PPOActorConfig(loss_aggregation="prompt_mean"),
        )
    with pytest.raises(ValueError, match="loss_aggregation must be"):
        PPOActorConfig(loss_aggregation="bogus")
    with pytest.raises(ValueError, match="loss_aggregation_divisor"):
        PPOActorConfig(loss_aggregation="constant")
    with pytest.raises(ValueError, match="loss_aggregation_divisor"):
        PPOActorConfig(loss_aggregation="constant", loss_aggregation_divisor=0)
    with pytest.raises(ValueError, match="loss_aggregation_divisor"):
        PPOActorConfig(
            loss_aggregation="constant", loss_aggregation_divisor=float("inf")
        )
    with pytest.raises(ValueError, match="only used"):
        PPOActorConfig(loss_aggregation="seq_mean", loss_aggregation_divisor=10)
    PPOActorConfig(loss_aggregation="constant", loss_aggregation_divisor=10)
    GRPOConfig(gconfig=GenerationHyperparameters(n_samples=1))
    GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=1),
        actor=PPOActorConfig(loss_aggregation="seq_mean"),
    )
