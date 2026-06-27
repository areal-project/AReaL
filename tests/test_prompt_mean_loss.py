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
from areal.trainer.ppo.actor import _make_actor_loss_normalizer_fn
from areal.utils.constants import (
    PROX_LOGP_METHOD_LOGLINEAR,
    PROX_LOGP_METHOD_RECOMPUTE,
)
from areal.utils.data import split_padded_tensor_dict_into_mb_list
from areal.utils.functional import (
    aggregate_pg_loss,
    aggregate_pg_loss_sum,
    make_pg_loss_normalizer_fn,
)

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


def test_prompt_mean_accepts_partial_group_sizes():
    pg = torch.tensor([[2.0, 2.0], [2.0, 2.0], [10.0, 10.0]])
    mask = torch.ones_like(pg)

    loss = aggregate_pg_loss(
        pg,
        mask,
        loss_aggregation="prompt_mean",
        group_size=2,
        group_sizes=[2, 1],
    )

    torch.testing.assert_close(loss, torch.tensor(6.0))


def test_prompt_mean_partial_group_packed_matches_padded():
    pg = torch.tensor([[2.0, 2.0], [2.0, 2.0], [10.0, 10.0]])
    mask = torch.ones_like(pg)
    cu_seqlens = torch.tensor([0, 2, 4, 6], dtype=torch.int32)

    padded = aggregate_pg_loss(
        pg,
        mask,
        loss_aggregation="prompt_mean",
        group_size=2,
        group_sizes=[2, 1],
    )
    packed = aggregate_pg_loss(
        pg.reshape(-1),
        mask.reshape(-1),
        loss_aggregation="prompt_mean",
        group_size=2,
        cu_seqlens=cu_seqlens,
        group_sizes=[2, 1],
    )

    torch.testing.assert_close(packed, padded)


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
def test_loss_normalizer_pairing_realizes_global_mean(aggregation):
    group_size = 2 if aggregation == "prompt_mean" else 1
    divisor = CONSTANT_DIVISOR if aggregation == "constant" else None
    normalizer_fn = make_pg_loss_normalizer_fn(aggregation, group_size)

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
        normalizer = normalizer_fn({"loss_mask": mb_mask})
        num = num + loss_mb * normalizer
        den = den + normalizer
    torch.testing.assert_close(num / den, full)


@pytest.mark.parametrize(
    "aggregation", ["token_mean", "seq_mean", "prompt_mean", "constant"]
)
def test_loss_sum_pairing_realizes_global_mean(aggregation):
    group_size = 2 if aggregation == "prompt_mean" else 1
    divisor = CONSTANT_DIVISOR if aggregation == "constant" else None
    normalizer_fn = make_pg_loss_normalizer_fn(aggregation, group_size)

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
        num = num + aggregate_pg_loss_sum(
            mb_pg,
            mb_mask,
            loss_aggregation=aggregation,
            group_size=group_size,
            loss_aggregation_divisor=divisor,
        )
        den = den + normalizer_fn({"loss_mask": mb_mask})
    torch.testing.assert_close(num / den, full)


@pytest.mark.parametrize(
    "aggregation", ["token_mean", "seq_mean", "prompt_mean", "constant"]
)
def test_loss_normalizer_pairing_realizes_global_mean_for_packed_inputs(aggregation):
    group_size = 2 if aggregation == "prompt_mean" else 1
    divisor = CONSTANT_DIVISOR if aggregation == "constant" else None
    normalizer_fn = make_pg_loss_normalizer_fn(aggregation, group_size)

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
        normalizer = normalizer_fn({"loss_mask": mb_mask, "cu_seqlens": mb_cu_seqlens})
        num = num + loss_mb * normalizer
        den = den + normalizer
    torch.testing.assert_close(num / den, full)


@pytest.mark.parametrize(
    "aggregation", ["token_mean", "seq_mean", "prompt_mean", "constant"]
)
def test_loss_sum_pairing_realizes_global_mean_for_packed_inputs(aggregation):
    group_size = 2 if aggregation == "prompt_mean" else 1
    divisor = CONSTANT_DIVISOR if aggregation == "constant" else None
    normalizer_fn = make_pg_loss_normalizer_fn(aggregation, group_size)

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
        num = num + aggregate_pg_loss_sum(
            mb_pg,
            mb_mask,
            loss_aggregation=aggregation,
            group_size=group_size,
            loss_aggregation_divisor=divisor,
            cu_seqlens=mb_cu_seqlens,
        )
        den = den + normalizer_fn({"loss_mask": mb_mask, "cu_seqlens": mb_cu_seqlens})
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


@pytest.mark.parametrize(
    ("aggregation", "group_size"), [("seq_mean", 1), ("prompt_mean", 2)]
)
def test_unit_mean_skips_units_without_denominator(aggregation, group_size):
    pg = torch.tensor([[2.0, 2.0], [2.0, 2.0], [100.0, 100.0], [100.0, 100.0]])
    loss_mask = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

    loss = aggregate_pg_loss(
        pg,
        loss_mask,
        loss_aggregation=aggregation,
        group_size=group_size,
    )

    torch.testing.assert_close(loss, torch.tensor(2.0))


@pytest.mark.parametrize(
    ("aggregation", "group_size"), [("seq_mean", 1), ("prompt_mean", 2)]
)
def test_packed_unit_mean_skips_units_without_denominator(aggregation, group_size):
    pg = torch.tensor([2.0, 2.0, 2.0, 2.0, 100.0, 100.0, 100.0, 100.0])
    loss_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    cu_seqlens = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32)

    loss = aggregate_pg_loss(
        pg,
        loss_mask,
        loss_aggregation=aggregation,
        group_size=group_size,
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(loss, torch.tensor(2.0))


def test_constant_skips_sequences_without_denominator():
    pg = torch.tensor([[2.0, 2.0], [100.0, 100.0]])
    loss_mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

    loss = aggregate_pg_loss(
        pg,
        loss_mask,
        loss_aggregation="constant",
        loss_aggregation_divisor=2.0,
    )

    torch.testing.assert_close(loss, torch.tensor(2.0))


def test_packed_constant_skips_sequences_without_denominator():
    pg = torch.tensor([2.0, 2.0, 100.0, 100.0])
    loss_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)

    loss = aggregate_pg_loss(
        pg,
        loss_mask,
        loss_aggregation="constant",
        loss_aggregation_divisor=2.0,
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(loss, torch.tensor(2.0))


@pytest.mark.parametrize(
    ("aggregation", "group_size"), [("seq_mean", 1), ("prompt_mean", 2)]
)
def test_denom_mask_keeps_empty_numerator_units_in_denominator(aggregation, group_size):
    pg = torch.tensor([[2.0, 2.0], [4.0, 4.0]])
    loss_mask = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    denom_mask = torch.ones_like(loss_mask)

    loss = aggregate_pg_loss(
        pg,
        loss_mask,
        loss_aggregation=aggregation,
        group_size=group_size,
        denom_mask=denom_mask,
    )
    without = aggregate_pg_loss(
        pg,
        loss_mask,
        loss_aggregation=aggregation,
        group_size=group_size,
    )

    torch.testing.assert_close(loss, torch.tensor(2.0))
    torch.testing.assert_close(without, torch.tensor(4.0))


def test_pg_loss_rejects_broadcastable_loss_mask_shape():
    pg = torch.ones(2, 3)
    loss_mask = torch.ones(2, 1)

    with pytest.raises(ValueError, match="loss_mask shape"):
        aggregate_pg_loss(pg, loss_mask, loss_aggregation="token_mean")


def test_pg_loss_sum_rejects_broadcastable_denom_mask_shape():
    pg = torch.ones(2, 3)
    loss_mask = torch.ones_like(pg)
    denom_mask = torch.ones(1, 3)

    with pytest.raises(ValueError, match="denom_mask shape"):
        aggregate_pg_loss_sum(
            pg,
            loss_mask,
            loss_aggregation="prompt_mean",
            group_size=2,
            denom_mask=denom_mask,
        )


def test_loss_normalizer_counts_active_units():
    seq_normalizer = make_pg_loss_normalizer_fn("seq_mean", 1)
    prompt_normalizer = make_pg_loss_normalizer_fn("prompt_mean", 2)
    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    torch.testing.assert_close(seq_normalizer({"loss_mask": mask}), torch.tensor(1.0))
    torch.testing.assert_close(
        prompt_normalizer({"loss_mask": mask}), torch.tensor(1.0)
    )


def test_prompt_normalizer_counts_partial_groups():
    prompt_normalizer = make_pg_loss_normalizer_fn("prompt_mean", 2)
    mask = torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])

    torch.testing.assert_close(
        prompt_normalizer({"loss_mask": mask, "group_sizes": [2, 1]}),
        torch.tensor(2.0),
    )


def test_m2po_normalizer_uses_post_filter_mask():
    normalizer = _make_actor_loss_normalizer_fn(
        "token_mean",
        group_size=1,
        m2_threshold=0.1,
        prox_logp_method=PROX_LOGP_METHOD_RECOMPUTE,
        current_version=3,
    )
    data = {
        "logprobs": torch.tensor([[0.0, 0.2, 1.0]]),
        "prox_logp": torch.zeros(1, 3),
        "loss_mask": torch.ones(1, 3, dtype=torch.bool),
    }

    torch.testing.assert_close(normalizer(data), torch.tensor(2))


def test_m2po_normalizer_rejects_missing_prox_logp_for_loglinear():
    normalizer = _make_actor_loss_normalizer_fn(
        "token_mean",
        group_size=1,
        m2_threshold=0.1,
        prox_logp_method=PROX_LOGP_METHOD_LOGLINEAR,
        current_version=3,
    )
    data = {
        "logprobs": torch.tensor([[0.0, 0.2, 1.0]]),
        "loss_mask": torch.ones(1, 3, dtype=torch.bool),
        "versions": torch.zeros(1, 3, dtype=torch.long),
    }

    with pytest.raises(ValueError, match="m2_threshold requires prox_logp"):
        normalizer(data)


def test_packed_loss_normalizer_counts_active_units():
    normalizer = make_pg_loss_normalizer_fn("seq_mean", 1)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)

    torch.testing.assert_close(
        normalizer({"loss_mask": mask, "cu_seqlens": cu_seqlens}),
        torch.tensor(1.0),
    )


def test_prompt_mean_group_size_one_equals_seq_mean():
    a = aggregate_pg_loss(PG, MASK, loss_aggregation="prompt_mean", group_size=1)
    b = aggregate_pg_loss(PG, MASK, loss_aggregation="seq_mean")
    torch.testing.assert_close(a, b)


def test_prompt_mean_rejects_ragged_group_count():
    pg = torch.ones(3, 2)
    mask = torch.ones(3, 2)
    with pytest.raises(ValueError, match="not divisible by group_size"):
        aggregate_pg_loss(pg, mask, loss_aggregation="prompt_mean", group_size=2)


def test_prompt_sum_pairing_realizes_global_mean_for_partial_groups():
    pg = torch.tensor([[2.0, 2.0], [2.0, 2.0], [10.0, 10.0]])
    mask = torch.ones_like(pg)
    normalizer_fn = make_pg_loss_normalizer_fn("prompt_mean", 2)
    full = aggregate_pg_loss(
        pg,
        mask,
        loss_aggregation="prompt_mean",
        group_size=2,
        group_sizes=[2, 1],
    )

    num = torch.tensor(0.0)
    den = torch.tensor(0.0)
    for seq_slice, group_sizes in ((slice(0, 2), [2]), (slice(2, 3), [1])):
        num = num + aggregate_pg_loss_sum(
            pg[seq_slice],
            mask[seq_slice],
            loss_aggregation="prompt_mean",
            group_size=2,
            group_sizes=group_sizes,
        )
        den = den + normalizer_fn(
            {"loss_mask": mask[seq_slice], "group_sizes": group_sizes}
        )

    torch.testing.assert_close(num / den, full)


def test_split_padded_tensor_dict_preserves_partial_group_sizes():
    data = {
        "attention_mask": torch.tensor(
            [[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool
        ),
        "input_ids": torch.arange(9).view(3, 3),
        "loss_mask": torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        "group_sizes": [2, 1],
    }

    mb_list = split_padded_tensor_dict_into_mb_list(
        data, MicroBatchSpec(n_mbs=2, granularity=2)
    )

    assert [mb["group_sizes"] for mb in mb_list.mbs] == [[2], [1]]


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


def test_config_granularity_is_not_mutated_for_prompt_mean():
    cfg = GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=4),
        actor=PPOActorConfig(
            loss_aggregation="prompt_mean", mb_spec=MicroBatchSpec(granularity=2)
        ),
    )
    assert cfg.actor.mb_spec.granularity == 2


def test_prompt_mean_keeps_partial_group_filtering_threshold():
    cfg = GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=4),
        rollout=InferenceEngineConfig(min_valid_group_size=2),
        actor=PPOActorConfig(
            loss_aggregation="prompt_mean", mb_spec=MicroBatchSpec(granularity=4)
        ),
    )
    assert cfg.rollout.min_valid_group_size == 2


def test_min_valid_group_size_cannot_exceed_n_samples():
    with pytest.raises(ValueError, match=r"cannot exceed gconfig\.n_samples"):
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
