# SPDX-License-Identifier: Apache-2.0

import torch

from areal.api import (
    LOSS_TERM_REDUCTION_SUM,
    LossReduction,
    LossTerm,
)
from areal.engine.core.train_engine import scale_loss_for_reduction


def _normalizer(_data):
    return torch.tensor(1.0)


def test_mean_scaling_preserves_local_mean_order():
    reduction = LossReduction.mean(
        loss_fn=lambda: torch.tensor(2.0), normalizer_fn=_normalizer
    )
    local_normalizer = torch.tensor(3.0)
    global_normalizer = torch.tensor(12.0)
    loss = torch.tensor(2.0)

    scaled = scale_loss_for_reduction(
        loss,
        reduction,
        {"loss": local_normalizer},
        {"loss": global_normalizer},
        loss_multiplier=1.0,
    )

    torch.testing.assert_close(
        scaled, loss * local_normalizer / global_normalizer, rtol=0, atol=0
    )


def test_sum_scaling_uses_global_normalizer_directly():
    reduction = LossReduction.sum(
        loss_fn=lambda: torch.tensor(6.0), normalizer_fn=_normalizer
    )
    local_sum = torch.tensor(6.0)

    scaled = scale_loss_for_reduction(
        local_sum,
        reduction,
        {"loss": torch.tensor(3.0)},
        {"loss": torch.tensor(12.0)},
        loss_multiplier=1.0,
    )

    torch.testing.assert_close(scaled, local_sum / 12.0, rtol=0, atol=0)


def test_multi_term_scaling_uses_each_terms_normalizer():
    reduction = LossReduction(
        loss_fn=lambda: {
            "pg": torch.tensor(6.0),
            "kd": torch.tensor(2.0),
        },
        terms=(
            LossTerm(
                "pg", normalizer_fn=_normalizer, reduction=LOSS_TERM_REDUCTION_SUM
            ),
            LossTerm(
                "kd", normalizer_fn=_normalizer, reduction=LOSS_TERM_REDUCTION_SUM
            ),
        ),
    )

    scaled = scale_loss_for_reduction(
        {"pg": torch.tensor(6.0), "kd": torch.tensor(2.0)},
        reduction,
        {"pg": torch.tensor(3.0), "kd": torch.tensor(2.0)},
        {"pg": torch.tensor(12.0), "kd": torch.tensor(4.0)},
        loss_multiplier=1.0,
    )

    torch.testing.assert_close(scaled, torch.tensor(1.0), rtol=0, atol=0)
