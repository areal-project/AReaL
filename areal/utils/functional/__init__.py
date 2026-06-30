# SPDX-License-Identifier: Apache-2.0

from areal.utils.functional.functional import (
    LOSS_AGGREGATION_CONSTANT,
    LOSS_AGGREGATION_PROMPT_MEAN,
    LOSS_AGGREGATION_SEQ_MEAN,
    LOSS_AGGREGATION_TOKEN_MEAN,
    LOSS_AGGREGATIONS_ALL,
    RejectionSamplingResult,
    aggregate_pg_loss,
    aggregate_pg_loss_sum,
    apply_rejection_sampling,
    cispo_loss_fn,
    dpo_pair_logratios,
    dpo_preference_loss,
    make_pg_loss_normalizer_fn,
    masked_normalization,
    ppo_actor_loss_fn,
    ppo_critic_loss_fn,
    reward_overlong_penalty,
    sapo_loss_fn,
)
from areal.utils.functional.vocab_parallel import (
    gather_logprobs,
    gather_logprobs_entropy,
)

__all__ = [
    # functional.py
    "LOSS_AGGREGATION_CONSTANT",
    "LOSS_AGGREGATION_PROMPT_MEAN",
    "LOSS_AGGREGATION_SEQ_MEAN",
    "LOSS_AGGREGATION_TOKEN_MEAN",
    "LOSS_AGGREGATIONS_ALL",
    "RejectionSamplingResult",
    "aggregate_pg_loss",
    "aggregate_pg_loss_sum",
    "apply_rejection_sampling",
    "cispo_loss_fn",
    "dpo_pair_logratios",
    "dpo_preference_loss",
    "make_pg_loss_normalizer_fn",
    "masked_normalization",
    "ppo_actor_loss_fn",
    "ppo_critic_loss_fn",
    "reward_overlong_penalty",
    "sapo_loss_fn",
    # vocab_parallel.py
    "gather_logprobs",
    "gather_logprobs_entropy",
]
