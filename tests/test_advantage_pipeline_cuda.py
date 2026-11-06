# SPDX-License-Identifier: Apache-2.0

"""Device parity for advantage stages; real engine tests cover training updates."""

import copy

import pytest
import torch

from areal.api.cli_args import AdvantageConfig, AdvantageFunctionConfig, PPOActorConfig
from areal.trainer.ppo.actor import PPOActor

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("unit", ["token", "turn"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("process", ["additive", "gvpo", "process_weighted"])
def test_advantage_stages_cuda_match_cpu(unit, dtype, process):
    """KL, process shaping, padding and explicit bootstrap agree across devices."""
    config = PPOActorConfig(
        kl_ctl=0.1,
        kl_estimator="k1",
        gae_lambda=0.7,
        gae_timestep_unit=unit,
        token_rewards_as_adv=True,
        advantage=AdvantageConfig(
            process_adv_shaping=AdvantageFunctionConfig(implementation=process)
        ),
    )
    actor = PPOActor(config, engine=None)
    data = {
        "input_ids": torch.ones(2, 6, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
        "loss_mask": torch.tensor([[0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 0, 0]]),
        "turn_ids": torch.tensor([[-1, 0, 0, -1, 1, 1], [-1, 0, 0, 1, -1, -1]]),
        "logprobs": torch.full((2, 6), -0.2, dtype=dtype),
        "ref_logp": torch.full((2, 6), -0.1, dtype=dtype),
        "values": torch.arange(12, dtype=dtype).reshape(2, 6) / 10,
        "rewards": torch.tensor([1.0, -1.0]),
        "is_truncated": torch.tensor([False, True]),
        "token_rewards": torch.full((2, 6), -0.25 if process == "gvpo" else 0.5),
    }
    expected = actor._compute_advantages(copy.deepcopy(data))
    actual = actor._compute_advantages({k: v.cuda() for k, v in data.items()})
    for key in expected:
        torch.testing.assert_close(
            actual[key].cpu(), expected[key], rtol=1e-5, atol=1e-6, msg=key
        )


def test_integer_outcome_cuda_preserves_reward_dtype():
    """The default stage still accepts integer task rewards on the accelerator."""
    actor = PPOActor(PPOActorConfig(kl_ctl=0), engine=None)
    data = {
        "input_ids": torch.ones(1, 3, dtype=torch.long, device="cuda"),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool, device="cuda"),
        "loss_mask": torch.tensor([[0, 1, 1]], device="cuda"),
        "logprobs": torch.zeros(1, 3, device="cuda"),
        "rewards": torch.tensor([1], device="cuda"),
    }
    output = actor._compute_advantages(data)
    assert output["rewards"].dtype == torch.int64
    torch.testing.assert_close(
        output["advantages"][:, :2], torch.ones(1, 2, device="cuda"), rtol=0, atol=0
    )
