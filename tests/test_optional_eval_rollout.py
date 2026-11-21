# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.trainer.rl_trainer import PPOTrainer


@pytest.mark.parametrize(
    "online_mode,has_valid_dataloader,expected_initialized",
    [
        (False, True, True),
        (False, False, False),
        (True, False, False),
    ],
)
def test_eval_rollout_is_initialized_only_when_validation_can_run(
    online_mode, has_valid_dataloader, expected_initialized
):
    trainer = object.__new__(PPOTrainer)
    trainer._online_mode = online_mode
    trainer.valid_dataloader = object() if has_valid_dataloader else None
    assert trainer._should_initialize_eval_rollout() is expected_initialized
