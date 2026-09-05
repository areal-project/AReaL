# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

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
    eval_rollout = object()
    trainer._init_rollout = Mock(return_value=eval_rollout)
    rollout_config = object()

    result = trainer._maybe_init_eval_rollout(rollout_config, lora_path="adapter")

    if expected_initialized:
        assert result is eval_rollout
        trainer._init_rollout.assert_called_once_with(
            rollout_config,
            is_eval=True,
            lora_path="adapter",
        )
    else:
        assert result is None
        trainer._init_rollout.assert_not_called()
