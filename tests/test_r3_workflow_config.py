# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from areal.trainer.rl_trainer import PPOTrainer


def _trainer_with_r3_rollout() -> PPOTrainer:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(return_routed_experts=True)
    )
    return trainer


def test_r3_workflow_kwargs_rejects_vision_workflow_string():
    trainer = _trainer_with_r3_rollout()

    with pytest.raises(ValueError, match="VisionRLVRWorkflow"):
        trainer._maybe_inject_r3_workflow_kwargs(
            "areal.workflow.vision_rlvr.VisionRLVRWorkflow",
            {},
        )


def test_r3_workflow_shape_support_excludes_vision_workflow_class():
    from areal.workflow.vision_rlvr import VisionRLVRWorkflow

    trainer = _trainer_with_r3_rollout()

    assert not trainer._supports_r3_workflow_shape(VisionRLVRWorkflow)
    assert trainer._is_unsupported_r3_vision_workflow(VisionRLVRWorkflow)
