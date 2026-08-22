# SPDX-License-Identifier: Apache-2.0

from areal.trainer.mopd.loss import compose_mopd_loss, mopd_loss_fn
from areal.trainer.mopd.targets import aggregate_mopd_targets
from areal.trainer.mopd.teacher_manager import (
    DrainReceipt,
    PersistentTeacherManager,
    TeacherManagerState,
)

__all__ = [
    "DrainReceipt",
    "PersistentTeacherManager",
    "TeacherManagerState",
    "aggregate_mopd_targets",
    "compose_mopd_loss",
    "mopd_loss_fn",
]
