# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.engine.r3.asserts import R3Error
from areal.engine.r3.discovery import NativeRouterReplayRef
from areal.engine.r3.orchestration import (
    clear_router_replay_action,
    clear_router_replay_state,
    enqueue_recorded_indices,
    set_router_replay_action,
    set_target_indices,
)


class _Replay:
    def __init__(self, recorded: torch.Tensor | None = None):
        self.action = None
        self.targets = []
        self.recorded = recorded
        self.cleared_indices = False

    def set_router_replay_action(self, action):
        self.action = action

    def clear_router_replay_action(self):
        self.action = None

    def set_target_indices(self, topk_indices):
        self.targets.append(topk_indices)

    def get_recorded_indices(self):
        return self.recorded

    def clear_indices(self):
        self.cleared_indices = True
        self.targets.clear()
        self.recorded = None


def _ref(name: str, replay: _Replay) -> NativeRouterReplayRef:
    return NativeRouterReplayRef(
        vp_stage=0,
        name=name,
        router=SimpleNamespace(),
        router_replay=replay,
        layer_number=0,
    )


def test_set_target_indices_casts_to_long():
    replay = _Replay()
    ref = _ref("router", replay)

    set_target_indices([ref], [torch.ones(3, 2, dtype=torch.int32)])

    assert replay.targets[0].dtype == torch.long


def test_enqueue_recorded_indices_preserves_fifo_entry():
    recorded = torch.arange(6, dtype=torch.int32).reshape(3, 2)
    replay = _Replay(recorded=recorded)
    ref = _ref("router", replay)

    enqueue_recorded_indices([ref])

    assert len(replay.targets) == 1
    torch.testing.assert_close(replay.targets[0], recorded.long(), rtol=0, atol=0)


def test_enqueue_recorded_indices_missing_record_raises():
    ref = _ref("router", _Replay(recorded=None))

    with pytest.raises(R3Error, match="RECORD did not produce"):
        enqueue_recorded_indices([ref])


def test_action_helpers_are_instance_local(monkeypatch):
    monkeypatch.setattr(
        "areal.engine.r3.orchestration.get_router_replay_action",
        lambda action_name: action_name,
    )
    replay_a = _Replay()
    replay_b = _Replay()
    ref_a = _ref("router_a", replay_a)

    set_router_replay_action([ref_a], "RECORD")

    assert replay_a.action == "RECORD"
    assert replay_b.action is None
    clear_router_replay_action([ref_a])
    assert replay_a.action is None


def test_clear_router_replay_state_clears_indices_and_action():
    replay = _Replay(recorded=torch.ones(1, 2, dtype=torch.long))
    replay.action = "REPLAY_FORWARD"
    replay.targets.append(torch.ones(1, 2, dtype=torch.long))
    ref = _ref("router", replay)

    clear_router_replay_state([ref])

    assert replay.action is None
    assert replay.cleared_indices
    assert replay.targets == []
