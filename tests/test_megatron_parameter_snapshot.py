# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.infra.platforms import current_platform

if current_platform.device_type != "npu" or current_platform.device_count() < 1:
    pytest.skip(
        "requires the target Ascend Megatron/MindSpeed stack", allow_module_level=True
    )

from areal.engine import megatron_engine as megatron_engine_module

pytestmark = pytest.mark.npu


class _DDPChunk(torch.nn.Module):
    """Real tensors/hooks with a small stand-in for DDP synchronization."""

    def __init__(self, index, events):
        super().__init__()
        self.index = index
        self.events = events
        self.use_forward_hook = True
        self.remove_forward_pre_hook_handles = {}
        self.fail_sync = False
        self.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.other_handle = self.register_forward_pre_hook(lambda *_args: None)
        self.enable_forward_pre_hook()

    def enable_forward_pre_hook(self):
        self.events.append(("enable", self.index))
        self.remove_forward_pre_hook_handles[self] = self.register_forward_pre_hook(
            lambda *_args: None
        )

    def disable_forward_pre_hook(self, param_sync=True):
        self.events.append(("disable", self.index))
        for module, handle in list(self.remove_forward_pre_hook_handles.items()):
            handle.remove()
            del self.remove_forward_pre_hook_handles[module]
        if param_sync:
            self.start_param_sync(force_sync=True)

    def start_param_sync(self, *, force_sync):
        assert force_sync
        self.events.append(("sync", self.index))
        if self.fail_sync:
            raise RuntimeError(f"sync {self.index}")


def _make_engine(monkeypatch):
    monkeypatch.setattr(megatron_engine_module, "DDP", _DDPChunk)
    events = []
    engine = megatron_engine_module.MegatronEngine.__new__(
        megatron_engine_module.MegatronEngine
    )
    engine.model = [_DDPChunk(0, events)]
    monkeypatch.setattr(
        current_platform._ensure_initialized(),
        "synchronize",
        lambda: events.append(("copies_done",)),
    )
    events.clear()
    return engine, events


@pytest.mark.parametrize("error_type", [None, RuntimeError])
def test_snapshot_sync_restores_weights_before_original_hooks(monkeypatch, error_type):
    """Sync precedes capture; normal/exception exits restore weights then hooks."""
    engine, events = _make_engine(monkeypatch)
    chunk = engine.model[0]
    engine.capture_parameter_snapshot()
    assert events.index(("sync", 0)) < events.index(("copies_done",))
    with torch.no_grad():
        chunk.weight.add_(10)
    current_weight = chunk.weight.detach().clone()
    events.clear()

    def run_forward():
        with engine.parameter_snapshot_context():
            assert not chunk.remove_forward_pre_hook_handles
            torch.testing.assert_close(
                chunk.weight, torch.arange(4, dtype=torch.float32), rtol=0, atol=0
            )
            events.append(("forward",))
            if error_type is not None:
                raise error_type("forward failed")

    if error_type is None:
        run_forward()
    else:
        with pytest.raises(error_type, match="forward failed"):
            run_forward()
    torch.testing.assert_close(chunk.weight, current_weight, rtol=0, atol=0)
    assert chunk.remove_forward_pre_hook_handles
    assert chunk.other_handle.id in chunk._forward_pre_hooks
    assert events.index(("sync", 0)) < events.index(("forward",))
    assert events.index(("copies_done",)) < events.index(("enable", 0))


def test_snapshot_sync_failure_restores_all_touched_chunks_without_gather(monkeypatch):
    """Restore touched hooks without another gather and propagate the sync error."""
    engine, events = _make_engine(monkeypatch)
    second = _DDPChunk(1, events)
    engine.model.append(second)
    second.fail_sync = True
    events.clear()

    with pytest.raises(RuntimeError, match="sync 1"):
        engine.capture_parameter_snapshot()

    # Cleanup must not submit another collective after one rank's sync fails.
    assert [event for event in events if event[0] == "sync"] == [
        ("sync", 0),
        ("sync", 1),
    ]
    for chunk in engine.model:
        assert chunk.remove_forward_pre_hook_handles
        assert chunk.other_handle.id in chunk._forward_pre_hooks
