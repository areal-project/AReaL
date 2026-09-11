# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang distributed MTP weight updates."""

from types import SimpleNamespace

import pytest

from areal.v2.inference_service.sglang import mtp_weight_update_bridge as bridge_mod
from areal.v2.inference_service.sglang.mtp_weight_update_bridge import (
    MTPDistributedWeightUpdateBridge,
)


class _RecordingRunner:
    def __init__(self):
        self.calls = []
        self.result = (True, "Success")

    def update_weights_from_tensor(self, named_tensors, load_format=None):
        self.calls.append((list(named_tensors), load_format))
        return self.result


def _make_scheduler(draft_runner=None, *, spec_v2=True):
    target_runner = _RecordingRunner()
    draft_runner = draft_runner or _RecordingRunner()
    if spec_v2:
        speculative_worker = SimpleNamespace(
            draft_worker=SimpleNamespace(draft_runner=draft_runner)
        )
    else:
        speculative_worker = SimpleNamespace(model_runner=draft_runner)
    scheduler = SimpleNamespace(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=speculative_worker,
    )
    return scheduler, target_runner, draft_runner


def test_bridge_non_nextn_keeps_original_update(monkeypatch):
    """Non-NEXTN inference must retain SGLang's original update path."""
    scheduler, target_runner, _ = _make_scheduler()
    original_update = object()
    target_runner.update_weights_from_distributed = original_update
    monkeypatch.setattr(bridge_mod, "_sglang_version", lambda: "unexpected")

    MTPDistributedWeightUpdateBridge(
        scheduler, SimpleNamespace(speculative_algorithm=None)
    ).bind()

    assert target_runner.update_weights_from_distributed is original_update


def test_bridge_nextn_does_not_require_model_type(monkeypatch):
    """NEXTN routing must be enabled without inspecting model metadata."""
    scheduler, target_runner, _ = _make_scheduler()
    original_update = object()
    target_runner.update_weights_from_distributed = original_update
    monkeypatch.setattr(
        bridge_mod, "_sglang_version", lambda: bridge_mod._SUPPORTED_SGLANG_VERSION
    )

    MTPDistributedWeightUpdateBridge(
        scheduler, SimpleNamespace(speculative_algorithm="NEXTN")
    ).bind()

    assert callable(target_runner.update_weights_from_distributed)
    assert target_runner.update_weights_from_distributed is not original_update


@pytest.mark.parametrize("spec_v2", [False, True])
def test_bridge_runtime_eagle_binds_builtin_mtp_for_both_spec_versions(
    monkeypatch, spec_v2
):
    """Normalized NEXTN must find the draft runner in spec v1 and v2."""
    scheduler, target_runner, _ = _make_scheduler(spec_v2=spec_v2)
    original_update = object()
    target_runner.update_weights_from_distributed = original_update
    monkeypatch.setattr(
        bridge_mod, "_sglang_version", lambda: bridge_mod._SUPPORTED_SGLANG_VERSION
    )

    MTPDistributedWeightUpdateBridge(
        scheduler,
        SimpleNamespace(
            speculative_algorithm="EAGLE",
            speculative_draft_model_path=None,
        ),
    ).bind()

    assert callable(target_runner.update_weights_from_distributed)
    assert target_runner.update_weights_from_distributed is not original_update


def test_bridge_external_eagle_keeps_original_update(monkeypatch):
    """An external EAGLE draft model must not use the built-in MTP bridge."""
    scheduler, target_runner, _ = _make_scheduler()
    original_update = object()
    target_runner.update_weights_from_distributed = original_update
    monkeypatch.setattr(bridge_mod, "_sglang_version", lambda: "unexpected")

    MTPDistributedWeightUpdateBridge(
        scheduler,
        SimpleNamespace(
            speculative_algorithm="EAGLE",
            speculative_draft_model_path="external/eagle-draft",
        ),
    ).bind()

    assert target_runner.update_weights_from_distributed is original_update


def test_bridge_rejects_unvalidated_sglang_version(monkeypatch):
    """NEXTN must fail fast instead of silently patching an unknown API."""
    scheduler, _, _ = _make_scheduler()
    monkeypatch.setattr(bridge_mod, "_sglang_version", lambda: "0.5.11")

    with pytest.raises(RuntimeError, match="supports only sglang==0.5.10.post1"):
        MTPDistributedWeightUpdateBridge(
            scheduler, SimpleNamespace(speculative_algorithm="NEXTN")
        ).bind()


def test_bridge_rejects_pipeline_parallelism(monkeypatch):
    """The focused NEXTN MTP patch currently supports inference PP=1 only."""
    scheduler, _, _ = _make_scheduler()
    monkeypatch.setattr(
        bridge_mod, "_sglang_version", lambda: bridge_mod._SUPPORTED_SGLANG_VERSION
    )

    with pytest.raises(NotImplementedError, match="requires SGLang pp_size=1"):
        MTPDistributedWeightUpdateBridge(
            scheduler,
            SimpleNamespace(speculative_algorithm="NEXTN", pp_size=2),
        ).bind()


def test_bridge_nextn_receives_once_and_updates_draft_then_target(monkeypatch):
    """One received bucket must be reused by the draft and target loaders."""
    scheduler, target_runner, draft_runner = _make_scheduler()
    received = [
        ("model.layers.0.weight", object()),
        ("mtp.fc.weight", object()),
        ("mtp.layers.0.self_attn.q_proj.weight", object()),
    ]
    receive_calls = []

    def fake_receive(*args):
        receive_calls.append(args)
        return received

    monkeypatch.setattr(
        bridge_mod, "_sglang_version", lambda: bridge_mod._SUPPORTED_SGLANG_VERSION
    )
    monkeypatch.setattr(bridge_mod, "_receive_named_tensors", fake_receive)

    MTPDistributedWeightUpdateBridge(
        scheduler, SimpleNamespace(speculative_algorithm="NEXTN")
    ).bind()
    success, message = target_runner.update_weights_from_distributed(
        [name for name, _ in received],
        ["bfloat16"] * len(received),
        [(1,)] * len(received),
        "update_weight_group_0",
        "flattened_bucket",
    )

    assert success is True
    assert "2 MTP draft tensors" in message
    assert len(receive_calls) == 1
    assert draft_runner.calls == [(received, None)]
    assert target_runner.calls == [(received, None)]
    assert draft_runner.calls[0][0][1][1] is received[1][1]


def test_receive_rejects_unvalidated_load_format():
    """Unknown transport load formats must not silently use default loading."""
    runner = SimpleNamespace(_model_update_group={"group": object()})

    with pytest.raises(NotImplementedError, match="flattened_bucket"):
        bridge_mod._receive_named_tensors(
            runner,
            names=[],
            dtypes=[],
            shapes=[],
            group_name="group",
            load_format="custom_loader",
        )


def test_bridge_nextn_without_draft_runner_fails_fast(monkeypatch):
    """A missing draft runner must not degrade into target-only updates."""
    target_runner = SimpleNamespace()
    scheduler = SimpleNamespace(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=None,
    )
    monkeypatch.setattr(
        bridge_mod, "_sglang_version", lambda: bridge_mod._SUPPORTED_SGLANG_VERSION
    )

    with pytest.raises(RuntimeError, match="draft ModelRunner"):
        MTPDistributedWeightUpdateBridge(
            scheduler, SimpleNamespace(speculative_algorithm="NEXTN")
        ).bind()


def test_bridge_draft_load_failure_skips_target(monkeypatch):
    """Draft load failures must fail before applying the target update."""
    scheduler, target_runner, draft_runner = _make_scheduler()
    draft_runner.result = (False, "bad MTP tensor")
    monkeypatch.setattr(
        bridge_mod, "_sglang_version", lambda: bridge_mod._SUPPORTED_SGLANG_VERSION
    )
    monkeypatch.setattr(
        bridge_mod,
        "_receive_named_tensors",
        lambda *args: [("mtp.fc.weight", object())],
    )

    MTPDistributedWeightUpdateBridge(
        scheduler, SimpleNamespace(speculative_algorithm="NEXTN")
    ).bind()
    success, message = target_runner.update_weights_from_distributed(
        ["mtp.fc.weight"],
        ["bfloat16"],
        [(1,)],
        "update_weight_group_0",
    )

    assert success is False
    assert message == "bad MTP tensor"
    assert target_runner.calls == []
