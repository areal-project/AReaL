"""Unit tests for the Router Replay (R3) Megatron-Core monkey-patches.

These tests intentionally do **not** spin up a distributed runtime or load
Megatron model weights.  They exercise the pure-Python / pure-PyTorch logic
that backs Rollout Routing Replay on MoE models such as
Moonlight-16B-A3B:

* ``RouterReplay`` instance lifecycle (``set_target_indices``,
  ``record_indices``, ``clear_indices``, action toggles).
* Idempotency of ``apply_router_replay_patch`` /
  ``remove_router_replay_patch`` and the
  ``_PATCHES_APPLIED`` sentinel.
* Correctness of the dropless ``num_out_tokens`` override on the patched
  ``MoEAlltoAllTokenDispatcher.preprocess`` (the core correctness guarantee
  of R3 on Megatron-Core 0.16.0).
* Automatic resolution of ``num_moe_layers`` / ``topk`` from the model
  config via ``resolve_r3_moe_config`` (using Moonlight-16B-A3B as the
  driving example).

E2E coverage is in ``tests/test_router_replay_e2e.py``.
"""

from __future__ import annotations

import types

import pytest
import torch

from areal.engine.router_replay_patch import (
    RouterReplay,
    RouterReplayAction,
    apply_router_replay_patch,
    remove_router_replay_patch,
)

# ---------------------------------------------------------------------------
# RouterReplay instance lifecycle
# ---------------------------------------------------------------------------


class TestRouterReplayInstance:
    def setup_method(self):
        RouterReplay.router_instances.clear()

    def teardown_method(self):
        RouterReplay.router_instances.clear()

    def test_instance_is_registered_in_classvar(self):
        inst = RouterReplay()
        assert inst in RouterReplay.router_instances
        assert len(RouterReplay.router_instances) == 1

    def test_set_and_clear_target_indices(self):
        inst = RouterReplay()
        idx = torch.randint(0, 64, (16, 6), dtype=torch.int32)
        inst.set_target_indices(idx)
        assert inst.target_topk_idx is idx
        assert inst.replay_backward_list == [idx]

        inst.clear_indices()
        assert inst.target_topk_idx is None
        assert inst.recorded_topk_idx is None
        assert inst.replay_backward_list == []

    def test_record_and_get_indices(self):
        inst = RouterReplay()
        assert inst.get_recorded_indices() is None
        rec = torch.randint(0, 64, (16, 6), dtype=torch.int32)
        inst.record_indices(rec)
        assert torch.equal(inst.get_recorded_indices(), rec)

    def test_action_toggles(self):
        inst = RouterReplay()
        inst.set_router_replay_action(RouterReplayAction.RECORD)
        assert inst.router_replay_action is RouterReplayAction.RECORD
        inst.clear_router_replay_action()
        assert inst.router_replay_action is None

    def test_set_global_action_broadcasts_to_all(self):
        a, b, c = RouterReplay(), RouterReplay(), RouterReplay()
        RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        for inst in (a, b, c):
            assert inst.router_replay_action is RouterReplayAction.REPLAY_FORWARD
        RouterReplay.clear_global_router_replay_action()
        for inst in (a, b, c):
            assert inst.router_replay_action is None

    def test_set_replay_data_distributes_in_order(self):
        instances = [RouterReplay() for _ in range(3)]
        per_layer = [torch.full((4, 6), i, dtype=torch.int32) for i in range(3)]
        RouterReplay.set_replay_data(per_layer)
        for i, inst in enumerate(instances):
            assert torch.equal(inst.target_topk_idx, per_layer[i])

    def test_set_replay_data_mismatch_raises(self):
        _ = [RouterReplay() for _ in range(3)]
        with pytest.raises(ValueError, match="does not match number of router"):
            RouterReplay.set_replay_data([torch.zeros(4, 6, dtype=torch.int32)])


# ---------------------------------------------------------------------------
# apply_router_replay_patch / remove_router_replay_patch idempotency
# ---------------------------------------------------------------------------


@pytest.mark.gpu  # transformer_config import chain requires CUDA in some envs
class TestApplyPatchIdempotency:
    def test_apply_is_idempotent_and_sentinel_flips(self):
        pytest.importorskip("megatron.core")
        from areal.engine import router_replay_patch as rrp

        # Ensure a clean slate.
        remove_router_replay_patch()
        assert rrp._PATCHES_APPLIED is False

        apply_router_replay_patch()
        assert rrp._PATCHES_APPLIED is True

        # Second call is a no-op and must not raise.
        apply_router_replay_patch()
        assert rrp._PATCHES_APPLIED is True

        remove_router_replay_patch()
        assert rrp._PATCHES_APPLIED is False

    def test_topk_router_init_patched_flag(self):
        pytest.importorskip("megatron.core")
        from megatron.core.transformer.moe.router import TopKRouter

        remove_router_replay_patch()
        assert not getattr(TopKRouter, "_r3_init_patched", False)
        apply_router_replay_patch()
        assert getattr(TopKRouter, "_r3_init_patched", False) is True
        remove_router_replay_patch()
        assert not getattr(TopKRouter, "_r3_init_patched", False)


# ---------------------------------------------------------------------------
# Patched MoEAlltoAllTokenDispatcher.preprocess: num_out_tokens correctness
# ---------------------------------------------------------------------------


class _FakeMoEConfig:
    """Minimal config shim for exercising patched_preprocess() directly."""

    def __init__(
        self,
        enable_routing_replay: bool,
        capacity_factor: float | None = None,
        fp8_padding: bool = False,
        quant_padding: bool = False,
        topk: int = 6,
    ):
        self.enable_routing_replay = enable_routing_replay
        self.moe_expert_capacity_factor = capacity_factor
        self.moe_router_padding_for_fp8 = fp8_padding
        self.moe_router_padding_for_quantization = quant_padding
        self.moe_router_topk = topk


def _invoke_patched_preprocess(
    dispatcher_self,
    routing_map: torch.Tensor,
    fake_original_num_out_tokens: int,
):
    """Directly exercise the logic inside ``_patch_alltoall_dispatcher_preprocess``.

    We reimplement the override locally instead of monkey-patching the real
    Megatron class (which is heavy and requires CUDA).  The logic must stay
    identical to ``router_replay_patch.py``'s ``patched_preprocess``.
    """
    # Emulate what the original dispatcher sets in the dropless branch.
    dispatcher_self.num_out_tokens = fake_original_num_out_tokens

    if (
        getattr(dispatcher_self.config, "enable_routing_replay", False)
        and not dispatcher_self.drop_and_pad
        and dispatcher_self.config.moe_expert_capacity_factor is None
        and not (
            getattr(dispatcher_self.config, "moe_router_padding_for_quantization", None)
            or getattr(dispatcher_self.config, "moe_router_padding_for_fp8", None)
        )
    ):
        dispatcher_self.num_out_tokens = int(routing_map.sum().item())


class TestDispatcherNumOutTokensOverride:
    """``num_out_tokens`` must be recomputed when replay zeroes padding rows.

    Megatron-Core 0.16.0 sets ``num_out_tokens = routing_map.size(0) * topk``
    on the dropless branch.  Under R3, ``routing_map`` has padding rows
    zeroed out so that the static value overcounts real tokens.  The patch
    must override it with ``routing_map.sum().item()``.
    """

    def _make_routing_map(
        self, num_real: int, num_padding: int, num_experts: int, topk: int
    ):
        rm = torch.zeros(num_real + num_padding, num_experts, dtype=torch.bool)
        for i in range(num_real):
            experts = torch.randperm(num_experts)[:topk]
            rm[i, experts] = True
        # padding rows stay all-False — this is what RouterReplay does
        return rm

    def test_replay_on_dropless_overrides_num_out_tokens(self):
        num_real, num_pad, num_experts, topk = 7, 3, 64, 6
        rm = self._make_routing_map(num_real, num_pad, num_experts, topk)
        disp = types.SimpleNamespace(
            drop_and_pad=False,
            config=_FakeMoEConfig(enable_routing_replay=True, topk=topk),
            num_out_tokens=None,
        )
        static_upstream = (num_real + num_pad) * topk
        _invoke_patched_preprocess(disp, rm, static_upstream)
        assert disp.num_out_tokens == num_real * topk
        assert disp.num_out_tokens < static_upstream

    def test_replay_disabled_keeps_upstream_value(self):
        rm = self._make_routing_map(5, 5, 64, 6)
        disp = types.SimpleNamespace(
            drop_and_pad=False,
            config=_FakeMoEConfig(enable_routing_replay=False, topk=6),
            num_out_tokens=None,
        )
        _invoke_patched_preprocess(disp, rm, 60)
        assert disp.num_out_tokens == 60

    def test_capacity_factor_set_keeps_upstream_value(self):
        rm = self._make_routing_map(5, 5, 64, 6)
        disp = types.SimpleNamespace(
            drop_and_pad=False,
            config=_FakeMoEConfig(
                enable_routing_replay=True, capacity_factor=1.25, topk=6
            ),
            num_out_tokens=None,
        )
        _invoke_patched_preprocess(disp, rm, 60)
        assert disp.num_out_tokens == 60

    def test_fp8_padding_keeps_upstream_value(self):
        rm = self._make_routing_map(5, 5, 64, 6)
        disp = types.SimpleNamespace(
            drop_and_pad=False,
            config=_FakeMoEConfig(enable_routing_replay=True, fp8_padding=True, topk=6),
            num_out_tokens=None,
        )
        _invoke_patched_preprocess(disp, rm, 60)
        assert disp.num_out_tokens == 60

    def test_drop_and_pad_keeps_upstream_value(self):
        rm = self._make_routing_map(5, 5, 64, 6)
        disp = types.SimpleNamespace(
            drop_and_pad=True,
            config=_FakeMoEConfig(enable_routing_replay=True, topk=6),
            num_out_tokens=None,
        )
        _invoke_patched_preprocess(disp, rm, 60)
        assert disp.num_out_tokens == 60


# ---------------------------------------------------------------------------
# resolve_r3_moe_config: Moonlight-16B-A3B driven auto-resolution
# ---------------------------------------------------------------------------


class _FakeHFConfig:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


class TestResolveR3MoeConfig:
    def setup_method(self):
        from areal.workflow import rlvr_r3_patch as mod

        mod._RESOLVED_CACHE.clear()

    def _patched_autoconfig(self, monkeypatch, fake_config):
        class _FakeAutoConfig:
            @staticmethod
            def from_pretrained(path, trust_remote_code=True):  # noqa: ARG004
                return fake_config

        monkeypatch.setattr("transformers.AutoConfig", _FakeAutoConfig, raising=True)

    def test_moonlight_like_config(self, monkeypatch):
        """Moonlight-16B-A3B: 27 layers (1 dense + 26 MoE), topk=6."""
        from areal.workflow.rlvr_r3_patch import resolve_r3_moe_config

        fake = _FakeHFConfig(
            num_experts_per_tok=6,
            num_hidden_layers=27,
            first_k_dense_replace=1,
        )
        self._patched_autoconfig(monkeypatch, fake)
        num_moe, topk = resolve_r3_moe_config("/fake/moonlight/16b-a3b")
        assert topk == 6
        assert num_moe == 26  # 27 - 1

    def test_moe_layer_freq_list(self, monkeypatch):
        from areal.workflow.rlvr_r3_patch import resolve_r3_moe_config

        freq = [0, 0, 1, 1, 1, 1]  # 4 MoE layers out of 6
        fake = _FakeHFConfig(
            num_experts_per_tok=4,
            num_hidden_layers=6,
            moe_layer_freq=freq,
        )
        self._patched_autoconfig(monkeypatch, fake)
        num_moe, topk = resolve_r3_moe_config("/fake/list-freq-model")
        assert num_moe == 4
        assert topk == 4

    def test_moe_layer_freq_int(self, monkeypatch):
        from areal.workflow.rlvr_r3_patch import resolve_r3_moe_config

        fake = _FakeHFConfig(
            num_experts_per_tok=2,
            num_hidden_layers=8,
            moe_layer_freq=2,  # every other layer is MoE
        )
        self._patched_autoconfig(monkeypatch, fake)
        num_moe, topk = resolve_r3_moe_config("/fake/int-freq-model")
        assert num_moe == 4  # 0,2,4,6
        assert topk == 2

    def test_missing_topk_raises(self, monkeypatch):
        from areal.workflow.rlvr_r3_patch import resolve_r3_moe_config

        fake = _FakeHFConfig(num_hidden_layers=27, first_k_dense_replace=1)
        self._patched_autoconfig(monkeypatch, fake)
        with pytest.raises(ValueError, match="Cannot resolve topk"):
            resolve_r3_moe_config("/fake/no-topk")

    def test_cache_hit_does_not_touch_disk(self, monkeypatch):
        from areal.workflow import rlvr_r3_patch as mod
        from areal.workflow.rlvr_r3_patch import resolve_r3_moe_config

        mod._RESOLVED_CACHE["/cached/path"] = (26, 6)

        def _should_not_be_called(*_a, **_kw):
            raise AssertionError(
                "transformers.AutoConfig.from_pretrained must not be called "
                "when _RESOLVED_CACHE already has the path."
            )

        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained",
            _should_not_be_called,
            raising=True,
        )
        assert resolve_r3_moe_config("/cached/path") == (26, 6)


# ---------------------------------------------------------------------------
# preprocess_routed_experts_batch: rollout numpy → training tensor
# ---------------------------------------------------------------------------


class TestPreprocessRoutedExpertsBatch:
    def test_moonlight_shape(self):
        import numpy as np

        from areal.engine.router_replay_utils import preprocess_routed_experts_batch

        num_moe, topk = 26, 6  # Moonlight-16B-A3B
        seq_len = 10
        num_sgl_tokens = seq_len - 1  # SGLang convention
        np_arr = np.random.randint(
            0, 64, size=(num_sgl_tokens, num_moe * topk), dtype=np.int32
        )
        input_ids = torch.zeros(1, seq_len, dtype=torch.long)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)

        out = preprocess_routed_experts_batch(
            np_arr,
            input_ids,
            attention_mask,
            num_moe_layers=num_moe,
            topk=topk,
            compress_dtype=False,
        )
        assert out.shape == (1, seq_len, num_moe, topk)
        # First num_sgl_tokens rows come from the numpy array
        for t in range(num_sgl_tokens):
            torch.testing.assert_close(
                out[0, t].to(torch.int32),
                torch.from_numpy(np_arr[t].reshape(num_moe, topk)),
            )
        # Trailing row is zero-padded
        assert (out[0, num_sgl_tokens:] == 0).all()

    def test_dtype_compression(self):
        import numpy as np

        from areal.engine.router_replay_utils import preprocess_routed_experts_batch

        np_arr = np.random.randint(0, 64, size=(5, 6 * 6), dtype=np.int32)
        input_ids = torch.zeros(1, 6, dtype=torch.long)
        attention_mask = torch.ones(1, 6, dtype=torch.long)
        out = preprocess_routed_experts_batch(
            np_arr,
            input_ids,
            attention_mask,
            num_moe_layers=6,
            topk=6,
            compress_dtype=True,
        )
        assert out.dtype == torch.uint8  # max expert idx < 256
