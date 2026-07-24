# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the unified Muon optimizer."""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_model() -> nn.Module:
    """A small model with both >=2D and <2D params."""
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.LayerNorm(32),
        nn.Linear(32, 8),
    )
    return model


def _make_param_groups(model: nn.Module) -> list[dict]:
    """Split parameters into muon (>=2D) and backend (<2D) groups."""
    muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    backend_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return [
        dict(params=muon_params, lr=1e-2, use_muon=True),
        dict(params=backend_params, lr=1e-3, use_muon=False),
    ]


# ---------------------------------------------------------------------------
# Tests for unified Muon optimizer
# ---------------------------------------------------------------------------


class TestMuonOptimizer:
    """Tests for the unified Muon optimizer with built-in Adam backend."""

    def test_step_and_zero_grad(self):
        from areal.engine.fsdp_utils.muon import Muon

        model = _make_simple_model()
        opt = Muon(_make_param_groups(model))

        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()

        opt.step()
        opt.zero_grad()

        for p in model.parameters():
            assert p.grad is None or (p.grad == 0).all()

    def test_param_groups_structure(self):
        from areal.engine.fsdp_utils.muon import Muon

        model = _make_simple_model()
        groups = _make_param_groups(model)
        opt = Muon(groups)

        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["use_muon"] is True
        assert opt.param_groups[1]["use_muon"] is False

    def test_state_dict_roundtrip(self):
        from areal.engine.fsdp_utils.muon import Muon

        model = _make_simple_model()
        opt = Muon(_make_param_groups(model))

        # Do a step to populate state
        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        sd = opt.state_dict()
        assert "state" in sd
        assert "param_groups" in sd

        # Reload into fresh optimizer
        opt2 = Muon(_make_param_groups(model))
        opt2.load_state_dict(sd)

        sd2 = opt2.state_dict()
        assert len(sd2["state"]) == len(sd["state"])

    def test_lr_scheduler_compat(self):
        from areal.engine.fsdp_utils.muon import Muon

        model = _make_simple_model()
        opt = Muon(_make_param_groups(model))

        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)

        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        scheduler.step()

        # LR should be halved for all groups
        assert abs(opt.param_groups[0]["lr"] - 5e-3) < 1e-9
        assert abs(opt.param_groups[1]["lr"] - 5e-4) < 1e-9

    def test_all_params_updated(self):
        from areal.engine.fsdp_utils.muon import Muon

        model = _make_simple_model()
        opt = Muon(_make_param_groups(model))

        params_before = {
            name: p.clone() for name, p in model.named_parameters() if p.requires_grad
        }

        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert not torch.equal(p.data, params_before[name]), (
                    f"Parameter {name} was not updated"
                )

    def test_convergence(self):
        """Test that Muon+Adam can minimize a simple quadratic."""
        from areal.engine.fsdp_utils.muon import Muon

        torch.manual_seed(42)
        model = nn.Linear(8, 1, bias=True)
        target_w = torch.randn(1, 8)
        target_b = torch.randn(1)

        opt = Muon(
            [
                dict(params=[model.weight], lr=0.02, use_muon=True),
                dict(params=[model.bias], lr=0.02, use_muon=False),
            ]
        )

        for _ in range(200):
            x = torch.randn(32, 8)
            pred = model(x)
            target = x @ target_w.T + target_b
            loss = ((pred - target) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        final_loss = loss.item()
        assert final_loss < 0.1, f"Muon did not converge, final loss={final_loss}"

    def test_multi_step_finite(self):
        """Multiple steps should keep producing finite, reasonable params."""
        from areal.engine.fsdp_utils.muon import Muon

        torch.manual_seed(42)
        model = _make_simple_model()
        opt = Muon(_make_param_groups(model))

        for step_idx in range(10):
            x = torch.randn(4, 16)
            loss = model(x).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        for p in model.parameters():
            assert torch.isfinite(p.data).all()
