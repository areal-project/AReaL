# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from areal.engine import megatron_engine as engine_module
from areal.engine.megatron_utils.activation_storage import (
    ActivationStorageTracker,
    install_activation_storage_tracking,
    track_activation_storage,
)


@pytest.mark.parametrize("awex_residency", [False, True])
def test_training_tracks_activation_storage_only_for_awex(monkeypatch, awex_residency):
    """Ordinary training bypasses tracking; AWEX releases after one backward."""
    engine = engine_module.MegatronEngine.__new__(engine_module.MegatronEngine)
    engine._weight_residency = (
        SimpleNamespace(ensure_grad_buffers=lambda: None) if awex_residency else None
    )
    engine.model = [torch.nn.Linear(2, 2)]
    engine.model[0].config = SimpleNamespace(calculate_per_token_loss=True)
    engine.hf_config = None
    engine._ensure_ready = lambda: None
    engine._normalize_batch_input = lambda batch: (batch, None)
    engine.optimizer_zero_grad = lambda: None
    engine._prepare_mb_list = lambda *args, **kwargs: SimpleNamespace(mbs=[{}])
    engine._per_token_loss_multiplier = lambda *args: 1.0
    engine._collect_mtp_loss = lambda count: None
    calls = []
    engine.forward_backward_batch = lambda *args, **kwargs: calls.append("backward")

    def optimizer_step():
        calls.append("optimizer")
        return {}

    engine.optimizer_step = optimizer_step
    monkeypatch.setattr(
        engine_module, "record_training_batch", lambda *args: nullcontext()
    )
    monkeypatch.setattr(
        engine_module, "tensor_container_to", lambda value, device: value
    )
    monkeypatch.setattr(
        engine_module,
        "install_activation_storage_tracking",
        lambda: calls.append("install"),
    )

    def release():
        calls.append("release")
        return 0, 0

    @contextmanager
    def tracking(protected):
        assert {id(t) for t in protected} == {
            id(t) for t in engine.model[0].parameters()
        }
        yield SimpleNamespace(release=release)

    monkeypatch.setattr(engine_module, "track_activation_storage", tracking)
    engine.train_batch({}, lambda *args: None, lambda *args: None)
    assert calls == (
        ["install", "backward", "release", "optimizer"]
        if awex_residency
        else ["backward", "optimizer"]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tracker_releases_activation_and_preserves_model_alias():
    """A tracked view of model storage must stay resident after release."""
    model_param = torch.nn.Parameter(torch.ones(8, device="cuda"))
    activation = torch.ones(8, device="cuda")
    tracker = ActivationStorageTracker([model_param])
    tracker.append(model_param.view(2, 4))
    tracker.append(activation)

    released, size = tracker.release()

    assert released == 1
    assert size == 8 * activation.element_size()
    assert activation.untyped_storage().nbytes() == 0
    torch.testing.assert_close(
        model_param, torch.ones_like(model_param), rtol=0, atol=0
    )


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_te_checkpoint_tracking_preserves_two_training_steps(
    dtype: torch.dtype,
) -> None:
    """Real TE recomputation preserves gradients, updates and persistent storage."""
    from megatron.core import utils as mcore_utils
    from megatron.core.tensor_parallel.random import checkpoint
    from transformer_engine.pytorch import ops

    class RecomputeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [
                    ops.Sequential(
                        ops.LayerNorm(16, device="cuda", dtype=dtype),
                        ops.BasicLinear(16, 32, device="cuda", dtype=dtype),
                        ops.GELU(),
                        ops.BasicLinear(32, 16, device="cuda", dtype=dtype),
                    )
                    for _ in range(2)
                ]
            )
            self.register_buffer("offset", torch.ones(16, device="cuda", dtype=dtype))
            self.forward_modes = []

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            hidden = inputs + self.offset
            for block in self.blocks:

                def recompute(value, block=block):
                    self.forward_modes.append(torch.is_grad_enabled())
                    viewless = mcore_utils.make_viewless_tensor(
                        value.view_as(value), requires_grad=True, keep_graph=True
                    )
                    return block(viewless) + value

                hidden = checkpoint(recompute, False, hidden)
            return hidden

    torch.manual_seed(123)
    baseline = RecomputeModel()
    tracked = RecomputeModel()
    tracked.load_state_dict(baseline.state_dict())
    models = [baseline, tracked]
    optimizers = [
        torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9) for model in models
    ]
    install_activation_storage_tracking()

    for step in range(2):
        inputs = torch.randn(8, 16, device="cuda", dtype=dtype)
        target = torch.randn_like(inputs)
        gradients = []
        for model, optimizer, enabled in zip(models, optimizers, [False, True]):
            optimizer.zero_grad(set_to_none=True)
            protected = [*model.parameters(), *model.buffers()]
            pointers = [tensor.untyped_storage().data_ptr() for tensor in protected]
            context = track_activation_storage(protected) if enabled else nullcontext()
            with context as tracker:
                output = model(inputs.detach().clone().requires_grad_())
                loss = (output.float() - target.float()).square().mean()
                loss.backward()
            if tracker is not None:
                assert tracker._nested_released_storages > 0
                count, size = tracker.release()
                assert count > 0 and size > 0
            gradients.append(
                [parameter.grad.clone() for parameter in model.parameters()]
            )
            assert model.forward_modes[step * 4 :] == [False, False, True, True]
            for tensor, pointer in zip(protected, pointers):
                assert tensor.untyped_storage().data_ptr() == pointer
                assert tensor.untyped_storage().nbytes() > 0
            optimizer.step()

        for reference, actual in zip(gradients[0], gradients[1]):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for reference, actual in zip(baseline.parameters(), tracked.parameters()):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            torch.testing.assert_close(
                optimizers[1].state[actual]["momentum_buffer"],
                optimizers[0].state[reference]["momentum_buffer"],
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(tracked.offset, baseline.offset, rtol=0, atol=0)
