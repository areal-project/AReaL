# SPDX-License-Identifier: Apache-2.0

"""Numerical and lifecycle regressions for chunk accumulation metadata."""

import copy
from types import SimpleNamespace

import pytest
import torch

from tests.utils import (
    _assert_adam_moments_close,
    _assert_optimizer_steps,
    _assert_pointwise_tensors_close,
    _optimizer_steps,
)

from areal.engine.core.train_engine import (
    _gradient_accumulation,
    _scale_local_gradients,
)
from areal.trainer.ppo.loss_reduction import prepare_policy_gradient_batch


@pytest.mark.parametrize(
    "mode", ["token-mean", "seq-mean", "traj-mean", "prompt-mean", "constant"]
)
def test_unequal_active_units_across_chunks_match_window_objective(mode):
    """Keep token, row, full-unit and constant objectives across chunk packing."""
    parameter = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
    engine = SimpleNamespace()
    chunks = []
    oracle_units, oracle_sum, oracle_tokens, oracle_rows = [], 0.0, 0, 0
    for counts, sizes in (((1, 3, 0), (2, 1)), ((2, 4, 1, 2), (1, 3))):
        mask = torch.arange(4)[None, :] < torch.tensor(counts)[:, None]
        targets = torch.arange(mask.numel(), dtype=torch.float64).reshape_as(mask) / 10
        data = {"loss_mask": mask, "targets": targets}
        groups = sizes if mode == "prompt-mean" else (1,) * len(counts)
        prepared = prepare_policy_gradient_batch(
            data,
            mode=mode,
            divisor=5.0 if mode == "constant" else None,
            group_sizes=groups,
        )
        chunks.append((data, prepared.reduction))
        values = (parameter - targets).square()
        oracle_sum = oracle_sum + values[mask].sum()
        oracle_tokens += sum(counts)
        oracle_rows += sum(count > 0 for count in counts)
        start = 0
        for size in groups:
            group_mask = mask[start : start + size]
            if group_mask.any():
                oracle_units.append(values[start : start + size][group_mask].mean())
            start += size
    if mode == "token-mean":
        expected = oracle_sum / oracle_tokens
    elif mode == "constant":
        expected = oracle_sum / (oracle_rows * 5.0)
    else:
        expected = torch.stack(oracle_units).mean()
    expected_grad = torch.autograd.grad(expected, parameter)[0]
    for index, (data, reduction) in enumerate(chunks):
        with _gradient_accumulation(engine, 2, index) as state:
            state.add_chunk(reduction.normalizer_fn(data), len(data["loss_mask"]))
            # Independent packing per chunk may split a trajectory or prompt.
            for row in range(len(data["loss_mask"])):
                mb = {key: tensor[row : row + 1] for key, tensor in data.items()}
                mean = reduction.aggregate(
                    (parameter - mb["targets"]).square(),
                    mb["loss_mask"],
                    unit_weights=mb.get("unit_weights"),
                )
                (
                    mean * reduction.normalizer_fn(mb) / state.reference_loss_weight
                ).backward()
            if state.is_last:
                _scale_local_gradients([parameter], state.gradient_scale)
    torch.testing.assert_close(parameter.grad, expected_grad, rtol=1e-6, atol=1e-7)


def test_unequal_chunks_match_full_window_adam_updates():
    """Reference weighting must preserve gradients, clipping and Adam moments."""
    torch.manual_seed(37)
    model = torch.nn.Linear(3, 1)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    engine = SimpleNamespace()
    for _ in range(2):
        chunks = [(torch.randn(n, 3), torch.randn(n, 1)) for n in (2, 7, 3)]
        initial_parameters = [p.detach().clone() for p in model.parameters()]
        reference_optimizer.zero_grad()
        total_weight = sum(len(x) for x, _ in chunks)
        sum(
            (reference(x) - y).square().sum() / total_weight for x, y in chunks
        ).backward()
        reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
        reference_optimizer.step()

        for index, (x, y) in enumerate(chunks):
            with _gradient_accumulation(engine, len(chunks), index) as state:
                if state.is_first:
                    optimizer.zero_grad()
                state.add_chunk(torch.tensor(float(len(x))), 1)
                ((model(x) - y).square().sum() / state.reference_loss_weight).backward()
                if not state.is_last:
                    for parameter, initial in zip(
                        model.parameters(), initial_parameters
                    ):
                        torch.testing.assert_close(parameter, initial, rtol=0, atol=0)
                    continue
                _scale_local_gradients(model.parameters(), state.gradient_scale)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                torch.testing.assert_close(norm, reference_norm, rtol=1e-5, atol=1e-6)
                optimizer.step()
        assert engine._gradient_accumulation_state is None
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-6)
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(
                    optimizer.state[parameter][key],
                    reference_optimizer.state[expected][key],
                    rtol=1e-5,
                    atol=1e-6,
                )


def test_reference_denominator_is_fixed_and_detached():
    """A reference need not predict the final token or microbatch count."""
    engine = SimpleNamespace()
    for index, (weight, count) in enumerate(((100.0, 2), (300.0, 3))):
        with _gradient_accumulation(engine, 2, index) as state:
            state.add_chunk(torch.tensor(weight, requires_grad=True), count)
            assert not state.reference_loss_weight.requires_grad
            assert not state.total_loss_weight.requires_grad
            torch.testing.assert_close(
                state.reference_loss_weight, torch.tensor(200.0), rtol=0, atol=0
            )
            assert state.reference_microbatches == 4
            if state.is_last:
                assert state.num_microbatches == 5
                torch.testing.assert_close(
                    state.gradient_scale, torch.tensor(0.5), rtol=0, atol=0
                )


@pytest.mark.parametrize("steps,index", [(0, 0), (2, -1), (2, 2), (2, 1)])
def test_invalid_window_calls_are_rejected(steps, index):
    with pytest.raises(ValueError):
        with _gradient_accumulation(SimpleNamespace(), steps, index):
            pytest.fail("invalid call reached the training body")


@pytest.mark.parametrize("steps,index", [(1, 0), (2, 0), (3, 1)])
def test_unfinished_window_cannot_be_overwritten(steps, index):
    engine = SimpleNamespace()
    with _gradient_accumulation(engine, 2, 0):
        pass
    with pytest.raises(ValueError):
        with _gradient_accumulation(engine, steps, index):
            pytest.fail("out-of-order call reached the training body")


def test_partial_backward_failure_blocks_rpc_retry():
    engine = SimpleNamespace()
    with pytest.raises(RuntimeError, match="backward failed"):
        with _gradient_accumulation(engine, 2, 0):
            raise RuntimeError("backward failed")
    for steps, index in ((2, 0), (2, 1), (1, 0)):
        with pytest.raises(RuntimeError, match="recreate the engine"):
            with _gradient_accumulation(engine, steps, index):
                pytest.fail("a retry must not reuse partial gradients")


def test_single_call_does_not_create_accumulation_state():
    engine = SimpleNamespace()
    with _gradient_accumulation(engine, 1, 0) as state:
        assert state is None
    assert not hasattr(engine, "_gradient_accumulation_state")


def test_gradient_scaling_preserves_storage_dtype_and_shared_parameters():
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    parameter.grad = torch.full_like(parameter, 3)
    grad = parameter.grad
    _scale_local_gradients([parameter, parameter], torch.tensor(0.5))
    assert parameter.grad is grad
    assert parameter.grad.dtype == torch.bfloat16
    torch.testing.assert_close(grad, torch.full_like(grad, 1.5), rtol=0, atol=0)


def _moment_tensors(*gradients):
    """First-update Adam moments for independently named parameter tensors."""
    return {
        ("state", index, moment): value
        for index, gradient in enumerate(gradients)
        for moment, value in (
            ("exp_avg", gradient * 0.1),
            ("exp_avg_sq", gradient.square() * 0.001),
        )
    }


def test_tiny_gradient_sign_change_does_not_imply_large_adam_moment_error():
    """Adam can amplify negligible gradient differences into opposite updates."""
    states, parameters = [], []
    for tiny_gradient in (1e-6, -1e-6):
        parameter = torch.nn.Parameter(torch.zeros(4))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        parameter.grad = torch.tensor([tiny_gradient, 1.0, -2.0, 3.0])
        optimizer.step()
        states.append(
            {(key,): value for key, value in optimizer.state[parameter].items()}
        )
        parameters.append(parameter.detach())

    _assert_adam_moments_close(states[1], states[0])
    with pytest.raises(AssertionError):
        torch.testing.assert_close(parameters[1], parameters[0], rtol=2e-2, atol=1e-6)


@pytest.mark.parametrize(
    "error", ["missing_chunk", "missing_correction", "scale", "direction"]
)
def test_moment_comparison_rejects_accumulation_errors(error):
    """Normwise comparison still detects lost, misweighted or rotated gradients."""
    first = torch.tensor([1.0, 2.0, -3.0])
    second = torch.tensor([-2.0, 4.0, 1.0])
    expected = (first + second) / 2
    actual = {
        "missing_chunk": first / 2,
        "missing_correction": expected * (1920 / 896),
        "scale": expected * 1.1,
        "direction": expected.roll(1),
    }[error]
    with pytest.raises(AssertionError, match="exp_avg"):
        _assert_adam_moments_close(_moment_tensors(actual), _moment_tensors(expected))


def test_strict_control_rejects_missing_small_parameter_moments():
    """The aligned control catches errors hidden by rank-wide vector norms."""
    large, small = torch.ones(10_000), torch.full((2,), 0.1)
    actual = _moment_tensors(large, torch.zeros_like(small))
    expected = _moment_tensors(large, small)
    # This is why the strict control is required alongside the raw BF16 test.
    _assert_adam_moments_close(actual, expected)
    with pytest.raises(AssertionError, match="tensor="):
        _assert_pointwise_tensors_close(actual, expected)


def test_moment_comparison_has_no_absolute_tolerance_for_tiny_vectors():
    expected = _moment_tensors(torch.full((4,), 1e-12))
    with pytest.raises(AssertionError, match="exp_avg"):
        _assert_adam_moments_close(_moment_tensors(torch.zeros(4)), expected)


@pytest.mark.parametrize("error", ["keys", "shape", "nonfinite", "negative_variance"])
def test_moment_comparison_rejects_invalid_optimizer_states(error):
    expected = _moment_tensors(torch.ones(4))
    actual = copy.deepcopy(expected)
    mean, variance = ("state", 0, "exp_avg"), ("state", 0, "exp_avg_sq")
    if error == "keys":
        del actual[mean]
    elif error == "shape":
        actual[mean] = actual[mean].reshape(2, 2)
    elif error == "nonfinite":
        actual[mean][0] = float("nan")
    else:
        actual[variance][0] = -1
    with pytest.raises(AssertionError):
        _assert_adam_moments_close(actual, expected)


def test_optimizer_steps_include_fused_group_counters_and_ignore_empty_groups():
    """Both Torch Adam and fused Adam counters must advance exactly once."""
    state = [
        {"state": {0: {"step": torch.tensor(1.0)}}, "param_groups": [{"params": [0]}]},
        {
            "state": {0: {}},
            "param_groups": [{"params": [], "step": 0}, {"params": [0], "step": 1}],
        },
    ]
    assert _assert_optimizer_steps(state, 1) == {
        (0, "state", 0, "step"): 1,
        (1, "param_groups", 1, "step"): 1,
    }
    state[1]["param_groups"][1]["step"] = 2
    with pytest.raises(AssertionError, match="expected Adam step=1"):
        _assert_optimizer_steps(state, 1)


def test_lazy_adam_initialization_has_no_steps_before_first_update():
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW([parameter])
    assert _optimizer_steps(optimizer.state_dict()) == {}
    with pytest.raises(AssertionError, match="missing Adam step"):
        _assert_optimizer_steps(optimizer.state_dict(), 1)
