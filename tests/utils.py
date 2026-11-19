"""Test utilities.

Re-exports common test utilities from areal.utils.testing_utils for use
by test files that import from tests.utils.
"""

from collections.abc import Generator
from contextlib import contextmanager

from areal.utils.testing_utils import (
    get_dataset_path,
    get_model_path,
)

__all__ = ["get_dataset_path", "get_model_path"]


def _assert_adam_moments_close(
    actual: dict, expected: dict, *, case: str = "", rtol: float = 2e-2
) -> None:
    """Compare moment vectors without amplifying near-zero coordinates.

    Check m and v separately over each rank's complete optimizer-shard vector.
    Tensor-level errors are diagnostics: the rank-wide relative budget is not
    a bound on every tensor's BF16 rounding error. The aligned common-reference
    test separately checks individual coordinates, including small tensors.
    """
    import math

    import torch

    assert actual.keys() == expected.keys(), f"{case}: optimizer tensor keys differ"
    for key, a in actual.items():
        b = expected[key]
        assert a.shape == b.shape and a.dtype == b.dtype, (
            f"{case}: optimizer shape/dtype differs for {key}"
        )
        if key[-1] not in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=f"{case}: {key}")

    for moment in ("exp_avg", "exp_avg_sq"):
        squared_error = squared_actual = squared_expected = dot = 0.0
        tensor_errors = []
        numel = 0
        for key, value in actual.items():
            if key[-1] != moment:
                continue
            a_flat, b_flat = value.reshape(-1), expected[key].reshape(-1)
            local_error = local_expected = 0.0
            # Bound temporary CPU memory for large optimizer shards.
            for start in range(0, a_flat.numel(), 1 << 20):
                a = a_flat[start : start + (1 << 20)].double()
                b = b_flat[start : start + (1 << 20)].double()
                assert torch.isfinite(a).all() and torch.isfinite(b).all(), (
                    f"{case}: nonfinite {moment} for {key}"
                )
                if moment == "exp_avg_sq":
                    assert (a >= 0).all() and (b >= 0).all(), (
                        f"{case}: negative second moment for {key}"
                    )
                local_error += (a - b).square().sum().item()
                local_expected += b.square().sum().item()
                squared_actual += a.square().sum().item()
                dot += (a * b).sum().item()
            local_relative = math.sqrt(local_error) / max(
                math.sqrt(local_expected), 1e-300
            )
            tensor_errors.append(
                (
                    local_relative,
                    repr(key),
                    math.sqrt(local_error),
                    math.sqrt(local_expected),
                )
            )
            squared_error += local_error
            squared_expected += local_expected
            numel += a_flat.numel()
        assert numel, f"{case}: missing {moment}"
        error, reference = math.sqrt(squared_error), math.sqrt(squared_expected)
        relative = error / max(reference, 1e-300)
        cosine = dot / max(math.sqrt(squared_actual * squared_expected), 1e-300)
        detail = (
            f"{case}, {moment}: rank_relative_l2={relative:.6g}, "
            f"cosine={cosine:.9g}, rtol={rtol}, "
            "largest_tensor_errors=(relative_l2, key, error_norm, reference_norm): "
            f"{sorted(tensor_errors, reverse=True)[:3]}"
        )
        assert math.isfinite(error) and math.isfinite(reference), detail
        # No absolute tolerance, even when the entire moment vector is tiny.
        assert error <= rtol * reference, detail


def _assert_pointwise_tensors_close(
    actual: dict, expected: dict, *, case: str = ""
) -> None:
    """Strict aligned-reference checks also cover low-energy parameter tensors."""
    import torch

    assert actual.keys() == expected.keys(), f"{case}: tensor keys differ"
    for key in actual:
        torch.testing.assert_close(
            actual[key],
            expected[key],
            rtol=2e-2,
            atol=1e-6,
            msg=lambda error: f"{case}, tensor={key}:\n{error}",
        )


def _optimizer_steps(value, path: tuple = ()) -> dict[tuple, int]:
    """Read active Adam counters, including Python-int fused group counters."""
    import math

    import torch

    def counter(step, key):
        if isinstance(step, torch.Tensor):
            assert step.numel() == 1, f"Non-scalar optimizer step: {key}"
            step = step.item()
        assert isinstance(step, (int, float)) and not isinstance(step, bool), key
        assert math.isfinite(step) and step >= 0 and int(step) == step, key
        return int(step)

    if isinstance(value, dict) and "param_groups" in value and "state" in value:
        steps = {}
        for index, group in enumerate(value["param_groups"]):
            # Distributed optimizers retain groups with no local parameters.
            if not group["params"]:
                continue
            if "step" in group:
                key = (*path, "param_groups", index, "step")
                steps[key] = counter(group["step"], key)
            else:
                for parameter in group["params"]:
                    state = value["state"].get(parameter, {})
                    if "step" in state:
                        key = (*path, "state", parameter, "step")
                        steps[key] = counter(state["step"], key)
        return steps
    if isinstance(value, dict):
        children = value.items()
    elif isinstance(value, (list, tuple)):
        children = enumerate(value)
    else:
        return {}
    return {
        key: step
        for name, child in children
        for key, step in _optimizer_steps(child, (*path, name)).items()
    }


def _assert_optimizer_steps(
    state, expected: int, *, case: str = ""
) -> dict[tuple, int]:
    steps = _optimizer_steps(state)
    assert steps, f"{case}: missing Adam step counters"
    assert all(step == expected for step in steps.values()), (
        f"{case}: expected Adam step={expected}, got {steps}"
    )
    return steps


@contextmanager
def _common_megatron_reference_scale(
    engine, batches, model_config
) -> Generator[dict[str, float], None, None]:
    """Align a single manual-loss schedule with chunk backward scaling.

    This test-only control uses independent packed-token counts, never the
    streaming accumulation state. Match the chunk path's pre-reduction
    correction and synchronous gradient communication on the existing buckets.
    """
    import torch
    import torch.distributed as dist

    if (
        model_config.calculate_per_token_loss
        or model_config.context_parallel_size != 1
        or engine.hf_config.model_type != "qwen3"
        or engine.mcore_config.enable_mtp_training
    ):
        raise ValueError(
            "Common-reference control requires dense Qwen3, CP=1, no MTP and "
            "calculate_per_token_loss=False"
        )
    weights = torch.stack(
        [
            torch.stack([mb["cu_seqlens"][-1] for mb in batch.mbs]).sum().float()
            for batch in batches
        ]
    ).to(engine.device)
    dist.all_reduce(weights, group=engine.data_parallel_group)
    reference_weight = weights[0] * len(batches)
    total_weight = weights.sum()
    correction = reference_weight / total_weight
    original_totals = engine._compute_loss_totals
    original_finalize = model_config.finalize_model_grads_func
    original_grad_sync = model_config.grad_sync_func
    ddp_configs = {
        id(model.ddp_config): model.ddp_config
        for model in engine._get_policy_model_parts()
    }
    original_overlap = [
        (config, config.overlap_grad_reduce) for config in ddp_configs.values()
    ]

    def compute_totals(mb_list, loss_weight_fn):
        actual_weight, tokens = original_totals(mb_list, loss_weight_fn)
        torch.testing.assert_close(actual_weight, total_weight, rtol=0, atol=0)
        return reference_weight, tokens

    @torch.no_grad()
    def finalize(models, *args, **kwargs):
        # Overlap is disabled before backward, so no reduction has started.
        # Apply the correction before native DP scaling/reduction, as in the
        # chunk path, retaining the original bucket layout and optimizer step.
        seen = set()
        for model in engine._get_policy_model_parts():
            for parameter in model.parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                grad = getattr(parameter, "main_grad", None)
                if grad is not None:
                    grad.mul_(correction)
        return original_finalize(models, *args, **kwargs)

    engine._compute_loss_totals = compute_totals
    try:
        for config, _ in original_overlap:
            config.overlap_grad_reduce = False
        model_config.grad_sync_func = None
        model_config.finalize_model_grads_func = finalize
        yield {"W_ref": reference_weight.item(), "W_total": total_weight.item()}
    finally:
        model_config.finalize_model_grads_func = original_finalize
        model_config.grad_sync_func = original_grad_sync
        for config, overlap in original_overlap:
            config.overlap_grad_reduce = overlap
        engine._compute_loss_totals = original_totals


def check_gradient_accumulation(
    engine_factory, *, case="", common_reference_scale=False
):
    """Compare identical packed microbatches and check consecutive updates.

    The normal 4/8-MB case compares first-update gradient norms and Adam moments
    from identical weights, then checks the second streaming update's lifecycle.
    The aligned 8/8-MB control also compares pointwise parameters and optimizer
    tensors for both updates. Both require exact update and snapshot boundaries.
    """
    import copy
    import math
    from contextlib import nullcontext

    import torch
    from torch.distributed.tensor import DTensor

    def loss_fn(logp, _entropy, _data, **_kwargs):
        return logp.float().mean()

    def local_copy(tensor):
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
        return tensor.detach().cpu().clone()

    def gradients(engine):
        return {
            (index, name): local_copy(grad)
            for index, model in enumerate(engine._get_policy_model_parts())
            for name, parameter in model.named_parameters()
            if (grad := getattr(parameter, "main_grad", parameter.grad)) is not None
        }

    def policy_parameters(engine):
        sync = getattr(engine, "_parameter_snapshot_sync_context", nullcontext)
        with sync():
            return {
                name: local_copy(tensor)
                for name, tensor in engine._iter_policy_state_tensors()
            }

    def assert_policy_unchanged(engine, expected, context):
        sync = getattr(engine, "_parameter_snapshot_sync_context", nullcontext)
        with sync():
            names = set()
            for name, tensor in engine._iter_policy_state_tensors():
                names.add(name)
                torch.testing.assert_close(
                    local_copy(tensor),
                    expected[name],
                    rtol=0,
                    atol=0,
                    msg=lambda error: f"{context}, parameter={name}:\n{error}",
                )
        assert names == expected.keys(), f"{context}: policy tensor keys differ"

    def optimizer_tensors(value, path=(), *, clone=True):
        if isinstance(value, torch.Tensor):
            return {path: local_copy(value) if clone else value}
        if isinstance(value, dict):
            return {
                name: tensor
                for key, child in value.items()
                for name, tensor in optimizer_tensors(
                    child, (*path, key), clone=clone
                ).items()
            }
        if isinstance(value, (list, tuple)):
            return {
                name: tensor
                for key, child in enumerate(value)
                for name, tensor in optimizer_tensors(
                    child, (*path, key), clone=clone
                ).items()
            }
        return {}

    def optimizer_state(optimizer):
        # MCore's distributed state_dict excludes the sharded Adam moments.
        if hasattr(optimizer, "chained_optimizers"):
            return [optimizer_state(child) for child in optimizer.chained_optimizers]
        if hasattr(optimizer, "optimizer"):
            return optimizer_state(optimizer.optimizer)
        return optimizer.state_dict()

    reference = []
    reference_scales = []
    engines = []
    initial_reference_parameters = None
    try:
        for streaming in (False, True):
            engine = engine_factory()
            original_step = engine.optimizer_step
            engines.append((engine, original_step))
            optimizer_calls = 0

            def counted_step(_step=original_step):
                nonlocal optimizer_calls
                optimizer_calls += 1
                return _step()

            engine.optimizer_step = counted_step
            assert all(
                step == 0
                for step in _optimizer_steps(optimizer_state(engine.optimizer)).values()
            ), f"{case}: nonzero initial Adam step"
            initial_state = optimizer_tensors(
                optimizer_state(engine.optimizer), clone=False
            )
            for key, tensor in initial_state.items():
                if key[-1] in ("exp_avg", "exp_avg_sq"):
                    if isinstance(tensor, DTensor):
                        tensor = tensor.to_local()
                    assert tensor.count_nonzero().item() == 0, (
                        f"{case}: nonzero initial Adam moment {key}"
                    )
            del initial_state
            if streaming:
                engine.capture_parameter_snapshot()
            # Raw BF16 Adam updates can diverge at near-zero coordinates. Their
            # second windows would then compare gradients at different weights.
            for window in range(2 if streaming or common_reference_scale else 1):
                window_case = f"{case}, window={window}"
                if streaming and common_reference_scale:
                    window_case += f", reference_scale={reference_scales[window]}"
                initial_parameters = policy_parameters(engine)
                initial_steps = _optimizer_steps(optimizer_state(engine.optimizer))
                if window == 0:
                    if streaming:
                        assert_policy_unchanged(
                            engine,
                            initial_reference_parameters,
                            f"{case}: engines must start from identical weights",
                        )
                        initial_reference_parameters = None
                    else:
                        initial_reference_parameters = initial_parameters
                chunks = []
                head = engine.current_data_parallel_head()
                for width in (32, 96):
                    lengths = (
                        width - (torch.arange(8, device=engine.device) + head) % 11
                    )
                    mask = (
                        torch.arange(width, device=engine.device)[None, :]
                        < lengths[:, None]
                    )
                    tokens = (
                        torch.arange(8 * width, device=engine.device).reshape(8, width)
                        + head
                        + window
                    ) % 100 + 3
                    chunks.append(
                        dict(
                            input_ids=tokens.masked_fill(~mask, 0), attention_mask=mask
                        )
                    )
                engine.train()
                if not streaming:
                    # Fix packing to the same MBs used by separate chunk calls.
                    original_prepare = engine._prepare_mb_list
                    lists = [original_prepare(chunk) for chunk in chunks]
                    expected_counts = [8, 8] if common_reference_scale else [4, 8]
                    assert [len(batch) for batch in lists] == expected_counts, (
                        f"{window_case}: expected chunk MB counts {expected_counts}"
                    )
                    merged = copy.copy(lists[0])
                    for name in (
                        "mbs",
                        "group_lens",
                        "padded_mbs",
                        "padding_lengths",
                        "padded_to_lengths",
                        "align_to_lengths",
                        "old_cu_seqlens_list",
                    ):
                        values = [getattr(batch, name) for batch in lists]
                        setattr(
                            merged, name, None if values[0] is None else sum(values, [])
                        )
                    merged._max_seqlen = None
                    engine._prepare_mb_list = lambda _input: merged
                    try:
                        if common_reference_scale:
                            from megatron.core.utils import get_model_config

                            reference_context = _common_megatron_reference_scale(
                                engine, lists, get_model_config(engine.model[0])
                            )
                        else:
                            reference_context = nullcontext()
                        with reference_context as scale:
                            reference_scales.append(scale)
                            stats = engine.train_batch(
                                chunks[0],
                                loss_fn=loss_fn,
                                loss_weight_fn=lambda data: data["cu_seqlens"][-1],
                            )
                    finally:
                        engine._prepare_mb_list = original_prepare
                else:
                    for index, chunk in enumerate(chunks):
                        stats = engine.train_batch(
                            chunk,
                            loss_fn=loss_fn,
                            loss_weight_fn=lambda data: data["cu_seqlens"][-1],
                            accumulation_steps=2,
                            accumulation_index=index,
                        )
                        if index == 0:
                            assert stats == {}
                            assert optimizer_calls == window, (
                                f"{window_case}: optimizer stepped before the last chunk"
                            )
                            assert (
                                _optimizer_steps(optimizer_state(engine.optimizer))
                                == initial_steps
                            )
                            assert_policy_unchanged(
                                engine, initial_parameters, window_case
                            )
                            pending = gradients(engine)
                            assert pending, (
                                "First chunk must leave gradients to accumulate"
                            )
                            with engine.parameter_snapshot_context():
                                engine.eval()
                                engine.forward(
                                    input_=chunk, aggregate_fn=lambda xs: torch.cat(xs)
                                )
                            restored = gradients(engine)
                            assert pending.keys() == restored.keys()
                            for key in pending:
                                torch.testing.assert_close(
                                    pending[key],
                                    restored[key],
                                    rtol=0,
                                    atol=0,
                                    msg=lambda error: (
                                        f"{window_case}, snapshot gradient "
                                        f"{key}:\n{error}"
                                    ),
                                )
                            del pending, restored
                            assert_policy_unchanged(
                                engine,
                                initial_parameters,
                                f"{window_case}, snapshot restore",
                            )
                            assert (
                                _optimizer_steps(optimizer_state(engine.optimizer))
                                == initial_steps
                            )
                            engine.train()
                assert stats["update_successful"] == 1
                assert math.isfinite(stats["grad_norm"]) and stats["grad_norm"] > 0
                assert optimizer_calls == window + 1, (
                    f"{window_case}: expected exactly one optimizer call per window"
                )
                parameters = policy_parameters(engine)
                assert parameters.keys() == initial_parameters.keys(), window_case
                for name, tensor in parameters.items():
                    assert torch.isfinite(tensor).all(), (
                        f"{window_case}: nonfinite parameter={name}"
                    )
                    assert tensor.shape == initial_parameters[name].shape
                    assert tensor.dtype == initial_parameters[name].dtype
                assert any(
                    not torch.equal(tensor, initial_parameters[name])
                    for name, tensor in parameters.items()
                ), f"{window_case}: optimizer did not update model parameters"
                del initial_parameters
                raw_state = optimizer_state(engine.optimizer)
                steps = _assert_optimizer_steps(raw_state, window + 1, case=window_case)
                state = optimizer_tensors(raw_state)
                for moment in ("exp_avg", "exp_avg_sq"):
                    assert any(moment in key for key in state), f"Missing {moment}"
                if not streaming:
                    # Only the strict control needs full reference model copies.
                    reference.append(
                        (
                            parameters if common_reference_scale else None,
                            state,
                            steps,
                            stats,
                        )
                    )
                elif common_reference_scale or window == 0:
                    (
                        expected_parameters,
                        expected_state,
                        expected_steps,
                        expected_stats,
                    ) = reference[window]
                    assert steps == expected_steps, f"{window_case}: Adam steps differ"
                    assert (
                        stats["num_micro_batches"]
                        == expected_stats["num_micro_batches"]
                    )
                    torch.testing.assert_close(
                        torch.tensor(stats["grad_norm"]),
                        torch.tensor(expected_stats["grad_norm"]),
                        rtol=2e-2,
                        atol=1e-5,
                        msg=lambda error: (
                            f"{window_case}, pre-clipping grad_norm:\n{error}"
                        ),
                    )
                    if common_reference_scale:
                        for kind, actual, expected in (
                            ("parameter", parameters, expected_parameters),
                            ("optimizer", state, expected_state),
                        ):
                            _assert_pointwise_tensors_close(
                                actual, expected, case=f"{window_case}, {kind}"
                            )
                    else:
                        # BF16 rounding can flip negligible gradients; Adam then
                        # turns those into opposite, almost-lr-sized updates.
                        # Compare gradient-sensitive moments, not model weights.
                        _assert_adam_moments_close(
                            state, expected_state, case=window_case
                        )
                del parameters, state, raw_state
            if streaming:
                engine.clear_parameter_snapshot()
    finally:
        for engine, original_step in engines:
            engine.optimizer_step = original_step
            engine.destroy()
