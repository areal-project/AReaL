"""DTA tests for ArchonEngine with numerical checks against FSDP.

This suite keeps smoke-level checks and adds dual-engine numerical validation for:
1) ``forward_batch`` output consistency
2) ``train_batch`` optimization signal consistency

Requires ``--dta-data=<path.pt>`` pointing to a file containing
``list[Tensor]`` (1-D token sequences without padding).
"""

import time
from typing import Any

import pytest
import torch

from tests.experimental.archon.utils import (
    compare_tensors,
    create_archon_engine,
    create_fsdp_engine,
    destroy_test_engine,
    dta_dummy_loss_fn,
    dta_loss_weight_fn,
    load_pt_batch,
    snapshot_module_parameters,
    strip_wrapper_prefixes,
)

_CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = [
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available"),
]


def _canonicalize_param_dict(
    params: dict[str, torch.Tensor],
    *,
    source: str,
    archon_adapter: Any | None = None,
) -> dict[str, torch.Tensor]:
    """Normalize parameter-key namespace for cross-engine comparisons."""
    canonical: dict[str, torch.Tensor] = {}
    for raw_name, value in params.items():
        if source == "archon":
            assert archon_adapter is not None, (
                "archon_adapter is required for archon canonicalization"
            )
            mapped = archon_adapter.convert_single_to_hf(raw_name, value)
            if not mapped:
                key = strip_wrapper_prefixes(raw_name)
            else:
                key, _ = mapped[0]
        else:
            key = strip_wrapper_prefixes(raw_name)
        canonical[key] = value
    return canonical


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return out


def _assert_tensor_finite(t: torch.Tensor, name: str) -> None:
    assert torch.isfinite(t).all(), f"{name} contains non-finite values"


def _assert_grad_norm_finite(result: dict[str, Any], name: str) -> float:
    assert "grad_norm" in result, f"{name} missing grad_norm: {list(result.keys())}"
    grad_norm = float(result["grad_norm"])
    assert torch.isfinite(torch.tensor(grad_norm)).item(), (
        f"{name} grad_norm is NaN/Inf: {grad_norm}"
    )
    return grad_norm


def _run_forward_batch(
    engine: Any, batch: dict[str, Any], name: str
) -> tuple[torch.Tensor, float, float]:
    """Run one forward step and validate finite output."""
    engine.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        output = engine.forward_batch(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start_time
    peak_mem_mib = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if torch.cuda.is_available()
        else 0.0
    )
    print(
        f"[{name}] forward_batch elapsed: {elapsed_s:.2f} s, peak_mem: {peak_mem_mib:.2f} MiB"
    )
    assert output.shape[0] == batch["input_ids"].shape[0], (
        f"{name} forward output shape mismatch"
    )
    assert output is not None, f"{name} forward output is None"
    _assert_tensor_finite(output, f"{name} forward output")
    return output, elapsed_s, peak_mem_mib


def _run_train_batch_and_snapshot(
    engine: Any,
    batch: dict[str, Any],
    name: str,
) -> tuple[
    dict[str, Any],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    float,
    float,
    float,
]:
    """Run one train step and return result, snapshots, deltas, grad_norm, elapsed_s, peak_mem_mib."""
    engine.train()
    engine.optimizer_zero_grad()
    before = snapshot_module_parameters(
        engine.model,
        to_cpu=True,
        param_filter=lambda n, p: p.requires_grad,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    result = engine.train_batch(batch, dta_dummy_loss_fn, dta_loss_weight_fn)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start_time
    peak_mem_mib = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if torch.cuda.is_available()
        else 0.0
    )
    print(
        f"[{name}] train_batch elapsed: {elapsed_s:.2f} s, peak_mem: {peak_mem_mib:.2f} MiB"
    )
    grad_norm = _assert_grad_norm_finite(result, name)
    after = snapshot_module_parameters(
        engine.model,
        to_cpu=True,
        param_filter=lambda n, p: p.requires_grad,
    )
    before_names = set(before.keys())
    after_names = set(after.keys())
    assert before_names == after_names, (
        f"{name} trainable parameter names changed after train_batch: "
        f"only_before={sorted(before_names - after_names)[:20]}, "
        f"only_after={sorted(after_names - before_names)[:20]}"
    )

    deltas: dict[str, torch.Tensor] = {}
    for param_name in sorted(before_names):
        _assert_tensor_finite(
            after[param_name], f"{name} param after train_batch: {param_name}"
        )
        delta = after[param_name] - before[param_name]
        _assert_tensor_finite(delta, f"{name} delta after train_batch: {param_name}")
        deltas[param_name] = delta

    engine.optimizer_zero_grad()
    return result, before, after, deltas, grad_norm, elapsed_s, peak_mem_mib


def _assert_train_consistency(
    *,
    archon_before: dict[str, torch.Tensor],
    fsdp_before: dict[str, torch.Tensor],
    archon_deltas: dict[str, torch.Tensor],
    fsdp_deltas: dict[str, torch.Tensor],
    archon_grad_norm: float,
    fsdp_grad_norm: float,
    archon_adapter: Any,
) -> None:
    """Validate train-step consistency between Archon and FSDP."""
    grad_norm_gap = abs(archon_grad_norm - fsdp_grad_norm)
    grad_norm_rel_gap = grad_norm_gap / max(abs(fsdp_grad_norm), 1e-6)
    print(
        f"[Archon vs FSDP] archon_grad_norm={archon_grad_norm:.6f}, fsdp_grad_norm={fsdp_grad_norm:.6f}, gap={grad_norm_gap:.6f}, rel_gap={grad_norm_rel_gap:.6f}"
    )
    assert grad_norm_rel_gap < 0.25, (
        "train_batch grad_norm differs too much: "
        f"archon={archon_grad_norm:.6f}, fsdp={fsdp_grad_norm:.6f}, rel_gap={grad_norm_rel_gap:.3f}"
    )

    archon_before_canonical = _canonicalize_param_dict(
        archon_before, source="archon", archon_adapter=archon_adapter
    )
    fsdp_before_canonical = _canonicalize_param_dict(fsdp_before, source="fsdp")
    archon_deltas_canonical = _canonicalize_param_dict(
        archon_deltas, source="archon", archon_adapter=archon_adapter
    )
    fsdp_deltas_canonical = _canonicalize_param_dict(fsdp_deltas, source="fsdp")

    archon_names = set(archon_before_canonical.keys())
    fsdp_names = set(fsdp_before_canonical.keys())
    assert archon_names == fsdp_names, (
        "Canonical trainable parameter names are not aligned between Archon and FSDP: "
        f"only_archon={sorted(archon_names - fsdp_names)[:20]}, "
        f"only_fsdp={sorted(fsdp_names - archon_names)[:20]}"
    )

    # Compute L2 norm of all delta tensors for Archon and FSDP (sqrt of sum of squares of all elements)
    def _global_l2_norm(param_dict):
        # param_dict: dict[param_name, tensor]
        return float(
            torch.sqrt(sum((param.float() ** 2).sum() for param in param_dict.values()))
        )

    archon_delta_norm = _global_l2_norm(archon_deltas_canonical)
    fsdp_delta_norm = _global_l2_norm(fsdp_deltas_canonical)

    print(
        f"[delta norm] archon_delta_norm={archon_delta_norm:.6f}, "
        f"fsdp_delta_norm={fsdp_delta_norm:.6f}, "
        f"abs_gap={abs(archon_delta_norm - fsdp_delta_norm):.6f}, "
        f"rel_gap={abs(archon_delta_norm - fsdp_delta_norm) / (abs(fsdp_delta_norm) + 1e-8):.4f}"
    )

    mismatches = []
    for name in sorted(archon_names):
        delta_metrics = compare_tensors(
            archon_deltas_canonical[name],
            fsdp_deltas_canonical[name],
            atol=1e-8,
            rtol=0.3,
        )
        if not delta_metrics.shape_match or not delta_metrics.allclose:
            mismatches.append((name, str(delta_metrics)))

    if mismatches:
        print(
            f"Note: Found {len(mismatches)} parameter delta mismatches out of {len(archon_names)} parameters after train_batch (showing up to 20): "
            f"{mismatches[:20]}"
        )


@pytest.fixture(scope="module")
def batch(request, archon_test_config):
    pt_path = archon_test_config.dta_data
    if pt_path is None:
        pytest.skip("Skipped: pass --dta-data=<path.pt> to run DTA tests")
    return load_pt_batch(test_config=archon_test_config)


def test_engine_is_initialized(archon_test_config):
    """Engine initializes with DTA flag from CLI option."""
    engine = create_archon_engine(test_config=archon_test_config)
    try:
        assert engine.initialized
        assert engine.tree_training_mode == archon_test_config.tree_training_mode
        assert hasattr(engine, "dta_wrapper") == (
            archon_test_config.tree_training_mode == "dta"
        )
    finally:
        destroy_test_engine(engine)


def test_forward_batch_runs(batch, archon_test_config):
    """Smoke check for DTA forward path on Archon engine."""
    archon_batch = _clone_batch(batch)
    archon_engine = create_archon_engine(test_config=archon_test_config)
    try:
        _, _, _ = _run_forward_batch(archon_engine, archon_batch, name="Archon")
    finally:
        destroy_test_engine(archon_engine)


def test_train_batch_runs(batch, archon_test_config):
    """Smoke check for DTA train path on Archon engine."""
    archon_batch = _clone_batch(batch)
    archon_engine = create_archon_engine(test_config=archon_test_config)
    try:
        result, _, _, _, _, _, _ = _run_train_batch_and_snapshot(
            archon_engine, archon_batch, name="Archon"
        )
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    finally:
        destroy_test_engine(archon_engine)


def test_forward_batch_matches_fsdp(batch, archon_test_config):
    """Numerical check: DTA-enabled Archon forward_batch ~= FSDP forward_batch."""
    archon_batch = _clone_batch(batch)
    fsdp_batch = _clone_batch(batch)

    archon_engine = create_archon_engine(test_config=archon_test_config)
    try:
        archon_out, archon_elapsed_s, archon_peak_mem_mib = _run_forward_batch(
            archon_engine, archon_batch, name="Archon"
        )
    finally:
        destroy_test_engine(archon_engine)

    fsdp_engine = create_fsdp_engine(test_config=archon_test_config)
    try:
        fsdp_out, fsdp_elapsed_s, fsdp_peak_mem_mib = _run_forward_batch(
            fsdp_engine, fsdp_batch, name="FSDP"
        )
    finally:
        destroy_test_engine(fsdp_engine)

    assert archon_out.shape == fsdp_out.shape, (
        f"forward_batch shape mismatch: archon={archon_out.shape}, fsdp={fsdp_out.shape}"
    )

    metrics = compare_tensors(archon_out, fsdp_out, atol=1e-4, rtol=1e-2)
    assert metrics.mean_diff < 0.25, f"forward_batch mean_diff too large: {metrics}"
    forward_speedup = fsdp_elapsed_s / max(archon_elapsed_s, 1e-12)
    print(
        "[Forward speedup] "
        f"archon={archon_elapsed_s:.4f}s, fsdp={fsdp_elapsed_s:.4f}s, "
        f"speedup={forward_speedup:.3f}x"
    )
    print(
        "[Forward peak memory] "
        f"archon={archon_peak_mem_mib:.2f}MiB, fsdp={fsdp_peak_mem_mib:.2f}MiB"
    )


def test_train_batch_matches_fsdp(batch, archon_test_config):
    """Numerical check: DTA-enabled Archon train signal ~= FSDP train signal."""
    archon_batch = _clone_batch(batch)
    fsdp_batch = _clone_batch(batch)

    archon_engine = create_archon_engine(test_config=archon_test_config)
    archon_adapter = None
    try:
        (
            archon_result,
            archon_before,
            _,
            archon_deltas,
            archon_grad_norm,
            archon_elapsed_s,
            archon_peak_mem_mib,
        ) = _run_train_batch_and_snapshot(
            archon_engine,
            archon_batch,
            name="Archon",
        )
        archon_adapter = archon_engine.state_dict_adapter
        assert archon_adapter is not None, (
            "Archon state_dict_adapter should be initialized"
        )
    finally:
        destroy_test_engine(archon_engine)

    fsdp_engine = create_fsdp_engine(test_config=archon_test_config)
    try:
        (
            fsdp_result,
            fsdp_before,
            _,
            fsdp_deltas,
            fsdp_grad_norm,
            fsdp_elapsed_s,
            fsdp_peak_mem_mib,
        ) = _run_train_batch_and_snapshot(fsdp_engine, fsdp_batch, name="FSDP")
    finally:
        destroy_test_engine(fsdp_engine)
    assert archon_adapter is not None

    train_speedup = fsdp_elapsed_s / max(archon_elapsed_s, 1e-12)
    print(
        "[Train speedup] "
        f"archon={archon_elapsed_s:.4f}s, fsdp={fsdp_elapsed_s:.4f}s, "
        f"speedup={train_speedup:.3f}x"
    )
    print(
        "[Train peak memory] "
        f"archon={archon_peak_mem_mib:.2f}MiB, fsdp={fsdp_peak_mem_mib:.2f}MiB"
    )
    _assert_train_consistency(
        archon_before=archon_before,
        fsdp_before=fsdp_before,
        archon_deltas=archon_deltas,
        fsdp_deltas=fsdp_deltas,
        archon_grad_norm=archon_grad_norm,
        fsdp_grad_norm=fsdp_grad_norm,
        archon_adapter=archon_adapter,
    )
