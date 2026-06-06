"""Torchrun-backed DTA engine step tests."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import AutoConfig

from tests.experimental.dta.engine_step_case import EngineStepCase
from tests.experimental.dta.sequence_data import build_cot_token_sequences

from areal.api.cli_args import MicroBatchSpec
from areal.experimental.dta import wrapper as dta_wrapper
from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports

RUNNER = "tests/experimental/dta/torchrun/run_engine_step.py"

_CUDA_AVAILABLE = torch.cuda.is_available()


def _run_engine_step(case: EngineStepCase) -> dict[str, Any]:
    if case.master_port is None:
        case = replace(case, master_port=find_free_ports(1)[0])
    payload_path = Path(case.payload_path)
    case_config = payload_path.with_suffix(".case.json")
    case.dump(case_config)
    cmd = [
        "torchrun",
        f"--nproc_per_node={case.n_gpus}",
        f"--nnodes={case.nnodes}",
        f"--master-addr={case.master_addr}",
        f"--master_port={case.master_port}",
        RUNNER,
        f"--case-config={case_config}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise AssertionError(
            f"torchrun failed for mode={case.mode}, dtype={case.dtype}\n"
            f"STDOUT:\n{exc.stdout}\nSTDERR:\n{exc.stderr}"
        ) from exc

    return torch.load(payload_path, map_location="cpu", weights_only=False)


def _save_cot_sequence_data(case: EngineStepCase) -> None:
    model_path = case.resolve_model_path()
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab_size = int(getattr(model_config, "vocab_size"))

    rng_state = torch.random.get_rng_state()
    torch.manual_seed(case.sequence_seed)
    try:
        sequences = build_cot_token_sequences(
            vocab_size,
            system_prompt_length=case.cot_system_prompt_length,
            thinking_token_length=case.cot_thinking_token_length,
            response_token_length=case.cot_response_token_length,
            turns=case.cot_turns,
        )
    finally:
        torch.random.set_rng_state(rng_state)

    sequence_data_path = Path(case.sequence_data_path)
    sequence_data_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vocab_size": vocab_size,
            "sequence_metadata": case.cot_sequence_metadata(),
            "sequences": sequences,
        },
        sequence_data_path,
    )


def _assert_finite_payload(payload: dict[str, Any]) -> None:
    assert float(payload["stats"]["update_successful"]) == 1.0
    grad_norm = torch.tensor(float(payload["stats"]["grad_norm"]))
    torch.testing.assert_close(grad_norm, grad_norm)
    for group_name in ("grads",):
        group = payload[group_name]
        assert group, f"{payload['mode']} produced no {group_name}"
        for name, tensor in group.items():
            assert torch.isfinite(tensor).all().item(), (
                f"{payload['mode']} {group_name} {name} non-finite"
            )


def _assert_tensor_groups_elementwise_close(
    baseline: dict[str, torch.Tensor],
    dta: dict[str, torch.Tensor],
    *,
    group_name: str,
    rtol: float,
    atol: float,
) -> int:
    baseline_names = set(baseline)
    dta_names = set(dta)
    assert baseline_names == dta_names, (
        f"{group_name} parameter-name mismatch: "
        f"baseline_only={sorted(baseline_names - dta_names)[:16]}, "
        f"dta_only={sorted(dta_names - baseline_names)[:16]}"
    )

    for name in sorted(baseline_names):
        b_tensor = baseline[name]
        d_tensor = dta[name]
        assert b_tensor.shape == d_tensor.shape, (
            f"{group_name} shape mismatch for {name}: "
            f"baseline={tuple(b_tensor.shape)}, dta={tuple(d_tensor.shape)}"
        )
        torch.testing.assert_close(
            d_tensor,
            b_tensor,
            rtol=rtol,
            atol=atol,
            msg=lambda msg, n=name: f"{group_name} tensor mismatch for {n}: {msg}",
        )

    return len(baseline_names)


class TinyEngineConfig:
    mb_spec = MicroBatchSpec(n_mbs=1, max_tokens_per_mb=32)


def test_dta_prepare_mb_list_creates_one_microbatch_per_sequence() -> None:
    """DTA keeps sequence-level independence when building micro-batches."""
    wrapper = object.__new__(dta_wrapper.DTAWrapper)
    wrapper.engine = SimpleNamespace(config=TinyEngineConfig())

    batch = {
        "input_ids": torch.tensor(
            [
                [11, 12, 13, 0],
                [21, 22, 0, 0],
                [31, 32, 33, 34],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
        "loss_mask": torch.ones((3, 4), dtype=torch.float32),
    }

    mb_list = dta_wrapper.DTAWrapper.prepare_mb_list(wrapper, batch)

    assert len(mb_list.mbs) == batch["input_ids"].shape[0]
    assert mb_list.group_lens == [3, 2, 4]
    for mb in mb_list.mbs:
        assert mb["input_ids"].shape[0] == 1
        assert mb["attention_mask"].shape[0] == 1


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.multi_gpu
@pytest.mark.slow
def test_dta_engine_fp32_grad_match_baseline_and_adam_step_succeeds(
    tmp_path: Path,
):
    """Compare fp32 forward logprobs and gradients between dense Archon and DTA."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")

    sequence_data_path = tmp_path / "engine_step_sequences.pt"
    baseline_case = EngineStepCase(
        mode="baseline",
        dtype="float32",
        payload_path=str(tmp_path / "baseline.pt"),
        sequence_data_path=str(sequence_data_path),
    )
    _save_cot_sequence_data(baseline_case)
    dta_case = replace(
        baseline_case,
        mode="dta",
        payload_path=str(tmp_path / "dta.pt"),
        gradient_checkpointing=False,
    )
    baseline = _run_engine_step(baseline_case)
    dta = _run_engine_step(dta_case)

    _assert_finite_payload(baseline)
    _assert_finite_payload(dta)

    assert baseline["forward_logprobs"].shape == dta["forward_logprobs"].shape
    forward_mask = baseline["forward_loss_mask"]
    torch.testing.assert_close(
        dta["forward_logprobs"][forward_mask],
        baseline["forward_logprobs"][forward_mask],
        rtol=baseline_case.forward_rtol,
        atol=baseline_case.forward_atol,
    )

    torch.testing.assert_close(
        torch.tensor(float(dta["stats"]["grad_norm"])),
        torch.tensor(float(baseline["stats"]["grad_norm"])),
        rtol=baseline_case.grad_norm_rtol,
        atol=baseline_case.grad_norm_atol,
    )
    torch.testing.assert_close(
        torch.tensor(float(dta["stats"]["global_loss"])),
        torch.tensor(float(baseline["stats"]["global_loss"])),
        rtol=baseline_case.forward_rtol,
        atol=baseline_case.forward_atol,
    )
    assert (
        _assert_tensor_groups_elementwise_close(
            baseline["grads"],
            dta["grads"],
            group_name="grads",
            rtol=baseline_case.grad_rtol,
            atol=baseline_case.grad_atol,
        )
        > 0
    )


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.multi_gpu
@pytest.mark.slow
def test_dta_engine_bf16_train_step_smoke(tmp_path: Path):
    """Smoke check that the DTA engine runs one bf16 train step."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")

    case = EngineStepCase(
        mode="dta",
        dtype="bfloat16",
        payload_path=str(tmp_path / "dta_bf16.pt"),
        sequence_data_path=str(tmp_path / "bf16_engine_step_sequences.pt"),
        gradient_checkpointing=False,
    )
    _save_cot_sequence_data(case)
    payload = _run_engine_step(case)

    _assert_finite_payload(payload)
