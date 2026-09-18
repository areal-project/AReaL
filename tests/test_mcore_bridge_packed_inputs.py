# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for ModelScope's wrapper-packed input path.

Only Megatron's metadata and rank API are stubbed. AReaL packing, partitioning,
payload extraction, and output layout execute normally; GPU model numerics
remain covered by distributed precision-alignment experiments.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from areal.engine.core.model import SequencePackingMode, resolve_sequence_packing_mode


@pytest.fixture
def packed_forward(monkeypatch):
    mpu = SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: 1,
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        is_pipeline_last_stage=lambda **_kwargs: True,
    )
    core = ModuleType("megatron.core")
    core.parallel_state = mpu
    metadata = ModuleType("megatron.core.packed_seq_params")
    metadata.PackedSeqParams = SimpleNamespace
    for name, module in (
        ("megatron", ModuleType("megatron")),
        ("megatron.core", core),
        ("megatron.core.packed_seq_params", metadata),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    path = (
        Path(__file__).resolve().parents[1]
        / "areal/engine/megatron_utils/packed_context_parallel.py"
    )
    spec = importlib.util.spec_from_file_location("_test_packed_inputs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("language_model_only", [False, True])
def test_qwen_text_packing_partitions_ids_and_positions_once(
    packed_forward, monkeypatch, cp_size, language_model_only
):
    module = packed_forward
    assert (
        resolve_sequence_packing_mode("qwen4_exp", "mcore-bridge")
        == SequencePackingMode.WRAPPER_THD
    )
    ids = torch.arange(24, dtype=torch.long) + 10
    cu_seqlens = torch.tensor([0, 8, 24], dtype=torch.int32)
    positions = torch.cat([torch.arange(8), torch.arange(16)])
    monkeypatch.setattr(module.mpu, "get_context_parallel_world_size", lambda: cp_size)
    outputs = []
    for cp_rank in range(cp_size):
        monkeypatch.setattr(
            module.mpu, "get_context_parallel_rank", lambda rank=cp_rank: rank
        )
        model = MagicMock(
            side_effect=lambda **inputs: (
                inputs["input_ids"] * 7 + inputs["position_ids"]
            ).unsqueeze(-1)
        )
        output = module.packed_context_parallel_forward(
            model,
            {"input_ids": ids, "cu_seqlens": cu_seqlens},
            gather_cp_output=False,
            is_vision_model=True,
            use_wrapper_packed_seq=True,
            language_model_only=language_model_only,
        )
        inputs = model.call_args.kwargs
        if cp_size == 1:
            expected_indices = torch.arange(24)
        else:
            # Megatron's balanced CP layout assigns chunk r and its mirror
            # independently within each packed document.
            expected_indices = torch.cat(
                [
                    doc.reshape(2 * cp_size, -1)[
                        [cp_rank, 2 * cp_size - cp_rank - 1]
                    ].flatten()
                    for doc in (torch.arange(8), torch.arange(8, 24))
                ]
            )
        assert inputs["input_ids"].shape == (1, 24 // cp_size)
        assert inputs["position_ids"].shape == inputs["input_ids"].shape
        assert inputs["attention_mask"] is None
        torch.testing.assert_close(
            inputs["input_ids"][0], ids[expected_indices], atol=0, rtol=0
        )
        torch.testing.assert_close(
            inputs["position_ids"][0], positions[expected_indices], atol=0, rtol=0
        )
        metadata = inputs["packed_seq_params"]
        assert metadata.qkv_format == "thd"
        assert metadata.max_seqlen_q == 16
        torch.testing.assert_close(metadata.cu_seqlens_q, cu_seqlens, atol=0, rtol=0)
        outputs.append(output.squeeze(-1))

    indices = module._build_cp_reassemble_indices(cu_seqlens, cp_size)
    reconstructed = torch.cat(outputs)[indices]
    torch.testing.assert_close(reconstructed, ids * 7 + positions, atol=0, rtol=0)


@pytest.mark.parametrize("position_shape", ["flat", "batched"])
def test_wrapper_packing_preserves_supplied_text_positions(
    packed_forward, monkeypatch, position_shape
):
    module = packed_forward
    monkeypatch.setattr(module.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(module.mpu, "get_context_parallel_rank", lambda: 1)
    model = MagicMock(return_value=torch.zeros(1, 4, 1))
    positions = torch.tensor([20, 21, 22, 23, 40, 41, 42, 43], dtype=torch.int32)
    module.packed_context_parallel_forward(
        model,
        {
            "input_ids": torch.arange(8),
            "cu_seqlens": torch.tensor([0, 4, 8], dtype=torch.int32),
            "position_ids": positions if position_shape == "flat" else positions[None],
        },
        gather_cp_output=False,
        is_vision_model=True,
        use_wrapper_packed_seq=True,
    )
    torch.testing.assert_close(
        model.call_args.kwargs["position_ids"],
        torch.tensor([[21, 22, 41, 42]]),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("key", ["pixel_values", "pixel_values_videos"])
def test_language_only_packing_rejects_real_vision_payload(packed_forward, key):
    model = MagicMock()
    mb = {"multi_modal_input": [{key: torch.ones(2, 4)}]}
    padded_mb = {
        **mb,
        "input_ids": torch.arange(8),
        "cu_seqlens": torch.tensor([0, 8], dtype=torch.int32),
    }
    packed_forward.extract_vision_from_multi_modal(mb, padded_mb)
    with pytest.raises(ValueError, match="language_model_only=True"):
        packed_forward.packed_context_parallel_forward(
            model,
            padded_mb,
            is_vision_model=True,
            use_wrapper_packed_seq=True,
            language_model_only=True,
        )
    model.assert_not_called()
    assert key not in mb


def test_wrapper_vision_requires_real_mrope_even_with_text_positions(packed_forward):
    model = MagicMock()
    with pytest.raises(ValueError, match="model-specific mRoPE"):
        packed_forward.packed_context_parallel_forward(
            model,
            {
                "input_ids": torch.arange(8),
                "cu_seqlens": torch.tensor([0, 8], dtype=torch.int32),
                "position_ids": torch.arange(8),
                "pixel_values": torch.ones(2, 4),
            },
            is_vision_model=True,
            use_wrapper_packed_seq=True,
        )
    model.assert_not_called()


def test_model_owned_thd_keeps_full_ids_and_video_payload(packed_forward):
    module = packed_forward
    model = MagicMock(return_value=torch.ones(1, 8, 2))
    pixels = torch.ones(2, 4)
    module.packed_context_parallel_forward(
        model,
        {
            "input_ids": torch.arange(8),
            "cu_seqlens": torch.tensor([0, 4, 8], dtype=torch.int32),
            "pixel_values_videos": pixels,
        },
        gather_cp_output=False,
        is_vision_model=True,
        use_model_packed_seq=True,
    )
    inputs = model.call_args.kwargs
    torch.testing.assert_close(inputs["input_ids"], torch.arange(8).reshape(2, 4))
    assert inputs["position_ids"] is None
    assert inputs["attention_mask"].all()
    assert inputs["pixel_values_videos"] is pixels


def test_wrapper_packing_rejects_already_cp_partitioned_positions(
    packed_forward, monkeypatch
):
    monkeypatch.setattr(
        packed_forward.mpu, "get_context_parallel_world_size", lambda: 2
    )
    with pytest.raises(ValueError, match="before context parallel partitioning"):
        packed_forward.packed_context_parallel_forward(
            MagicMock(),
            {
                "input_ids": torch.arange(8),
                "cu_seqlens": torch.tensor([0, 8], dtype=torch.int32),
                "position_ids": torch.arange(4),
            },
            is_vision_model=True,
            use_wrapper_packed_seq=True,
        )


def test_wrapper_packing_rejects_conflicting_contracts(packed_forward):
    with pytest.raises(ValueError, match="mutually exclusive"):
        packed_forward.packed_context_parallel_forward(
            MagicMock(),
            {"input_ids": torch.arange(8)},
            use_wrapper_packed_seq=True,
            use_model_packed_seq=True,
        )


def test_wrapper_mtp_preserves_thd_labels_without_padding_mask(packed_forward):
    model = MagicMock(return_value=torch.zeros(1, 8, 1))
    labels = torch.arange(8)
    mask = torch.tensor([1, 1, 1, 0, 1, 1, 1, 0], dtype=torch.float32)
    packed_forward.packed_context_parallel_forward(
        model,
        {
            "input_ids": torch.arange(8),
            "cu_seqlens": torch.tensor([0, 4, 8], dtype=torch.int32),
            "mtp_kwargs": {"mtp_labels": labels, "mtp_loss_mask": mask},
        },
        gather_cp_output=False,
        is_vision_model=True,
        use_wrapper_packed_seq=True,
    )
    kwargs = model.call_args.kwargs
    assert kwargs["attention_mask"] is None
    torch.testing.assert_close(kwargs["mtp_kwargs"]["mtp_labels"].reshape(-1), labels)
    torch.testing.assert_close(kwargs["mtp_kwargs"]["mtp_loss_mask"].reshape(-1), mask)
