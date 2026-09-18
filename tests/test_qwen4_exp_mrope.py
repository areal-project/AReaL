# SPDX-License-Identifier: Apache-2.0

"""Qwen4Exp mRoPE contracts against the real Transformers implementation.

The reference tests require Transformers with Qwen4Exp support (5.16.1). They
skip explicitly on older runtimes; the upstream position math is never mocked.
"""

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from tests import test_mcore_bridge_packed_inputs

from areal.engine.megatron_utils.qwen4_exp_mrope import prepare_qwen4_exp_mrope_inputs
from areal.utils.data import (
    MicroBatchSpec,
    concat_padded_tensors,
    pack_tensor_dict,
    pad_mb_list,
    split_padded_tensor_dict_into_mb_list,
    unpad_logits,
)

packed_forward = test_mcore_bridge_packed_inputs.packed_forward


@pytest.fixture
def qwen_config():
    return SimpleNamespace(
        image_token_id=248056,
        video_token_id=248057,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )


@pytest.fixture
def hf_qwen4_exp(qwen_config):
    module = pytest.importorskip(
        "transformers.models.qwen4_exp.modeling_qwen4_exp",
        reason="Actual Qwen4Exp mRoPE reference requires Transformers 5.16.1.",
    )
    model = object.__new__(module.Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model.config = qwen_config
    return model


@pytest.fixture
def vision_batch(hf_qwen4_exp, qwen_config):
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
        Qwen2VLImageProcessorPil,
    )

    processor = Qwen2VLImageProcessorPil(do_resize=False)
    image = Image.new("RGB", (84, 56), color=(12, 34, 56))
    image_inputs = processor(images=image, return_tensors="pt")
    torch.testing.assert_close(
        image_inputs["image_grid_thw"], torch.tensor([[1, 4, 6]]), atol=0, rtol=0
    )
    # Two timestamp-separated video frames share the same spatial grid.
    frame_inputs = processor(
        images=Image.new("RGB", (56, 56), color=(78, 90, 123)), return_tensors="pt"
    )
    video_pixels = frame_inputs["pixel_values"].repeat(2, 1)
    sample_ids = [
        [10, 11] + [qwen_config.image_token_id] * 6 + [12, 13],
        [20]
        + [qwen_config.video_token_id] * 4
        + [21, 22]
        + [qwen_config.video_token_id] * 4
        + [23],
        [30, 31, 32],
    ]
    sample_types = [
        [0, 0] + [1] * 6 + [0, 0],
        [0] + [2] * 4 + [0, 0] + [2] * 4 + [0],
        [0, 0, 0],
    ]
    sample_payloads = [
        dict(image_inputs),
        {
            "pixel_values_videos": video_pixels,
            "video_grid_thw": torch.tensor([[2, 4, 4]]),
        },
        {},
    ]
    return concat_padded_tensors(
        [
            {
                "input_ids": torch.tensor(ids, dtype=torch.int32)[None],
                "attention_mask": torch.ones(1, len(ids), dtype=torch.bool),
                "mm_token_type_ids": torch.tensor(types)[None],
                "multi_modal_input": [payload],
            }
            for ids, types, payload in zip(sample_ids, sample_types, sample_payloads)
        ]
    )


def test_mrope_uses_real_upstream_positions_without_constructing_model(
    hf_qwen4_exp, vision_batch, qwen_config, monkeypatch
):
    def reject_model_construction(*_args, **_kwargs):
        raise AssertionError("mRoPE preparation must not instantiate the model.")

    monkeypatch.setattr(type(hf_qwen4_exp), "__init__", reject_model_construction)
    expected, _ = hf_qwen4_exp.get_rope_index(
        input_ids=vision_batch["input_ids"].long(),
        mm_token_type_ids=vision_batch["mm_token_type_ids"],
        image_grid_thw=vision_batch["multi_modal_input"][0]["image_grid_thw"],
        video_grid_thw=vision_batch["multi_modal_input"][1]["video_grid_thw"],
        attention_mask=vision_batch["attention_mask"],
    )
    prepared = prepare_qwen4_exp_mrope_inputs(vision_batch, qwen_config)
    torch.testing.assert_close(
        prepared["position_ids"], expected.permute(1, 2, 0), atol=0, rtol=0
    )
    torch.testing.assert_close(
        prepared["position_ids"][0, 2:8],
        torch.tensor(
            [[2, 2, 2], [2, 2, 3], [2, 2, 4], [2, 3, 2], [2, 3, 3], [2, 3, 4]]
        ),
        atol=0,
        rtol=0,
    )
    assert "position_ids" not in vision_batch
    assert vision_batch["input_ids"].dtype == torch.int32
    assert prepared["input_ids"].dtype == torch.long


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_mrope_survives_microbatch_reorder_padding_and_cp_partition(
    hf_qwen4_exp, vision_batch, qwen_config, packed_forward, monkeypatch, cp_size
):
    prepared = prepare_qwen4_exp_mrope_inputs(vision_batch, qwen_config)
    reference, _ = hf_qwen4_exp.get_rope_index(
        input_ids=vision_batch["input_ids"].long(),
        mm_token_type_ids=vision_batch["mm_token_type_ids"],
        image_grid_thw=vision_batch["multi_modal_input"][0]["image_grid_thw"],
        video_grid_thw=vision_batch["multi_modal_input"][1]["video_grid_thw"],
        attention_mask=vision_batch["attention_mask"],
    )
    mb_list = split_padded_tensor_dict_into_mb_list(
        prepared, MicroBatchSpec(n_mbs=2, max_tokens_per_mb=64)
    )
    mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
    mb_list = pad_mb_list(mb_list, seq_align_to=2 * cp_size)
    module = packed_forward
    monkeypatch.setattr(module.mpu, "get_context_parallel_world_size", lambda: cp_size)

    outputs = []
    for index, (mb, padded_mb) in enumerate(zip(mb_list.mbs, mb_list.padded_mbs)):
        payloads = padded_mb["multi_modal_input"]
        expected_payload = {
            key: torch.cat([item[key] for item in payloads if key in item])
            for key in module._VLM_FORWARD_KEYS
            if any(key in item for item in payloads)
        }
        module.extract_vision_from_multi_modal(mb, padded_mb)
        cp_outputs = []
        for cp_rank in range(cp_size):
            monkeypatch.setattr(
                module.mpu, "get_context_parallel_rank", lambda rank=cp_rank: rank
            )
            model = MagicMock(
                side_effect=lambda **inputs: inputs["position_ids"].permute(1, 2, 0)
            )
            cp_outputs.append(
                module.packed_context_parallel_forward(
                    model,
                    padded_mb,
                    gather_cp_output=False,
                    is_vision_model=True,
                    use_wrapper_packed_seq=True,
                )
            )
            inputs = model.call_args.kwargs
            assert inputs["position_ids"].shape == (3, 1, inputs["input_ids"].shape[-1])
            for key, value in expected_payload.items():
                torch.testing.assert_close(inputs[key], value, atol=0, rtol=0)
            assert "mm_token_type_ids" in padded_mb
        indices = module._build_cp_reassemble_indices(padded_mb["cu_seqlens"], cp_size)
        reassembled = torch.cat(cp_outputs)[indices]
        outputs.append(
            unpad_logits(
                reassembled,
                mb_list.padding_lengths[index],
                cu_seqlens=padded_mb["cu_seqlens"],
                old_cu_seqlens=mb_list.old_cu_seqlens_list[index],
            )
        )
    order = mb_list.forward_indices
    expected = torch.cat(
        [reference[:, i, vision_batch["attention_mask"][i]].T for i in order]
    )
    torch.testing.assert_close(torch.cat(outputs), expected, atol=0, rtol=0)


def test_mrope_missing_token_types_uses_upstream_processor_method(
    hf_qwen4_exp, vision_batch, qwen_config
):
    from transformers.processing_utils import ProcessorMixin

    processor = SimpleNamespace(
        image_token_ids=[qwen_config.image_token_id],
        video_token_ids=[qwen_config.video_token_id],
        audio_token_ids=[],
    )
    processor.create_mm_token_type_ids = MethodType(
        ProcessorMixin.create_mm_token_type_ids, processor
    )
    inputs = dict(vision_batch)
    expected = inputs.pop("mm_token_type_ids")
    prepared = prepare_qwen4_exp_mrope_inputs(inputs, qwen_config, processor)
    torch.testing.assert_close(prepared["mm_token_type_ids"], expected, atol=0, rtol=0)


def _minimal_vision_input(qwen_config):
    return {
        "input_ids": torch.tensor([[1, qwen_config.image_token_id, 2]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
        "multi_modal_input": [
            {
                "pixel_values": torch.ones(4, 12),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }
        ],
    }


def test_language_only_rejects_vision_before_importing_model(qwen_config):
    with pytest.raises(ValueError, match="language_model_only=True"):
        prepare_qwen4_exp_mrope_inputs(
            _minimal_vision_input(qwen_config), qwen_config, language_model_only=True
        )


def test_missing_modality_ids_and_processor_is_rejected(qwen_config):
    inputs = _minimal_vision_input(qwen_config)
    inputs.pop("mm_token_type_ids")
    with pytest.raises(ValueError, match="processor-produced mm_token_type_ids"):
        prepare_qwen4_exp_mrope_inputs(inputs, qwen_config)


def test_modality_ids_must_match_real_placeholder_ids(qwen_config):
    inputs = _minimal_vision_input(qwen_config)
    inputs["mm_token_type_ids"].zero_()
    with pytest.raises(ValueError, match="disagree"):
        prepare_qwen4_exp_mrope_inputs(inputs, qwen_config)


def test_grid_from_another_sample_cannot_satisfy_missing_image(qwen_config):
    inputs = _minimal_vision_input(qwen_config)
    inputs["input_ids"] = inputs["input_ids"].repeat(2, 1)
    inputs["attention_mask"] = inputs["attention_mask"].repeat(2, 1)
    inputs["mm_token_type_ids"] = inputs["mm_token_type_ids"].repeat(2, 1)
    payload = inputs["multi_modal_input"][0]
    inputs["multi_modal_input"] = [{}, payload]
    with pytest.raises(ValueError, match="Sample 0 requires both"):
        prepare_qwen4_exp_mrope_inputs(inputs, qwen_config)


def test_visual_patch_count_mismatch_is_rejected(qwen_config):
    inputs = _minimal_vision_input(qwen_config)
    inputs["multi_modal_input"][0]["pixel_values"] = torch.ones(8, 12)
    with pytest.raises(ValueError, match="placeholder counts differ"):
        prepare_qwen4_exp_mrope_inputs(inputs, qwen_config)


def test_image_groups_cannot_consume_each_others_grid_tokens(qwen_config):
    image_token = qwen_config.image_token_id
    inputs = {
        "input_ids": torch.tensor([[image_token, image_token, 1, image_token]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "mm_token_type_ids": torch.tensor([[1, 1, 0, 1]]),
        "multi_modal_input": [
            {
                "pixel_values": torch.ones(12, 12),
                "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 4]]),
            }
        ],
    }
    with pytest.raises(ValueError, match="does not match modality token groups"):
        prepare_qwen4_exp_mrope_inputs(inputs, qwen_config)
