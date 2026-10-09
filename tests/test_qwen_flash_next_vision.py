# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from examples.swe.qwen38_flash_next.batch_snapshot import (
    capture_training_batches,
    load_batch_snapshot,
    save_batch_snapshot,
)
from examples.swe.qwen38_flash_next.gdn_cp_compat import patch_packed_cp_forward
from examples.swe.qwen38_flash_next.grad_norm_guard import (
    guard_grad_norm,
    local_gradient_diagnostics,
)
from examples.swe.qwen38_flash_next.ple_chunked import chunked_ple
from examples.swe.qwen38_flash_next.train_rl import (
    configure_training_rpc,
    select_evaluation_rows,
    validate_evaluation_only,
)
from tests import test_qwen_flash_next_training

from areal.engine.megatron_utils.qwen4_exp_mrope import (
    install_qwen4_exp_visual_token_mask,
    prepare_qwen4_exp_mrope_inputs,
    require_qwen4_exp_vision_runtime,
)
from areal.utils.data import (
    MicroBatchSpec,
    RolloutGroup,
    TrajBatchMeta,
    concat_padded_tensors,
    pack_tensor_dict,
    pad_mb_list,
    split_padded_tensor_dict_into_mb_list,
    unpad_logits,
)

packed_forward = test_qwen_flash_next_training.packed_forward


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
            torch.testing.assert_close(
                inputs["mm_token_type_ids"],
                padded_mb["mm_token_type_ids"].reshape(1, -1),
                atol=0,
                rtol=0,
            )
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


def _compute_vision_batch_advantages(inputs):
    from areal.api.cli_args import PPOActorConfig
    from areal.trainer.ppo.actor import PPOActor

    actor = PPOActor(
        PPOActorConfig(
            reward_norm=None,
            adv_norm=None,
            kl_ctl=0,
            recompute_logprob=False,
            mask_no_eos_with_zero=False,
        ),
        MagicMock(),
    )
    inputs["logprobs"] = torch.zeros_like(inputs["input_ids"], dtype=torch.float32)
    inputs["rewards"] = torch.ones(inputs["input_ids"].shape[0])
    return actor._compute_advantages(inputs)


def test_vision_runtime_without_model_fails_with_actionable_error(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "transformers.models.qwen4_exp.modeling_qwen4_exp", None
    )
    with pytest.raises(RuntimeError, match="Transformers 5.16.1"):
        require_qwen4_exp_vision_runtime()


@pytest.mark.parametrize("after_advantages", [False, True])
@pytest.mark.parametrize("token_kind", ["image_token_id", "video_token_id"])
def test_generated_visual_special_tokens_remain_text(
    hf_qwen4_exp, vision_batch, qwen_config, token_kind, after_advantages
):
    original = prepare_qwen4_exp_mrope_inputs(vision_batch, qwen_config)
    inputs = dict(vision_batch)
    inputs["input_ids"] = inputs["input_ids"].clone()
    inputs["loss_mask"] = torch.zeros_like(inputs["attention_mask"])
    for row in range(inputs["input_ids"].shape[0]):
        last = int(inputs["attention_mask"][row].sum()) - 1
        inputs["input_ids"][row, last] = getattr(qwen_config, token_kind)
        inputs["loss_mask"][row, last] = True
    if after_advantages:
        inputs = _compute_vision_batch_advantages(inputs)
    prepared = prepare_qwen4_exp_mrope_inputs(inputs, qwen_config)
    torch.testing.assert_close(prepared["position_ids"], original["position_ids"])
    torch.testing.assert_close(
        prepared["mm_token_type_ids"], original["mm_token_type_ids"]
    )
    torch.testing.assert_close(prepared["input_ids"], inputs["input_ids"].long())


def test_visual_scatter_preserves_generated_token_embeddings_and_gradients():
    class Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hf_config = SimpleNamespace(image_token_id=2, video_token_id=3)

        def get_inputs_embeds(self, inputs_embeds, **kwargs):
            ids = kwargs["input_ids"]
            mask = (ids == 2) | (ids == 3)
            return inputs_embeds.masked_scatter(
                mask.unsqueeze(-1), kwargs["pixel_values"]
            )

    model = SimpleNamespace(visual=Visual())
    install_qwen4_exp_visual_token_mask(model)
    installed = model.visual.get_inputs_embeds
    install_qwen4_exp_visual_token_mask(model)
    assert model.visual.get_inputs_embeds is installed
    ids = torch.tensor([[1, 2, 2, 3]])
    types = torch.tensor([[0, 1, 0, 0]])
    embeddings = torch.arange(8.0).reshape(1, 4, 2).requires_grad_()
    pixels = torch.tensor([[[10.0, 11.0]]], requires_grad=True)
    output = model.visual.get_inputs_embeds(
        embeddings, input_ids=ids, mm_token_type_ids=types, pixel_values=pixels
    )
    expected = embeddings.detach().clone()
    expected[:, 1] = pixels.detach()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    output.sum().backward()
    expected_grad = torch.ones_like(embeddings)
    expected_grad[:, 1] = 0
    torch.testing.assert_close(embeddings.grad, expected_grad, atol=0, rtol=0)
    torch.testing.assert_close(pixels.grad, torch.ones_like(pixels), atol=0, rtol=0)
    torch.testing.assert_close(ids, torch.tensor([[1, 2, 2, 3]]), atol=0, rtol=0)
    assert not hasattr(Visual(), "_areal_visual_token_mask")


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


@pytest.mark.parametrize("after_advantages", [False, True])
@pytest.mark.parametrize("language_model_only", [False, True])
@pytest.mark.parametrize("modality", ["image", "video"])
def test_generated_special_token_without_pixels_remains_text(
    qwen_config, language_model_only, modality, after_advantages
):
    special = getattr(qwen_config, f"{modality}_token_id")
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12, special]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 0, 1, 1]]),
        "mm_token_type_ids": torch.zeros(1, 4, dtype=torch.long),
    }
    if after_advantages:
        inputs = _compute_vision_batch_advantages(inputs)
    result = prepare_qwen4_exp_mrope_inputs(
        inputs, qwen_config, language_model_only=language_model_only
    )
    assert "position_ids" not in result
    torch.testing.assert_close(result["input_ids"], inputs["input_ids"], rtol=0, atol=0)


@pytest.mark.parametrize("method", ["_train_lm", "_evaluate_lm"])
def test_sft_preserves_token_provenance_before_loss_alignment(
    qwen_config, method, monkeypatch
):
    from areal.trainer.sft import lm_engine

    engine = MagicMock()
    engine.train_batch.return_value = {}
    monkeypatch.setattr(lm_engine, "stage_batch_for_engine", lambda *_: None)
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12, qwen_config.image_token_id]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 0, 1, 1]]),
    }
    getattr(lm_engine.LMEngine(engine), method)(inputs)
    call = engine.train_batch if method == "_train_lm" else engine.eval_batch
    batch = call.call_args.kwargs["input_"]
    result = prepare_qwen4_exp_mrope_inputs(batch, qwen_config)
    assert "position_ids" not in result
    torch.testing.assert_close(
        batch["loss_mask"], torch.tensor([[False, True, True, False]]), rtol=0, atol=0
    )


def test_prompt_placeholder_without_pixels_still_rejected(qwen_config):
    inputs = {
        "input_ids": torch.tensor([[qwen_config.image_token_id, 10]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1]]),
    }
    with pytest.raises(ValueError, match="language_model_only"):
        prepare_qwen4_exp_mrope_inputs(inputs, qwen_config, language_model_only=True)


def _causal_reference(h, k, v, nk, nq, nc, weight, n, eps, dilation, seq_len):
    x = (h * nk + k * nq + v * nc).reshape(-1, seq_len, h.shape[-1])
    x = x.transpose(1, 2)
    halo = (weight.shape[-1] - 1) * dilation
    return (
        F.conv1d(
            F.pad(x, (halo, 0)), weight.float(), groups=h.shape[-1], dilation=dilation
        )
        .transpose(1, 2)
        .reshape_as(h)
    )


@pytest.mark.parametrize("chunk_tokens,dilation", [(5, 1), (2, 3), (32, 1)])
def test_ple_chunks_preserve_rows_halo_and_gradients(chunk_tokens, dilation):
    torch.manual_seed(1234)
    shape = (26, 3)  # Two rows; the last chunk is shorter than the others.
    data = [torch.randn(shape) for _ in range(3)]
    data += [torch.randn(3).bfloat16() for _ in range(3)]
    data += [torch.randn(3, 1, 4).bfloat16()]
    actual_inputs = [x.clone().requires_grad_() for x in data]
    reference_inputs = [x.clone().requires_grad_() for x in data]
    actual = chunked_ple(
        _causal_reference,
        *actual_inputs,
        4,
        1e-6,
        dilation,
        13,
        chunk_tokens=chunk_tokens,
    )
    h, k, v, *weights = reference_inputs
    reference = _causal_reference(
        h, k, v, *(w.float() for w in weights), 4, 1e-6, dilation, 13
    )
    grad = torch.randn_like(actual)
    actual.backward(grad)
    reference.backward(grad)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    for actual_input, reference_input in zip(actual_inputs, reference_inputs):
        torch.testing.assert_close(
            actual_input.grad, reference_input.grad, rtol=1e-4, atol=1e-4
        )


def _legacy_forward(self, cu_seqlens, *, cp_size=2):
    return cu_seqlens // self.cp_size


def _fixed_forward(self, cu_seqlens, *, cp_size=2):
    return cu_seqlens // cp_size


def _unknown_forward(self, cu_seqlens):
    return cu_seqlens


def test_packed_cp_uses_local_size_and_patch_is_idempotent():
    patched = patch_packed_cp_forward(_legacy_forward)
    # Legacy self.cp_size may be missing or stale.
    torch.testing.assert_close(
        patched(SimpleNamespace(), torch.tensor([0, 16, 40])),
        torch.tensor([0, 8, 20]),
        rtol=0,
        atol=0,
    )
    assert patch_packed_cp_forward(patched) is patched
    assert patch_packed_cp_forward(_fixed_forward) is _fixed_forward
    with pytest.raises(RuntimeError, match="divisor"):
        patch_packed_cp_forward(_unknown_forward)


def test_training_rpc_disables_optimizer_replay_only():
    calls = []

    async def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "ok"

    scheduler = configure_training_rpc(SimpleNamespace(async_call_engine=original))
    assert (
        asyncio.run(
            scheduler.async_call_engine(
                "actor/0", "ppo_update", http_timeout=1, max_retries=3
            )
        )
        == "ok"
    )
    assert calls[-1][1] == dict(http_timeout=28800, max_retries=1)
    asyncio.run(
        scheduler.async_call_engine(
            "actor/0", "get_version", http_timeout=17, max_retries=2
        )
    )
    assert calls[-1][1] == dict(http_timeout=17, max_retries=2)


def _optimizer(grad):
    return SimpleNamespace(get_main_grads_for_grad_norm=lambda: [grad])


def test_large_finite_gradients_diagnose_fp32_norm_overflow():
    grad = torch.tensor([1e20, -1e20], dtype=torch.float32)
    assert torch.isinf(grad.norm())
    before = grad.clone()
    optimizer = _optimizer(grad)
    report = local_gradient_diagnostics(optimizer, chunk_size=1)
    assert report["nonfinite_elements"] == 0
    assert report["finite_fp64_norm"] == pytest.approx(grad.double().norm().item())
    with pytest.raises(RuntimeError, match="before clipping"):
        guard_grad_norm(lambda _: grad.norm().item())(optimizer)
    torch.testing.assert_close(grad, before, rtol=0, atol=0)


def test_snapshot_real_rollout_metadata_roundtrips_without_mutating_batch(tmp_path):
    group = RolloutGroup((1, 2, 1, 1), (0.0, 1.0, 1.0, 1.0))
    pixels = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    batch = [
        {
            "rollout_group": group,
            "pixel_values": pixels,
            "alias": pixels,
            "logprobs": torch.zeros(5, 3),
            "attention_mask": torch.ones(5, 3),
        }
    ]
    actor = SimpleNamespace(
        prepare_batch=lambda: batch, compute_advantages=lambda data: data
    )
    with capture_training_batches(actor, tmp_path, {}):
        assert actor.prepare_batch() is batch
        assert actor.compute_advantages(batch) is batch
    path = tmp_path / "prepare_batch-0000.output.pt"
    assert torch.load(path, weights_only=True)["schema_version"] == 2
    restored = load_batch_snapshot(path)["batch"][0]
    assert restored["rollout_group"].validate_rows(5) == group
    assert batch[0]["rollout_group"] is group
    assert restored["pixel_values"] is restored["alias"]
    torch.testing.assert_close(restored["pixel_values"], pixels, rtol=0, atol=0)
    meta = TrajBatchMeta(1, [5], [3], [group])
    save_batch_snapshot(tmp_path / "meta.pt", {"meta": meta}, {})
    assert load_batch_snapshot(tmp_path / "meta.pt")["batch"]["meta"] == meta


def evaluation_config():
    return SimpleNamespace(
        total_train_steps=0,
        evaluator=SimpleNamespace(eval_before_train=True),
        recover=SimpleNamespace(mode="disabled"),
        eval_gconfig=SimpleNamespace(n_samples=1),
        valid_dataset=SimpleNamespace(shuffle=False, drop_last=False, batch_size=3),
    )


@pytest.mark.parametrize(
    "section,field,value",
    [
        (None, "total_train_steps", 1),
        ("evaluator", "eval_before_train", False),
        ("recover", "mode", "auto"),
        ("eval_gconfig", "n_samples", 8),
        (None, "valid_dataset", None),
        ("valid_dataset", "shuffle", True),
        ("valid_dataset", "drop_last", True),
        ("valid_dataset", "batch_size", 2),
    ],
)
def test_evaluation_unsafe_config_rejected_before_initialization(section, field, value):
    config = evaluation_config()
    setattr(config if section is None else getattr(config, section), field, value)

    with pytest.raises(ValueError, match="swe-eval"):
        validate_evaluation_only(config, ["task-a", "task-b", "task-c"])


def test_evaluation_pins_historical_version_and_preserves_stream_routing():
    rows = [
        {"data_id": "env:a@new", "stream_id": "stream", "arena_task_type": "swe"},
        {"data_id": "env:b@v1", "stream_id": "stream", "arena_task_type": "swe"},
    ]

    selected = select_evaluation_rows(rows, ["env:b@v1", "env:a@old"])

    assert [row["data_id"] for row in selected] == ["env:b@v1", "env:a@old"]
    assert all(row["stream_id"] == "stream" for row in selected)
    assert all(row["arena_task_type"] == "swe" for row in selected)
    assert rows[0]["data_id"] == "env:a@new"


@pytest.mark.parametrize("task_count", [1, 3])
def test_shared_mm_recipe_evaluation_disables_training_and_keeps_all_tasks(task_count):
    from examples.swe.qwen38_flash_next.train_rl import configure_evaluation_only
    from examples.swe.utils import SWEPPOConfig

    from areal.api.cli_args import OptimizerConfig, TrainDatasetConfig

    config = SWEPPOConfig(
        train_dataset=TrainDatasetConfig(batch_size=2, drop_last=True)
    )
    config.actor.optimizer = OptimizerConfig(lr=3e-6)
    configure_evaluation_only(config, task_count)
    validate_evaluation_only(config, list(range(task_count)))

    assert config.total_train_steps == 0
    assert config.recover.mode == "disabled"
    assert config.actor.optimizer.lr == 0
    assert config.gconfig.n_samples == config.eval_gconfig.n_samples == 1
    assert config.rollout.consumer_batch_size == task_count
    assert config.actor.mb_spec.n_mbs % config.actor.mb_spec.n_mbs_divisor == 0
    assert config.valid_dataset is not config.train_dataset
    assert config.train_dataset.drop_last  # Only validation must keep the final batch.
    assert not config.valid_dataset.drop_last
    with pytest.raises(ValueError, match="nonempty"):
        configure_evaluation_only(config, 0)
