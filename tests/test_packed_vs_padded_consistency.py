import os
import random

import pytest
import torch
import torch.distributed as dist
from torch.testing import assert_close

from tests.utils import get_model_path

from areal.api.cli_args import TrainEngineConfig
from areal.engine import FSDPEngine
from areal.infra.platforms import current_platform
from areal.utils.data import concat_padded_tensors, tensor_container_to
from areal.utils.functional import gather_logprobs
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.network import find_free_ports
from areal.utils.seeding import set_random_seed

BS = 4
MAX_ANSWER_LEN = 16
MAX_PROMPT_LEN = 8
VOCAB_SIZE = 100


@pytest.fixture
def mock_padded_llm_data():
    """Generate mock padded input data."""
    prompt_lens = torch.randint(1, MAX_PROMPT_LEN, size=(BS,))
    answer_lens = torch.randint(1, MAX_ANSWER_LEN, size=(BS,))
    all_data = []

    for prompt_len, ans_len in zip(prompt_lens, answer_lens):
        prompt_len = int(prompt_len)
        ans_len = int(ans_len)
        seq = dict(
            input_ids=torch.randint(
                0, VOCAB_SIZE, size=(prompt_len + ans_len,)
            ).unsqueeze(0),
            loss_mask=torch.tensor([0] * prompt_len + [1] * ans_len).unsqueeze(0),
            attention_mask=torch.tensor([1] * (prompt_len + ans_len)).unsqueeze(0),
        )
        all_data.append(seq)

    return tensor_container_to(
        concat_padded_tensors(all_data), current_platform.device_type
    )


QWEN3_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)
QWEN25_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen2.5-1.5B/", "Qwen/Qwen2.5-1.5B"
)


@pytest.mark.parametrize(
    "model_path",
    [QWEN3_PATH, QWEN25_PATH],
)
def test_llm_consistency(model_path, mock_padded_llm_data):
    os.environ["RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_ports(1)[0])
    os.environ["LOCAL_RANK"] = str(0)

    config = TrainEngineConfig(
        backend="fsdp:d1",
        path=model_path,
        dtype="bfloat16",
        optimizer_dtype="bfloat16",
        attn_impl="flash_attention_2",
        gradient_checkpointing=False,
        disable_dropout=True,
        init_from_scratch=True,
        optimizer=None,
    )
    engine = FSDPEngine(config)
    engine.create_process_group()
    engine._create_device_model()
    engine._initialized = True

    try:
        # Prepare padded input
        padded_input = mock_padded_llm_data

        # Get packed input using _prepare_mb_list
        mb_list = engine._prepare_mb_list(padded_input)
        assert len(mb_list.mbs) == 1

        with torch.no_grad():
            padded_logits = engine.model(
                input_ids=padded_input["input_ids"],
                attention_mask=padded_input["attention_mask"],
            ).logits
            seqlens = padded_input["attention_mask"].sum(1)
            x1 = []
            for i, s in enumerate(seqlens):
                x1.append(padded_logits[i, :s])
            x1 = torch.cat(x1)

            mb = mb_list.padded_mbs[0]
            pad_len = mb_list.padding_lengths[0]
            x2 = engine.model(**mb).logits.squeeze(0)[:-pad_len]

            assert x1.shape == x2.shape, (x1.shape, x2.shape)

            assert_close(x1, x2, atol=2e-1, rtol=2e-1)
    finally:
        engine.destroy()
        assert not dist.is_initialized()


QWEN25_VL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"
)
QWEN3_VL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-2B-Instruct"
)
# GEMMA3 requires special authentication to download
# Use it if it's available, but do not make it a hard dependency for the test
GEMMA3_PATH = None
if os.path.exists(
    os.getenv("AREAL_CI_GEMMA3_PATH", "/storage/openpsi/models/google__gemma-3-4b-it/")
):
    GEMMA3_PATH = get_model_path(
        "/storage/openpsi/models/google__gemma-3-4b-it/", "google/gemma-3-4b-it"
    )


def mock_padded_vlm_data(model_path):
    if model_path == GEMMA3_PATH:
        model_type = "gemma3"
    elif model_path == QWEN25_VL_PATH:
        model_type = "qwen25"
    elif model_path == QWEN3_VL_PATH:
        model_type = "qwen3"
    else:
        raise NotImplementedError()
    # TODO: create mock vlm image data
    prompt_lens = torch.randint(1, MAX_PROMPT_LEN, size=(BS,))
    answer_lens = torch.randint(1, MAX_ANSWER_LEN, size=(BS,))

    all_data = []

    processor, tokenizer = load_hf_processor_and_tokenizer(model_path)

    for prompt_len, ans_len in zip(prompt_lens, answer_lens):
        num_images = random.randint(1, 4)
        images = []
        image_tokens = []

        for _ in range(num_images):
            VISION_H = random.randint(4, 32) * 28
            VISION_W = random.randint(4, 32) * 28

            image = torch.randint(0, 255, size=(3, VISION_H, VISION_W)).float() / 255.0
            images.append(image)
            if model_type == "qwen25" or model_type == "qwen3":
                image_tokens.append("<|vision_start|><|image_pad|><|vision_end|>")
            else:
                assert model_type == "gemma3"
                image_tokens.append(processor.boi_token)

        combined_image_token = "".join(image_tokens)

        processed_input = processor(
            text=combined_image_token, images=images, return_tensors="pt"
        )

        image_input_id = (
            processed_input["input_ids"].squeeze(0)
            if processed_input["input_ids"].dim() > 1
            else processed_input["input_ids"]
        )

        text_prompt_len = int(prompt_len)
        prompt_len = text_prompt_len + int(image_input_id.shape[0])
        ans_len = int(ans_len)
        total_len = prompt_len + ans_len

        # Realistic layout: [text prompt][image tokens][text answer], loss on the
        # answer only.
        input_ids = torch.cat(
            [
                torch.randint(0, VOCAB_SIZE, size=(text_prompt_len,)),
                image_input_id,
                torch.randint(0, VOCAB_SIZE, size=(ans_len,)),
            ]
        )

        seq = dict(
            input_ids=input_ids.unsqueeze(0),
            loss_mask=torch.tensor([0] * prompt_len + [1] * ans_len).unsqueeze(0),
            attention_mask=torch.tensor([1] * total_len).unsqueeze(0),
            multi_modal_input=[
                {
                    "pixel_values": processed_input["pixel_values"],
                }
            ],
        )

        # Qwen-VL M-RoPE needs per-token modality types; text tokens are 0.
        if "mm_token_type_ids" in processed_input:
            image_token_types = processed_input["mm_token_type_ids"].reshape(-1)
            dtype = image_token_types.dtype
            seq["mm_token_type_ids"] = torch.cat(
                [
                    torch.zeros(text_prompt_len, dtype=dtype),
                    image_token_types,
                    torch.zeros(ans_len, dtype=dtype),
                ]
            ).unsqueeze(0)

        if "image_grid_thw" in processed_input:
            seq["multi_modal_input"][0]["image_grid_thw"] = processed_input[
                "image_grid_thw"
            ]

        td = seq

        all_data.append(td)

    padded_data = tensor_container_to(
        concat_padded_tensors(all_data), current_platform.device_type
    )
    if "multi_modal_input" in padded_data:
        for item in padded_data["multi_modal_input"]:
            if isinstance(item, dict):
                for k, v in item.items():
                    if torch.is_tensor(v):
                        item[k] = v.to(current_platform.device_type)
    return padded_data


vlm_model_paths = [pytest.param(QWEN25_VL_PATH), pytest.param(QWEN3_VL_PATH)]
if GEMMA3_PATH is not None:
    vlm_model_paths.append(pytest.param(GEMMA3_PATH, marks=pytest.mark.slow))


@pytest.mark.slow
@pytest.mark.parametrize("model_path", vlm_model_paths)
def test_vlm_consistency(model_path):
    # Set random seed for reproducibility.
    # This test might fail on other seed on A100, so do not change this line.
    set_random_seed(27010, key="test_vlm_consistency")

    os.environ["RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_ports(1)[0])
    os.environ["LOCAL_RANK"] = str(0)

    config = TrainEngineConfig(
        backend="fsdp:d1",
        path=model_path,
        dtype="bfloat16",
        optimizer_dtype="bfloat16",
        attn_impl="flash_attention_2",
        gradient_checkpointing=False,
        disable_dropout=True,
        init_from_scratch=False,
        optimizer=None,
    )

    engine = FSDPEngine(config)
    engine.create_process_group()
    engine._create_device_model()
    engine._initialized = True

    padded_input = mock_padded_vlm_data(model_path)

    try:
        # Get packed input
        mb_list = engine._prepare_mb_list(padded_input)
        assert len(mb_list.mbs) == 1

        with torch.no_grad():
            # Padded logits
            if "multi_modal_input" in padded_input:
                image_grid_thw_list = [
                    item["image_grid_thw"]
                    for item in padded_input["multi_modal_input"]
                    if "image_grid_thw" in item
                ]
                if image_grid_thw_list:
                    padded_input["image_grid_thw"] = torch.cat(
                        image_grid_thw_list, dim=0
                    )
                pixel_values_list = [
                    item["pixel_values"]
                    for item in padded_input["multi_modal_input"]
                    if "pixel_values" in item
                ]
                if pixel_values_list:
                    padded_input["pixel_values"] = torch.cat(pixel_values_list, dim=0)
            padded_logits = engine.model(**padded_input).logits

            # Extract valid sequence logits
            seqlens = padded_input["attention_mask"].sum(1)
            x1 = []
            for i, s in enumerate(seqlens):
                x1.append(padded_logits[i, :s])
            x1 = torch.cat(x1)

            # Packed logits
            mb = mb_list.padded_mbs[0]
            pad_len = mb_list.padding_lengths[0]
            x2 = engine.model(**mb).logits.squeeze(0)[:-pad_len]

            assert x1.shape == x2.shape, f"Shape mismatch: {x1.shape} vs {x2.shape}"

            # Compare what training consumes: the log-prob of each answer token
            # (loss_mask=1), scored from the previous position. An element-wise
            # check on raw logits is too strict for VLMs: bf16 rounding from the
            # different batch shapes moves a few of the ~150k logits per token
            # past atol=0.2, even between two unpacked layouts. Measured on XPU
            # (Qwen2.5-VL-3B, Qwen3-VL-2B): correct packing stays within 0.30 max
            # / 0.062 mean |dlogp|; removing a sequence boundary or using 1-D
            # positions for images raises the mean to >= 0.15.
            input_ids = torch.cat(
                [padded_input["input_ids"][i, :s] for i, s in enumerate(seqlens)]
            )
            loss_mask = torch.cat(
                [padded_input["loss_mask"][i, :s] for i, s in enumerate(seqlens)]
            ).bool()
            ends = torch.cumsum(seqlens, 0).tolist()
            pred_pos = torch.cat(
                [torch.arange(e - s, e - 1) for s, e in zip(seqlens.tolist(), ends)]
            ).to(input_ids.device)
            pred_pos = pred_pos[loss_mask[pred_pos + 1]]
            labels = input_ids[pred_pos + 1]

            logp_padded = gather_logprobs(x1[pred_pos].float(), labels)
            logp_packed = gather_logprobs(x2[pred_pos].float(), labels)
            assert_close(logp_packed, logp_padded, atol=5e-1, rtol=0)
            mean_diff = (logp_packed - logp_padded).abs().mean().item()
            assert mean_diff <= 1e-1, f"mean |dlogp| {mean_diff:.4f} > 0.1"
    finally:
        engine.destroy()
        assert not dist.is_initialized()
