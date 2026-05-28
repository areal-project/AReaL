"""Test comparison between FP8 and BF16 models using Megatron Engine.

This test verifies:
1. Load FP8 model with fp8_param enabled and BF16 model using Megatron Engine
2. Compare logprobs from forward pass
3. Compare logits from forward pass
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

from tests.fp8.comparison_utils import (
    compare_logits,
    compare_tensors_dict,
    log_problematic_operations,
)
from tests.fp8.engine_utils import (
    create_engine,
    decode_with_megatron_forward,
    forward_with_logits_and_logprobs,
)
from tests.fp8.model_hooks import (
    collect_gradients_after_train_batch,
    forward_backward_model_with_hooks,
)
from tests.utils import get_model_path

from areal.infra.platforms import current_platform
from areal.utils import logging

MODEL_PATH_BF16 = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)
MODEL_PATH_FP8 = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B-FP8/", "Qwen/Qwen3-0.6B-FP8"
)

logger = logging.getLogger("FP8 BF16 Comparison Test")


@pytest.fixture(scope="module")
def fixed_input(
    questions: list[str] | None = None,
    answers: list[str] | None = None,
    model_path: str = MODEL_PATH_BF16,
    device=current_platform.device_type,
) -> dict[str, Any]:
    """Create fixed input data for SFT training with question and answer.

    Args:
        questions: List of question strings. If None, uses default questions.
        answers: List of answer strings. If None, uses default answers.
        model_path: Path to the model for loading tokenizer.
        device: Device to place tensors on.

    Returns:
        Dictionary with 'input_ids', 'attention_mask', and 'loss_mask' tensors.
        loss_mask: 0 for prompt tokens, 1 for answer tokens (including EOS).
    """
    if questions is None:
        questions = [
            "What is 2+2?",
            "Count from 1 to 5:",
        ]
    if answers is None:
        answers = [
            " 2+2 equals 4.",
            " 1, 2, 3, 4, 5",
        ]

    assert len(questions) == len(answers), (
        "Questions and answers must have the same length"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token = tokenizer.eos_token if tokenizer.eos_token else ""
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_ids_list = []
    loss_mask_list = []

    for question, answer in zip(questions, answers):
        # Encode full sequence: question + answer + eos_token
        full_text = question + answer + eos_token
        seq_token = tokenizer.encode(full_text, add_special_tokens=False)

        # Encode prompt (question) to determine where loss_mask should start
        prompt_token = tokenizer.encode(question, add_special_tokens=False)

        # Create loss_mask: 0 for prompt, 1 for answer (including EOS)
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))

        input_ids_list.append(torch.tensor(seq_token, dtype=torch.long, device=device))
        loss_mask_list.append(torch.tensor(loss_mask, dtype=torch.long, device=device))

    # Pad to same length
    max_length = max(ids.shape[0] for ids in input_ids_list)

    padded_input_ids = []
    padded_loss_masks = []
    attention_masks = []

    for input_ids, loss_mask in zip(input_ids_list, loss_mask_list):
        seq_len = input_ids.shape[0]
        padding_length = max_length - seq_len

        padded_ids = F.pad(input_ids, (0, padding_length), value=pad_token_id)
        padded_input_ids.append(padded_ids)

        padded_loss_mask = F.pad(loss_mask, (0, padding_length), value=0)
        padded_loss_masks.append(padded_loss_mask)

        attention_mask = F.pad(
            torch.ones(seq_len, dtype=torch.bool, device=device),
            (0, padding_length),
            value=0,
        )
        attention_masks.append(attention_mask)

    # Stack into batch
    input_ids = torch.stack(padded_input_ids).to(device)
    loss_mask = torch.stack(padded_loss_masks).to(device)
    attention_mask = torch.stack(attention_masks).to(device)

    logger.info("Using fixed input:")
    for i, (q, a) in enumerate(zip(questions, answers)):
        logger.info(f"  Sample {i}:")
        logger.info(f"    Question: {q}")
        logger.info(f"    Answer: {a}")
        logger.info(f"    Input IDs shape: {input_ids[i].shape}")
        logger.info(f"    Loss mask sum: {loss_mask[i].sum().item()}")

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
    )


@pytest.fixture(scope="module")
def engine_bf16():
    engine = create_engine(
        MODEL_PATH_BF16, fp8_enabled=False, fp8_param=False, port=7777
    )
    try:
        yield engine
    finally:
        engine.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture(scope="module")
def engine_fp8():
    engine = create_engine(MODEL_PATH_FP8, fp8_enabled=True, fp8_param=True, port=7778)
    try:
        yield engine
    finally:
        engine.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_megatron_decode_output(engine_bf16, engine_fp8):
    """Test decode using Megatron forward pass and print model output."""
    # Test prompts
    test_prompts = [
        "What is 2+2?",
        "The capital of France is",
        "Once upon a time",
    ]

    top_k = None
    temperature = 0.7
    max_new_tokens = 100

    logger.info("=" * 80)
    logger.info("Testing decode with BF16 model")
    logger.info("=" * 80)

    for prompt in test_prompts:
        logger.info(f"{'=' * 80}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"{'=' * 80}")
        generated = decode_with_megatron_forward(
            engine_bf16,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        logger.info(f"BF16 Final output: {generated}\n")

    logger.info("=" * 80)
    logger.info("Testing decode with FP8 model")
    logger.info("=" * 80)

    for prompt in test_prompts:
        logger.info(f"{'=' * 80}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"{'=' * 80}")
        generated = decode_with_megatron_forward(
            engine_fp8,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        logger.info(f"FP8 Final output: {generated}\n")


def test_fp8_bf16_logits_logprobs_comparison(fixed_input, engine_bf16, engine_fp8):
    """Compare both logits and logprobs between FP8 and BF16 models."""
    logits_bf16, logprobs_bf16 = forward_with_logits_and_logprobs(
        engine_bf16, fixed_input
    )

    logits_fp8, logprobs_fp8 = forward_with_logits_and_logprobs(engine_fp8, fixed_input)

    # Get attention mask to filter out padding positions
    attention_mask = fixed_input["attention_mask"]  # [batch, seq_len]

    # Compare logprobs first
    assert logprobs_bf16.shape == logprobs_fp8.shape, "Logprob shapes don't match"
    # Only compute differences for non-padding positions
    valid_logprobs_mask = attention_mask  # [batch, seq_len]
    logprob_diff = (logprobs_bf16 - logprobs_fp8).abs()
    logprob_max_diff = (logprob_diff * valid_logprobs_mask).max().item()
    logprob_mean_diff = (
        logprob_diff * valid_logprobs_mask
    ).sum().item() / valid_logprobs_mask.sum().item()
    logger.info(
        f"Logprob comparison (non-padding only): max_diff={logprob_max_diff:.6f}, mean_diff={logprob_mean_diff:.6f}"
    )

    # Compare logits
    assert logits_bf16.shape == logits_fp8.shape, "Logits shapes don't match"
    # Only compute differences for non-padding positions
    valid_logits_mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
    logits_diff = (logits_bf16 - logits_fp8).abs()
    logits_max_diff = (logits_diff * valid_logits_mask).max().item()
    logits_mean_diff = (
        logits_diff * valid_logits_mask
    ).sum().item() / valid_logits_mask.sum().item()
    logger.info(
        f"Logits comparison (non-padding only): max_diff={logits_max_diff:.6f}, mean_diff={logits_mean_diff:.6f}"
    )

    # Sequence-level and token-level cosine similarity (only use valid tokens)
    batch_size, seq_len = attention_mask.shape

    # Collect data for both sequence-level and token-level cosine similarity
    cos_sim_logprobs_seq_list = []
    cos_sim_logits_seq_list = []
    logprobs_bf16_valid_list = []
    logprobs_fp8_valid_list = []
    logits_bf16_valid_list = []
    logits_fp8_valid_list = []

    for i in range(batch_size):
        valid_mask_i = attention_mask[i]  # [seq_len]
        valid_indices = valid_mask_i.nonzero(as_tuple=False).squeeze(-1)  # [num_valid]

        # Extract valid token positions: [num_valid, vocab_size]
        logprobs_bf16_valid = logprobs_bf16[i, valid_indices]  # [num_valid, vocab_size]
        logprobs_fp8_valid = logprobs_fp8[i, valid_indices]  # [num_valid, vocab_size]
        logits_bf16_valid = logits_bf16[i, valid_indices]  # [num_valid, vocab_size]
        logits_fp8_valid = logits_fp8[i, valid_indices]  # [num_valid, vocab_size]

        # For sequence-level: flatten valid tokens: [num_valid * vocab_size]
        logprobs_bf16_flat = logprobs_bf16_valid.flatten()
        logprobs_fp8_flat = logprobs_fp8_valid.flatten()
        logits_bf16_flat = logits_bf16_valid.flatten()
        logits_fp8_flat = logits_fp8_valid.flatten()

        # Compute sequence-level cosine similarity for this sample
        cos_sim_logprobs_i = F.cosine_similarity(
            logprobs_bf16_flat.unsqueeze(0), logprobs_fp8_flat.unsqueeze(0), dim=1
        ).item()
        cos_sim_logits_i = F.cosine_similarity(
            logits_bf16_flat.unsqueeze(0), logits_fp8_flat.unsqueeze(0), dim=1
        ).item()

        cos_sim_logprobs_seq_list.append(cos_sim_logprobs_i)
        cos_sim_logits_seq_list.append(cos_sim_logits_i)

        # For token-level: collect individual valid tokens
        logprobs_bf16_valid_list.append(logprobs_bf16_valid)  # [num_valid, vocab_size]
        logprobs_fp8_valid_list.append(logprobs_fp8_valid)  # [num_valid, vocab_size]
        logits_bf16_valid_list.append(logits_bf16_valid)  # [num_valid, vocab_size]
        logits_fp8_valid_list.append(logits_fp8_valid)  # [num_valid, vocab_size]

    # Sequence-level statistics
    cos_sim_logprobs_seq_mean = sum(cos_sim_logprobs_seq_list) / len(
        cos_sim_logprobs_seq_list
    )
    cos_sim_logits_seq_mean = sum(cos_sim_logits_seq_list) / len(
        cos_sim_logits_seq_list
    )

    logger.info(
        f"Seq cosine similarity of logprobs (valid tokens only): {cos_sim_logprobs_seq_mean}"
    )
    logger.info(
        f"Seq cosine similarity of logits (valid tokens only): {cos_sim_logits_seq_mean}"
    )

    # Stack token-level tensors: [num_valid_tokens, vocab_size]
    logprobs_bf16_valid = torch.cat(logprobs_bf16_valid_list, dim=0)
    logprobs_fp8_valid = torch.cat(logprobs_fp8_valid_list, dim=0)
    logits_bf16_valid = torch.cat(logits_bf16_valid_list, dim=0)
    logits_fp8_valid = torch.cat(logits_fp8_valid_list, dim=0)

    # Compute cosine similarity only for valid tokens
    cos_sim_logprobs_valid = F.cosine_similarity(
        logprobs_bf16_valid, logprobs_fp8_valid, dim=-1
    )  # [num_valid_tokens]
    cos_sim_logits_valid = F.cosine_similarity(
        logits_bf16_valid, logits_fp8_valid, dim=-1
    )  # [num_valid_tokens]

    cos_sim_logprobs_mean = cos_sim_logprobs_valid.mean().item()
    cos_sim_logprobs_min = cos_sim_logprobs_valid.min().item()
    cos_sim_logprobs_max = cos_sim_logprobs_valid.max().item()

    cos_sim_logits_mean = cos_sim_logits_valid.mean().item()
    cos_sim_logits_min = cos_sim_logits_valid.min().item()
    cos_sim_logits_max = cos_sim_logits_valid.max().item()

    logger.info(
        f"Token cosine similarity of logprobs (valid tokens only): {cos_sim_logprobs_mean}"
    )
    logger.info(
        f"Token cosine similarity of logits (valid tokens only): {cos_sim_logits_mean}"
    )
    logger.info(
        f"Token cosine similarity of logprobs - min: {cos_sim_logprobs_min:.6f}, max: {cos_sim_logprobs_max:.6f}"
    )
    logger.info(
        f"Token cosine similarity of logits - min: {cos_sim_logits_min:.6f}, max: {cos_sim_logits_max:.6f}"
    )

    if cos_sim_logprobs_mean < 0.99:
        raise AssertionError(
            f"Token cosine similarity of logprobs is less than 0.99: {cos_sim_logprobs_mean}"
        )
    if cos_sim_logits_mean < 0.99:
        raise AssertionError(
            f"Token cosine similarity of logits is less than 0.99: {cos_sim_logits_mean}"
        )


def test_fp8_bf16_gradient_comparison(fixed_input, engine_bf16, engine_fp8):
    """Compare gradients between FP8 and BF16 models after train_batch.

    This test verifies that gradients computed from FP8 and BF16 models
    are consistent across all layers.
    """
    engine_bf16.train()
    gradients_bf16 = collect_gradients_after_train_batch(engine_bf16, fixed_input)
    logger.info(f"BF16 model: collected {len(gradients_bf16)} parameter gradients")

    engine_fp8.train()
    gradients_fp8 = collect_gradients_after_train_batch(engine_fp8, fixed_input)
    logger.info(f"FP8 model: collected {len(gradients_fp8)} parameter gradients")

    # Compare gradients
    assert len(gradients_bf16) == len(gradients_fp8), (
        f"Number of parameters with gradients don't match: "
        f"BF16={len(gradients_bf16)}, FP8={len(gradients_fp8)}"
    )

    # Get common parameter names
    common_names = set(gradients_bf16.keys()) & set(gradients_fp8.keys())
    logger.info(f"Comparing {len(common_names)} common parameters")

    # Statistics for all layers
    all_max_diffs = []
    all_mean_diffs = []
    all_cos_sims = []
    layer_stats = []

    for name in sorted(common_names):
        grad_bf16 = gradients_bf16[name]
        grad_fp8 = gradients_fp8[name]

        # Check shapes match
        assert grad_bf16.shape == grad_fp8.shape, (
            f"Gradient shapes don't match for {name}: "
            f"BF16={grad_bf16.shape}, FP8={grad_fp8.shape}"
        )

        # Compute differences
        grad_diff = (grad_bf16 - grad_fp8).abs()
        max_diff = grad_diff.max().item()
        mean_diff = grad_diff.mean().item()

        # Compute cosine similarity
        grad_bf16_flat = grad_bf16.flatten()
        grad_fp8_flat = grad_fp8.flatten()
        cos_sim = F.cosine_similarity(
            grad_bf16_flat.unsqueeze(0), grad_fp8_flat.unsqueeze(0), dim=1
        ).item()

        all_max_diffs.append(max_diff)
        all_mean_diffs.append(mean_diff)
        all_cos_sims.append(cos_sim)

        # Extract layer index from parameter name for grouping
        layer_match = re.search(r"layers\.(\d+)", name)
        layer_idx = int(layer_match.group(1)) if layer_match else -1

        layer_stats.append(
            {
                "name": name,
                "layer_idx": layer_idx,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "cos_sim": cos_sim,
                "shape": grad_bf16.shape,
            }
        )

    # Log statistics by layer
    layer_indices = sorted(
        set(s["layer_idx"] for s in layer_stats if s["layer_idx"] >= 0)
    )
    for layer_idx in layer_indices:
        layer_grads = [s for s in layer_stats if s["layer_idx"] == layer_idx]
        layer_max_diffs = [s["max_diff"] for s in layer_grads]
        layer_mean_diffs = [s["mean_diff"] for s in layer_grads]
        layer_cos_sims = [s["cos_sim"] for s in layer_grads]

        logger.info(
            f"Layer {layer_idx}: "
            f"max_diff={max(layer_max_diffs):.6f}, "
            f"mean_diff={sum(layer_mean_diffs) / len(layer_mean_diffs):.6f}, "
            f"cos_sim={sum(layer_cos_sims) / len(layer_cos_sims):.6f}, "
            f"n_params={len(layer_grads)}, "
            f"names={','.join([s['name'] for s in layer_grads])}"
        )
    # log lay_idx < 0
    layer_stats_less_than_0 = [s for s in layer_stats if s["layer_idx"] < 0]
    logger.info(f"Do not have layer indices: {len(layer_stats_less_than_0)} params")
    for stat in layer_stats_less_than_0:
        name_str = f"Layer {stat['name']}"
        logger.info(
            f"{name_str:<70} "
            f"max_diff={stat['max_diff']:>12.6f}, "
            f"mean_diff={stat['mean_diff']:>12.6f}, "
            f"cos_sim={stat['cos_sim']:>10.6f}"
        )

    # Overall statistics
    overall_max_diff = max(all_max_diffs)
    overall_mean_diff = sum(all_mean_diffs) / len(all_mean_diffs)
    overall_cos_sim = sum(all_cos_sims) / len(all_cos_sims)
    overall_min_cos_sim = min(all_cos_sims)

    logger.info("=" * 80)
    logger.info("Overall gradient comparison statistics:")
    logger.info(f"  Max difference: {overall_max_diff:.6f}")
    logger.info(f"  Mean difference: {overall_mean_diff:.6f}")
    logger.info(f"  Mean cosine similarity: {overall_cos_sim:.6f}")
    logger.info(f"  Min cosine similarity: {overall_min_cos_sim:.6f}")
    logger.info("=" * 80)

    # Log parameters with lowest cosine similarity
    layer_stats_sorted = sorted(layer_stats, key=lambda x: x["cos_sim"], reverse=False)
    logger.info("Top 10 parameters with lowest gradient cosine similarity:")
    for i, stat in enumerate(layer_stats_sorted[:10]):
        logger.info(
            f"  {i + 1}. {stat['name']}: "
            f"cos_sim={stat['cos_sim']:.6f}, "
            f"max_diff={stat['max_diff']:.6f}, "
            f"mean_diff={stat['mean_diff']:.6f}"
        )

    # Assertions - allow some tolerance for FP8 quantization
    assert overall_cos_sim > 0.94, (
        f"Overall cosine similarity too low: {overall_cos_sim:.6f}. "
        f"This suggests gradients are not consistent between BF16 and FP8 models."
    )
    assert overall_min_cos_sim > 0.60, (
        f"Minimum cosine similarity too low: {overall_min_cos_sim:.6f}. "
        f"Some parameters have very different gradients."
    )


@pytest.mark.skip(reason="This test is only for debugging")
def test_profile_gemm_kernels(fixed_input, engine_bf16, engine_fp8):
    """Profile and print GEMM kernels used in forward and backward pass.

    This test profiles the GEMM kernels (matrix multiplication operations) used
    during forward and backward passes to understand which operators are being used.
    """
    logger.info("=" * 80)
    logger.info("Profiling GEMM kernels - BF16 Model")
    logger.info("=" * 80)

    # Profile forward pass
    logger.info("\n>>> Profiling FORWARD pass...")
    logits_bf16, logprobs_bf16 = forward_with_logits_and_logprobs(
        engine_bf16, fixed_input, profile_gemm=True
    )

    # Profile backward pass
    logger.info("\n>>> Profiling BACKWARD pass...")
    engine_bf16.train()
    gradients_bf16 = collect_gradients_after_train_batch(
        engine_bf16, fixed_input, profile_gemm=True
    )
    logger.info(f"Collected {len(gradients_bf16)} parameter gradients")

    logger.info("\n" + "=" * 80)
    logger.info("Profiling GEMM kernels - FP8 Model")
    logger.info("=" * 80)

    # Profile forward pass
    logger.info("\n>>> Profiling FORWARD pass...")
    logits_fp8, logprobs_fp8 = forward_with_logits_and_logprobs(
        engine_fp8, fixed_input, profile_gemm=True
    )

    # Profile backward pass
    logger.info("\n>>> Profiling BACKWARD pass...")
    engine_fp8.train()
    gradients_fp8 = collect_gradients_after_train_batch(
        engine_fp8, fixed_input, profile_gemm=True
    )
    logger.info(f"Collected {len(gradients_fp8)} parameter gradients")


def test_fp8_bf16_partial_layers_comparison(
    fixed_input, engine_bf16, engine_fp8, save_data: bool = False
):
    """Compare FP8 and BF16 on a model reduced to specified layers.

    This test reduces the model to specified transformer layers while keeping the full
    structure (embedding, rotary_pos_emb, final_layernorm, output_layer), performs
    forward and backward with real loss computation, and compares activations and
    gradients between FP8 and BF16 models to identify which operations have precision issues.
    """
    # Test specific layers - can be a single layer index or a list of indices
    layer_indices = list(
        range(2)
    )  # Test the first layer, or use [0, 1] to test first two layers

    logger.info("=" * 80)
    logger.info(f"Testing model with layers {layer_indices} - BF16 Model")
    logger.info("=" * 80)

    # Forward and backward on model with specified layers
    logits_bf16, activations_bf16, gradients_bf16, output_gradients_bf16 = (
        forward_backward_model_with_hooks(
            engine_bf16,
            fixed_input,
            layer_indices=layer_indices,
        )
    )

    logger.info(f"BF16 - Logits shape: {logits_bf16.shape}")
    logger.info(f"BF16 - Collected {len(activations_bf16)} activations")
    logger.info(f"BF16 - Collected {len(gradients_bf16)} gradients")
    logger.info(f"BF16 - Collected {len(output_gradients_bf16)} output gradients")

    logger.info("\n" + "=" * 80)
    logger.info(f"Testing model with layers {layer_indices} - FP8 Model")
    logger.info("=" * 80)

    # Forward and backward on model with specified layers
    logits_fp8, activations_fp8, gradients_fp8, output_gradients_fp8 = (
        forward_backward_model_with_hooks(
            engine_fp8,
            fixed_input,
            layer_indices=layer_indices,
        )
    )

    logger.info(f"FP8 - Logits shape: {logits_fp8.shape}")
    logger.info(f"FP8 - Collected {len(activations_fp8)} activations")
    logger.info(f"FP8 - Collected {len(gradients_fp8)} gradients")
    logger.info(f"FP8 - Collected {len(output_gradients_fp8)} output gradients")

    # Compare logits
    compare_logits(logits_bf16, logits_fp8)

    # Compare activations by op type
    activation_comparison = compare_tensors_dict(
        activations_bf16,
        activations_fp8,
        title="Activation Comparison",
        check_nan_inf=False,
        check_zero_norm=False,
        group_by_op_type=True,
        name_width=50,
    )

    # Compare gradients by op type
    gradient_comparison = compare_tensors_dict(
        gradients_bf16,
        gradients_fp8,
        title="Gradient Comparison",
        check_nan_inf=True,
        check_zero_norm=True,
        group_by_op_type=True,
        name_width=80,
    )

    # Compare output gradients by op type
    output_gradient_comparison = compare_tensors_dict(
        output_gradients_bf16,
        output_gradients_fp8,
        title="Output Gradient Comparison",
        check_nan_inf=False,
        check_zero_norm=False,
        group_by_op_type=True,
        name_width=80,
    )

    # Log problematic operations
    log_problematic_operations(
        activation_comparison["stats_by_type"],
        threshold=0.95,
        title="Problematic Activations",
    )
    log_problematic_operations(
        gradient_comparison["stats_by_type"],
        threshold=0.95,
        title="Problematic Gradients",
    )
    log_problematic_operations(
        output_gradient_comparison["stats_by_type"],
        threshold=0.95,
        title="Problematic Output Gradients",
    )

    if save_data:
        # Save q_layernorm and k_layernorm inputs and output gradients for separate testing
        layernorm_inputs_bf16 = {}
        layernorm_inputs_fp8 = {}
        layernorm_output_grads_bf16 = {}
        layernorm_output_grads_fp8 = {}
        for name in activations_bf16.keys():
            if name.endswith(".q_layernorm.input") or name.endswith(
                ".k_layernorm.input"
            ):
                layernorm_inputs_bf16[name] = activations_bf16[name]
        for name in activations_fp8.keys():
            if name.endswith(".q_layernorm.input") or name.endswith(
                ".k_layernorm.input"
            ):
                layernorm_inputs_fp8[name] = activations_fp8[name]
        for name in output_gradients_bf16.keys():
            if name.endswith(".q_layernorm.output_grad") or name.endswith(
                ".k_layernorm.output_grad"
            ):
                layernorm_output_grads_bf16[name] = output_gradients_bf16[name]
        for name in output_gradients_fp8.keys():
            if name.endswith(".q_layernorm.output_grad") or name.endswith(
                ".k_layernorm.output_grad"
            ):
                layernorm_output_grads_fp8[name] = output_gradients_fp8[name]

        if layernorm_inputs_bf16 or layernorm_inputs_fp8:
            logger.info("\n" + "=" * 80)
            logger.info("Found layernorm inputs for separate testing")
            logger.info(f"BF16 layernorm inputs: {list(layernorm_inputs_bf16.keys())}")
            logger.info(f"FP8 layernorm inputs: {list(layernorm_inputs_fp8.keys())}")
            logger.info(
                f"BF16 layernorm output grads: {list(layernorm_output_grads_bf16.keys())}"
            )
            logger.info(
                f"FP8 layernorm output grads: {list(layernorm_output_grads_fp8.keys())}"
            )
            logger.info("=" * 80)

            # Save activation inputs to files
            save_dir = Path("activation_inputs")
            save_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Also save a combined file with metadata
            combined_data = {
                "bf16_inputs": layernorm_inputs_bf16,
                "fp8_inputs": layernorm_inputs_fp8,
                "bf16_output_grads": layernorm_output_grads_bf16,
                "fp8_output_grads": layernorm_output_grads_fp8,
                "timestamp": timestamp,
                "layer_indices": layer_indices,
            }
            combined_save_path = save_dir / f"layernorm_inputs_combined_{timestamp}.pt"
            torch.save(combined_data, combined_save_path)
            logger.info(f"Saved combined layernorm inputs to: {combined_save_path}")
            logger.info(
                f"  Total size: {combined_save_path.stat().st_size / 1024 / 1024:.2f} MB"
            )

    logger.info("=" * 80)
