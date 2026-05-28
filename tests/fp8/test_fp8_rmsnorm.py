"""RMSNorm testing utilities for FP8/BF16 comparison tests.

This module contains RMSNorm-related classes, functions, and tests.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core.fp8_utils import get_fp8_context, is_float8tensor
from megatron.core.utils import get_model_config
from torch import nn
from torch.autograd import Function
from transformers import PretrainedConfig

from tests.fp8.engine_utils import create_engine
from tests.fp8.model_hooks import get_model_from_engine
from tests.utils import get_model_path

from areal.engine import MegatronEngine
from areal.utils import logging

logger = logging.getLogger("FP8 BF16 RMSNorm Test")

MODEL_PATH_BF16 = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)
MODEL_PATH_FP8 = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B-FP8/", "Qwen/Qwen3-0.6B-FP8"
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


def dequantize_fp8_param(tensor: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 tensor to bfloat16."""
    if is_float8tensor(tensor):
        return tensor.dequantize(dtype=torch.bfloat16)
    else:
        logger.info("Not a quantized tensor, converting to bfloat16")
        return tensor.to(torch.bfloat16)


class Qwen3RMSNormFunction(Function):
    """Custom autograd Function for Qwen3RMSNorm backward."""

    @staticmethod
    def forward(ctx, hidden_states, weight, variance_epsilon):
        """
        Forward pass for RMSNorm.

        Args:
            hidden_states: Input tensor of shape [..., hidden_size]
            weight: Weight parameter of shape [hidden_size]
            variance_epsilon: Epsilon value for numerical stability

        Returns:
            Normalized and weighted output tensor
        """
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(torch.float32)

        # Compute variance: mean(x^2) along last dimension
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)

        # Compute normalized: x / sqrt(variance + eps)
        inv_std = torch.rsqrt(variance + variance_epsilon)
        normalized = hidden_states_fp32 * inv_std

        # Apply weight and convert back to input dtype
        output = (weight * normalized).to(input_dtype)

        # Save tensors for backward
        ctx.save_for_backward(hidden_states_fp32, weight, inv_std, normalized)
        ctx.variance_epsilon = variance_epsilon
        ctx.input_dtype = input_dtype

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for RMSNorm.

        Args:
            grad_output: Gradient w.r.t. output, shape [..., hidden_size]

        Returns:
            grad_input: Gradient w.r.t. input
            grad_weight: Gradient w.r.t. weight
            grad_eps: None (variance_epsilon is not a tensor)
        """
        hidden_states, weight, inv_std, normalized = ctx.saved_tensors
        input_dtype = ctx.input_dtype

        # Convert grad_output to float32 for computation
        grad_output_fp32 = grad_output.to(torch.float32)

        # Gradient w.r.t. weight: sum over all dimensions except last
        grad_weight = (grad_output_fp32 * normalized).sum(
            dim=tuple(range(grad_output_fp32.dim() - 1))
        )

        # Gradient w.r.t. normalized: weight * grad_output
        grad_normalized = grad_output_fp32 * weight.unsqueeze(0)

        # Gradient w.r.t. variance
        inv_std_pow3 = inv_std.pow(3)
        grad_variance = (grad_normalized * hidden_states * -0.5 * inv_std_pow3).sum(
            -1, keepdim=True
        )

        # Gradient w.r.t. hidden_states
        hidden_size = hidden_states.shape[-1]
        grad_input_from_variance = grad_variance * 2.0 * hidden_states / hidden_size

        # d(normalized)/d(hidden_states) = inv_std (direct contribution)
        grad_input_from_normalized = grad_normalized * inv_std

        # Total gradient w.r.t. input
        grad_input = grad_input_from_normalized + grad_input_from_variance

        # Convert back to input dtype
        grad_input = grad_input.to(input_dtype)
        grad_weight = grad_weight.to(input_dtype)

        return grad_input, grad_weight, None


class Qwen3RMSNorm(nn.Module):
    """Qwen3RMSNorm is equivalent to T5LayerNorm."""

    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return Qwen3RMSNormFunction.apply(
            hidden_states, self.weight, self.variance_epsilon
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def forward_backward_rmsnorm_module(
    layernorm_module: torch.nn.Module,
    input_activation: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    name: str = "rmsnorm",
    collect_gradients: bool = True,
    output_grad: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Forward and backward a single RMSNorm module with given input activation.

    This function tests a RMSNorm module in isolation by:
    1. Setting the module to train mode (for gradients)
    2. Converting input to the specified dtype
    3. Running forward pass
    4. Running backward pass with a dummy loss
    5. Collecting output statistics and gradients

    Args:
        layernorm_module: The RMSNorm module to test
        input_activation: Input activation tensor
        dtype: Data type to use (torch.bfloat16 or torch.float16)
        name: Name identifier for logging
        collect_gradients: Whether to collect gradients (requires backward pass)
        output_grad: Optional gradient from downstream layers for backward pass

    Returns:
        Dictionary with output tensor, statistics, and gradients
    """
    layernorm_module.train()  # Set to train mode for gradients

    # Convert input to specified dtype and ensure it requires grad
    input_activation = input_activation.to(dtype=dtype)
    if collect_gradients:
        input_activation = input_activation.clone().detach().requires_grad_(True)

    # Forward pass
    output = layernorm_module(input_activation)

    # Calculate statistics
    output_norm = output.norm().item()
    output_max = output.abs().max().item()
    output_mean = output.mean().item()
    output_std = output.std().item()

    gradients = {}
    if collect_gradients:
        # Zero gradients first
        layernorm_module.zero_grad()
        if input_activation.grad is not None:
            input_activation.grad.zero_()

        # Use provided output gradient if available, otherwise use dummy loss
        if output_grad is not None:
            # Use the real gradient from downstream layers
            output_grad = output_grad.to(dtype=dtype, device=output.device)
            output.backward(output_grad)
        else:
            # Create a dummy loss (sum of output)
            loss = output.sum()
            # Backward pass
            loss.backward()

        # Collect gradients from module parameters
        for param_name, param in layernorm_module.named_parameters():
            if param.requires_grad:
                grad = None
                # Check different gradient storage locations
                if hasattr(param, "main_grad") and param.main_grad is not None:
                    grad = param.main_grad.clone().detach()
                elif hasattr(param, "grad") and param.grad is not None:
                    grad = param.grad.clone().detach()
                else:
                    raise ValueError(f"No gradient found for {param_name}")
                if grad is not None:
                    gradients[param_name + "_grad"] = grad
                    logger.debug(
                        f"{name} gradient {param_name}: "
                        f"shape={grad.shape}, norm={grad.norm().item():.6f}, "
                        f"min={grad.min().item():.6f}, max={grad.max().item():.6f}"
                    )

        gradients["input"] = input_activation.clone().detach()
        gradients["output"] = output.clone().detach()

        if output_grad is not None:
            gradients["output_grad"] = output_grad.clone().detach()

    logger.info(
        f"{name} ({dtype}): "
        f"input_shape={input_activation.shape}, output_shape={output.shape}, "
        f"output_norm={output_norm:.6f}, output_max={output_max:.6f}, "
        f"output_mean={output_mean:.6f}, output_std={output_std:.6f}, "
        f"n_gradients={len(gradients)}"
    )

    return {
        "output": output,
        "output_norm": output_norm,
        "output_max": output_max,
        "output_mean": output_mean,
        "output_std": output_std,
        "input_shape": input_activation.shape,
        "output_shape": output.shape,
        "gradients": gradients,
    }


def load_layernorm_inputs_from_file(file_path: str | Path) -> dict[str, Any]:
    """Load layernorm activation inputs from saved file.

    Args:
        file_path: Path to the saved .pt file (can be combined file or individual file)

    Returns:
        Dictionary with 'bf16_inputs', 'fp8_inputs', 'timestamp', 'layer_indices'
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = torch.load(file_path, map_location="cpu")

    # Check if it's a combined file or individual file
    if isinstance(data, dict) and "bf16_inputs" in data and "fp8_inputs" in data:
        # Combined file
        return data
    elif isinstance(data, dict):
        # Individual file - determine if BF16 or FP8 based on keys or filename
        if "bf16" in file_path.name.lower():
            return {
                "bf16_inputs": data,
                "fp8_inputs": {},
                "timestamp": file_path.stem.split("_")[-1]
                if "_" in file_path.stem
                else "",
                "layer_indices": [],
            }
        elif "fp8" in file_path.name.lower():
            return {
                "bf16_inputs": {},
                "fp8_inputs": data,
                "timestamp": file_path.stem.split("_")[-1]
                if "_" in file_path.stem
                else "",
                "layer_indices": [],
            }
        else:
            # Assume it's BF16 if can't determine
            return {
                "bf16_inputs": data,
                "fp8_inputs": {},
                "timestamp": file_path.stem.split("_")[-1]
                if "_" in file_path.stem
                else "",
                "layer_indices": [],
            }
    else:
        raise ValueError(f"Unexpected file format in {file_path}")


def get_custom_rmsnorm(
    layernorm_module: torch.nn.Module,
    hf_config: PretrainedConfig,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    weight: torch.Tensor | None = None,
) -> torch.nn.Module:
    """Create a custom RMSNorm module with dequantized FP8 params."""
    # Extract weight parameter
    if hasattr(layernorm_module, "weight"):
        weight_param = layernorm_module.weight
    else:
        # Try to find weight in named_parameters
        weight_param = None
        for name, param in layernorm_module.named_parameters():
            if "weight" in name.lower():
                weight_param = param
                break

    if weight_param is None:
        raise ValueError(f"Cannot find weight parameter in {layernorm_module}")

    # Dequantize if FP8, or convert to bfloat16
    dequantized_weight_data = dequantize_fp8_param(weight_param.data)

    # Get hidden_size from weight shape
    hidden_size = hf_config.head_dim
    eps = hf_config.rms_norm_eps

    # Create custom RMSNorm module
    custom_rmsnorm = Qwen3RMSNorm(hidden_size, eps=eps)
    if weight is not None:
        custom_rmsnorm.weight.data = (
            weight.clone().detach().to(device=device, dtype=dtype)
        )
    else:
        custom_rmsnorm.weight.data = dequantized_weight_data.clone().detach()
    custom_rmsnorm = custom_rmsnorm.to(device=device, dtype=dtype)

    logger.info(
        f"Using custom Qwen3RMSNorm for to replace {layernorm_module} with dtype {dtype}"
    )

    return custom_rmsnorm


def compare_rmsnorm_bf16_fp8(
    engine_bf16: MegatronEngine,
    engine_fp8: MegatronEngine,
    q_layernorm_input_bf16: torch.Tensor,
    q_layernorm_input_fp8: torch.Tensor,
    layer_path: str,
    output_grad_bf16: torch.Tensor | None = None,
    output_grad_fp8: torch.Tensor | None = None,
    use_custom_rmsnorm: bool = False,
    save_data: bool = False,
) -> dict[str, Any]:
    """Compare RMSNorm module outputs between BF16 and FP8 engines.

    This function extracts the q_layernorm module from both engines and compares
    their outputs when given the respective input activations.

    Args:
        engine_bf16: BF16 MegatronEngine
        engine_fp8: FP8 MegatronEngine
        q_layernorm_input_bf16: Input activation from BF16 model
        q_layernorm_input_fp8: Input activation from FP8 model
        layer_path: Path to identify the layer (e.g., "layer_0.self_attention.q_layernorm")
        output_grad_bf16: Optional output gradient for BF16
        output_grad_fp8: Optional output gradient for FP8
        use_custom_rmsnorm: Whether to use custom RMSNorm
        save_data: Whether to save data to file

    Returns:
        Dictionary with comparison results
    """
    logger.info("=" * 80)
    logger.info(f"Testing RMSNorm module: {layer_path}")
    logger.info("=" * 80)

    # Extract q_layernorm module from both engines
    model_bf16 = get_model_from_engine(engine_bf16)
    model_fp8 = get_model_from_engine(engine_fp8)

    # Parse layer path
    matches = re.match(
        r"layer_(\d+)\.self_attention\.(q_layernorm|k_layernorm)", layer_path
    )
    if not matches:
        raise ValueError(
            f"Invalid layer path: {layer_path}. Expected format: layer_X.self_attention.(q_layernorm|k_layernorm)"
        )
    layer_idx = int(matches.group(1))
    layernorm_type = matches.group(2)

    fp8_context = get_fp8_context(get_model_config(model_fp8), layer_no=layer_idx)

    # Get decoder and layer
    decoder_bf16 = model_bf16.decoder if hasattr(model_bf16, "decoder") else None
    decoder_fp8 = model_fp8.decoder if hasattr(model_fp8, "decoder") else None

    if decoder_bf16 is None or decoder_fp8 is None:
        raise ValueError("Cannot find decoder in model")

    if layer_idx >= len(decoder_bf16.layers) or layer_idx >= len(decoder_fp8.layers):
        raise ValueError(f"Layer index {layer_idx} out of range")

    layer_bf16 = decoder_bf16.layers[layer_idx]
    layer_fp8 = decoder_fp8.layers[layer_idx]

    if not hasattr(layer_bf16.self_attention, layernorm_type) or not hasattr(
        layer_fp8.self_attention, layernorm_type
    ):
        raise ValueError(f"Layer {layer_idx} does not have {layernorm_type}")

    layernorm_bf16 = getattr(layer_bf16.self_attention, layernorm_type)
    layernorm_fp8 = getattr(layer_fp8.self_attention, layernorm_type)

    # Test BF16
    logger.info("Testing BF16 RMSNorm...")
    if use_custom_rmsnorm:
        layernorm_bf16 = get_custom_rmsnorm(
            layernorm_bf16, engine_bf16.hf_config, engine_bf16.device, torch.bfloat16
        )
    result_bf16 = forward_backward_rmsnorm_module(
        layernorm_bf16,
        q_layernorm_input_bf16,
        output_grad=output_grad_bf16,
        dtype=torch.bfloat16,
        name=f"{layer_path} (BF16)",
        collect_gradients=True,
    )

    # Test FP8
    logger.info("Testing FP8 RMSNorm...")
    if use_custom_rmsnorm:
        # For custom RMSNorm, we dequantize params first, so no need for FP8 context
        layernorm_fp8 = get_custom_rmsnorm(
            layernorm_fp8, engine_fp8.hf_config, engine_fp8.device, torch.bfloat16
        )
        result_fp8 = forward_backward_rmsnorm_module(
            layernorm_fp8,
            q_layernorm_input_fp8,
            output_grad=output_grad_fp8,
            dtype=torch.bfloat16,  # Will use dequantized params
            name=f"{layer_path} (FP8, dequantized)",
            collect_gradients=True,
        )
    else:
        # Use original FP8 module with FP8 context
        with fp8_context:
            result_fp8 = forward_backward_rmsnorm_module(
                layernorm_fp8,
                q_layernorm_input_fp8,
                output_grad=output_grad_fp8,
                dtype=torch.bfloat16,  # Input will be converted, but module may use FP8 internally
                name=f"{layer_path} (FP8)",
                collect_gradients=True,
            )

    if save_data:
        # save input, weight, output_grad for both BF16 and FP8
        save_dir = Path("layernorm_inputs")
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"layernorm_inputs_{layer_path}_{timestamp}.pt"
        torch.save(
            {
                "bf16": {
                    "input": q_layernorm_input_bf16,
                    "weight": layernorm_bf16.weight.data.clone().detach(),
                    "output_grad": output_grad_bf16.clone().detach()
                    if output_grad_bf16 is not None
                    else None,
                },
                "fp8": {
                    "input": q_layernorm_input_fp8,
                    "weight": layernorm_fp8.weight.data.clone().detach(),
                    "output_grad": output_grad_fp8.clone().detach()
                    if output_grad_fp8 is not None
                    else None,
                },
            },
            save_path,
        )
        logger.info(f"Saved layernorm inputs to: {save_path}")

    # Compare outputs
    output_bf16 = result_bf16["output"]
    output_fp8 = result_fp8["output"]

    if output_bf16.shape != output_fp8.shape:
        logger.warning(
            f"Output shapes don't match: BF16={output_bf16.shape}, FP8={output_fp8.shape}"
        )
        return {
            "layer_path": layer_path,
            "shape_mismatch": True,
            "bf16_shape": output_bf16.shape,
            "fp8_shape": output_fp8.shape,
        }

    # Calculate differences
    output_diff = (output_bf16 - output_fp8).abs()
    max_diff = output_diff.max().item()
    mean_diff = output_diff.mean().item()

    # Cosine similarity
    output_bf16_flat = output_bf16.flatten()
    output_fp8_flat = output_fp8.flatten()
    cos_sim = F.cosine_similarity(
        output_bf16_flat.unsqueeze(0), output_fp8_flat.unsqueeze(0), dim=1
    ).item()

    logger.info("=" * 80)
    logger.info(f"RMSNorm Comparison Results for {layer_path}")
    logger.info("=" * 80)
    logger.info(
        f"Output - max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.6f}"
    )
    logger.info(
        f"BF16 output_norm={result_bf16['output_norm']:.6f}, FP8 output_norm={result_fp8['output_norm']:.6f}"
    )
    logger.info(
        f"BF16 output_max={result_bf16['output_max']:.6f}, FP8 output_max={result_fp8['output_max']:.6f}"
    )

    # Compare gradients
    gradients_bf16 = result_bf16.get("gradients", {})
    gradients_fp8 = result_fp8.get("gradients", {})

    gradient_comparison = {}
    common_gradient_names = set(gradients_bf16.keys()) & set(gradients_fp8.keys())

    if common_gradient_names:
        logger.info("\n" + "-" * 80)
        logger.info("Gradient Comparison")
        logger.info("-" * 80)

        for grad_name in sorted(common_gradient_names):
            grad_bf16 = gradients_bf16[grad_name]
            grad_fp8 = gradients_fp8[grad_name]

            if grad_bf16.shape != grad_fp8.shape:
                logger.warning(
                    f"Gradient {grad_name} shapes don't match: "
                    f"BF16={grad_bf16.shape}, FP8={grad_fp8.shape}"
                )
                continue

            # Calculate differences
            grad_diff = (grad_bf16 - grad_fp8).abs()
            grad_max_diff = grad_diff.max().item()
            grad_mean_diff = grad_diff.mean().item()

            # Cosine similarity
            grad_bf16_flat = grad_bf16.flatten()
            grad_fp8_flat = grad_fp8.flatten()
            grad_cos_sim = F.cosine_similarity(
                grad_bf16_flat.unsqueeze(0), grad_fp8_flat.unsqueeze(0), dim=1
            ).item()

            # Norms
            grad_bf16_norm = grad_bf16.norm().item()
            grad_fp8_norm = grad_fp8.norm().item()

            gradient_comparison[grad_name] = {
                "max_diff": grad_max_diff,
                "mean_diff": grad_mean_diff,
                "cos_sim": grad_cos_sim,
                "bf16_norm": grad_bf16_norm,
                "fp8_norm": grad_fp8_norm,
            }

            logger.info(
                f"{layer_path + '.' + grad_name:<80} "
                f"max_diff={grad_max_diff:>12.6f}, "
                f"mean_diff={grad_mean_diff:>12.6f}, "
                f"cos_sim={grad_cos_sim:>10.6f}, "
                f"BF16_norm={grad_bf16_norm:>12.6f}, FP8_norm={grad_fp8_norm:>12.6f}"
            )

        # Summary
        if gradient_comparison:
            avg_cos_sim = sum(g["cos_sim"] for g in gradient_comparison.values()) / len(
                gradient_comparison
            )
            max_grad_diff = max(g["max_diff"] for g in gradient_comparison.values())
            logger.info("-" * 80)
            logger.info(
                f"Gradient Summary: "
                f"avg_cos_sim={avg_cos_sim:.6f}, "
                f"max_diff={max_grad_diff:.6f}, "
                f"n_gradients={len(gradient_comparison)}"
            )
    else:
        logger.warning("No common gradients found for comparison")

    logger.info("=" * 80)

    return {
        "layer_path": layer_path,
        "output_max_diff": max_diff,
        "output_mean_diff": mean_diff,
        "output_cos_sim": cos_sim,
        "bf16_output_norm": result_bf16["output_norm"],
        "fp8_output_norm": result_fp8["output_norm"],
        "bf16_output_max": result_bf16["output_max"],
        "fp8_output_max": result_fp8["output_max"],
        "output_bf16": output_bf16,
        "output_fp8": output_fp8,
        "gradient_comparison": gradient_comparison,
    }


@pytest.mark.skip(reason="This test is only for debugging")
@pytest.mark.parametrize("use_custom_rmsnorm", [True, False])
@pytest.mark.parametrize(
    "activation_inputs_file",
    [
        "activation_inputs/layernorm_inputs_combined_20251216_170822.pt",
    ],
)
def test_rmsnorm_from_file(
    use_custom_rmsnorm: bool,
    activation_inputs_file: str | Path | None,
    engine_bf16,
    engine_fp8,
    layer_path: str | None = None,
    save_data: bool = False,
):
    """Test RMSNorm modules using activation inputs loaded from file.

    This test loads previously saved activation inputs from file and tests
    RMSNorm modules (q_layernorm and k_layernorm) in isolation.

    Args:
        activation_inputs_file: Path to the saved activation inputs file.
        layer_path: Specific layer path to test (e.g., "layer_0.self_attention.q_layernorm").
                   If None, will test all available layers.
        use_custom_rmsnorm: If True, use custom Qwen3RMSNorm with dequantized FP8 params.
                           For FP8, params will be dequantized to bfloat16 before RMSNorm.
    """

    # Load activation inputs
    logger.info("=" * 80)
    logger.info(f"Loading activation inputs from: {activation_inputs_file}")
    logger.info("=" * 80)

    data = load_layernorm_inputs_from_file(activation_inputs_file)
    bf16_inputs = data.get("bf16_inputs", {})
    fp8_inputs = data.get("fp8_inputs", {})
    bf16_output_grads = data.get("bf16_output_grads", {})
    fp8_output_grads = data.get("fp8_output_grads", {})

    logger.info(f"Loaded BF16 inputs: {list(bf16_inputs.keys())}")
    logger.info(f"Loaded FP8 inputs: {list(fp8_inputs.keys())}")

    # Find matching layer paths
    common_keys = set(bf16_inputs.keys()) & set(fp8_inputs.keys())
    if not common_keys:
        logger.warning("No common layer paths found between BF16 and FP8 inputs")
        return

    # Filter by layer_path if specified
    if layer_path:
        # Convert layer_path to input key format
        if layer_path.endswith(".q_layernorm"):
            input_key = layer_path.replace(".q_layernorm", ".q_layernorm.input")
        elif layer_path.endswith(".k_layernorm"):
            input_key = layer_path.replace(".k_layernorm", ".k_layernorm.input")
        else:
            input_key = f"{layer_path}.input"

        if input_key not in common_keys:
            logger.warning(f"Layer path {layer_path} not found in loaded inputs")
            logger.info(f"Available keys: {sorted(common_keys)}")
            return

        common_keys = {input_key}

    # Only test q_layernorm
    common_keys = {k for k in common_keys if k.endswith(".q_layernorm.input")}

    # Test each matching layer
    results = []
    for input_key in sorted(common_keys):
        # Extract layer path from input key
        if input_key.endswith(".q_layernorm.input"):
            test_layer_path = input_key.replace(".input", "")
            layernorm_type = "q_layernorm"
        elif input_key.endswith(".k_layernorm.input"):
            test_layer_path = input_key.replace(".input", "")
            layernorm_type = "k_layernorm"
        else:
            logger.warning(f"Unexpected input key format: {input_key}")
            continue

        logger.info("\n" + "=" * 80)
        logger.info(f"Testing {layernorm_type} for {test_layer_path}")
        logger.info("=" * 80)

        # Get input activations
        q_layernorm_input_bf16 = bf16_inputs[input_key]
        q_layernorm_input_fp8 = fp8_inputs[input_key]

        # Get output gradients (from downstream layers)
        output_grad_key = input_key.replace(".input", ".output_grad")
        output_grad_bf16 = bf16_output_grads.get(output_grad_key, None)
        output_grad_fp8 = fp8_output_grads.get(output_grad_key, None)

        q_layernorm_input_bf16 = q_layernorm_input_bf16.to(engine_bf16.device)
        q_layernorm_input_fp8 = q_layernorm_input_fp8.to(engine_fp8.device)
        if output_grad_bf16 is not None:
            output_grad_bf16 = output_grad_bf16.to(engine_bf16.device)
        if output_grad_fp8 is not None:
            output_grad_fp8 = output_grad_fp8.to(engine_fp8.device)

        # Compare RMSNorm
        result = compare_rmsnorm_bf16_fp8(
            engine_bf16,
            engine_fp8,
            q_layernorm_input_bf16,
            q_layernorm_input_fp8,
            test_layer_path,
            output_grad_bf16=output_grad_bf16,
            output_grad_fp8=output_grad_fp8,
            use_custom_rmsnorm=use_custom_rmsnorm,
            save_data=save_data,
        )
        results.append(result)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("RMSNorm Test Summary")
    logger.info("=" * 80)
    for result in results:
        if "shape_mismatch" in result and result["shape_mismatch"]:
            logger.warning(
                f"{result['layer_path']}: Shape mismatch - "
                f"BF16={result['bf16_shape']}, FP8={result['fp8_shape']}"
            )
        else:
            logger.info(
                f"{result['layer_path']}: "
                f"output_max_diff={result['output_max_diff']:.6f}, "
                f"output_mean_diff={result['output_mean_diff']:.6f}, "
                f"output_cos_sim={result['output_cos_sim']:.6f}"
            )

            # Gradient summary
            if "gradient_comparison" in result and result["gradient_comparison"]:
                grad_comp = result["gradient_comparison"]
                avg_grad_cos_sim = sum(g["cos_sim"] for g in grad_comp.values()) / len(
                    grad_comp
                )
                max_grad_diff = max(g["max_diff"] for g in grad_comp.values())
                logger.info(
                    f"  Gradients: "
                    f"avg_cos_sim={avg_grad_cos_sim:.6f}, "
                    f"max_diff={max_grad_diff:.6f}, "
                    f"n_gradients={len(grad_comp)}"
                )

    logger.info("=" * 80)
