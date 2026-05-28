"""Model manipulation utilities for FP8/BF16 comparison tests.

This module contains functions for extracting layers, reducing models,
and collecting activations/gradients using hooks.
"""

from typing import Any

import torch
from megatron.core import parallel_state as mpu

from tests.fp8.engine_utils import (
    extract_gemm_kernels,
    print_gemm_profile,
)

from areal.engine import MegatronEngine
from areal.engine.core.train_engine import compute_total_loss_weight
from areal.engine.megatron_utils.megatron import get_named_parameters
from areal.utils import logging

logger = logging.getLogger("FP8 BF16 Model Utils")


def get_model_from_engine(engine: MegatronEngine):
    """Get the actual model module from engine, unwrapping DDP and Float16Module."""
    assert engine.model is not None, "Model is not initialized."
    model = engine.model[0]
    if hasattr(model, "module"):
        model = model.module
    # Handle Float16Module wrapper
    if hasattr(model, "module"):
        model = model.module
    return model


def reduce_model_to_layers(engine: MegatronEngine, layer_indices: list[int] | int):
    """Reduce the model to specified transformer layers while keeping full structure.

    This function modifies the model in-place by replacing decoder.layers (ModuleList)
    with a new ModuleList containing only the specified layers. This allows the model
    to maintain its full structure (embedding, rotary_pos_emb, final_layernorm, output_layer)
    so that forward pass and loss computation work correctly.

    Args:
        engine: MegatronEngine instance
        layer_indices: Index or list of indices of layers to keep (0-based).
                      If int, keeps only that layer. If list, keeps multiple layers.

    Returns:
        The original number of layers (for potential restoration)
    """
    model = get_model_from_engine(engine)

    # Get decoder
    decoder = None
    if hasattr(model, "decoder"):
        decoder = model.decoder
    elif hasattr(model, "module") and hasattr(model.module, "decoder"):
        decoder = model.module.decoder

    if decoder is None or not hasattr(decoder, "layers"):
        raise ValueError("Cannot find decoder.layers")

    original_layers = decoder.layers
    original_num_layers = len(original_layers)

    # Convert single int to list
    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]

    # Validate layer indices
    for layer_idx in layer_indices:
        if layer_idx >= original_num_layers:
            raise ValueError(
                f"Layer index {layer_idx} out of range. Model has {original_num_layers} layers."
            )

    # Remove duplicates and sort to maintain order
    layer_indices = sorted(list(set(layer_indices)))

    # Create new ModuleList with only the specified layers
    selected_layers = [original_layers[idx] for idx in layer_indices]
    new_layers = torch.nn.ModuleList(selected_layers)

    # Replace the layers ModuleList
    decoder.layers = new_layers

    if len(layer_indices) == 1:
        logger.info(
            f"Reduced model from {original_num_layers} layers to 1 layer (keeping layer {layer_indices[0]})"
        )
    else:
        logger.info(
            f"Reduced model from {original_num_layers} layers to {len(layer_indices)} layers (keeping layers {layer_indices})"
        )

    return original_num_layers


def collect_gradients_after_train_batch(
    engine: MegatronEngine, input_: dict[str, Any], profile_gemm: bool = False
) -> dict[str, torch.Tensor]:
    """Execute train_batch but collect gradients before optimizer.step().

    This function replicates the train_batch logic but stops before optimizer.step()
    to collect gradients for comparison.

    Args:
        engine: MegatronEngine instance
        input_: Input dictionary
        profile_gemm: If True, profile GEMM kernels during forward and backward pass

    Returns:
        Dictionary mapping parameter names to their gradients.
    """
    engine._ensure_ready()
    assert engine.optimizer is not None, "Optimizer is not initialized."
    engine.optimizer.zero_grad()
    for model in engine.model:
        model.zero_grad_buffer()

    # Step 1: Prepare micro-batches
    mb_list = engine._prepare_mb_list(input_).to(engine.device)

    # Step 2: Define loss functions
    def sft_loss_fn(logprobs, entropy, input_):
        """SFT loss function based on compute_packed_sft_loss."""
        del entropy  # SFT does not use entropy

        # Get loss_mask from input
        loss_mask = input_["loss_mask"].bool()

        # Shift loss_mask to align with next-token prediction
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)

        # Apply loss_mask to logprobs
        logprobs = torch.where(loss_mask, logprobs, 0)

        # Compute loss: negative log likelihood averaged over valid tokens
        device = logprobs.device
        num_valid = loss_mask.count_nonzero()
        if num_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -logprobs.sum() / num_valid
        return loss

    def loss_weight_fn(mb):
        """Loss weight function based on number of valid tokens."""
        return mb["loss_mask"].count_nonzero()

    # Step 3: Compute total loss weight
    total_loss_weight = compute_total_loss_weight(
        mb_list, loss_weight_fn, mpu.get_data_parallel_group()
    )

    # Step 4: Forward-backward using Megatron's pipeline function
    loss_multiplier = (
        mpu.get_data_parallel_world_size() * engine.optimizer.get_loss_scale().item()
    )

    def process_output(output: torch.Tensor, inputs: dict[str, Any]) -> torch.Tensor:
        return engine._compute_logprobs_and_loss(
            output,
            inputs,
            loss_fn=sft_loss_fn,
            loss_weight_fn=loss_weight_fn,
            total_loss_weight=total_loss_weight,
            loss_multiplier=loss_multiplier,
        )

    # Profile GEMM kernels if requested
    if profile_gemm:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU,
            ],
            record_shapes=True,
            with_stack=False,
            profile_memory=False,
        ) as prof:
            engine.forward_backward_batch(mb_list, process_output, forward_only=False)
            torch.cuda.synchronize()

        # Extract and print GEMM kernels
        gemm_profile = extract_gemm_kernels(prof, phase="backward")
        print_gemm_profile(gemm_profile)
    else:
        engine.forward_backward_batch(mb_list, process_output, forward_only=False)

    # Step 5: Collect gradients before optimizer.step()
    gradients = {}
    for name, param in get_named_parameters(engine.model, num_experts=None):
        if param.requires_grad:
            # Try to get gradient from param.grad or param.main_grad
            grad = None
            if hasattr(param, "main_grad") and param.main_grad is not None:
                grad = param.main_grad.clone()
            elif hasattr(param, "grad") and param.grad is not None:
                grad = param.grad.clone()
            else:
                raise ValueError(f"No gradient found for {name}")

            if grad is not None:
                if mpu.get_tensor_model_parallel_world_size() > 1:
                    raise NotImplementedError("TP gradients are not supported yet")
                gradients[name] = grad

    return gradients


def categorize_op_name(name: str) -> str:
    """Categorize operation name into op type.

    Args:
        name: Parameter or activation name

    Returns:
        Op type category: 'attention', 'mlp', 'layernorm', 'embedding', 'other'
    """
    name_lower = name.lower()
    if "attn" in name_lower or "attention" in name_lower:
        if (
            "qkv" in name_lower
            or "q_proj" in name_lower
            or "k_proj" in name_lower
            or "v_proj" in name_lower
        ):
            return "attention_proj"
        elif (
            "linear_proj" in name_lower
            or "o_proj" in name_lower
            or "out_proj" in name_lower
        ):
            return "attention_out"
        elif "core_attention" in name_lower:
            return "attention_core"
        else:
            return "attention"
    elif "mlp" in name_lower or "feedforward" in name_lower or "ffn" in name_lower:
        if "activation" in name_lower:
            return "mlp_activation"
        elif "fc1" in name_lower or "gate" in name_lower or "up" in name_lower:
            return "mlp_gate_up"
        elif "fc2" in name_lower or "down" in name_lower:
            return "mlp_down"
        else:
            return "mlp"
    elif "rotary" in name_lower or "rope" in name_lower:
        return "rope"
    elif "layernorm" in name_lower or "norm" in name_lower:
        # Distinguish Q/K layernorms from regular layernorms
        if "q_layernorm" in name_lower or "k_layernorm" in name_lower:
            return "qk_layernorm"
        return "layernorm"
    elif "embedding" in name_lower or "embed" in name_lower:
        return "embedding"
    else:
        return "other"


def forward_backward_model_with_hooks(
    engine: MegatronEngine,
    input_: dict[str, Any],
    layer_indices: list[int] | int = 0,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Perform forward and backward pass on model with specified layers and activation hooks.

    This function reduces the model to specified layers, then performs forward and backward
    using the full model structure (embedding -> layers -> final_layernorm -> output_layer),
    allowing for real loss computation.

    Args:
        engine: MegatronEngine instance
        input_: Input dictionary with 'input_ids', 'attention_mask', 'loss_mask'
        layer_indices: Index or list of indices of layers to keep (0-based).
                      If int, keeps only that layer. If list, keeps multiple layers.

    Returns:
        tuple: (logits, activations_dict, gradients_dict, output_gradients_dict)
        - logits: Output logits from the model
        - activations_dict: Dictionary mapping op names to their output activations
        - gradients_dict: Dictionary mapping parameter names to their gradients
        - output_gradients_dict: Dictionary mapping op names to their output gradients
    """
    # Convert single int to list for consistency
    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]

    # Reduce model to specified layers
    _ = reduce_model_to_layers(engine, layer_indices)

    activations = {}
    gradients = {}
    output_gradients = {}  # Gradients flowing back to module outputs
    hooks = []

    def make_activation_hook(name):
        def hook(module, input, output):
            try:
                if isinstance(output, tuple):
                    activations[name] = (
                        output[0].clone().detach() if len(output) > 0 else None
                    )
                else:
                    activations[name] = output.clone().detach()
                logger.info(
                    f"Captured activation for {name}: {activations[name].dtype}"
                )
            except Exception as e:
                logger.warning(f"Failed to capture activation for {name}: {e}")

        return hook

    def make_input_pre_hook(activation_key: str, log_name: str):
        """Create a pre-hook to capture module input."""

        def input_hook(module, input):
            try:
                if isinstance(input, tuple):
                    activations[activation_key] = (
                        input[0].clone().detach() if len(input) > 0 else None
                    )
                else:
                    activations[activation_key] = input.clone().detach()
                logger.info(
                    f"Captured {log_name} input: {activations[activation_key].shape}"
                )
            except Exception as e:
                logger.warning(f"Failed to capture {log_name} input: {e}")

        return input_hook

    def make_output_grad_hook(grad_key: str, log_name: str):
        """Create a backward hook to capture module output gradient."""

        def backward_hook(module, grad_input, grad_output):
            try:
                if grad_output is not None and len(grad_output) > 0:
                    if grad_output[0] is not None:
                        output_gradients[grad_key] = grad_output[0].clone().detach()
                        logger.info(
                            f"Captured {log_name} output grad: {output_gradients[grad_key].shape}"
                        )
            except Exception as e:
                logger.warning(f"Failed to capture {log_name} output grad: {e}")

        return backward_hook

    def register_layernorm_hooks(
        module, layer_prefix: str, layernorm_name: str
    ) -> list:
        """Register input pre-hook and backward hook for a layernorm module."""
        registered_hooks = []
        activation_key = f"{layer_prefix}.self_attention.{layernorm_name}.input"
        grad_key = f"{layer_prefix}.self_attention.{layernorm_name}.output_grad"

        pre_hook = module.register_forward_pre_hook(
            make_input_pre_hook(activation_key, layernorm_name)
        )
        registered_hooks.append(pre_hook)

        backward_hook = module.register_full_backward_hook(
            make_output_grad_hook(grad_key, layernorm_name)
        )
        registered_hooks.append(backward_hook)

        return registered_hooks

    # Get model and register hooks
    model = get_model_from_engine(engine)

    # Register hooks for components
    hook_names = []

    # Embedding
    if hasattr(model, "embedding"):
        hook_names.append(("embedding", model.embedding))
        if hasattr(model.embedding, "word_embeddings"):
            hook_names.append(
                ("embedding.word_embeddings", model.embedding.word_embeddings)
            )

    # Rotary position embedding
    if hasattr(model, "rotary_pos_emb"):
        hook_names.append(("rotary_pos_emb", model.rotary_pos_emb))

    # Decoder and layers
    if hasattr(model, "decoder"):
        decoder = model.decoder
        hook_names.append(("decoder", decoder))

        # Selected layers (after reduction)
        if hasattr(decoder, "layers") and len(decoder.layers) > 0:
            # Register hooks for each layer
            for layer_idx_in_reduced, layer in enumerate(decoder.layers):
                layer_prefix = f"layer_{layer_idx_in_reduced}"

                hook_names.append((f"{layer_prefix}", layer))

                # Input layernorm
                if hasattr(layer, "input_layernorm"):
                    hook_names.append(
                        (f"{layer_prefix}.input_layernorm", layer.input_layernorm)
                    )

                # Self attention
                if hasattr(layer, "self_attention"):
                    hook_names.append(
                        (f"{layer_prefix}.self_attention", layer.self_attention)
                    )
                    if hasattr(layer.self_attention, "linear_qkv"):
                        hook_names.append(
                            (
                                f"{layer_prefix}.self_attention.linear_qkv",
                                layer.self_attention.linear_qkv,
                            )
                        )
                    if hasattr(layer.self_attention, "linear_proj"):
                        hook_names.append(
                            (
                                f"{layer_prefix}.self_attention.linear_proj",
                                layer.self_attention.linear_proj,
                            )
                        )
                    if hasattr(layer.self_attention, "core_attention"):
                        hook_names.append(
                            (
                                f"{layer_prefix}.self_attention.core_attention",
                                layer.self_attention.core_attention,
                            )
                        )
                    # Register hooks for q_layernorm and k_layernorm
                    for layernorm_name in ["q_layernorm", "k_layernorm"]:
                        if hasattr(layer.self_attention, layernorm_name):
                            layernorm_module = getattr(
                                layer.self_attention, layernorm_name
                            )
                            hook_names.append(
                                (
                                    f"{layer_prefix}.self_attention.{layernorm_name}",
                                    layernorm_module,
                                )
                            )
                            hooks.extend(
                                register_layernorm_hooks(
                                    layernorm_module, layer_prefix, layernorm_name
                                )
                            )

                # Post attention layernorm
                if hasattr(layer, "post_attention_layernorm"):
                    hook_names.append(
                        (
                            f"{layer_prefix}.post_attention_layernorm",
                            layer.post_attention_layernorm,
                        )
                    )
                elif hasattr(layer, "pre_mlp_layernorm"):
                    hook_names.append(
                        (f"{layer_prefix}.pre_mlp_layernorm", layer.pre_mlp_layernorm)
                    )

                # MLP
                if hasattr(layer, "mlp"):
                    hook_names.append((f"{layer_prefix}.mlp", layer.mlp))
                    if hasattr(layer.mlp, "linear_fc1"):
                        hook_names.append(
                            (f"{layer_prefix}.mlp.linear_fc1", layer.mlp.linear_fc1)
                        )
                    if hasattr(layer.mlp, "linear_fc2"):
                        hook_names.append(
                            (f"{layer_prefix}.mlp.linear_fc2", layer.mlp.linear_fc2)
                        )

                    # Add pre-hook to capture activation output
                    if hasattr(layer.mlp, "linear_fc2"):
                        activation_key = f"{layer_prefix}.mlp.activation_output"
                        activation_hook = (
                            layer.mlp.linear_fc2.register_forward_pre_hook(
                                make_input_pre_hook(activation_key, "MLP activation")
                            )
                        )
                        hooks.append(activation_hook)

        # Final layernorm
        if hasattr(decoder, "final_layernorm"):
            hook_names.append(("decoder.final_layernorm", decoder.final_layernorm))

    # Output layer
    if hasattr(model, "output_layer"):
        hook_names.append(("output_layer", model.output_layer))

    # Register forward hooks and backward hooks for all modules
    for name, module in hook_names:
        try:
            # Register forward hook
            hook = module.register_forward_hook(make_activation_hook(name))
            hooks.append(hook)

            # Register backward hook to capture output gradients
            def make_backward_hook(hook_name):
                def backward_hook(module, grad_input, grad_output):
                    try:
                        if grad_output is not None and len(grad_output) > 0:
                            if grad_output[0] is not None:
                                output_gradients[f"{hook_name}.output_grad"] = (
                                    grad_output[0].clone().detach()
                                )
                                logger.debug(
                                    f"Captured output grad for {hook_name}: {output_gradients[f'{hook_name}.output_grad'].shape}"
                                )
                    except Exception as e:
                        logger.warning(
                            f"Failed to capture output grad for {hook_name}: {e}"
                        )

                return backward_hook

            backward_hook = module.register_full_backward_hook(make_backward_hook(name))
            hooks.append(backward_hook)
        except Exception as e:
            logger.warning(f"Failed to register hook for {name}: {e}")

    # Forward and backward using engine's train_batch method
    engine.train()

    # Prepare loss function
    def sft_loss_fn(logprobs, entropy, input_):
        del entropy
        loss_mask = input_["loss_mask"].bool()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        logprobs = torch.where(loss_mask, logprobs, 0)
        device = logprobs.device
        num_valid = loss_mask.count_nonzero()
        if num_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        loss = -logprobs.sum() / num_valid
        return loss

    def loss_weight_fn(mb):
        return mb["loss_mask"].count_nonzero()

    # Use engine's train_batch but collect gradients before optimizer step
    engine.optimizer.zero_grad()
    for model_chunk in engine.model:
        model_chunk.zero_grad_buffer()

    # Forward and backward
    engine.train_batch(input_, sft_loss_fn, loss_weight_fn)

    # Collect gradients from all components (focusing on the selected layers)
    model = get_model_from_engine(engine)

    # Collect gradients from all selected layers
    if (
        hasattr(model, "decoder")
        and hasattr(model.decoder, "layers")
        and len(model.decoder.layers) > 0
    ):
        for layer_idx_in_reduced, layer in enumerate(model.decoder.layers):
            layer_prefix = f"layer_{layer_idx_in_reduced}"
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    grad = None
                    if hasattr(param, "main_grad") and param.main_grad is not None:
                        grad = param.main_grad.clone().detach()
                    elif hasattr(param, "grad") and param.grad is not None:
                        grad = param.grad.clone().detach()
                    else:
                        raise ValueError(f"No gradient found for {layer_prefix}.{name}")

                    if grad is not None:
                        # Use layer_X. prefix to match activation naming
                        gradients[f"{layer_prefix}.{name}"] = grad
                    else:
                        logger.warning(f"No gradient found for {layer_prefix}.{name}")

    # Get logits by doing a forward pass
    engine.eval()
    logits = engine.forward(input_)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return logits, activations, gradients, output_gradients
