# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.
# Apache-2.0; wrappers copied unchanged from Megatron-LM f007db77b9d86517af2f2cfc15b610338cbe3434.
# Compatibility for Megatron-Core 0.17 and 0.19; scoped to this recipe.

import torch


def tensor_a2a_cp2hp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: list[int] | None = None,
    undo_attention_load_balancing: bool = True,
):
    """All-to-all context parallel to hidden parallel.

    Args:
        tensor (torch.Tensor): The tensor to all-to-all.
            Currently only support (seq_len, batch, head_dim) shaped tensor.
        seq_dim (int): The dimension of sequence length. Currently only supports seq_dim == 0.
        head_dim (int): The dimension of head. Currently only supports head_dim == -1 or 2.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None, split the tensor along the dimension
            head_dim into sections first, then do all-to-all for each section separately,
            finally concatenate the separated tensors along the dimension head_dim.
        undo_attention_load_balancing (bool): Whether to undo the attention load balancing of CP.

    Returns:
        torch.Tensor: The all-to-all tensor.
    """

    from megatron.core.ssm.mamba_context_parallel import (
        _all_to_all_cp2hp,
        _undo_attention_load_balancing,
    )

    cp_size = cp_group.size()

    # No need to all-to-all if CP size is 1.
    if cp_size == 1:
        return tensor

    # Limitations of mamba_context_parallel._all_to_all_cp2hp.
    assert seq_dim == 0, (
        f"tensor_a2a_cp2hp only supports seq_dim == 0 for now, but got {seq_dim=}"
    )
    assert head_dim == -1 or head_dim == 2, (
        f"tensor_a2a_cp2hp only supports head_dim == -1 or 2 for now, but got {head_dim=}"
    )
    assert tensor.dim() == 3, (
        f"tensor_a2a_cp2hp only supports 3-d input tensor for now, but got {tensor.dim()=}"
    )

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_cp2hp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                undo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_cp2hp(tensor, cp_group)

    # Undo attention load balancing last if needed.
    if undo_attention_load_balancing:
        tensor = _undo_attention_load_balancing(tensor, cp_size)
    return tensor


def tensor_a2a_hp2cp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: list[int] | None = None,
    redo_attention_load_balancing: bool = True,
):
    """All-to-all hidden parallel to context parallel.

    Args:
        tensor (torch.Tensor): The tensor to all-to-all.
            Currently only support (seq_len, batch, head_dim) shaped tensor.
        seq_dim (int): The dimension of sequence length. Currently only supports seq_dim == 0.
        head_dim (int): The dimension of head. Currently only supports head_dim == -1 or 2.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None, first split the tensor along the
            dimension head_dim into sections, then do all-to-all for each section separately,
            finally concatenate the separated tensors along the dimension head_dim.
        redo_attention_load_balancing (bool): Whether to redo the attention load balancing of HP.

    Returns:
        torch.Tensor: The all-to-all tensor.
    """

    from megatron.core.ssm.mamba_context_parallel import (
        _all_to_all_hp2cp,
        _redo_attention_load_balancing,
    )

    cp_size = cp_group.size()

    # No need to all-to-all if CP size is 1.
    if cp_size == 1:
        return tensor

    # Limitations of mamba_context_parallel._all_to_all_hp2cp.
    assert seq_dim == 0, (
        f"tensor_a2a_hp2cp only supports seq_dim == 0 for now, but got {seq_dim=}"
    )
    assert head_dim == -1 or head_dim == 2, (
        f"tensor_a2a_hp2cp only supports head_dim == -1 or 2 for now, but got {head_dim=}"
    )
    assert tensor.dim() == 3, (
        f"tensor_a2a_hp2cp only supports 3-d input tensor for now, but got {tensor.dim()=}"
    )

    # Redo attention load balancing first if needed.
    if redo_attention_load_balancing:
        tensor = _redo_attention_load_balancing(tensor, cp_size)

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_hp2cp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                redo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_hp2cp(tensor, cp_group)

    return tensor


def install_config_compat():
    """Relax only the legacy GDN CP guard, only for bridge Qwen4 configs.

    All other TransformerConfig validation runs with the real CP size.
    Core 0.19 supports GDN CP natively and needs no config rewrite. Fail closed
    for unknown runtimes or changed legacy guards; this is not a general CP override.
    """
    import ast
    import inspect
    import textwrap

    from megatron.core.transformer import TransformerConfig

    original = TransformerConfig.__post_init__
    if getattr(original, "_qwen_gdn_cp_compat", False):
        return
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and node.msg is not None:
            if (
                "Gated delta net does not support context parallel for now"
                in ast.unparse(node.msg)
            ):
                matches.append(node)
    if not matches:
        import megatron.core as core
        from megatron.core.ssm import gated_delta_net as native

        # Check the imported runtime, including PYTHONPATH overrides. Keep its
        # original validation intact, including TP * CP head divisibility.
        if getattr(core, "__version__", None) == "0.19.0" and all(
            callable(getattr(native, name, None))
            for name in ("tensor_a2a_cp2hp", "tensor_a2a_hp2cp")
        ):
            return
    if len(matches) != 1:
        raise RuntimeError(
            "Unexpected TransformerConfig guard; review CP compatibility before enabling"
        )
    guard = matches[0]
    if ast.unparse(guard.test) != "self.context_parallel_size == 1":
        raise RuntimeError("Unexpected GDN CP guard expression")
    scoped = ast.parse(
        "getattr(self, 'hf_model_type', None) == 'qwen4_exp' and type(self).__module__ == 'mcore_bridge.config.model_config'",
        mode="eval",
    ).body
    guard.test = ast.BoolOp(op=ast.Or(), values=[guard.test, scoped])
    # Compile inside a class to preserve zero-argument super's __class__ cell.
    import types

    holder = ast.ClassDef(
        name="_CompatibilityHolder",
        bases=[],
        keywords=[],
        body=tree.body,
        decorator_list=[],
    )
    wrapped = ast.Module(body=[holder], type_ignores=[])
    ast.fix_missing_locations(wrapped)
    scope = dict(original.__globals__)
    exec(compile(wrapped, "<qwen-gdn-cp-config-compat>", "exec"), scope)
    code = scope["_CompatibilityHolder"].__post_init__.__code__
    if code.co_freevars != original.__code__.co_freevars:
        raise RuntimeError("Unexpected TransformerConfig closure")
    replacement = types.FunctionType(
        code,
        original.__globals__,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    replacement._qwen_gdn_cp_compat = True
    TransformerConfig.__post_init__ = replacement


def patch_packed_cp_forward(original):
    """Repair the pinned bridge's packed-CP divisor; reject source drift."""
    import ast
    import inspect
    import textwrap
    import types

    if getattr(original, "_qwen_packed_cp_compat", False):
        return original
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    if any(
        isinstance(node, ast.ImportFrom) and node.module == "qwen_gdn_cp_compat"
        for node in ast.walk(tree)
    ):
        raise RuntimeError(
            "Use the clean pinned bridge, not the experiment-only CP patch"
        )
    legacy = []
    fixed = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
            if ast.unparse(node.left) == "cu_seqlens":
                if ast.unparse(node.right) == "self.cp_size":
                    legacy.append(node)
                elif ast.unparse(node.right) == "cp_size":
                    fixed.append(node)
    if not legacy and len(fixed) == 1:
        return original
    if len(legacy) != 1 or fixed:
        raise RuntimeError("Unexpected bridge packed-CP divisor; review compatibility")
    legacy[0].right = ast.Name(id="cp_size", ctx=ast.Load())
    ast.fix_missing_locations(tree)
    if original.__code__.co_freevars:
        raise RuntimeError("Unexpected bridge forward closure")
    scope = dict(original.__globals__)
    exec(compile(tree, "<qwen-packed-cp-compat>", "exec"), scope)
    replacement = types.FunctionType(
        scope[original.__name__].__code__,
        original.__globals__,
        original.__name__,
        original.__defaults__,
    )
    replacement.__kwdefaults__ = original.__kwdefaults__
    replacement._qwen_packed_cp_compat = True
    return replacement


def install():
    """Enable the validated Qwen CP path without changing installed packages."""
    from mcore_bridge.model.modules.gated_delta_net import GatedDeltaNet
    from megatron.core.ssm import gated_delta_net as native

    # Prefer native helpers as a pair; a partial API is an unknown runtime.
    available = [
        hasattr(native, name) for name in ("tensor_a2a_cp2hp", "tensor_a2a_hp2cp")
    ]
    if any(available) and not all(available):
        raise RuntimeError("Incomplete native GDN CP helper API")
    patched_forward = patch_packed_cp_forward(GatedDeltaNet.forward)
    install_config_compat()
    if not any(available):
        native.tensor_a2a_cp2hp = tensor_a2a_cp2hp
        native.tensor_a2a_hp2cp = tensor_a2a_hp2cp
    GatedDeltaNet.forward = patched_forward
