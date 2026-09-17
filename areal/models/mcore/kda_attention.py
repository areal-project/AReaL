# SPDX-License-Identifier: Apache-2.0

"""KDA (Kimi Delta Attention) module for megatron-core, using fla Triton kernels.

BailingMoeV3 (Ling V3) uses a heterogeneous architecture where most layers use KDA
(a gated delta-rule linear attention with a short causal convolution) and every
``layer_group_size``-th layer uses standard MLA. KDA replaces the Lightning Attention
used in BailingMoeV2.5.

This module ports ant-megatron's ``megatron/core/ssm/kda.py`` (``KimiDeltaAttention``)
into AReaL's open-source megatron-core path. It is numerically faithful to that
reference but strips the ant-only FP8 / mxfp8 fast paths and the transpose/rms-norm
fusion optimizations (bf16 bring-up).

Reference kernels (flash-linear-attention + causal-conv1d):
    - fla.ops.kda.chunk_kda          (chunked training kernel, gated delta rule)
    - fla.ops.kda.gate.fused_kda_gate (decay gate from A_log / dt_bias)
    - fla.modules.l2norm.l2_norm
    - causal_conv1d.causal_conv1d_fn  (depthwise causal short conv + silu)

Architecture (no_kda_lora=True, the v3 setting):
    in_proj(hidden) -> [qkv | g | gate]
    qkv -> causal_conv1d (depthwise, silu) -> split [q, k, v]
    beta = sigmoid(beta_proj(hidden))
    g (decay) via fused_kda_gate(A_log, dt_bias) or inside chunk_kda
    core = chunk_kda(q, k, v, g, beta, A_log, dt_bias, qk_l2norm)
    out  = out_norm(core) * sigmoid(gate)            # per-head RMSNorm over v_head_dim
    out_proj(out)

Notes:
    - KDA does NOT use RoPE (it is a delta-rule SSM); ``rotary_pos_emb`` is ignored.
    - Context parallelism uses HybridEngine's all2all strategy: CP sequence shards are
      exchanged into head shards before KDA, then exchanged back before output norm.
"""

import math
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from torch import Tensor

from areal.utils import logging

logger = logging.getLogger("KDAAttention")

try:
    from torch.distributed._functional_collectives import all_to_all_single_autograd
except ImportError:  # pragma: no cover
    all_to_all_single_autograd = None

# Optional ant-megatron extension: packed-seq params that carry an explicit seq_idx.
try:
    from megatron.core.packed_seq_params import PackedSeqParamsWithSeqidx
except ImportError:  # pragma: no cover - not present in upstream megatron-core
    PackedSeqParamsWithSeqidx = None

# fla / causal-conv1d kernels (optional import; required at runtime for KDA).
try:
    from fla.modules.l2norm import l2_norm as l2norm
    from fla.ops.kda import chunk_kda, fused_recurrent_kda

    HAVE_FLA = True
except ImportError:  # pragma: no cover
    l2norm = None
    chunk_kda = None
    fused_recurrent_kda = None
    HAVE_FLA = False

try:
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:  # pragma: no cover
    fused_kda_gate = None

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:  # pragma: no cover
    causal_conv1d_fn = None


def _get_tp_world_size() -> int:
    """Tensor-model-parallel world size, with a fallback for uninitialized mpu."""
    try:
        if mpu.model_parallel_is_initialized():
            return mpu.get_tensor_model_parallel_world_size()
    except (RuntimeError, AttributeError):
        pass
    return 1


def _get_cp_world_size() -> int:
    """Context-parallel world size, with a fallback for uninitialized mpu."""
    try:
        if mpu.model_parallel_is_initialized():
            return mpu.get_context_parallel_world_size()
    except (RuntimeError, AttributeError):
        pass
    return 1


def _get_cp_rank() -> int:
    """Context-parallel rank, with a fallback for uninitialized mpu."""
    try:
        if mpu.model_parallel_is_initialized():
            return mpu.get_context_parallel_rank()
    except (RuntimeError, AttributeError):
        pass
    return 0


def _get_cp_group():
    """Context-parallel process group, with a fallback for uninitialized mpu."""
    try:
        if mpu.model_parallel_is_initialized():
            return mpu.get_context_parallel_group()
    except (RuntimeError, AttributeError):
        pass
    return None


class _AllToAll(torch.autograd.Function):
    """Autograd fallback for equal-size all-to-all."""

    @staticmethod
    def forward(ctx, group, input_: Tensor):
        ctx.group = group
        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return input_
        input_ = input_.contiguous()
        output = torch.empty_like(input_)
        dist.all_to_all_single(output, input_, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, _AllToAll.apply(ctx.group, grad_output)


def _all_to_all_equal(input_: Tensor, cp_group) -> Tensor:
    """Equal-size all-to-all over the CP group with autograd support."""
    if cp_group is None:
        return input_
    world_size = dist.get_world_size(group=cp_group)
    if world_size == 1:
        return input_
    input_shape = input_.shape
    flat = input_.contiguous().reshape(-1)
    if all_to_all_single_autograd is not None:
        exchanged = all_to_all_single_autograd(flat, None, None, group=cp_group)
    else:  # pragma: no cover - old torch fallback
        exchanged = _AllToAll.apply(cp_group, flat)
    return exchanged.reshape(input_shape)


def _all_to_all_cp2hp(
    input_: Tensor,
    cp_group,
    split_size_or_sections: list[int] | None = None,
) -> Tensor:
    """CP sequence shard -> head shard.

    Shape: ``[S/CP, B, H] -> [S, B, H/CP]``.
    """
    if cp_group is None:
        return input_
    cp_size = dist.get_world_size(group=cp_group)
    if cp_size == 1:
        return input_
    if split_size_or_sections is not None:
        chunks = torch.split(input_, split_size_or_sections, dim=-1)
        return torch.cat(
            [_all_to_all_cp2hp(chunk, cp_group) for chunk in chunks],
            dim=-1,
        )
    assert input_.dim() == 3, input_.shape
    seq_len, batch, hidden = input_.shape
    if hidden % cp_size != 0:
        raise ValueError(
            f"KDA all2all CP requires hidden dim {hidden} divisible by CP {cp_size}."
        )
    hidden_per_cp = hidden // cp_size
    flat = input_.reshape(seq_len * batch, hidden)
    flat = torch.cat(torch.split(flat, hidden_per_cp, dim=-1), dim=0)
    flat = _all_to_all_equal(flat, cp_group)
    return flat.reshape(seq_len * cp_size, batch, hidden_per_cp)


def _all_to_all_hp2cp(input_: Tensor, cp_group) -> Tensor:
    """Head shard -> CP sequence shard.

    Shape: ``[S, B, H/CP] -> [S/CP, B, H]``.
    """
    if cp_group is None:
        return input_
    cp_size = dist.get_world_size(group=cp_group)
    if cp_size == 1:
        return input_
    assert input_.dim() == 3, input_.shape
    seq_len, batch, hidden_per_cp = input_.shape
    if seq_len % cp_size != 0:
        raise ValueError(
            f"KDA all2all CP requires sequence dim {seq_len} divisible by CP {cp_size}."
        )
    seq_per_cp = seq_len // cp_size
    flat = input_.reshape(seq_len * batch, hidden_per_cp)
    flat = _all_to_all_equal(flat, cp_group)
    chunks = torch.split(flat, seq_per_cp * batch, dim=0)
    return torch.cat(chunks, dim=-1).reshape(seq_per_cp, batch, hidden_per_cp * cp_size)


def _get_parameter_local_cp(
    param: Tensor,
    dim: int,
    cp_rank: int,
    cp_size: int,
    split_size_or_sections: list[int] | None = None,
) -> Tensor:
    """Slice a TP-local parameter for HybridEngine-style all2all CP."""
    if cp_size == 1:
        return param
    if split_size_or_sections is not None:
        chunks = torch.split(param, split_size_or_sections, dim=dim)
        return torch.cat(
            [_get_parameter_local_cp(chunk, dim, cp_rank, cp_size) for chunk in chunks],
            dim=dim,
        )
    dim_size = param.size(dim)
    if dim_size % cp_size != 0:
        raise ValueError(
            f"Cannot CP-slice parameter dim {dim} of size {dim_size} by CP {cp_size}."
        )
    per_rank = dim_size // cp_size
    slices = [slice(None)] * param.dim()
    slices[dim] = slice(cp_rank * per_rank, (cp_rank + 1) * per_rank)
    return param[tuple(slices)]


def _build_zigzag_undo_indices(
    total_len: int,
    cp_size: int,
    cu_seqlens: Tensor | None,
    device: torch.device,
) -> Tensor:
    """Undo AReaL packed CP zigzag ordering after CP->HP all-to-all.

    Fully vectorized: for every canonical position the source index is
    computed with tensor ops, so the only host sync is the single
    divisibility check (and callers cache the result per microbatch via
    ``_get_zigzag_undo_redo_indices``).
    """
    indices = torch.arange(total_len, dtype=torch.long, device=device)
    if cp_size <= 1:
        return indices

    if cu_seqlens is None:
        cu = torch.tensor([0, total_len], dtype=torch.long, device=device)
    else:
        cu = cu_seqlens.to(device=device, dtype=torch.long)
    lens = cu[1:] - cu[:-1]
    if bool(((lens % (2 * cp_size)) != 0).any()):
        raise ValueError(
            f"Packed sequence lengths {lens.tolist()} must be divisible by "
            f"2*CP={2 * cp_size} for KDA CP zigzag reorder."
        )

    t_per_cp = total_len // cp_size
    # For canonical (undone) position p in sequence i with offset o and
    # zigzag chunk size c_i: chunk index k = o // c selects rank k for the
    # front half (k < cp) and rank 2*cp-1-k for the mirrored back half; the
    # source row lives in that rank's all-to-all block at the sequence's
    # CP-local offset.
    seq = torch.searchsorted(cu, indices, right=True) - 1
    o = indices - cu[seq]
    c = (lens // (2 * cp_size))[seq]
    cu_s = (cu[:-1] // cp_size)[seq]
    k = o // c
    j = o % c
    front = k < cp_size
    rank = torch.where(front, k, 2 * cp_size - 1 - k)
    return rank * t_per_cp + cu_s + torch.where(front, j, j + c)


def _build_zigzag_redo_indices(undo_indices: Tensor) -> Tensor:
    """Inverse permutation of ``_build_zigzag_undo_indices``."""
    redo = torch.empty_like(undo_indices)
    redo[undo_indices] = torch.arange(undo_indices.numel(), device=undo_indices.device)
    return redo


def _get_zigzag_undo_redo_indices(
    total_len: int,
    cp_size: int,
    cu_seqlens: Tensor | None,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Build (or fetch cached) zigzag undo/redo index pairs.

    The permutation only depends on (total_len, cp_size, cu_seqlens), which
    are identical for every KDA layer — and every recompute replay — of the
    same microbatch, so cache on the cu_seqlens tensor object instead of
    rebuilding (and syncing) per layer.
    """
    key = (int(total_len), int(cp_size), str(device))
    cache = (
        getattr(cu_seqlens, "_kda_zigzag_cache", None)
        if cu_seqlens is not None
        else None
    )
    if cache is not None and key in cache:
        return cache[key]
    undo = _build_zigzag_undo_indices(total_len, cp_size, cu_seqlens, device)
    redo = _build_zigzag_redo_indices(undo)
    if cu_seqlens is not None:
        if cache is None:
            cache = {}
            cu_seqlens._kda_zigzag_cache = cache
        cache[key] = (undo, redo)
    return undo, redo


@dataclass
class KimiDeltaAttentionSubmodules:
    """Module specs for the input/output projections and the output norm.

    For the v3 setting (``no_kda_lora=True``) ``f_b_proj`` and ``g_b_proj`` are
    unused (Identity) because ``g`` and ``gate`` are produced directly by the fused
    ``in_proj``.
    """

    in_proj: ModuleSpec | type = IdentityOp
    beta_proj: ModuleSpec | type = IdentityOp
    f_b_proj: ModuleSpec | type = IdentityOp
    g_b_proj: ModuleSpec | type = IdentityOp
    out_norm: ModuleSpec | type = IdentityOp
    out_proj: ModuleSpec | type = IdentityOp


class KimiDeltaAttention(MegatronModule):
    """KDA layer: input ``[s, b, h]`` -> output ``[s, b, h]``.

    Ported from ant-megatron ``megatron/core/ssm/kda.py``. FP8/mxfp8 fast paths and
    the transpose / rms-norm fusion optimizations are intentionally omitted for the
    bf16 bring-up; the numerical chain is preserved.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: KimiDeltaAttentionSubmodules,
        layer_number: int = None,
        attn_mask_type=None,
        *,
        head_dim: int | None = None,
        conv_kernel_dim: int = 4,
        no_kda_lora: bool = True,
        use_qk_l2norm: bool = True,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        A_init_range: tuple[float, float] = (1, 16),
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: float | None = None,
        **kwargs,
    ):
        if not HAVE_FLA:  # pragma: no cover
            raise ImportError(
                "flash-linear-attention (fla) with KDA kernels is required. "
                "Install a version exposing fla.ops.kda.{chunk_kda,fused_recurrent_kda} "
                "and fla.ops.kda.gate.fused_kda_gate."
            )

        super().__init__(config)

        # Attributes from arguments
        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        self.use_qk_l2norm = use_qk_l2norm
        self.no_kda_lora = no_kda_lora
        # KDA decay-gate clamping. BailingMoeV3 HF checkpoints are trained with
        # ``kda_safe_gate=True`` / ``kda_lower_bound=-5.0``: the decay gate is the
        # bounded ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` rather than the
        # default unbounded ``-exp(A_log) * softplus(g + dt_bias)``. These must be
        # forwarded to the kernel or the loaded weights will not reproduce their forward.
        self.safe_gate = safe_gate
        self.lower_bound = lower_bound
        self.tp_size = _get_tp_world_size()
        self.cp_size = _get_cp_world_size()
        self.cp_rank = _get_cp_rank()
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        # Attributes from config
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim if head_dim is not None else config.kv_channels
        self.act_fn = config.activation_func
        self.activation = getattr(self.act_fn, "__name__", "silu")
        self.conv_kernel_dim = conv_kernel_dim
        self.key_head_dim = self.head_dim
        self.value_head_dim = self.head_dim
        self.num_key_heads = config.num_attention_heads
        self.num_value_heads = config.num_attention_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads
        assert self.num_value_heads % self.num_key_heads == 0
        if self.cp_size > 1:
            heads_per_tp = self.num_key_heads // self.tp_size
            if heads_per_tp % self.cp_size != 0:
                raise ValueError(
                    "KDA all2all CP requires num_attention_heads / TP divisible "
                    f"by CP, got heads={self.num_key_heads}, TP={self.tp_size}, "
                    f"CP={self.cp_size}."
                )
            logger.info(
                f"KDA all2all CP enabled: cp_size={self.cp_size}, "
                f"heads_per_tp={heads_per_tp}, heads_per_cp={heads_per_tp // self.cp_size}"
            )

        # nGPT value normalization is unused by released v3 configs.
        self.use_nGPT = getattr(config, "use_nGPT", False)
        self.value_norm = getattr(config, "value_norm", False)

        # Whether the decay gate is computed inside chunk_kda or via fused_kda_gate.
        # If fla exposes kda_gate_ref we may apply the gate externally
        # (use_gate_in_kernel=False).
        try:
            from fla.ops.kda.gate import kda_gate_ref  # noqa: F401

            self.use_gate_in_kernel = False
        except ImportError:
            self.use_gate_in_kernel = True

        # When the checkpoint requires a clamped (safe) gate, force the in-kernel path:
        # it matches the HF BailingMoeV3 forward exactly (chunk_kda(use_gate_in_kernel=
        # True, safe_gate=..., lower_bound=...)). The external fused_kda_gate path is
        # avoided here because its clamping signature varies across fla forks.
        #
        # GUARD: chunk_kda only grew safe_gate/lower_bound in Arc fla >= v0.4.2. Older
        # forks (e.g. the v1.5.0-pinned e131287, or v0.4.0) silently swallow them via
        # **kwargs and revert to the unbounded softplus decay -> wrong (silent) loss.
        # Fail loudly instead so the fla version mismatch is caught at build time.
        if self.safe_gate or self.lower_bound is not None:
            import inspect

            try:
                _kda_params = inspect.signature(chunk_kda).parameters
            except (TypeError, ValueError):  # pragma: no cover - builtin/extension
                _kda_params = {}
            if "lower_bound" not in _kda_params or "safe_gate" not in _kda_params:
                raise ImportError(
                    "This BailingMoeV3 checkpoint needs the clamped KDA gate "
                    f"(safe_gate={self.safe_gate}, lower_bound={self.lower_bound}), but "
                    "the installed flash-linear-attention's chunk_kda does not accept "
                    "these arguments (they would be silently ignored, giving the wrong "
                    "unbounded-softplus decay). Use Arc flash-linear-attention >= v0.4.2 "
                    "(e.g. branch bailing_fla_v0.4.2_env_autotune)."
                )
            self.use_gate_in_kernel = True
        logger.info(
            f"KDA use_gate_in_kernel={self.use_gate_in_kernel} "
            f"safe_gate={self.safe_gate} lower_bound={self.lower_bound}"
        )

        # Input projection (fused). no_kda_lora -> [qkv | g(qk_dim) | gate(v_dim)].
        if not self.no_kda_lora:
            self.in_proj_dim = self.qk_dim * 2 + self.v_dim + self.value_head_dim * 2
        else:
            self.in_proj_dim = self.qk_dim * 2 + self.v_dim + self.qk_dim + self.v_dim

        self.in_proj = build_module(
            submodules.in_proj,
            self.hidden_size,
            self.in_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
        )

        self.beta_proj = build_module(
            submodules.beta_proj,
            self.hidden_size,
            self.num_key_heads,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
        )

        if not self.no_kda_lora:
            self.f_b_proj = build_module(
                submodules.f_b_proj,
                self.value_head_dim,
                self.qk_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="f_b_proj",
            )
        else:
            self.f_b_proj = nn.Identity()

        # Depthwise causal Conv1d over the (q, k, v) channels.
        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.conv_dim_local_tp = self.conv_dim // self.tp_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim_local_tp,
            out_channels=self.conv_dim_local_tp,
            bias=conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim_local_tp,
            padding=self.conv_kernel_dim - 1,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        # partition_dim is required alongside tensor_model_parallel: the mbridge
        # HF-export fallback concatenates TP shards with
        # ``torch.cat(shards, dim=param.partition_dim)``, and mbridge's get_model
        # fills an unset partition_dim with -1 — which silently merges the 3-D
        # conv1d weight along the kernel axis at TP>1.
        setattr(self.conv1d.weight, "tensor_model_parallel", True)
        setattr(self.conv1d.weight, "partition_dim", 0)
        if conv_bias:
            setattr(self.conv1d.bias, "tensor_model_parallel", True)
            setattr(self.conv1d.bias, "partition_dim", 0)

        self.num_k_heads_local_tp = self.num_key_heads // self.tp_size

        with get_cuda_rng_tracker().fork():
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max.
            # Keep dt_bias in fp32 (like A_log below): the HF ckpt stores it as fp32 and
            # it feeds the decay gate exp(A_log)*(g+dt_bias), which is sensitive to
            # precision. Also, hf_load's fp32-buffer exemption only fires when the target
            # param is already fp32 — a bf16 dt_bias would be silently loaded in bf16.
            dt = torch.exp(
                torch.rand(
                    self.qk_dim // self.tp_size,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            self.dt_bias._no_reinit = True
            self.dt_bias._no_weight_decay = True
            setattr(self.dt_bias, "tensor_model_parallel", True)
            setattr(self.dt_bias, "partition_dim", 0)

            A = torch.empty(
                self.num_k_heads_local_tp,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            ).uniform_(*A_init_range)
            A_log = torch.log(A)  # keep A_log in fp32
            self.A_log = nn.Parameter(A_log)
            self.A_log._no_weight_decay = True
            setattr(self.A_log, "tensor_model_parallel", True)
            setattr(self.A_log, "partition_dim", 0)

        if not self.no_kda_lora:
            self.g_b_proj = build_module(
                submodules.g_b_proj,
                self.value_head_dim,
                self.v_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="g_b_proj",
            )
        else:
            self.g_b_proj = nn.Identity()

        # Output norm (per-head RMSNorm over value_head_dim) applied before gating.
        self.out_norm = build_module(
            submodules.out_norm,
            config=self.config,
            hidden_size=self.value_head_dim,
            eps=self.config.layernorm_epsilon,
        )

        self.out_proj = build_module(
            submodules.out_proj,
            self.v_dim,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the convolution parameters if a custom init range is provided."""
        if self.config.perform_initialization and self.conv_init is not None:
            with get_cuda_rng_tracker().fork():
                nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

    def _pad_packed_qkv(self, qkv: Tensor, cu_seqlens: Tensor) -> tuple[Tensor, int]:
        """Pad a packed qkv tensor ``[Total_Seq, Dim]`` into ``[B, Dim, Max_Seq]``."""
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        batch_size = len(seqlens)
        max_seqlen = seqlens.max().item()

        padded = torch.zeros(
            batch_size, max_seqlen, qkv.shape[-1], dtype=qkv.dtype, device=qkv.device
        )
        total_tokens = cu_seqlens[-1].item()
        batch_indices = torch.arange(
            batch_size, device=qkv.device, dtype=torch.long
        ).repeat_interleave(seqlens)
        offsets = cu_seqlens[:-1].repeat_interleave(seqlens)
        seq_indices = (
            torch.arange(total_tokens, device=qkv.device, dtype=torch.long) - offsets
        )
        padded[batch_indices, seq_indices] = qkv
        return padded.transpose(1, 2).contiguous(), max_seqlen

    def _unpad_packed_qkv(self, padded_qkv: Tensor, cu_seqlens: Tensor) -> Tensor:
        """Unpad ``[B, Dim, Max_Seq]`` convolution output back to ``[Total_Seq, Dim]``."""
        padded_qkv = padded_qkv.transpose(1, 2)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        batch_size = len(seqlens)
        total_tokens = cu_seqlens[-1].item()
        batch_indices = torch.arange(
            batch_size, device=padded_qkv.device, dtype=torch.long
        ).repeat_interleave(seqlens)
        offsets = cu_seqlens[:-1].repeat_interleave(seqlens)
        seq_indices = (
            torch.arange(total_tokens, device=padded_qkv.device, dtype=torch.long)
            - offsets
        )
        return padded_qkv[batch_indices, seq_indices]

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        key_value_states: Tensor | None = None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params=None,
        **kwargs,
    ):
        """Forward pass. ``hidden_states`` is ``[S, B, H]``; RoPE is ignored (KDA is an SSM)."""
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError("KDA does not support inference for now.")

        seq_len_hidden_states, batch, _ = hidden_states.shape
        seq_len = seq_len_hidden_states * self.sp_size * self.cp_size

        cu_seqlens_q = None
        if packed_seq_params is not None:
            cu_seqlens_q = packed_seq_params.cu_seqlens_q

        # Fused input projection -> [qkv | g | gate]
        qkvfg, _ = self.in_proj(hidden_states)
        qkv, g, gate = torch.split(
            qkvfg,
            [
                (self.qk_dim * 2 + self.v_dim) // self.tp_size,
                (self.value_head_dim if not self.no_kda_lora else self.qk_dim)
                // self.tp_size,
                (self.value_head_dim if not self.no_kda_lora else self.v_dim)
                // self.tp_size,
            ],
            dim=-1,
        )

        beta, _ = self.beta_proj(hidden_states)

        cp_group = _get_cp_group() if self.cp_size > 1 else None
        undo_idx = None
        redo_idx = None
        qkv_split_sections = [
            self.qk_dim // self.tp_size,
            self.qk_dim // self.tp_size,
            self.v_dim // self.tp_size,
        ]

        if self.cp_size > 1:
            # Convert CP sequence shards into head shards before convolution/KDA.
            # q/k/v must be exchanged independently to preserve fused channel layout:
            # [S/CP, B, (Q|K|V)/TP] -> [S, B, (Q|K|V)/(TP*CP)].
            qkv = _all_to_all_cp2hp(
                qkv,
                cp_group,
                split_size_or_sections=qkv_split_sections,
            )
            beta = _all_to_all_cp2hp(beta, cp_group)
            undo_idx, redo_idx = _get_zigzag_undo_redo_indices(
                qkv.shape[0], self.cp_size, cu_seqlens_q, qkv.device
            )
            qkv = qkv[undo_idx]
            beta = beta[undo_idx]

        # seq-first [S, B, *] -> batch-first [B, S, *]
        beta = beta.transpose(0, 1)
        qkv = qkv.transpose(0, 1)

        qk_dim_local = self.qk_dim // (self.tp_size * self.cp_size)
        v_dim_local = self.v_dim // (self.tp_size * self.cp_size)
        conv1d_weight = self.conv1d.weight
        conv1d_bias = self.conv1d.bias
        if self.cp_size > 1:
            conv1d_weight = _get_parameter_local_cp(
                conv1d_weight,
                dim=0,
                cp_rank=self.cp_rank,
                cp_size=self.cp_size,
                split_size_or_sections=qkv_split_sections,
            )
            if conv1d_bias is not None:
                conv1d_bias = _get_parameter_local_cp(
                    conv1d_bias,
                    dim=0,
                    cp_rank=self.cp_rank,
                    cp_size=self.cp_size,
                    split_size_or_sections=qkv_split_sections,
                )

        # Depthwise causal short convolution (+ silu) over the q/k/v channels.
        if packed_seq_params is not None:
            b, s, d = qkv.shape
            if causal_conv1d_fn is None:
                qkv_flat = qkv.reshape(-1, d)
                qkv_padded, max_seqlen_q = self._pad_packed_qkv(qkv_flat, cu_seqlens_q)
                qkv_conv = self.act_fn(
                    F.conv1d(
                        qkv_padded,
                        conv1d_weight,
                        conv1d_bias,
                        padding=self.conv_kernel_dim - 1,
                        groups=conv1d_weight.shape[0],
                    )
                )[..., :max_seqlen_q]
                qkv = self._unpad_packed_qkv(qkv_conv, cu_seqlens_q).reshape(b, s, d)
            else:
                assert self.activation in ["silu", "swish"]
                if (
                    PackedSeqParamsWithSeqidx is not None
                    and isinstance(packed_seq_params, PackedSeqParamsWithSeqidx)
                    and packed_seq_params.seq_idx is not None
                ):
                    seq_idx = packed_seq_params.seq_idx
                else:
                    seqlens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                    seq_idx = (
                        torch.repeat_interleave(
                            torch.arange(
                                len(seqlens), device=qkv.device, dtype=torch.int32
                            ),
                            seqlens,
                        )
                        .unsqueeze(0)
                        .contiguous()
                    )
                qkv_input = qkv.reshape(1, -1, d).contiguous().transpose(1, 2)
                qkv_conv = causal_conv1d_fn(
                    x=qkv_input,
                    weight=conv1d_weight.squeeze(1),
                    bias=conv1d_bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                )
                qkv = qkv_conv.transpose(1, 2).reshape(b, s, d)
        else:
            qkv = qkv.transpose(1, 2).contiguous()  # [B, Dim, S]
            if causal_conv1d_fn is None:
                qkv = self.act_fn(
                    F.conv1d(
                        qkv,
                        conv1d_weight,
                        conv1d_bias,
                        padding=self.conv_kernel_dim - 1,
                        groups=conv1d_weight.shape[0],
                    )
                )[..., :seq_len]
            else:
                assert self.activation in ["silu", "swish"]
                qkv = causal_conv1d_fn(
                    x=qkv,
                    weight=conv1d_weight.squeeze(1),
                    bias=conv1d_bias,
                    activation=self.activation,
                )
            qkv = qkv.transpose(1, 2)  # [B, S, Dim]

        query, key, value = torch.split(
            qkv,
            [
                qk_dim_local,
                qk_dim_local,
                v_dim_local,
            ],
            dim=-1,
        )

        # Decay-gate features `g` (no_kda_lora -> Identity, already produced by in_proj).
        if not self.no_kda_lora and self.tp_size > 1:
            g = gather_from_tensor_model_parallel_region(g)
            if self.config.sequence_parallel:
                g = scatter_to_sequence_parallel_region(g)
        if not self.no_kda_lora:
            g, _ = self.f_b_proj(g)
        if self.cp_size > 1:
            g = _all_to_all_cp2hp(g, cp_group)
            g = g[undo_idx]
        g = g.transpose(0, 1)  # [B, S, *]

        seq_len_for_kda = qkv.shape[1]
        if packed_seq_params is not None:
            query = query.reshape(
                1, seq_len_for_kda * batch, -1, self.key_head_dim
            ).contiguous()
            key = key.reshape(
                1, seq_len_for_kda * batch, -1, self.key_head_dim
            ).contiguous()
            value = value.reshape(
                1, seq_len_for_kda * batch, -1, self.value_head_dim
            ).contiguous()
            beta = beta.reshape(1, seq_len_for_kda * batch, -1).contiguous()
            if self.use_gate_in_kernel:
                g = g.reshape(
                    1, seq_len_for_kda * batch, -1, self.key_head_dim
                ).contiguous()
            else:
                g = g.reshape(1, seq_len_for_kda * batch, -1).contiguous()
        else:
            query = query.reshape(
                batch, seq_len_for_kda, -1, self.key_head_dim
            ).contiguous()
            key = key.reshape(
                batch, seq_len_for_kda, -1, self.key_head_dim
            ).contiguous()
            value = value.reshape(
                batch, seq_len_for_kda, -1, self.value_head_dim
            ).contiguous()
            beta = beta.reshape(batch, seq_len_for_kda, -1).contiguous()
            if self.use_gate_in_kernel:
                g = g.reshape(
                    batch, seq_len_for_kda, -1, self.key_head_dim
                ).contiguous()

        if self.use_qk_l2norm and self.use_nGPT and self.value_norm:
            value = l2norm(value.contiguous())

        A_log = self.A_log
        dt_bias = self.dt_bias
        if self.cp_size > 1:
            A_log = _get_parameter_local_cp(
                self.A_log, dim=0, cp_rank=self.cp_rank, cp_size=self.cp_size
            )
            dt_bias = _get_parameter_local_cp(
                self.dt_bias, dim=0, cp_rank=self.cp_rank, cp_size=self.cp_size
            )

        if not self.use_gate_in_kernel:
            g = fused_kda_gate(
                g, A_log.view(1, 1, -1, 1), self.key_head_dim, g_bias=dt_bias
            )
        beta = beta.float().sigmoid()

        core_attn_out, _ = chunk_kda(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            use_gate_in_kernel=self.use_gate_in_kernel,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            cu_seqlens=cu_seqlens_q,
        )

        # Output gate `gate` (no_kda_lora -> Identity, already produced by in_proj).
        if not self.no_kda_lora and self.tp_size > 1:
            gate = gather_from_tensor_model_parallel_region(gate)
            if self.config.sequence_parallel:
                gate = scatter_to_sequence_parallel_region(gate)
        if not self.no_kda_lora:
            gate, _ = self.g_b_proj(gate)

        if self.cp_size > 1:
            core_attn_out = core_attn_out.reshape(batch, seq_len_for_kda, -1)
            core_attn_out = core_attn_out.transpose(0, 1).contiguous()
            core_attn_out = core_attn_out[redo_idx]
            core_attn_out = _all_to_all_hp2cp(core_attn_out, cp_group)

            local_seq_len = core_attn_out.shape[0]
            core_attn_out = core_attn_out.reshape(
                local_seq_len, batch, -1, self.value_head_dim
            )
            gate = gate.contiguous().reshape(
                local_seq_len, batch, -1, self.value_head_dim
            )
            norm_out = self._apply_gated_norm(core_attn_out, gate)
            norm_out = norm_out.reshape(local_seq_len, batch, -1)
        else:
            gate = gate.transpose(0, 1)
            gate = gate.contiguous().reshape(
                batch, seq_len_for_kda, -1, self.value_head_dim
            )
            norm_out = self._apply_gated_norm(core_attn_out, gate)
            norm_out = norm_out.reshape(batch, seq_len_for_kda, -1)
            norm_out = norm_out.transpose(0, 1).contiguous()  # [S, B, v_dim_local]

        out, out_bias = self.out_proj(norm_out)
        return out, out_bias

    def _apply_gated_norm(self, x: Tensor, gate: Tensor) -> Tensor:
        """``out_norm(x) * sigmoid(gate)`` with per-head RMSNorm over ``value_head_dim``."""
        x_shape = x.shape
        x_dtype = x.dtype
        x = x.reshape(-1, self.value_head_dim)
        y = self.out_norm(x)
        gate = gate.reshape(-1, self.value_head_dim)
        y = y * torch.sigmoid(gate.float())
        return y.to(x_dtype).reshape(x_shape)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharded state dict for distributed checkpointing.

        ``A_log`` / ``dt_bias`` are TP-sharded on axis 0. The fused ``in_proj`` and the
        depthwise ``conv1d`` weights are further split into named logical chunks
        (query/key/value[/g/gate]) so they can be resharded across TP independently.
        """
        sharded_state_dict = {}
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={"A_log": 0, "dt_bias": 0},
            sharded_offsets=sharded_offsets,
        )
        for name, module in self.named_children():
            if name == "conv1d":
                module_sd = module.state_dict(prefix="", keep_vars=True)
                tp_sharding_map = {"weight": 0}
                if self.conv_bias:  # pragma: no cover
                    tp_sharding_map["bias"] = 0
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd, f"{prefix}{name}.", tp_sharding_map, sharded_offsets
                )
            elif name == "out_norm":
                # The KDA output norm is shared across heads and replicated across
                # tensor-parallel ranks. Some norm implementations expose their own
                # sharded_state_dict but do not encode the TP rank in replica_id,
                # which makes Megatron DCP see duplicate replicated shards at TP>1.
                module_sd = module.state_dict(prefix="", keep_vars=True)
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd, f"{prefix}{name}.", {}, sharded_offsets
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata
                )
            sharded_state_dict.update(module_sharded_sd)

        in_proj_dim_local_tp = self.in_proj_dim // self.tp_size
        assert (
            sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0)
            == in_proj_dim_local_tp
        ), (in_proj_dim_local_tp, sharded_state_dict[f"{prefix}in_proj.weight"])
        sharded_state_dict[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_state_dict[f"{prefix}in_proj.weight"],
            [
                self.qk_dim // self.tp_size,
                self.qk_dim // self.tp_size,
                self.v_dim // self.tp_size,
                (self.value_head_dim if not self.no_kda_lora else self.qk_dim)
                // self.tp_size,
                (self.value_head_dim if not self.no_kda_lora else self.v_dim)
                // self.tp_size,
            ],
            ["query", "key", "value", "g", "gate"],
            0,
        )

        conv_layer_name_list = ["conv1d.weight"]
        assert (
            sharded_state_dict[f"{prefix}conv1d.weight"].data.size(0)
            == self.conv_dim_local_tp
        ), (self.conv_dim_local_tp, sharded_state_dict[f"{prefix}conv1d.weight"])
        if self.conv_bias:  # pragma: no cover
            conv_layer_name_list.append("conv1d.bias")
            assert (
                sharded_state_dict[f"{prefix}conv1d.bias"].data.size(0)
                == self.conv_dim_local_tp
            )
        for conv_layer_name in conv_layer_name_list:
            sharded_state_dict[f"{prefix}{conv_layer_name}"] = _split_tensor_factory(
                sharded_state_dict[f"{prefix}{conv_layer_name}"],
                [
                    self.qk_dim // self.tp_size,
                    self.qk_dim // self.tp_size,
                    self.v_dim // self.tp_size,
                ],
                ["query", "key", "value"],
                0,
            )

        return sharded_state_dict


def _split_tensor_factory(
    orig_sh_ten: ShardedTensor,
    split_sections: list[int],
    split_names: list[str],
    split_dim: int,
) -> ShardedTensorFactory:
    """Build a factory that splits a ShardedTensor into named independent chunks."""
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    orig_sh_ten_no_data = orig_sh_ten.without_data()

    if sum(split_sections) != orig_sh_ten_no_data.local_shape[split_dim]:
        raise ValueError(
            f"Split sections must cover the whole dimension size, "
            f"got {split_sections=} vs dimension size "
            f"{orig_sh_ten_no_data.local_shape[split_dim]}"
        )
    assert not isinstance(split_sections, int)
    assert len(split_sections) == len(split_names)

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str,
        t: torch.Tensor,
        replica_id: ReplicaId,
        flattened_range: slice | None,
    ):
        factory_sh_ten = replace(
            orig_sh_ten_no_data,
            key=key,
            data=t,
            dtype=t.dtype,
            replica_id=replica_id,
            flattened_range=flattened_range,
        )
        chunk_sh_tens = []
        split_start = 0
        for split_size, split_name in zip(split_sections, split_names):
            split_chunks = factory_sh_ten.narrow(split_dim, split_start, split_size)
            for sh_ten in split_chunks:
                sh_ten.key = f"{sh_ten.key}.{split_name}"
            chunk_sh_tens.extend(split_chunks)
            split_start += split_size

        assert split_start == orig_sh_ten_no_data.local_shape[split_dim]
        assert sum(sh_ten.data.numel() for sh_ten in chunk_sh_tens) == t.numel()
        return chunk_sh_tens

    @torch.no_grad()
    def sh_ten_merge_fn(sub_state_dict):
        return torch.cat(sub_state_dict)

    return ShardedTensorFactory(
        orig_sh_ten.key,
        orig_sh_ten.data,
        sh_ten_build_fn,
        sh_ten_merge_fn,
        orig_sh_ten.replica_id,
    )
