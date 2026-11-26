# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 decision readout for Megatron-Core.

``PointerHead`` replaces ``GPTModel.output_layer`` on the last pipeline stage.
It projects each token hidden state with two dense maps and returns
``[S, B, 2 * dp]`` in fp32. Checkpoints are written as ``head.pt``.
"""

import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from torch import nn

from areal.utils import logging

logger = logging.getLogger("Qwen35Decision")

HEAD_CHECKPOINT_NAME = "head.pt"
POINTER_HEAD_CHECKPOINT_NAME = "pointer_head.pt"


@dataclass
class DecisionHeadMeta:
    """On-disk ``head.pt`` record.

    Known fields overwrite ``extra`` on write. Unknown keys survive a
    read-modify-write round trip.
    """

    KNOWN: ClassVar[tuple[str, ...]] = (
        "base",
        "head",
        "base_revision",
        "lora",
        "head_dim",
        "option_isolation",
        "special_embeddings",
        "weights_dtype",
        "temperature",
        "holdout",
        "weights",
    )

    base: str = ""
    head: dict | None = None
    base_revision: str | None = None
    lora: int = 0
    head_dim: int = 256
    option_isolation: bool = False
    special_embeddings: bool = False
    weights_dtype: str = "fp32"
    temperature: float = 1.0
    holdout: list = field(default_factory=list)
    weights: str = "lora"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionHeadMeta":
        known = {key: payload[key] for key in cls.KNOWN if key in payload}
        extra = {key: value for key, value in payload.items() if key not in cls.KNOWN}
        meta = cls(**known)
        meta.extra = extra
        return meta

    def to_dict(self) -> dict[str, Any]:
        # Known fields win over a stray key of the same name in extra.
        return {**self.extra, **{key: getattr(self, key) for key in self.KNOWN}}


class PointerHead(nn.Module):
    """Megatron output layer that emits q and k pointer projections.

    Parameters stay in fp32. ``Float16Module`` casts the wrapped trunk with
    ``module.bfloat16()``; this module keeps its own parameters in fp32 so the
    readout stays in full precision. Megatron DDP buckets parameters
    by dtype, so the fp32 head gets its own gradient buffer.

    When ``config.sequence_parallel`` is set, ``q`` and ``k`` are marked
    ``sequence_parallel`` so ``finalize_model_grads`` all-reduces their
    gradients across the tensor-parallel ranks. The forward gather rebuilds
    the full sequence on the last pipeline stage.
    """

    def __init__(self, d: int, dp: int = 256, *, config: Any = None) -> None:
        super().__init__()
        self.hidden_size = int(d)
        self.dp = int(dp)
        self.q = nn.Linear(self.hidden_size, self.dp, bias=True)
        self.k = nn.Linear(self.hidden_size, self.dp, bias=True)
        self.scale = 1.0 / math.sqrt(self.dp)
        # Inference divides logits by this. Training always sees 1.0.
        self.temperature = 1.0
        self.sequence_parallel = bool(
            getattr(config, "sequence_parallel", False) if config is not None else False
        )
        self._mark_sequence_parallel()

    def _mark_sequence_parallel(self) -> None:
        if not hasattr(self, "q"):
            return
        flag = bool(getattr(self, "sequence_parallel", False))
        for linear in (self.q, self.k):
            for name in ("weight", "bias"):
                param = getattr(linear, name, None)
                if param is not None:
                    setattr(param, "sequence_parallel", flag)

    def _apply(self, fn, recurse=True):
        """Apply ``fn`` but keep floating parameters in fp32.

        Device moves still go through. A dtype cast such as ``bfloat16()``
        keeps the fp32 storage, so ``Float16Module`` does not round the head.
        """

        def keep_fp32(tensor):
            converted = fn(tensor)
            if (
                not isinstance(tensor, torch.Tensor)
                or not isinstance(converted, torch.Tensor)
                or not tensor.is_floating_point()
                or not converted.is_floating_point()
                or converted.dtype == tensor.dtype
            ):
                return converted
            kept = tensor.detach()
            if kept.device != converted.device:
                kept = kept.to(device=converted.device)
            return kept

        out = super()._apply(keep_fp32, recurse=recurse)
        self._mark_sequence_parallel()
        return out

    def _project(self, linear: nn.Linear, hidden: torch.Tensor) -> torch.Tensor:
        """fp32 linear. This is ``linear(hidden.float())`` when weights are fp32."""
        inputs = hidden.float()
        weight_dtype = linear.weight.dtype
        bias_dtype = None if linear.bias is None else linear.bias.dtype
        if weight_dtype == inputs.dtype and (
            bias_dtype is None or bias_dtype == inputs.dtype
        ):
            return linear(inputs)
        bias = None if linear.bias is None else linear.bias.float()
        return F.linear(inputs, linear.weight.float(), bias)

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Return concatenated q and k projections, shape ``[S, B, 2 * dp]``."""
        del weight, runtime_gather_output
        projected = torch.cat(
            (self._project(self.q, input_), self._project(self.k, input_)),
            dim=-1,
        )
        if self.sequence_parallel:
            projected = _gather_sequence_parallel(projected)
        return projected, None

    def _with_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training or self.temperature == 1.0:
            return logits
        return logits / self.temperature

    def score_single(
        self, q_decide: torch.Tensor, k_opts: torch.Tensor
    ) -> torch.Tensor:
        """Dot-product logits of one query against option keys.

        ``q_decide`` is ``[dp]`` (or broadcastable to ``k_opts``) and
        ``k_opts`` is ``[K, dp]``. Returns ``[K]``.
        """
        logits = (k_opts * q_decide).sum(dim=-1) * self.scale
        return self._with_temperature(logits)

    def score_many(
        self,
        q_decide: torch.Tensor,
        k_opts: torch.Tensor,
        owner: torch.Tensor,
    ) -> torch.Tensor:
        """Dot-product logits for many queries.

        ``q_decide`` is ``[Q, dp]``, ``k_opts`` is ``[N, dp]``, and ``owner``
        indexes the query that owns each option. Returns ``[N]``.
        """
        logits = (k_opts * q_decide[owner]).sum(dim=-1) * self.scale
        return self._with_temperature(logits)


def replace_output_layer_with_pointer_head(
    model: nn.Module,
    tf_config: Any,
    head_dim: int = 256,
) -> PointerHead | None:
    """Swap the last-stage language-model head for a ``PointerHead``.

    Earlier pipeline stages (``post_process`` false, or no ``output_layer``)
    are left unchanged. The new head is fp32 and ``vocab_size`` becomes
    ``2 * head_dim``, the width of the concatenated projections.
    """
    gpt = _unwrap_gpt_model(model)
    if not getattr(gpt, "post_process", True) or not hasattr(gpt, "output_layer"):
        return None
    device = _module_device(gpt.output_layer)
    # TP>1 shards the sequence. Gather in the head even if the passed
    # config still says sequence_parallel is false.
    tp_size = 1
    try:
        if dist.is_initialized():
            tp_size = int(mpu.get_tensor_model_parallel_world_size())
    except Exception:
        tp_size = 1
    if tp_size > 1:
        try:
            tf_config.sequence_parallel = True
        except Exception:
            pass
    head = PointerHead(int(tf_config.hidden_size), dp=head_dim, config=tf_config)
    if tp_size > 1:
        head.sequence_parallel = True
        head._mark_sequence_parallel()
    if device is None:
        head = head.to(dtype=torch.float32)
    else:
        head = head.to(device=device, dtype=torch.float32)
    head._mark_sequence_parallel()
    gpt.output_layer = head
    gpt.vocab_size = 2 * int(head_dim)
    logger.info(
        "Replaced output_layer with PointerHead"
        f"(hidden={tf_config.hidden_size}, dp={head_dim})."
    )
    return head


def save_decision_pointer_head(
    models: nn.Module | Iterable[nn.Module],
    weights_path: str,
    meta: Any = None,
) -> str | None:
    """Write ``head.pt`` from the last pipeline stage's tensor- and data-parallel rank 0.

    ``meta`` may be a :class:`DecisionHeadMeta` or a dict. The live fp32
    ``q`` / ``k`` state dict is stored under ``head``.
    Other ranks return ``None`` without writing.
    """
    if not _is_checkpoint_writer():
        return None
    heads = _pointer_heads(models)
    if not heads:
        logger.warning("No PointerHead to save.")
        return None
    if len(heads) > 1:
        logger.warning("Multiple PointerHeads found; saving the first.")
    head = heads[0]
    payload = _coerce_meta(meta, head, _fp32_state_dict(head)).to_dict()
    os.makedirs(weights_path, exist_ok=True)
    path = os.path.join(weights_path, HEAD_CHECKPOINT_NAME)
    torch.save(payload, path)
    logger.info(f"Saved PointerHead weights to {path}")
    return path


def load_decision_pointer_head(
    models: nn.Module | Iterable[nn.Module],
    weights_path: str,
) -> DecisionHeadMeta | None:
    """Load ``head.pt`` or ``pointer_head.pt`` into every attached ``PointerHead``.

    Call this on every rank. Only the last pipeline stage owns the head. A
    ``head.pt`` file is a meta dict (state under ``head``). A
    ``pointer_head.pt`` file is a raw module state dict. After the load, head
    parameters are broadcast across tensor- and context-parallel ranks. A
    missing checkpoint broadcasts the fresh initialization across tensor,
    context, and data parallel so every replica starts from the same weights.
    """
    if not _is_pipeline_last_stage():
        return None
    heads = _pointer_heads(models)
    path = _checkpoint_file(weights_path)
    if not heads:
        if path is not None and _is_checkpoint_writer():
            logger.warning(
                f"Found {path} but no PointerHead is attached on the last pipeline stage."
            )
        return None

    present = _agree_flag(path is not None, heads[0].q.weight.device)
    meta: DecisionHeadMeta | None = None
    apply_temperature = False
    if present:
        assert path is not None
        meta, apply_temperature = _read_checkpoint(path)
        _install_checkpoint(heads, meta, apply_temperature=apply_temperature)
    elif path is not None:
        logger.warning(
            f"Ignoring {path} because it is not visible on every last-stage rank."
        )

    groups = _replica_groups(include_data_parallel=not present)
    for head in heads:
        _broadcast_head(head, groups)

    if _is_checkpoint_writer():
        if present:
            logger.info(f"Loaded PointerHead weights from {path}")
        else:
            logger.info(
                "PointerHead checkpoint not found under "
                f"{weights_path}; broadcasting initialization."
            )
    return meta


def _gather_sequence_parallel(projected: torch.Tensor) -> torch.Tensor:
    """Gather the sequence with Megatron's sequence-parallel output rule.

    ``gather_from_sequence_parallel_region`` allocates its buffer with
    ``torch.cuda.current_device()``. Ascend builds have no CUDA, so that
    lookup is pointed at the projection's device for this call. The backward
    still splits the gradient along the sequence, matching
    ``tensor_parallel_output_grad=False``.
    """
    gather = tensor_parallel.gather_from_sequence_parallel_region
    if torch.cuda.is_available() or projected.device.type != "npu":
        return gather(projected, tensor_parallel_output_grad=False)
    original = torch.cuda.current_device
    torch.cuda.current_device = lambda: projected.device
    try:
        return gather(projected, tensor_parallel_output_grad=False)
    finally:
        torch.cuda.current_device = original


def _unwrap_gpt_model(model: nn.Module):
    from areal.models.mcore.registry import unwrap_to_gpt_model

    return unwrap_to_gpt_model(model)


def _module_device(module: nn.Module) -> torch.device | None:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def _model_list(models: nn.Module | Iterable[nn.Module]) -> list[nn.Module]:
    if isinstance(models, nn.Module):
        return [models]
    return list(models)


def _pointer_heads(models: nn.Module | Iterable[nn.Module]) -> list[PointerHead]:
    heads: list[PointerHead] = []
    for model in _model_list(models):
        gpt = _unwrap_gpt_model(model)
        head = getattr(gpt, "output_layer", None)
        if isinstance(head, PointerHead):
            heads.append(head)
    return heads


def _fp32_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for key, value in module.state_dict().items()
    }


def _coerce_meta(
    meta: Any, head: PointerHead, state: dict[str, torch.Tensor]
) -> DecisionHeadMeta:
    if meta is None:
        built = DecisionHeadMeta(head_dim=head.dp, temperature=float(head.temperature))
    elif isinstance(meta, DecisionHeadMeta):
        built = DecisionHeadMeta.from_dict(meta.to_dict())
    elif hasattr(meta, "to_dict") and callable(meta.to_dict):
        built = DecisionHeadMeta.from_dict(dict(meta.to_dict()))
    elif isinstance(meta, Mapping):
        built = DecisionHeadMeta.from_dict(meta)
    else:
        raise TypeError(
            "meta must be a DecisionHeadMeta, a mapping, or an object with to_dict(), "
            f"got {type(meta).__name__}."
        )
    built.head = state
    built.head_dim = head.dp
    return built


def _checkpoint_file(weights_path: str) -> str | None:
    head_path = os.path.join(weights_path, HEAD_CHECKPOINT_NAME)
    if os.path.isfile(head_path):
        return head_path
    pointer_path = os.path.join(weights_path, POINTER_HEAD_CHECKPOINT_NAME)
    if os.path.isfile(pointer_path):
        return pointer_path
    return None


def _parse_checkpoint(payload: Any, source: str) -> tuple[DecisionHeadMeta, bool]:
    """Return meta and whether the file stored a temperature."""
    if not isinstance(payload, dict):
        raise TypeError(f"{source} must contain a dict, got {type(payload).__name__}.")
    nested = payload.get("head")
    if isinstance(nested, dict) and "q.weight" in nested:
        meta = DecisionHeadMeta.from_dict(payload)
        meta.head = nested
        return meta, True
    if "q.weight" in payload and "k.weight" in payload:
        state = {
            key: value
            for key, value in payload.items()
            if isinstance(value, torch.Tensor)
        }
        dp = int(state["q.weight"].shape[0])
        return DecisionHeadMeta(head=state, head_dim=dp), False
    raise ValueError(
        f"{source} is neither a head.pt payload nor a PointerHead state dict."
    )


def _read_checkpoint(path: str) -> tuple[DecisionHeadMeta, bool]:
    # head.pt holds strings and lists beside tensors, so weights_only cannot be used.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return _parse_checkpoint(payload, path)


def _install_checkpoint(
    heads: list[PointerHead],
    meta: DecisionHeadMeta,
    *,
    apply_temperature: bool,
) -> None:
    if not isinstance(meta.head, dict) or "q.weight" not in meta.head:
        raise ValueError("pointer-head checkpoint is missing 'q.weight'.")
    state = meta.head
    loaded_dp = int(state["q.weight"].shape[0])
    if int(meta.head_dim) != loaded_dp:
        raise ValueError(
            f"head.pt head_dim={meta.head_dim} does not match "
            f"q.weight shape {tuple(state['q.weight'].shape)}."
        )
    for head in heads:
        if head.dp != loaded_dp:
            raise ValueError(
                f"PointerHead dp={head.dp} does not match checkpoint head_dim={loaded_dp}."
            )
        head.load_state_dict(state)
        if apply_temperature:
            head.temperature = float(meta.temperature)


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _mpu_ready() -> bool:
    return _distributed() and mpu.model_parallel_is_initialized()


def _is_pipeline_last_stage() -> bool:
    if not _mpu_ready():
        return True
    return bool(mpu.is_pipeline_last_stage())


def _is_checkpoint_writer() -> bool:
    """Last pipeline stage, tensor-parallel rank 0, and data-parallel rank 0.

    Data-parallel rank is taken with context parallel included, matching
    ``save_critic_value_head``: one rank in the stage writes the file.
    """
    if not _mpu_ready():
        return True
    if not mpu.is_pipeline_last_stage():
        return False
    return (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    )


def _group_world_size(group) -> int:
    if group is None or not _distributed():
        return 1
    return int(dist.get_world_size(group))


def _replica_groups(*, include_data_parallel: bool) -> list:
    """Tensor- and context-parallel groups, plus data parallel when requested.

    Broadcast tensor parallel first, then context parallel, then data parallel
    with context parallel included. Each group rank 0 is the source, so the
    copies on a pipeline stage converge to one initialization.
    """
    if not _mpu_ready():
        return []
    groups = []
    for group in (
        mpu.get_tensor_model_parallel_group(),
        mpu.get_context_parallel_group(),
    ):
        if _group_world_size(group) > 1:
            groups.append(group)
    if include_data_parallel:
        group = mpu.get_data_parallel_group(with_context_parallel=True)
        if _group_world_size(group) > 1:
            groups.append(group)
    return groups


def _broadcast_head(head: PointerHead, groups: list) -> None:
    if not groups or not _distributed():
        return
    device = head.q.weight.device
    for group in groups:
        src = dist.get_global_rank(group, 0)
        for param in head.parameters():
            dist.broadcast(param.data, src=src, group=group)
        temperature = torch.tensor(
            [head.temperature], dtype=torch.float32, device=device
        )
        dist.broadcast(temperature, src=src, group=group)
        head.temperature = float(temperature.item())


def _agree_flag(flag: bool, device: torch.device) -> bool:
    """True only when every last-stage rank passed True.

    Ranks that disagree about whether ``head.pt`` exists must take the same
    branch, or the parameter broadcast deadlocks.
    """
    if not _mpu_ready():
        return flag
    group = mpu.get_tensor_and_data_parallel_group(with_context_parallel=True)
    if _group_world_size(group) <= 1:
        return flag
    token = torch.tensor([1 if flag else 0], dtype=torch.int32, device=device)
    dist.all_reduce(token, op=dist.ReduceOp.MIN, group=group)
    return bool(int(token.item()))
