# SPDX-License-Identifier: Apache-2.0

# Pad/unpad operations are modified from flash-attention under BSD-3 license.
# Copyright (c) 2023, Tri Dao.

import copy
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import MicroBatchSpec, NormConfig
from areal.infra.platforms import current_platform
from areal.utils import logging, seqpack
from areal.utils.math import align
from areal.utils.seqpack import get_allocate_fn

logger = logging.getLogger("DataUtils")


def get_batch_size(data: dict[str, Any]) -> int:
    if not data:
        return 0

    am = data.get("attention_mask")
    if torch.is_tensor(am) and am.ndim >= 1:
        return int(am.shape[0])

    cu = data.get("cu_seqlens")
    if torch.is_tensor(cu) and cu.ndim >= 1 and cu.numel() >= 1:
        return max(int(cu.shape[0]) - 1, 0)

    mmi = data.get("multi_modal_input")
    if isinstance(mmi, list):
        return len(mmi)

    for v in data.values():
        if torch.is_tensor(v) and v.ndim >= 1:
            return int(v.shape[0])

    return 0


def extract_valid_token_sequences(
    input_ids_batch: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[list[torch.Tensor], int]:
    """Extract unpadded token sequences from a [B, S] batch."""
    if not (torch.is_tensor(input_ids_batch) and torch.is_tensor(attention_mask)):
        raise TypeError("input_ids_batch and attention_mask must be torch.Tensor.")
    if input_ids_batch.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids_batch and attention_mask must be rank-2 tensors.")
    if input_ids_batch.shape != attention_mask.shape:
        raise ValueError(
            "input_ids_batch and attention_mask must have identical shapes."
        )

    max_seq_len = 0
    input_ids_list: list[torch.Tensor] = []
    for i in range(input_ids_batch.shape[0]):
        valid_length = int(attention_mask[i].sum().item())
        max_seq_len = max(max_seq_len, valid_length)
        input_ids_list.append(input_ids_batch[i, :valid_length])
    return input_ids_list, max_seq_len


def extract_single_valid_token_sequence(
    trajectory: dict[str, Any],
) -> torch.Tensor:
    """Extract one unpadded token sequence from a trajectory dict.

    Raises
    ------
    ValueError
        If required fields are missing, malformed, or trajectory batch size is not 1.
    """
    if "input_ids" not in trajectory or "attention_mask" not in trajectory:
        raise ValueError(
            "trajectory must contain both 'input_ids' and 'attention_mask'."
        )

    input_ids = trajectory["input_ids"]
    attention_mask = trajectory["attention_mask"]
    seqs, _ = extract_valid_token_sequences(input_ids, attention_mask)
    if len(seqs) != 1:
        raise ValueError(
            f"trajectory must contain exactly one sequence, got {len(seqs)}."
        )
    return seqs[0]


def get_total_valid_tokens(trajectory: dict[str, Any]) -> int:
    """Return total valid token count inferred from attention_mask when available."""
    attention_mask = trajectory.get("attention_mask")
    if torch.is_tensor(attention_mask):
        return int(attention_mask.sum().item())
    return 0


def reorder_list(xs: Sequence, indices: list[int]) -> list:
    assert len(set(indices)) == len(xs)
    return [xs[i] for i in indices]


def dict_map(x: dict, fn: Callable) -> dict:
    return {k: fn(v) for k, v in x.items()}


def dict_of_list2list_of_dict(
    dict_of_lists: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    if not dict_of_lists:
        return []
    keys = list(dict_of_lists.keys())
    length = len(dict_of_lists[keys[0]])
    for key, value_list in dict_of_lists.items():
        if len(value_list) != length:
            raise ValueError(
                f"All lists must have the same length. Key '{key}' has length {len(value_list)}, expected {length}"
            )
    return [{key: dict_of_lists[key][i] for key in keys} for i in range(length)]


def list_of_dict2dict_of_list(
    list_of_dicts: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    if not list_of_dicts:
        return {}
    keys = list(list_of_dicts[0].keys())
    for i, dict_item in enumerate(list_of_dicts):
        if set(dict_item.keys()) != set(keys):
            raise ValueError(
                f"All dictionaries must have the same keys. Dictionary at index {i} has keys {set(dict_item.keys())}, expected {set(keys)}"
            )
    return {key: [dict_item[key] for dict_item in list_of_dicts] for key in keys}


def is_multi_modal_key(key: str) -> bool:
    # Any key matching: multi_modal_input*
    return key.startswith("multi_modal_input")


def _get_first_non_multimodal_seq(item: dict[str, Any]) -> Any:
    """Get the first non-multimodal sequence from a dict item."""
    for key, seq in item.items():
        if not is_multi_modal_key(key):
            return seq
    raise ValueError("No non-multimodal key found in item")


def _make_attention_mask(seq_len: int, max_len: int) -> list[int]:
    return [1] * seq_len + [0] * (max_len - seq_len)


def pad_sequences_to_tensors(
    sequence_list: list[dict[str, Any]], pad_value: float = 0.0
) -> dict[str, Any]:
    if not sequence_list:
        return {}
    max_length = max(
        len(seq)
        for item in sequence_list
        for key, seq in item.items()
        if not is_multi_modal_key(key)
    )
    result = {}
    for key in sequence_list[0].keys():
        padded = []
        if is_multi_modal_key(key):
            for i in range(len(sequence_list)):
                if sequence_list[i][key]:
                    item = sequence_list[i][key][0]
                    for k, v in item.items():
                        if not torch.is_tensor(v):
                            item[k] = torch.tensor(v)
            # list concat
            result[key] = sum(
                [sequence_list[i][key] for i in range(len(sequence_list))], []
            )
            continue
        for item in sequence_list:
            x = item[key]
            if not torch.is_tensor(x):
                x = torch.tensor(x)
            padded_x = torch.nn.functional.pad(
                x, (0, max_length - len(item[key])), value=pad_value
            )
            padded.append(padded_x)
        result[key] = torch.stack(padded)
    attention_mask = [
        _make_attention_mask(len(_get_first_non_multimodal_seq(item)), max_length)
        for item in sequence_list
    ]
    result["attention_mask"] = torch.tensor(attention_mask, dtype=torch.bool)
    return result


def collate_samples_to_list(
    sequence_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert raw dataset samples into per-sample tensor dicts.

    Each sample is converted to a dict of 2-D tensors ``[1, seqlen]``
    with an ``attention_mask`` added.  Returns ``list[dict[str, Tensor]]``,
    the canonical per-trajectory format expected by :func:`batched_call` /
    :func:`concat_batch`.
    """
    result: list[dict[str, Any]] = []
    for item in sequence_list:
        sample: dict[str, Any] = {}
        seqlen: int | None = None
        for key, value in item.items():
            if is_multi_modal_key(key):
                if isinstance(value, list):
                    for v in value:
                        if isinstance(v, dict):
                            for k, t in v.items():
                                if not torch.is_tensor(t):
                                    v[k] = torch.tensor(t)
                sample[key] = value if isinstance(value, list) else [value]
                continue
            if not torch.is_tensor(value):
                value = torch.tensor(value)
            if seqlen is None:
                seqlen = value.shape[0]
            sample[key] = value.unsqueeze(0)  # [seqlen] -> [1, seqlen]
        if "attention_mask" not in sample and seqlen is not None:
            sample["attention_mask"] = torch.ones(1, seqlen, dtype=torch.bool)
        result.append(sample)
    return result


def unpad_input(
    hidden_states, attention_mask
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        rearrange(hidden_states, "b s ... -> (b s) ...")[indices],
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    output = hidden_states.new_zeros(batch * seqlen)
    output[indices] = hidden_states
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def _pad_cat_dim0(tensors: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """Pad tensors to the same non-batch dims and concatenate along dim 0.

    For 0D/1D tensors no padding is needed — they are simply concatenated.
    For ≥2D tensors, dimensions 1…N-1 are right-padded to the per-dimension
    maximum across *tensors* before concatenation along dim 0.
    """
    if not tensors:
        raise ValueError("_pad_cat_dim0 requires a non-empty list of tensors.")
    ndim = tensors[0].ndim
    if ndim <= 1:
        return torch.cat(tensors, dim=0)

    # Compute the maximum shape for dims 1 … N-1
    max_shape = [0] * (ndim - 1)
    for t in tensors:
        if t.ndim != ndim:
            raise ValueError(
                f"Dimension mismatch: expected {ndim}D tensors, got {t.ndim}D"
            )
        for i in range(1, ndim):
            max_shape[i - 1] = max(max_shape[i - 1], t.shape[i])

    padded_tensors = []
    for t in tensors:
        pad_sizes = [max_shape[i - 1] - t.shape[i] for i in range(1, ndim)]
        if any(pad_sizes):
            # F.pad expects sizes in reversed dimension order (innermost first)
            pad = []
            for ps in reversed(pad_sizes):
                pad.extend([0, ps])
            t = F.pad(t, tuple(pad), "constant", pad_value)
        padded_tensors.append(t)

    return torch.cat(padded_tensors, dim=0)


def concat_padded_tensors(
    tensor_dicts: list[dict[str, Any]], pad_value: float = 0.0
) -> dict[str, Any]:
    """Concatenate and pad tensors from multiple dictionaries.

    For each key present in the input dicts:

    * **Tensor values** — all non-batch dimensions (dims 1…N-1) are right-padded
      to the per-dimension maximum across dicts, then concatenated along dim 0.
      ``attention_mask`` is always zero-padded regardless of *pad_value*.
    * **List values** — flat-concatenated.
    * **Multimodal keys** (``multi_modal_input*``) — list-extended with per-dict
      batch-size awareness.
    * **Other values** — the first dict's value is kept (assumed identical).

    All input dicts must share the same set of keys.
    """
    if not tensor_dicts:
        return {}
    if len(tensor_dicts) == 1:
        return dict(tensor_dicts[0])

    # Validate key consistency
    first_keys = set(tensor_dicts[0].keys())
    for i, d in enumerate(tensor_dicts[1:], 1):
        if set(d.keys()) != first_keys:
            raise ValueError(
                f"concat_padded_tensors: dict[{i}] has different keys than dict[0]. "
                f"Expected {sorted(first_keys)}, got {sorted(d.keys())}"
            )

    result: dict[str, Any] = {}

    multimodal_keys = {
        key for td in tensor_dicts for key in td if is_multi_modal_key(key)
    }
    # Merge multimodal keys
    for mm_key in multimodal_keys:
        merged_multi_modal = []
        for td in tensor_dicts:
            bs = get_batch_size(td)
            merged_multi_modal.extend(td.get(mm_key, [{} for _ in range(bs)]))
        result[mm_key] = merged_multi_modal

    # Process remaining keys
    for key in tensor_dicts[0]:
        if key in multimodal_keys:
            continue
        values = [td[key] for td in tensor_dicts]
        if isinstance(values[0], torch.Tensor):
            pv = 0.0 if key == "attention_mask" else pad_value
            result[key] = _pad_cat_dim0(values, pad_value=pv)
        elif isinstance(values[0], list):
            result[key] = [item for v in values for item in v]
        else:
            result[key] = values[0]

    return result


def _unpad_splits(
    splits: list[torch.Tensor], traj_seqlens: list[int] | None
) -> list[torch.Tensor]:
    """Trim each split tensor's last dim to its original sequence length."""
    if traj_seqlens is None:
        return splits
    for i, s in enumerate(splits):
        if s.ndim >= 2 and s.shape[-1] > traj_seqlens[i]:
            splits[i] = s[..., : traj_seqlens[i]]
    return splits


def split_and_unpad_tensor(
    result: Any,
    n_trajs: int,
    traj_group_sizes: list[int] | int = 1,
    traj_seqlens: list[int] | None = None,
) -> Any:
    """Inverse of concat_padded_tensors: split batched result into per-trajectory
    list and trim trailing padding. Handles Tensor, dict, and None inputs.

    When traj_seqlens is None and result is a dict with attention_mask,
    seqlens are auto-derived via attention_mask.sum(-1).max() per group.
    """
    if result is None:
        return None
    # Normalize to list for uniform handling
    if isinstance(traj_group_sizes, int):
        traj_group_sizes = [traj_group_sizes] * n_trajs
    total = sum(traj_group_sizes)

    # Auto-derive traj_seqlens from attention_mask when not provided
    if traj_seqlens is None and isinstance(result, dict):
        attn_mask = result.get("attention_mask")
        if isinstance(attn_mask, torch.Tensor) and attn_mask.ndim >= 2:
            am_splits = list(attn_mask.split(traj_group_sizes, dim=0))
            derived = [int(am.sum(-1).max().item()) for am in am_splits]
            # Only apply if there's actual padding to trim
            if any(sl < attn_mask.shape[-1] for sl in derived):
                traj_seqlens = derived
    if isinstance(result, torch.Tensor):
        splits = list(result.split(traj_group_sizes, dim=0))
        return _unpad_splits(splits, traj_seqlens)
    if isinstance(result, dict):
        split_result = [{} for _ in range(n_trajs)]
        for key, value in result.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == total:
                splits = _unpad_splits(
                    list(value.split(traj_group_sizes, dim=0)), traj_seqlens
                )
                for i, s in enumerate(splits):
                    split_result[i][key] = s
            else:
                for i in range(n_trajs):
                    split_result[i][key] = copy.deepcopy(value)
        return split_result
    return result


def unpack_groups_to_sequences(
    item_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten grouped trajectories into fully independent sequence-level dicts.

    For example, if an item in item_list has shape [8, seq_len, ...] for 8 samples
    (group_size=8), it will be split into 8 separate dictionaries, each with
    shape [1, seq_len, ...]. This is required for algorithms like DTA that operate
    on individual sequences rather than groups.

    Args:
        item_list: List of trajectory dictionaries.

    Returns:
        A new list where every dictionary represents a single sequence.
    """
    flat_item_list = []
    for item in item_list:
        attn_mask = item.get("attention_mask")
        if (
            attn_mask is not None
            and isinstance(attn_mask, torch.Tensor)
            and attn_mask.ndim >= 2
        ):
            n_seqs = attn_mask.shape[0]
            if n_seqs > 1:
                splits = split_and_unpad_tensor(
                    item, n_trajs=n_seqs, traj_group_sizes=1
                )
                if isinstance(splits, list):
                    flat_item_list.extend(splits)
                else:
                    flat_item_list.append(splits)
            else:
                flat_item_list.append(item)
        else:
            flat_item_list.append(item)
    return flat_item_list


@dataclass
class TrajBatchMeta:
    """Metadata for reversing concat_batch: traj counts, group sizes, seqlens."""

    n_trajs: int
    traj_group_sizes: list[int]
    traj_seqlens: list[int]


def concat_batch(
    data: list[dict[str, Any]],
) -> tuple[dict[str, Any], "TrajBatchMeta"]:
    """Concat list[dict] trajectories into a single batched dict with metadata."""
    assert isinstance(data, list) and all(isinstance(d, dict) for d in data), (
        f"Expected list[dict], got {type(data)}"
    )
    traj_group_sizes = []
    for d in data:
        first_tensor = next(
            (v for v in d.values() if isinstance(v, torch.Tensor)), None
        )
        traj_group_sizes.append(
            first_tensor.shape[0] if first_tensor is not None else 1
        )
    traj_seqlens = [d["attention_mask"].shape[-1] for d in data]
    meta = TrajBatchMeta(
        n_trajs=len(data),
        traj_group_sizes=traj_group_sizes,
        traj_seqlens=traj_seqlens,
    )
    return concat_padded_tensors(data), meta


def split_batch(
    result: Any,
    meta: TrajBatchMeta,
) -> list[Any] | None:
    """Inverse of concat_batch: split batched result back into per-trajectory list."""
    return split_and_unpad_tensor(
        result, meta.n_trajs, meta.traj_group_sizes, meta.traj_seqlens
    )


def batched_call(
    fn: Callable[[dict[str, Any]], Any],
    data: list[dict[str, Any]],
    *,
    unpack: bool = True,
) -> Any:
    """Concatenate per-trajectory dicts into one batch, call *fn*, optionally unpack.

    This is the canonical way to bridge the per-trajectory data representation
    (``list[dict[str, Any]]``) used by trainers/workflows with the single-batch
    representation (``dict[str, Any]``) expected by engine forward/backward methods.

    Parameters
    ----------
    fn : Callable[[dict[str, Any]], Any]
        Implementation function that receives the batched dict.
    data : list[dict[str, Any]]
        Per-trajectory dicts to be concatenated.
    unpack : bool
        If True (default), split the result back into a per-trajectory list
        via :func:`split_batch`.
    """
    batched, meta = concat_batch(data)
    result = fn(batched)
    if unpack:
        return split_batch(result, meta)
    return result


def unpack_sequence(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    lens: list[int] | None = None,
    dim: int = 0,
):
    """Unpack a sequence tensor into a list of tensors based on cumulative sequence lengths."""
    if lens is not None:
        return torch.split(x, lens, dim=dim)
    if cu_seqlens is not None:
        return torch.split(
            x, (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist(), dim=dim
        )
    raise ValueError("Either cu_seqlens or input_lens must be provided.")


def allocate_balanced_mbs(mb_spec: MicroBatchSpec, lens: list[int]) -> list[list[int]]:
    """Allocate sequences into balanced micro-batches using the configured algorithm.

    The packing algorithm is determined by ``mb_spec.packing_algorithm``:
      - ``"ffd"`` (default): First Fit Decreasing — fast greedy heuristic.
      - ``"kk"``: Karmarkar-Karp — produces more balanced partitions at a
        slight computational cost.

    Args:
        mb_spec: MicroBatchSpec containing packing configuration.
        lens: List of sequence lengths to allocate.

    Returns:
        List of lists of indices, one per micro-batch.
    """
    assert mb_spec.max_tokens_per_mb is not None
    allocate_fn = get_allocate_fn(getattr(mb_spec, "packing_algorithm", "ffd"))
    group_indices = allocate_fn(
        lens,
        mb_spec.max_tokens_per_mb,
        min_groups=mb_spec.n_mbs,
        n_groups_divisor=mb_spec.n_mbs_divisor,
    )
    group_indices = sorted([sorted(g) for g in group_indices])
    return group_indices


def allocate_balanced_mbs_synced(
    mb_spec: MicroBatchSpec,
    lens: list[int],
    group: dist.ProcessGroup | None = None,
) -> list[list[int]]:
    group_indices = allocate_balanced_mbs(mb_spec, lens)
    if not dist.is_initialized():
        return group_indices
    all_n_mbs = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(all_n_mbs, len(group_indices), group=group)
    if all(mbs == len(group_indices) for mbs in all_n_mbs):
        return group_indices
    return allocate_balanced_mbs_synced(
        MicroBatchSpec.new(mb_spec, n_mbs=max(all_n_mbs)), lens, group=group
    )


def pack_tensor_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Pack a dict of tensors of shape [B, S, ...] into [total_length, ...], leaving other keys unchanged.

    Args:
        data (Dict[str, Any]): Dictionary containing tensors to be packed. Should contain key "attention_mask" with shape [B, S].

    Returns:
        Dict[str, Any]: Dictionary with packed tensors. The "attention_mask" key will be replaced by "cu_seqlens" with shape [B+1].
    """

    assert "attention_mask" in data, "Input data must contain 'attention_mask' key."
    attention_mask = data["attention_mask"]
    assert attention_mask.ndim == 2, "Attention mask must be a 2D tensor."
    bs = attention_mask.shape[0]
    seq_len = attention_mask.shape[1]

    # Calculate cumulative sequence lengths
    lens = attention_mask.sum(dim=1, dtype=torch.int32)
    max_seqlen = lens.max().item()
    cu_seqlens = torch.cumsum(lens, dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    total_length = int(cu_seqlens[-1].item())
    # Pack tensors
    packed_data = {}
    for key, value in data.items():
        if key == "attention_mask":
            packed_data["cu_seqlens"] = cu_seqlens
            packed_data["max_seqlen"] = max_seqlen
            continue
        # tensor and of shape [B, S, ...]
        if (
            torch.is_tensor(value)
            and value.ndim >= 2
            and value.shape[0] == bs
            and value.shape[1] == seq_len
        ):
            packed_tensor = torch.empty(
                (total_length, *value.shape[2:]), dtype=value.dtype, device=value.device
            )
            # Fill the packed tensor with values from the original tensor
            for i in range(bs):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                packed_tensor[start:end] = value[i][: end - start]
            packed_data[key] = packed_tensor
        else:
            packed_data[key] = value

    return packed_data


def pad_and_stack_tensors_along_first_dim(tensor_list: list[torch.Tensor]):
    max_length = max(tensor.shape[0] for tensor in tensor_list)
    n_dim = tensor_list[0].ndim
    if not all(tensor.ndim == n_dim for tensor in tensor_list):
        raise ValueError("All tensors must have the same number of dimensions.")

    padded_tensors = []
    for tensor in tensor_list:
        pad_mode = (0,) * (2 * (n_dim - 1)) + (0, max_length - tensor.shape[0])
        padded_tensor = F.pad(tensor, pad_mode, value=0.0)
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors, dim=0)


def tensor_container_to(
    d: dict[str, Any] | torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    *args,
    **kwargs,
):
    """Apply `t.to(*args, **kwargs)` to all tensors in the dictionary.
    Support nested dictionaries.
    """
    if torch.is_tensor(d):
        return d.to(*args, **kwargs)

    if isinstance(d, list) or isinstance(d, tuple):
        return [tensor_container_to(v, *args, **kwargs) for v in d]

    if isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            if isinstance(value, dict) or isinstance(value, list):
                new_dict[key] = tensor_container_to(value, *args, **kwargs)
            elif torch.is_tensor(value):
                new_dict[key] = value.to(*args, **kwargs)
            else:
                new_dict[key] = value
        return new_dict

    return d


class MicroBatchItem(NamedTuple):
    """A single micro-batch item from MicroBatchList iteration.

    Attributes:
        orig_mb: Original micro-batch dict (for loss_weight_fn, context)
        padded_mb: Padded micro-batch dict (for model forward)
        padding_length: Batch-level padding added to this micro-batch
        old_cu_seqlens: Original cu_seqlens before sequence alignment (or None)
        padded_to_length: The padded sequence length for this micro-batch (or None)
    """

    orig_mb: dict[str, Any]
    padded_mb: dict[str, Any]
    padding_length: int
    old_cu_seqlens: torch.Tensor | None
    padded_to_length: int | None = None


@dataclass
class MicroBatchList:
    data: dict[str, Any]
    mb_spec: MicroBatchSpec
    mbs: list[dict[str, Any]]
    group_lens: list[int]
    forward_indices: list[int] | None = None
    backward_indices: list[int] | None = None
    padded_mbs: list[dict[str, Any]] | None = None
    _max_seqlen: int | None = None
    # Batch-level padding information
    padding_lengths: list[int] | None = None
    padded_to_lengths: list[int] | None = None
    # sequence-level padding information
    align_to_lengths: list[int] | None = None
    old_cu_seqlens_list: list[torch.Tensor] | None = None

    @property
    def max_seqlen(self) -> int:
        """Return the maximum sequence length across all padded micro-batches."""
        if self.padded_mbs is None:
            raise ValueError("padded_mbs is None. Call pad_mb_list first.")
        if self._max_seqlen is None:
            assert all("cu_seqlens" in m for m in self.padded_mbs), (
                "cu_seqlens not found in some padded micro-batches."
            )
            self._max_seqlen = max(m["cu_seqlens"][-1].item() for m in self.padded_mbs)
        return self._max_seqlen

    def __len__(self) -> int:
        return len(self.mbs)

    def __iter__(self) -> Iterator[MicroBatchItem]:
        """Iterate over micro-batches, yielding MicroBatchItem named tuples.

        Yields:
            MicroBatchItem containing:
                - orig_mb: Original micro-batch dict (for loss_weight_fn, context)
                - padded_mb: Padded micro-batch dict (for model forward)
                - padding_length: Batch-level padding added to this micro-batch
                - old_cu_seqlens: Original cu_seqlens before sequence alignment (or None)
                - padded_to_length: The padded sequence length for this micro-batch (or None)
        """
        if self.padded_mbs is None:
            raise ValueError("padded_mbs is None. Call pad_mb_list first.")
        for i in range(len(self.mbs)):
            old_cu_seqlens = (
                self.old_cu_seqlens_list[i] if self.old_cu_seqlens_list else None
            )
            padded_to_length = (
                self.padded_to_lengths[i] if self.padded_to_lengths else None
            )
            yield MicroBatchItem(
                orig_mb=self.mbs[i],
                padded_mb=self.padded_mbs[i],
                padding_length=self.padding_lengths[i],
                old_cu_seqlens=old_cu_seqlens,
                padded_to_length=padded_to_length,
            )

    def to(self, *args, **kwargs):
        mbs = [tensor_container_to(mb, *args, **kwargs) for mb in self.mbs]
        data = tensor_container_to(self.data, *args, **kwargs)
        padded_mbs = None
        if self.padded_mbs is not None:
            padded_mbs = [
                tensor_container_to(mb, *args, **kwargs) for mb in self.padded_mbs
            ]
        old_cu_seqlens_list = None
        if self.old_cu_seqlens_list is not None:
            old_cu_seqlens_list = [
                t.to(*args, **kwargs) for t in self.old_cu_seqlens_list
            ]
        return MicroBatchList(
            data=data,
            mb_spec=self.mb_spec,
            mbs=mbs,
            forward_indices=self.forward_indices,
            backward_indices=self.backward_indices,
            group_lens=self.group_lens,
            padded_mbs=padded_mbs,
            _max_seqlen=self._max_seqlen,
            padding_lengths=self.padding_lengths,
            padded_to_lengths=self.padded_to_lengths,
            old_cu_seqlens_list=old_cu_seqlens_list,
            align_to_lengths=self.align_to_lengths,
        )


DEFAULT_MAX_TOKENS_PER_MB = int(1e12)


def split_padded_tensor_dict_into_mb_list(
    data: dict[str, Any],
    mb_spec: MicroBatchSpec,
    group: dist.ProcessGroup | None = None,
    one_seq_per_mb: bool = False,
) -> MicroBatchList:
    """Split a padded dict of tensors into micro-batches based on the attention mask.

    Args:
        data (Dict): Dictionary containing padded tensors.
        mb_spec (MicroBatchSpec): Specification for micro-batch splitting.
        group (Optional[dist.ProcessGroup]): Process group for distributed synchronization.
        one_seq_per_mb (bool): If True, each micro-batch contains exactly one sequence
            and skips cross-rank synchronized micro-batch count alignment.
            Requires every row's valid token count ``<= mb_spec.max_tokens_per_mb``
            and ``granularity=1`` (errors in this path mention ``one_seq_per_mb``).

    Returns:
        MicroBatchList: A structure containing the split micro-batches and metadata.
    """
    # TODO: should align sequences first and then split, needs refactor
    if "attention_mask" not in data:
        raise ValueError("Input data must be padded and contain 'attention_mask' key.")
    if mb_spec.max_tokens_per_mb is None:
        mb_spec = MicroBatchSpec.new(
            mb_spec, max_tokens_per_mb=DEFAULT_MAX_TOKENS_PER_MB
        )
    granularity = mb_spec.granularity
    bs = data["attention_mask"].shape[0]
    if bs % granularity != 0:
        raise RuntimeError(f"Batch size {bs} cannot divide granularity {granularity}.")
    max_seqlen = data["attention_mask"].shape[1]
    seq_lens = data["attention_mask"].sum(1).long().cpu().numpy().tolist()
    if one_seq_per_mb:
        if granularity != 1:
            raise RuntimeError(
                f"split_padded_tensor_dict_into_mb_list: one_seq_per_mb=True requires "
                f"granularity=1, but got granularity={granularity}."
            )
        cap = mb_spec.max_tokens_per_mb
        for i, L in enumerate(seq_lens):
            if L > cap:
                raise RuntimeError(
                    f"split_padded_tensor_dict_into_mb_list: one_seq_per_mb=True, "
                    f"but sequence at index {i} has {L} valid tokens, which exceeds "
                    f"max_tokens_per_mb={cap}."
                )
    input_lens = (
        data["attention_mask"]
        .view(bs // granularity, granularity, -1)
        .sum(dim=(1, 2))
        .long()
        .cpu()
        .numpy()
    )

    # check for multimodal input data
    multimodal_keys = {key for key in data if is_multi_modal_key(key)}

    # check tensor shape, split only 1d tensors with length "total_lens"
    to_split = {}
    not_to_split = {}
    for key, value in data.items():
        if key in multimodal_keys:
            continue
        if key == "position_ids" or (
            torch.is_tensor(value) and value.numel() == bs * max_seqlen
        ):
            # NOTE: qwen2.5-vl position_ids.numel() == bs * max_seqlen * 3
            to_split[key] = value
        else:
            not_to_split[key] = value

    # split
    if one_seq_per_mb:
        group_indices = [[i] for i in range(bs)]
    else:
        group_indices = allocate_balanced_mbs_synced(mb_spec, input_lens, group=group)
        group_indices = [
            seqpack.flat2d(
                [
                    list(range(i * granularity, (i + 1) * granularity))
                    for i in group_index
                ]
            )
            for group_index in group_indices
        ]
    splitted_lens = [
        [seq_lens[i] for i in group_index] for group_index in group_indices
    ]
    group_n_seqs = [len(x) for x in splitted_lens]
    group_lens = [sum(x) for x in splitted_lens]

    forward_indices = seqpack.flat2d(group_indices)
    backward_indices = np.zeros(bs, dtype=np.int64)
    backward_indices[forward_indices] = np.arange(bs)

    def _split(tensor):
        """Split and pad a tensor based on forward indices and lens."""
        # Unpack the sequence
        unpacked = [tensor[i] for i in range(bs)]
        # Reorder according to forward indices
        reordered = reorder_list(unpacked, forward_indices)
        reordered = torch.stack(reordered)
        # Unpack again according to split lens
        splitted = []
        offset = 0
        for _n_seqs in group_n_seqs:
            splitted.append(reordered[offset : offset + _n_seqs])
            offset += _n_seqs
        return splitted

    to_split = dict_map(to_split, lambda x: _split(x))

    for key in multimodal_keys:
        multi_modal_input = data[key]

        # Prepare the pixel_values and image_grid_thw for each group
        multi_modal_input_split = []

        for group_index in group_indices:
            group_pixel_multi_modal_input = [multi_modal_input[i] for i in group_index]
            # Stack pixel_values for each group (assuming pixel_values is a list of tensors)
            multi_modal_input_split.append(group_pixel_multi_modal_input)
        # Pack the split pixel_values and image_grid_thw back into the data
        to_split[key] = multi_modal_input_split
    mbs = dict_of_list2list_of_dict(to_split)

    results = []
    # organize splitted micro batches
    assert len(mbs) == len(splitted_lens), (len(mbs), len(splitted_lens))
    for i, (mb, lens) in enumerate(zip(mbs, splitted_lens)):
        results.append({**mb, **not_to_split})

    return MicroBatchList(
        data=data,
        mb_spec=mb_spec,
        mbs=results,
        forward_indices=forward_indices,
        backward_indices=backward_indices.tolist(),
        group_lens=group_lens,
    )


N_TOKENS_PER_PAGE = 256


def pad_packed_tensor_dict(
    data: dict[str, Any],
    pad_to_length: int,
    pad_value: float = 0.0,
    seq_align_to: int | None = None,
) -> tuple[dict[str, Any], int, torch.Tensor, int]:
    """Pad a packed dict of tensors to a specified length.
    This function assumes that the input data contains "cu_seqlens" and "max_seqlen" key,
    and all other tensors of shape [total_length, ] will be padded to `pad_to_length`.
    This function will pad a new sequence filled with `pad_value` to the end of each tensor,
    and update the "cu_seqlens" and "max_seqlen" keys accordingly.

    Args:
        data (Dict): Dictionary containing tensors to be packed.
        pad_to_length (int): The length to pad the tensors to. All tensors

    Returns:
        Dict: Dictionary with padded tensors and modified "cu_seqlens" and
            "max_seqlen".
        int: The pad length.
    """
    assert "cu_seqlens" in data, "Input data must contain 'cu_seqlens' key."
    assert "max_seqlen" in data, "Input data must contain 'max_seqlen' key."
    cu_seqlens = data["cu_seqlens"]
    max_seqlen = data["max_seqlen"]
    old_cu_seqlens = cu_seqlens.clone()
    total_length = data["cu_seqlens"][-1].item()
    # First pad sequences
    sequence_padded_data = {}
    align_to_length = None
    if seq_align_to is not None:
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        batch_size = input_lens.shape[0]
        # Align sequences to an integer multiple of seq_align_to
        pad_size = (-input_lens) % seq_align_to
        input_lens_padded = input_lens + pad_size
        cu_seqlens_padded = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=cu_seqlens.device
        )
        cu_seqlens_padded[1:] = torch.cumsum(input_lens_padded, dim=0)
        max_seqlens_padded = input_lens_padded.max().item()
        padded_shape = (input_lens_padded.sum().item(),)
        for key, value in data.items():
            if key == "cu_seqlens":
                sequence_padded_data["cu_seqlens"] = cu_seqlens_padded
            elif key == "max_seqlen":
                sequence_padded_data["max_seqlen"] = max_seqlens_padded
            elif key == "position_ids":
                if len(value.shape) == 2 and value.shape[1] == 3:
                    # [total_seq_len, channel] for qwen2.5 vl, channel==3 for t,h,w
                    new_value = torch.zeros(
                        (padded_shape[0], 3), dtype=value.dtype, device=value.device
                    )
                    for i in range(batch_size):
                        new_start = cu_seqlens_padded[i]
                        new_end = cu_seqlens_padded[i + 1]
                        old_start = cu_seqlens[i]
                        old_end = cu_seqlens[i + 1]
                        length = old_end - old_start
                        pad_length = new_end - new_start - length
                        new_value[new_start : new_start + length] = value[
                            old_start:old_end
                        ]
                        new_value[new_start + length : new_end] = (
                            torch.arange(
                                pad_length, dtype=torch.long, device=value.device
                            )
                            .unsqueeze(1)
                            .expand(-1, 3)
                        )
                else:
                    new_value = torch.zeros(
                        padded_shape, dtype=value.dtype, device=value.device
                    )
                    for i in range(batch_size):
                        new_start = cu_seqlens_padded[i]
                        new_end = cu_seqlens_padded[i + 1]
                        new_value[new_start:new_end] = torch.arange(
                            new_end - new_start, dtype=value.dtype, device=value.device
                        )
                sequence_padded_data[key] = new_value
            elif torch.is_tensor(value) and value.numel() == total_length:
                new_value = torch.full(
                    padded_shape,
                    fill_value=pad_value,
                    dtype=value.dtype,
                    device=value.device,
                )
                for i in range(batch_size):
                    new_start = cu_seqlens_padded[i]
                    start = cu_seqlens[i]
                    end = cu_seqlens[i + 1]
                    length = end - start
                    new_value[new_start : new_start + length] = value[start:end]
                sequence_padded_data[key] = new_value
            else:
                sequence_padded_data[key] = value

        data = sequence_padded_data
        align_to_length = cu_seqlens_padded[-1].item()
        # ensure pad_to_length is a integer multiple of both seq_align_to and N_TOKENS_PER_PAGE
        lcm = np.lcm(seq_align_to, N_TOKENS_PER_PAGE).item()
        pad_to_length = (pad_to_length + lcm - 1) // lcm * lcm

        cu_seqlens = data["cu_seqlens"]
        max_seqlen = data["max_seqlen"]
        total_length = data["cu_seqlens"][-1].item()
        if pad_to_length < total_length:
            # NOTE: In some occasion where sequence lengths, sequence padding will make total length
            # exceed expected `pad_to_length`. This happens more often when sequence lengths are small.
            # In this case, we increase pad_to_length.
            pad_to_length = (total_length + lcm - 1) // lcm * lcm

    # Pad batch
    pad_length = pad_to_length - total_length
    if pad_length < 0:
        raise ValueError(
            f"pad_to_length {pad_to_length} is smaller than total length {total_length}."
        )
    elif pad_length == 0:
        return (
            data,
            pad_length,
            old_cu_seqlens,
            align_to_length,
        )
    new_cu_seqlens = F.pad(cu_seqlens, (0, 1), value=pad_to_length)
    new_max_seqlen = max(max_seqlen, pad_length)
    padded_data = {}
    for key, value in data.items():
        if key == "cu_seqlens":
            padded_data[key] = new_cu_seqlens
        elif key == "max_seqlen":
            padded_data[key] = new_max_seqlen
        elif key == "position_ids":
            # [total_seqlen, channel] for qwen2.5 vl, channel==3 for t,h,w
            if len(value.shape) == 2 and value.shape[1] == 3:
                pad = (
                    torch.arange(pad_length, dtype=torch.long, device=value.device)
                    .unsqueeze(1)
                    .expand(-1, 3)
                )
                padded_tensor = torch.cat([value, pad])
            else:
                pad = torch.arange(pad_length, dtype=torch.long, device=value.device)
                padded_tensor = torch.cat([value, pad])
            padded_data[key] = padded_tensor
        elif torch.is_tensor(value) and value.numel() == total_length:
            # Pad the tensor to the new total length
            padded_tensor = torch.nn.functional.pad(
                value, (0, pad_length), value=pad_value
            )
            padded_data[key] = padded_tensor
        else:
            padded_data[key] = value
    return (
        padded_data,
        pad_length,
        old_cu_seqlens,
        align_to_length,
    )


def pad_mb_list(
    mb_list: MicroBatchList,
    pad_value: float = 0.0,
    pad_to_maximum: bool = False,
    batch_align_to: int | None = None,
    seq_align_to: int | None = None,
) -> MicroBatchList:
    """Pad the micro-batch list to the maximum length or to a specific size to:
        1. Reduce memory fragmentation.
        2. Align sequences to an integer multiple of `seq_align_to`
        to be equally sliced into context and sequence parallel ranks.
        3. Align batch total length to an integer multiple of `batch_align_to`.

    Args:
        mb_list (MicroBatchList): The micro-batch list to pad.
        pad_value (float, optional): The value to pad the tensors with. Defaults to 0.0.
        pad_to_maximum (bool, optional): Whether to pad to the maximum length specified in `mb_spec`. Defaults to False.
        batch_align_to (int, optional): The size to align batch total length to. Defaults to None.
        seq_align_to (int, optional): The size to align each sequence length to. Defaults to None.

    Returns:
        MicroBatchList: The padded micro-batch list.
    """
    padded_mb_inputs, pad_lengths = [], []
    pad_to_lengths = []
    old_cu_seqlens_list = []
    align_to_lengths = []
    if pad_to_maximum and (
        mb_list.mb_spec.max_tokens_per_mb is None
        or mb_list.mb_spec.max_tokens_per_mb == DEFAULT_MAX_TOKENS_PER_MB
    ):
        logger.warning(
            "Unable to pad to maximum because max_tokens_per_mb is not properly set."
        )
        pad_to_maximum = False
    for mb, length in zip(mb_list.mbs, mb_list.group_lens):
        if pad_to_maximum and mb_list.mb_spec.max_tokens_per_mb is not None:
            pad_to_length = mb_list.mb_spec.max_tokens_per_mb
        else:
            # NOTE: GPU page size is 2MB
            # Take hidden size 4096 with bf16 dtype as an example,
            # the batch size of a page is 256
            pad_to_length = align(length, N_TOKENS_PER_PAGE)
            if batch_align_to is not None:
                pad_to_length = align(pad_to_length, batch_align_to)
        padded_mb, pad_len, old_cu_seqlens, align_to_length = pad_packed_tensor_dict(
            mb,
            pad_to_length,
            pad_value=pad_value,
            seq_align_to=seq_align_to,
        )
        padded_mb_inputs.append(padded_mb)
        pad_lengths.append(pad_len)
        pad_to_lengths.append(pad_to_length)
        old_cu_seqlens_list.append(old_cu_seqlens)
        align_to_lengths.append(align_to_length)
    mb_list.padded_mbs = padded_mb_inputs
    mb_list.padding_lengths = pad_lengths
    mb_list.padded_to_lengths = pad_to_lengths
    if seq_align_to is not None:
        mb_list.old_cu_seqlens_list = old_cu_seqlens_list
        mb_list.align_to_lengths = align_to_lengths
    return mb_list


def unpad_logits(
    logits: torch.Tensor,
    padding_length: int,
    cu_seqlens: torch.Tensor | None = None,
    old_cu_seqlens: torch.Tensor | None = None,
):
    # TODO: when using megatron, logits are in fp32,
    # create new logits in bucket to reduce peak memory usage
    # First unpad batch
    if padding_length > 0:
        logits = logits[:-padding_length]

    # Then unpad according to old_cu_seqlens
    if old_cu_seqlens is not None:
        new_logits = torch.empty(
            (old_cu_seqlens[-1].item(), *logits.shape[1:]),
            dtype=logits.dtype,
            device=logits.device,
        )
        batch_size = old_cu_seqlens.shape[0] - 1
        for i in range(batch_size):
            old_start = old_cu_seqlens[i].item()
            old_end = old_cu_seqlens[i + 1].item()
            start = cu_seqlens[i].item()
            length = old_end - old_start
            new_logits[old_start:old_end] = logits[start : start + length]
        return new_logits

    return logits


def unsqueeze_packed_tensor_dict(data: dict[str, Any]) -> dict[str, Any]:
    assert "cu_seqlens" in data, "Input data must contain 'cu_seqlens' key."
    assert "max_seqlen" in data, "Input data must contain 'max_seqlen' key."

    total_length = data["cu_seqlens"][-1].item()
    new_data = {}
    for key, value in data.items():
        if key == "position_ids" or (
            key
            not in [
                "cu_seqlens",
                "max_seqlen",
            ]
            and torch.is_tensor(value)
            and value.numel() == total_length
        ):
            new_data[key] = value.unsqueeze(dim=0)
        else:
            new_data[key] = value
    return new_data


def unsqueeze_mb_list(
    mb_list: MicroBatchList,
) -> MicroBatchList:
    """Unsqueeze the packed dict of tensors in the micro-batch list."""
    new_padded_mbs = []
    for i, mb in enumerate(mb_list.mbs):
        if mb_list.padded_mbs is not None:
            new_padded_mbs.append(unsqueeze_packed_tensor_dict(mb_list.padded_mbs[i]))
    mb_list.padded_mbs = new_padded_mbs if mb_list.padded_mbs is not None else None
    return mb_list


def amend_position_ids(data: dict) -> dict:
    assert "attention_mask" in data, "Input data must contain 'attention_mask' key."

    attn_mask = data["attention_mask"]
    bs, seqlen = attn_mask.shape[:2]
    position_ids = (
        torch.arange(0, seqlen, dtype=torch.long, device=attn_mask.device)
        .unsqueeze(0)
        .expand(bs, -1)
    )
    position_ids.masked_fill(~attn_mask.bool(), 0)
    data["position_ids"] = position_ids
    return data


def broadcast_tensor(tensor: torch.Tensor | None, src_rank=0, group=None):
    """
    Broadcast a tensor from source rank to all other ranks in the process group.

    Args:
        tensor: Tensor on source rank, None on non-source ranks
        src_rank: The rank that holds the tensor to broadcast (default: 0)
        group: The process group to use for broadcasting (default: None, uses the default group)
        device: The device of the output tensor.

    Returns:
        Tensor: The broadcasted tensor on all ranks
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")

    current_rank = dist.get_rank()

    # On source rank, prepare the tensor for broadcasting
    if current_rank == src_rank:
        if tensor is None:
            raise ValueError(f"Tensor cannot be None on source rank {src_rank}")

        tensor = tensor.contiguous()
        device = tensor.device
        # Prepare metadata as Python objects
        metadata = {
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
            "device_type": device.type,
        }

        # Broadcast metadata using broadcast_object_list
        metadata_list = [metadata]
        dist.broadcast_object_list(metadata_list, src=src_rank, group=group)

        # Broadcast the actual tensor
        tensor = tensor.contiguous()
        dist.broadcast(tensor, src=src_rank, group=group)

        return tensor
    else:
        # On non-source ranks, receive metadata
        metadata_list = [None]
        dist.broadcast_object_list(metadata_list, src=src_rank, group=group)

        metadata = metadata_list[0]
        tensor_shape = metadata["shape"]
        dtype = metadata["dtype"]
        device_type = metadata["device_type"]
        device = (
            torch.device("cpu")
            if device_type == "cpu"
            else current_platform.current_device()
        )
        # Create tensor with the received shape and dtype
        tensor = torch.empty(tensor_shape, dtype=dtype, device=device)

        # Receive the actual tensor data
        dist.broadcast(tensor, src=src_rank, group=group)

        return tensor


def _unpad_unflatten(x, shape):
    assert len(x.shape) == 1
    pad_size = x.numel() - np.prod(shape)
    assert pad_size >= 0, pad_size
    return x[: x.numel() - pad_size].view(*shape)


def _flatten_pad_to_max_numel(x, shapes):
    pad_size = max(np.prod(shape) for shape in shapes) - x.numel()
    assert pad_size >= 0, pad_size
    return torch.nn.functional.pad(x.view(-1), (0, pad_size), value=0)


def all_gather_tensor_container(data, group=None) -> list:
    world_size = dist.get_world_size(group)
    if torch.is_tensor(data):
        local_shape = list(data.shape)
        shapes = [None for _ in range(dist.get_world_size(group))]
        dist.all_gather_object(shapes, local_shape, group=group)

        y = _flatten_pad_to_max_numel(data, shapes)

        ys = [torch.empty_like(y) for _ in range(dist.get_world_size(group=group))]
        dist.all_gather(ys, y, group=group)

        return [_unpad_unflatten(y, shape) for y, shape in zip(ys, shapes)]

    if isinstance(data, list):
        lengths = [None for _ in range(world_size)]
        dist.all_gather_object(lengths, len(data), group=group)
        if not len(set(lengths)) == 1:
            raise RuntimeError(
                f"Trying to all-gather lists with mismatched lengths: {lengths}"
            )

        data = [all_gather_tensor_container(d, group=group) for d in data]
        return list(zip(*data))

    if isinstance(data, dict):
        all_keys = [None for _ in range(world_size)]
        local_keys = set(data.keys())
        dist.all_gather_object(all_keys, local_keys, group=group)
        if any(keys != local_keys for keys in all_keys):
            raise RuntimeError(
                f"Trying to all-gather dicts with mismatched keys: {all_keys}"
            )

        results = {
            k: all_gather_tensor_container(v, group=group) for k, v in data.items()
        }
        results = [
            {k: v[i] for k, v in results.items()}
            for i in range(dist.get_world_size(group))
        ]
        return results

    results = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(results, data, group=group)
    return results


def broadcast_tensor_container(data, src_rank=0, group=None):
    if dist.get_rank() != src_rank:
        metadata = [None]
        dist.broadcast_object_list(metadata, src=src_rank, group=group)
        data_type, info = metadata[0]
        if data_type == "none":
            return None
        if data_type == "tensor":
            return broadcast_tensor(data, src_rank=src_rank, group=group)
        elif data_type == "list":
            length = info
            return [
                broadcast_tensor_container(None, src_rank=src_rank, group=group)
                for _ in range(length)
            ]
        elif data_type == "dict":
            keys = info
            return {
                k: broadcast_tensor_container(None, src_rank=src_rank, group=group)
                for k in keys
            }
        elif data_type == "object":
            to_broadcast = [None]
            dist.broadcast_object_list(to_broadcast, src=src_rank, group=group)
            return to_broadcast[0]
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    else:
        if data is None:
            metadata = [("none", None)]
            dist.broadcast_object_list(metadata, src=src_rank, group=group)
            return None
        elif torch.is_tensor(data):
            metadata = [("tensor", None)]
            dist.broadcast_object_list(metadata, src=src_rank, group=group)
            return broadcast_tensor(data, src_rank=src_rank, group=group)
        elif isinstance(data, list):
            metadata = [("list", len(data))]
            dist.broadcast_object_list(metadata, src=src_rank, group=group)
            return [
                broadcast_tensor_container(d, src_rank=src_rank, group=group)
                for d in data
            ]
        elif isinstance(data, dict):
            metadata = [("dict", list(data.keys()))]
            dist.broadcast_object_list(metadata, src=src_rank, group=group)
            return {
                k: broadcast_tensor_container(v, src_rank=src_rank, group=group)
                for k, v in data.items()
            }
        else:
            metadata = [("object", None)]
            dist.broadcast_object_list(metadata, src=src_rank, group=group)
            to_broadcast = [data]
            dist.broadcast_object_list(to_broadcast, src=src_rank, group=group)
            return to_broadcast[0]


def bcast_mb_list(
    mb_list: MicroBatchList | None, src_rank=0, group=None
) -> MicroBatchList:
    if dist.get_rank() == src_rank:
        assert mb_list is not None
    # bcast tensor container attributes
    data = broadcast_tensor_container(
        mb_list.data if mb_list else None, src_rank=src_rank, group=group
    )
    mbs = broadcast_tensor_container(
        mb_list.mbs if mb_list else None, src_rank=src_rank, group=group
    )
    padded_mbs = broadcast_tensor_container(
        mb_list.padded_mbs if mb_list else None, src_rank=src_rank, group=group
    )
    old_cu_seqlens_list = broadcast_tensor_container(
        mb_list.old_cu_seqlens_list if mb_list else None, src_rank=src_rank, group=group
    )
    # bcast other attributes
    to_broadcast = (
        [
            mb_list.mb_spec,
            mb_list.forward_indices,
            mb_list.backward_indices,
            mb_list.group_lens,
            mb_list.padding_lengths,
            mb_list.padded_to_lengths,
            mb_list.align_to_lengths,
        ]
        if mb_list
        else [None for _ in range(7)]
    )
    dist.broadcast_object_list(to_broadcast, src=src_rank, group=group)
    (
        mb_spec,
        forward_indices,
        backward_indices,
        group_lens,
        padding_lengths,
        padded_to_lengths,
        align_to_lengths,
    ) = to_broadcast
    return MicroBatchList(
        data=data,
        mb_spec=mb_spec,
        mbs=mbs,
        forward_indices=forward_indices,
        backward_indices=backward_indices,
        group_lens=group_lens,
        padded_mbs=padded_mbs,
        padding_lengths=padding_lengths,
        padded_to_lengths=padded_to_lengths,
        old_cu_seqlens_list=old_cu_seqlens_list,
        align_to_lengths=align_to_lengths,
    )


def cycle_dataloader(dataloader: StatefulDataLoader, num_cycles: int = -1):
    """Cycle through a dataloader indefinitely."""
    epoch = 0
    while True:
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)
        yield from dataloader
        epoch += 1
        if num_cycles > 0 and epoch >= num_cycles:
            break


class Normalization:
    """
    Adaptive normalization with different levels.

    Supports independent specification of normalization level for mean and std:
    - "batch": normalize across entire batch (with optional all_reduce in distributed setting)
    - "group": normalize within fixed-size groups
    - None: no centering or no std scaling
    """

    def __init__(self, config: NormConfig):
        self.mean_level = config.mean_level
        self.mean_leave1out = config.mean_leave1out
        self.std_level = config.std_level
        self.std_unbiased = config.std_unbiased
        self.group_size = config.group_size
        self.eps = config.eps

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        high_precision: bool = True,
        reduce_group=None,
    ) -> torch.Tensor:
        bs = x.size(0)
        eps = self.eps

        # Early return if no elements are active (all masked out)
        if loss_mask is not None and loss_mask.sum().item() == 0:
            return x.float()

        # Step 1: Compute mean
        if self.mean_level == "batch":
            mean = self._compute_mean(
                x,
                loss_mask,
                high_precision=high_precision,
                leave_one_out=self.mean_leave1out,
                all_reduce=True,
                reduce_group=reduce_group,
            )
            mean = mean.expand_as(x)
        elif self.mean_level == "group":
            mean = torch.zeros_like(x)
            for i in range(0, bs // self.group_size):
                s = slice(i * self.group_size, (i + 1) * self.group_size)
                xx = x[s]
                m = loss_mask[s] if loss_mask is not None else None

                # Special case: with group_size=1 and leave_one_out=True, mean should be 0
                if self.group_size == 1 and self.mean_leave1out:
                    dtype = torch.float64 if high_precision else torch.float32
                    group_mean = torch.zeros(
                        (1, *xx.shape[1:]), dtype=dtype, device=xx.device
                    )
                else:
                    group_mean = self._compute_mean(
                        xx,
                        m,
                        high_precision=high_precision,
                        leave_one_out=self.mean_leave1out,
                        all_reduce=False,
                        reduce_group=None,
                    )
                mean[s] = group_mean.expand_as(xx)
        else:  # mean_level == "none"
            mean = torch.zeros_like(x)

        # Subtract mean
        x_centered = x - mean
        # mask unrelevant elements as 0
        if loss_mask is not None:
            x_centered = x_centered * loss_mask

        # Step 2: Compute std
        if self.std_level == "batch":
            std = self._compute_std(
                x,
                loss_mask,
                mean,
                unbiased=self.std_unbiased,
                high_precision=high_precision,
                all_reduce=True,
                reduce_group=reduce_group,
            )
            std = std.expand_as(x)
        elif self.std_level == "group":
            std = torch.zeros_like(x)
            for i in range(0, bs // self.group_size):
                s = slice(i * self.group_size, (i + 1) * self.group_size)
                xx = x[s]
                m = loss_mask[s] if loss_mask is not None else None
                group_mean_slice = mean[s]  # already computed and expanded

                # Special case: with group_size=1 and std_unbiased=True, std should be 1 for numerical stability
                if self.group_size == 1 and self.std_unbiased:
                    dtype = torch.float64 if high_precision else torch.float32
                    group_std = torch.ones(
                        (1, *xx.shape[1:]), dtype=dtype, device=xx.device
                    )
                else:
                    group_std = self._compute_std(
                        xx,
                        m,
                        group_mean_slice,
                        unbiased=self.std_unbiased,
                        high_precision=high_precision,
                        all_reduce=False,
                        reduce_group=reduce_group,
                    )
                std[s] = group_std.expand_as(xx)
        else:
            std = torch.ones_like(x)
            eps = 0.0

        # Normalize
        return (x_centered / (std + eps)).float()

    @staticmethod
    def _compute_mean(
        x: torch.Tensor,
        mask: torch.Tensor | None,
        high_precision: bool,
        leave_one_out: bool,
        all_reduce: bool,
        reduce_group,
    ) -> torch.Tensor:
        """Compute mean only, using masked_normalization internals."""
        dtype = torch.float64 if high_precision else torch.float32
        x = x.to(dtype)
        dim = tuple(range(len(x.shape)))

        if mask is None:
            factor = torch.tensor(
                np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
            )
            x_masked = x
            x_sum = x.sum(dim=dim, keepdim=True)
        else:
            mask = mask.to(dtype)
            x_masked = x * mask
            factor = mask.sum(dim, keepdim=True)
            x_sum = x_masked.sum(dim=dim, keepdim=True)

        if dist.is_initialized() and all_reduce:
            dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
            dist.all_reduce(x_sum, op=dist.ReduceOp.SUM, group=reduce_group)

        if leave_one_out:
            if factor.item() <= 1:
                return torch.zeros_like(x_sum)
            # For leave-one-out, we need to compute mean excluding each element individually
            # This requires broadcasting: (total_sum - each_element) / (count - 1)
            if mask is None:
                # Broadcast x_sum to original shape and subtract each element
                x_sum_broadcast = x_sum.expand_as(x)
                leave_one_out_sum = x_sum_broadcast - x
                return leave_one_out_sum / (factor - 1)
            else:
                # For masked case, only subtract where mask is 1
                x_sum_broadcast = x_sum.expand_as(x)
                leave_one_out_sum = x_sum_broadcast - x_masked
                # Only compute leave-one-out where mask is 1, elsewhere return global mean
                regular_mean = x_sum / factor
                leave_one_out_mean = leave_one_out_sum / torch.clamp(
                    factor - mask, min=1.0
                )
                return torch.where(
                    mask > 0, leave_one_out_mean, regular_mean.expand_as(x)
                )

        if factor.item() == 0:
            return torch.zeros_like(x_sum)
        return x_sum / factor

    @staticmethod
    def _compute_std(
        x: torch.Tensor,
        mask: torch.Tensor | None,
        mean: torch.Tensor,
        unbiased: bool,
        high_precision: bool,
        all_reduce: bool,
        reduce_group,
    ) -> torch.Tensor:
        """Compute std only, given precomputed mean."""
        dtype = torch.float64 if high_precision else torch.float32
        x = x.to(dtype)
        mean = mean.to(dtype)
        dim = tuple(range(len(x.shape)))

        if mask is None:
            factor = torch.tensor(
                np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
            )
            x_centered = x - mean
            x_sum_sq = (x_centered**2).sum(dim=dim, keepdim=True)
        else:
            mask = mask.to(dtype)
            x_masked = x * mask
            factor = mask.sum(dim, keepdim=True)
            x_centered = x_masked - mean * mask  # only apply mean where mask is 1
            x_sum_sq = (x_centered**2).sum(dim=dim, keepdim=True)

        if dist.is_initialized() and all_reduce:
            dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
            dist.all_reduce(x_sum_sq, op=dist.ReduceOp.SUM, group=reduce_group)

        if unbiased:
            if factor.item() <= 1:
                return torch.ones_like(x_sum_sq)
            return (x_sum_sq / (factor - 1)).sqrt()

        if factor.item() == 0:
            return torch.ones_like(x_sum_sq)
        return (x_sum_sq / factor).sqrt()


class KLEstimator:
    """
    KL divergence estimator, supports k1, k2 and k3.
    """

    def __init__(self, kl_estimator: str = "k1", apply_clamp: bool = True):
        self.kl_estimator = kl_estimator
        if kl_estimator not in ["k1", "k2", "k3"]:
            raise ValueError(
                f"Invalid KL estimator: {kl_estimator}. Valid choices: k1, k2, k3"
            )
        self.apply_clamp = apply_clamp

    def __call__(
        self, log_probs: torch.Tensor, log_probs_base: torch.Tensor
    ) -> torch.Tensor:
        return self._compute_approx_kl(
            log_probs, log_probs_base, self.kl_estimator, self.apply_clamp
        )

    # adapted from https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/utils.py#L7
    @staticmethod
    def _compute_approx_kl(
        log_probs: torch.Tensor,
        log_probs_base: torch.Tensor,
        kl_estimator: str = "k1",
        apply_clamp: bool = True,
    ) -> torch.Tensor:
        """
        Compute the approximate KL divergence between two distributions.
        Schulman blog: http://joschu.net/blog/kl-approx.html

        Args:
            log_probs: Log probabilities of the new distribution.
            log_probs_base: Log probabilities of the base distribution.
        """

        if kl_estimator == "k1":
            log_ratio = log_probs.float() - log_probs_base.float()

        # The k2 estimator is the non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        # The k2_loss is approximately equivalent to the
        # one-step KL divergence penalty with the k1 estimator
        # used in https://arxiv.org/pdf/2310.10505.
        if kl_estimator == "k2":
            log_ratio = log_probs.float() - log_probs_base.float()
            log_ratio = log_ratio**2 / 2.0

        # The k3 estimator is the non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        if kl_estimator == "k3":
            log_ratio = log_probs.float() - log_probs_base.float()
            log_ratio = -log_ratio
            log_ratio = log_ratio.exp() - 1 - log_ratio

        if apply_clamp:
            log_ratio = log_ratio.clamp(min=-10, max=10)
        return log_ratio


def make_dummy_eval_item(template: dict[str, Any]) -> dict[str, Any]:
    """Create a zero-contribution dummy item matching *template*'s schema.

    Every tensor field is replaced with a minimal all-zeros tensor that
    preserves dtype and device.  ``attention_mask`` and ``loss_mask`` are
    set to zero so that downstream loss/metric code treats the item as
    contributing nothing.
    """

    def _zero_tensor_like(tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, 1), dtype=tensor.dtype, device=tensor.device)

    dummy: dict[str, Any] = {}
    for key, value in template.items():
        if key in {"attention_mask", "loss_mask"}:
            if isinstance(value, torch.Tensor):
                dummy[key] = _zero_tensor_like(value)
            else:
                dummy[key] = torch.zeros((1, 1), dtype=torch.bool)
            continue

        if key.startswith("multi_modal_input"):
            dummy[key] = [{}]
            continue

        if isinstance(value, torch.Tensor):
            dummy[key] = _zero_tensor_like(value)
        else:
            dummy[key] = copy.deepcopy(value)

    return dummy
