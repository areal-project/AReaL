# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from mbridge.core import Bridge
from mbridge.core.util import unwrap_model
from megatron.core import parallel_state as mpu
from megatron.core.fp8_utils import is_float8tensor
from safetensors.torch import save_file
from torch.distributed._functional_collectives import all_gather_into_tensor_coalesced

from areal.engine.megatron_utils.fp8 import (
    FP8BlockwiseTensorHelper,
    convert_fp8_helper_to_pytorch_fp8,
    get_block_size_from_config,
    quantize_params,
)
from areal.infra.platforms import current_platform
from areal.models.mcore.registry import unwrap_to_gpt_model
from areal.utils import logging

logger = logging.getLogger("HFSaver")

HF_MODEL_CONFIG_FILES = [
    "generation_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
]


def _maybe_convert_from_te_fp8_params(
    params, fp8_direct_convert: bool, weight_block_size: int
):
    """Convert TE FP8 parameters to the appropriate format.

    Args:
        params: Parameter or list of parameters to convert
        fp8_direct_convert: If True, convert TE FP8 to FP8BlockwiseTensorHelper;
                            If False, dequantize to bf16
        weight_block_size: Block size for FP8 quantization

    Returns:
        Converted parameter(s) in the same structure as input
    """
    is_single_param = not isinstance(params, list)
    param_list = [params] if is_single_param else params

    if fp8_direct_convert:
        # Direct FP8 conversion: convert TE FP8 to FP8BlockwiseTensorHelper
        converted = [
            FP8BlockwiseTensorHelper.from_te(p, weight_block_size)
            if is_float8tensor(p)
            else p
            for p in param_list
        ]
    else:
        # Dequantize TE FP8 tensors to bf16
        converted = [p.dequantize() if is_float8tensor(p) else p for p in param_list]

    return converted[0] if is_single_param else converted


def _maybe_convert_to_torch_fp8_params(
    global_name: str,
    converted_names: list,
    converted_params: list,
    quantization_config,
    fp8_direct_convert: bool,
):
    """Convert HF-format parameters to PyTorch FP8 format if quantization is configured.

    This function handles three cases:
    1. No quantization_config: Returns parameters unchanged
    2. fp8_direct_convert=True: Direct conversion (FP8BlockwiseTensorHelper -> PyTorch FP8)
    3. fp8_direct_convert=False: Quantization (bf16 -> PyTorch FP8)

    Args:
        global_name: Global name of the parameter (for logging/tracking)
        converted_names: List of HF-format parameter names
        converted_params: List of HF-format parameters (bf16 or FP8BlockwiseTensorHelper)
        quantization_config: Quantization configuration. If None, returns unchanged
        fp8_direct_convert: If True, direct FP8 conversion; if False, quantize from bf16

    Returns:
        Tuple of (converted_names, converted_params), optionally in PyTorch FP8 format
    """
    if quantization_config is None:
        return converted_names, converted_params

    if fp8_direct_convert:
        # Direct FP8 conversion: convert FP8BlockwiseTensorHelper to PyTorch FP8
        converted_named_params = convert_fp8_helper_to_pytorch_fp8(
            list(zip(converted_names, converted_params))
        )
    else:
        # Quantize from bf16 to PyTorch FP8
        converted_named_params = list(zip(converted_names, converted_params))
        converted_named_params = quantize_params(
            global_name, converted_named_params, quantization_config
        )

    converted_names = [name for name, _ in converted_named_params]
    converted_params = [param for _, param in converted_named_params]

    return converted_names, converted_params


def copy_hf_configs(src_model_dir, dst_model_dir):
    for file in HF_MODEL_CONFIG_FILES:
        try:
            shutil.copy(
                os.path.join(src_model_dir, file),
                os.path.join(dst_model_dir, file),
            )
            logger.info(f"copied {file} from {src_model_dir} to {dst_model_dir}")
        except FileNotFoundError:
            logger.info(f"{file} not exist in {src_model_dir} skipping.")
    # Copy remote codes and chat template files
    for file in os.listdir(src_model_dir):
        copy = False
        for prefix in ["chat_format", "configuration_", "modeling_", "tokenization_"]:
            if file.startswith(prefix) and file.endswith(".py"):
                copy = True
                break
        # Chat template files (e.g. chat_template.jinja)
        if file.startswith("chat_template"):
            copy = True
        if copy:
            shutil.copy(
                os.path.join(src_model_dir, file),
                os.path.join(dst_model_dir, file),
            )
            logger.info(f"copied {file} from {src_model_dir} to {dst_model_dir}")


def _patch_saved_config(base_model_path, saved_path):
    """Patch saved config.json to preserve model_type and torch_dtype.

    Some HF config classes lack a ``model_type`` class attribute, causing
    ``save_pretrained()`` to lose the field (``PretrainedConfig.to_dict()``
    reads the class attribute, not the instance value).  This restores
    critical fields from the original model's config.json.
    """
    orig_config_path = os.path.join(base_model_path, "config.json")
    saved_config_path = os.path.join(saved_path, "config.json")

    if not os.path.exists(orig_config_path) or not os.path.exists(saved_config_path):
        return

    with open(orig_config_path) as f:
        orig_config = json.load(f)
    with open(saved_config_path) as f:
        saved_config = json.load(f)

    patched_fields = []

    # Restore model_type if missing or null
    if not saved_config.get("model_type") and orig_config.get("model_type"):
        saved_config["model_type"] = orig_config["model_type"]
        patched_fields.append(f"model_type={orig_config['model_type']}")

    # Restore torch_dtype if missing
    if "torch_dtype" not in saved_config and "torch_dtype" in orig_config:
        saved_config["torch_dtype"] = orig_config["torch_dtype"]
        patched_fields.append(f"torch_dtype={orig_config['torch_dtype']}")

    if patched_fields:
        with open(saved_config_path, "w") as f:
            json.dump(saved_config, f, indent=2)
        logger.info(f"Patched config.json: {', '.join(patched_fields)}")


def _bridge_uses_stacked_experts(bridge: Bridge) -> bool:
    """Detect bridges whose HF format keeps experts grouped under a single
    stacked tensor (e.g., Qwen3-VL-MoE ``mlp.experts.gate_up_proj`` shape
    ``[E, hidden, 2*expert_dim]``) rather than per-expert flat keys.

    Inferred from the bridge's ``_MLP_MAPPING``: per-expert-flat bridges (Qwen3-MoE,
    BailingMoeV2, DeepSeekV3) map ``mlp.experts.linear_fc1.weight`` to HF names that
    template ``{expert_id}``. Stacked bridges map to a single HF name without that
    template, and their ``_weight_to_hf_format`` buffers per-expert tensors and
    only emits the stacked output when ``num_moe_experts`` are accumulated. Under
    EP>1 this means the caller must arrange to feed all ``num_moe_experts`` in a
    single rank's loop — the per-EP-rank loop in the existing fast path leaves
    the buffer permanently under-filled.
    """
    mapping = getattr(bridge, "_MLP_MAPPING", None)
    if not mapping:
        return False
    for mcore_pat, hf_names in mapping.items():
        if "mlp.experts.linear_fc" not in mcore_pat:
            continue
        for n in hf_names:
            if "{expert_id}" not in n:
                return True
    return False


def split_state_dict_into_shards(state_dict: dict, n_shards: int) -> list[dict]:
    if n_shards == 1:
        return [state_dict]

    keys = list(state_dict.keys())
    if len(keys) < n_shards:
        raise ValueError(f"state_dict has {len(keys)} keys, but n_shards={n_shards}")

    shard_size = len(keys) // n_shards
    extra = len(keys) % n_shards
    shard_size_list = [shard_size for _ in range(n_shards)]
    shard_size_list[-1] = shard_size + extra
    start, shards = 0, []
    for i, size in enumerate(shard_size_list):
        shard = {}
        for j in range(start, start + size):
            shard[keys[j]] = state_dict[keys[j]]
        start += size
        shards.append(shard)
    return shards


@dataclass
class McoreDistributedWeightSpec:
    param: torch.Tensor
    local_name: str
    global_name: str
    local_shape: list[int]
    dtype: str
    tensor_model_parallel: bool
    pp_rank: int
    vpp_rank: int

    def full_param_size_byte(self) -> int:
        s = np.prod(self.local_shape) * getattr(torch, self.dtype).itemsize
        if self.tensor_model_parallel:
            if ".mlp.experts.linear_fc" in self.global_name:
                s *= mpu.get_expert_tensor_parallel_world_size()
            else:
                s *= mpu.get_tensor_model_parallel_world_size()
        return int(s)


def _emit_stacked_moe_expert_sd(
    bridge: Bridge,
    expert_specs: list["McoreDistributedWeightSpec"],
    *,
    all_gather_outputs: dict,
    etp_size: int,
    ep_size: int,
    ep_rank: int,
    ep_group,
    quantization_config,
    fp8_direct_convert: bool,
    weight_block_size: int,
) -> dict:
    """Stacked-MoE save path (e.g., Qwen3-VL-MoE).

    mbridge buffers per-expert tensors and only emits the stacked HF tensor
    once ``num_moe_experts`` have arrived. Each EP rank only owns
    ``num_moe_experts / ep_size`` experts, so this helper EP-gathers merged
    tensors per ``(layer, fc)`` group, feeds mbridge in global-index order so
    its buffer fills exactly once, and consolidates writes onto ``ep_rank==0``.
    Other EP ranks still participate in collectives but return an empty dict.

    Stacked outputs are moved to CPU memory immediately to bound peak GPU
    footprint by one stacked tensor at a time. Without this, all
    ``num_layers × 2`` stacked tensors (~74 GB for Qwen3-VL-30B-A3B)
    accumulate on the EP-rank-0 GPU and trigger CUDA OOM.
    """
    expert_sd: dict = {}
    num_experts = bridge.config.num_moe_experts
    num_experts_per_rank = num_experts // ep_size
    assert num_experts_per_rank * ep_size == num_experts, (
        f"num_moe_experts={num_experts} not divisible by ep_size={ep_size}"
    )

    # Group local specs by (layer_idx, fc_kind) and order by local-expert-id
    # within each group. ``local_name`` carries the rank-local TEGroupedLinear
    # suffix (``weight0..weight{num_experts_per_rank-1}``); ``global_name`` was
    # rewritten by the caller to bake in the global expert id, but for sort
    # order the local idx is sufficient and avoids the rank-dependent offset.
    _expert_re = re.compile(
        r"\.layers\.(\d+)\.mlp\.experts\.(linear_fc\d+)\.weight(\d+)$"
    )
    groups: dict[tuple[str, str], list[tuple[int, McoreDistributedWeightSpec]]] = {}
    for s in expert_specs:
        m = _expert_re.search(s.local_name)
        if not m:
            raise ValueError(
                f"stacked-expert path: cannot parse layer/fc/expert "
                f"from local_name={s.local_name!r}"
            )
        layer_idx, fc_kind, local_idx_str = m.groups()
        local_id = int(local_idx_str)
        groups.setdefault((layer_idx, fc_kind), []).append((local_id, s))
    for k in groups:
        groups[k].sort(key=lambda t: t[0])

    # Invariant check across the EP group: every rank must see the same set of
    # (layer, fc) groups with the same per-group expert count. Without this, a
    # future refactor that injects a vision-tower expert spec on some ranks
    # only — or a VPP layout where local layers differ across EP ranks — would
    # silently mismatch the ``all_gather_into_tensor`` collective below and
    # either hang or scramble tensors across layers. One cheap
    # ``all_gather_object`` eliminates that class of
    # silent-corruption-on-future-refactor failures.
    group_keys = tuple(sorted(groups.keys()))
    ep_keys: list[tuple | None] = [None] * ep_size
    dist.all_gather_object(ep_keys, group_keys, group=ep_group)
    assert all(k == group_keys for k in ep_keys), (
        f"EP rank divergence in (layer, fc) groups: "
        f"rank {ep_rank} sees {group_keys}, peers see {ep_keys}"
    )
    for k, specs_with_idx in groups.items():
        assert len(specs_with_idx) == num_experts_per_rank, (
            f"layer {k} has {len(specs_with_idx)} local experts on "
            f"ep_rank {ep_rank}, expected {num_experts_per_rank}"
        )

    for (layer_idx, fc_kind), specs_with_idx in groups.items():
        local_specs = [s for _, s in specs_with_idx]
        # Pre-merge each spec's TP shard into a per-expert tensor.
        merged_local: list[torch.Tensor] = []
        for s in local_specs:
            if etp_size > 1:
                params = all_gather_outputs[s.global_name].chunk(etp_size, dim=0)
            else:
                params = [s.param]
            params = _maybe_convert_from_te_fp8_params(
                params, fp8_direct_convert, weight_block_size
            )
            merged = bridge._weight_merge_across_tp(s.global_name, params, s.param)
            merged_local.append(merged.contiguous())

        # EP all-gather: stack local-merged on dim 0, all-gather, split.
        # All ranks contribute the same number of experts (uniform EP shard).
        stacked_local = torch.stack(merged_local, dim=0)
        gathered = torch.empty(
            (ep_size,) + tuple(stacked_local.shape),
            dtype=stacked_local.dtype,
            device=stacked_local.device,
        )
        dist.all_gather_into_tensor(gathered, stacked_local, group=ep_group)
        # gathered shape: [ep_size, num_experts_per_rank, *expert_shape].
        # Flatten the first two dims to get global expert order:
        # ep_rank0 [0..n-1], ep_rank1 [n..2n-1], ...
        gathered = gathered.view(num_experts, *stacked_local.shape[1:])

        # Feed mbridge in global expert order. Build the synthetic mcore name
        # with the appropriate global index. mbridge's buffer fills after
        # num_experts calls and emits the stacked HF tensor; intermediate
        # calls return ``[], []``.
        # Use the first local spec's name as the template — the prefix is
        # rank-invariant, only the .weight{N} suffix varies per call.
        name_prefix = local_specs[0].global_name.split(".weight")[0]
        last_converted_names: list[str] = []
        last_converted_params: list[torch.Tensor] = []
        for global_idx in range(num_experts):
            name = f"{name_prefix}.weight{global_idx}"
            converted_names, converted_params = bridge._weight_to_hf_format(
                name, gathered[global_idx]
            )
            if converted_names:
                last_converted_names = converted_names
                last_converted_params = converted_params
        converted_names, converted_params = _maybe_convert_to_torch_fp8_params(
            name_prefix,
            last_converted_names,
            last_converted_params,
            quantization_config,
            fp8_direct_convert,
        )
        # Only ep_rank==0 contributes to the on-disk shard; other EP ranks
        # discard so their expert_sd stays empty (collectives have already
        # happened above, so this is safe).
        if ep_rank == 0:
            for n, p in zip(converted_names, converted_params):
                assert n not in expert_sd, n
                if isinstance(p, torch.Tensor) and p.is_cuda:
                    expert_sd[n] = p.detach().to(device="cpu", non_blocking=False)
                else:
                    expert_sd[n] = p
        # Free the gathered buffer before the next group.
        del gathered, stacked_local

    return expert_sd


def _emit_per_expert_flat_expert_sd(
    bridge: Bridge,
    expert_specs: list["McoreDistributedWeightSpec"],
    *,
    all_gather_outputs: dict,
    etp_size: int,
    quantization_config,
    fp8_direct_convert: bool,
    weight_block_size: int,
) -> dict:
    """Per-expert-flat save path (Qwen3-MoE, BailingMoeV2, DeepSeekV3).

    HF format keys experts as ``mlp.experts.{idx}.{gate,up,down}_proj.weight``,
    so each EP rank emits its local-experts subset independently and the
    union across ranks forms the full set without any cross-EP gather.
    """
    expert_sd: dict = {}
    for s in expert_specs:
        param = s.param
        if etp_size > 1:
            params = all_gather_outputs[s.global_name].chunk(etp_size, dim=0)
        else:
            params = [param]

        params = _maybe_convert_from_te_fp8_params(
            params, fp8_direct_convert, weight_block_size
        )
        merge_params = bridge._weight_merge_across_tp(s.global_name, params, param)
        converted_names, converted_params = bridge._weight_to_hf_format(
            s.global_name, merge_params
        )
        converted_names, converted_params = _maybe_convert_to_torch_fp8_params(
            s.global_name,
            converted_names,
            converted_params,
            quantization_config,
            fp8_direct_convert,
        )
        for n, p in zip(converted_names, converted_params):
            assert n not in expert_sd, n
            expert_sd[n] = p
    return expert_sd


def save_weights_to_hf_with_mbridge_fast(
    bridge: Bridge,
    models: list,
    weights_path: str,
    base_model_path: str | None = None,
    max_shard_size_byte: int = int(3e9),
    max_workers: int | None = None,
    fp8_direct_convert: bool = False,
):
    # 1. Prepare some global metadata required for saving the model.
    models = [unwrap_model(model) for model in models]
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_group = mpu.get_pipeline_model_parallel_group()

    local_to_global_maps = [
        bridge._weight_name_mapping_mcore_local_to_global(model, consider_ep=False)
        for model in models
    ]
    state_dicts = [m.state_dict() for m in models]

    # 2. Get all weights specification used to decide how we should save the model.
    device = None
    weight_specs = []
    for vpp_rank, model in enumerate(models):
        existing_keys = set()
        for name, param in model.named_parameters():
            device = param.device
            existing_keys.add(name)
            global_name = local_to_global_maps[vpp_rank][name]
            weight_specs.append(
                McoreDistributedWeightSpec(
                    param=param,
                    local_name=name,
                    global_name=global_name,
                    pp_rank=mpu.get_pipeline_model_parallel_rank(),
                    vpp_rank=vpp_rank,
                    local_shape=list(param.shape),
                    dtype=str(param.dtype).lstrip("torch."),
                    tensor_model_parallel=(
                        hasattr(param, "tensor_model_parallel")
                        and param.tensor_model_parallel
                    ),
                )
            )
        # note
        # there is a bug in megatron GPTModel
        # decoder.layers[n].mlp.router.expert_bias" in GPTModel is not registered in named_parameter, but in state_dict().
        # for now we patch it by adding those keys to extra_keys.
        extra_keys = [
            x
            for x in state_dicts[vpp_rank].keys()
            if "_extra_state" not in x and "expert_bias" in x and x not in existing_keys
        ]
        for name in extra_keys:
            param = state_dicts[vpp_rank][name].to(current_platform.current_device())
            global_name = local_to_global_maps[vpp_rank][name]
            weight_specs.append(
                McoreDistributedWeightSpec(
                    param=param,
                    local_name=name,
                    global_name=global_name,
                    pp_rank=pp_rank,
                    vpp_rank=vpp_rank,
                    local_shape=list(param.shape),
                    dtype=str(param.dtype).lstrip("torch."),
                    tensor_model_parallel=(
                        hasattr(param, "tensor_model_parallel")
                        and param.tensor_model_parallel
                    ),
                )
            )

    # 3. Separate parameters to be saved into expert/non-expert groups.
    # Non-expert parameters can be collectively saved by the (dp, cp, tp) group
    # while different pp ranks save their own disjoint parameters independently.
    non_expert_specs = list(
        filter(lambda s: ".mlp.experts.linear_fc" not in s.global_name, weight_specs)
    )
    non_expert_param_size = [s.full_param_size_byte() for s in non_expert_specs]
    non_expert_shards_this_stage = torch.tensor(
        min(
            len(non_expert_param_size),
            (sum(non_expert_param_size) + max_shard_size_byte - 1)
            // max_shard_size_byte,
        ),
        dtype=torch.int32,
        device=device,
    )
    pp_stage_non_expert_shards = [
        torch.zeros_like(non_expert_shards_this_stage) for _ in range(pp_size)
    ]
    dist.all_gather(
        pp_stage_non_expert_shards,
        non_expert_shards_this_stage,
        group=pp_group,
    )
    pp_stage_non_expert_shards = [int(x) for x in pp_stage_non_expert_shards]
    assert all(x >= 1 for x in pp_stage_non_expert_shards)

    # Expert parameters can be collectively saved by the the (edp, etp) group
    # while different (pp, ep) ranks save their own disjoint parameter independently.
    expert_specs = list(
        filter(lambda s: ".mlp.experts.linear_fc" in s.global_name, weight_specs)
    )
    # Stacked-experts bridges (e.g., Qwen3-VL-MoE) emit a single stacked HF
    # tensor per (layer, fc) that requires all ``num_moe_experts`` per-expert
    # tensors to flow through ``bridge._weight_to_hf_format`` on the SAME rank
    # so its internal buffer fills. Under EP>1 we EP-gather and consolidate
    # writes onto ep_rank==0, so total expert shards == per-PP-stage count
    # (no ``* ep_size`` multiplication; one rank writes them all).
    stacked_experts = _bridge_uses_stacked_experts(bridge)
    if stacked_experts:
        # Total bytes after EP consolidation = local bytes × ep_size.
        ep_size_for_size = mpu.get_expert_model_parallel_world_size()
        expert_param_size = [
            s.full_param_size_byte() * ep_size_for_size for s in expert_specs
        ]
    else:
        expert_param_size = [s.full_param_size_byte() for s in expert_specs]
    expert_shards_this_stage = torch.tensor(
        min(
            len(expert_param_size),
            (sum(expert_param_size) + max_shard_size_byte - 1) // max_shard_size_byte,
        ),
        dtype=torch.int32,
        device=device,
    )
    pp_stage_expert_shards = [
        torch.zeros_like(expert_shards_this_stage) for _ in range(pp_size)
    ]
    dist.all_gather(
        pp_stage_expert_shards,
        expert_shards_this_stage,
        group=pp_group,
    )
    ep_size = mpu.get_expert_model_parallel_world_size()
    # Per-expert-flat bridges: each EP rank writes its own disjoint shards →
    # total shards across the EP group is ``per_rank * ep_size``.
    # Stacked bridges: only ep_rank==0 writes after EP-gather → no multiplication.
    if not stacked_experts:
        pp_stage_expert_shards = [int(x) for x in pp_stage_expert_shards] * ep_size
    else:
        pp_stage_expert_shards = [int(x) for x in pp_stage_expert_shards]
    if len(expert_param_size) > 0:
        assert all(x >= 1 for x in pp_stage_expert_shards)

    # 4. Compute the number of totoal required model shards.
    total_n_shards = sum(pp_stage_non_expert_shards) + sum(pp_stage_expert_shards)
    output_filename = "model" + "-{shard:05d}" + f"-of-{total_n_shards:05d}.safetensors"
    bin_index = {}
    bin_index["metadata"] = dict(
        total_size=sum(non_expert_param_size) + sum(expert_param_size)
    )
    bin_index["weight_map"] = {}
    weight_map = {}

    # 5. Save non-expert weights.
    # Each process independently saves its own portion. The following logic computes
    # which portion should this process save.
    shard_offset = sum(pp_stage_non_expert_shards[:pp_rank])
    g = mpu.get_tensor_and_data_parallel_group(with_context_parallel=True)
    mesh_size = dist.get_world_size(g)
    mesh_idx = dist.get_rank(group=g)
    n_shards = int(non_expert_shards_this_stage)
    n_shards_per_gpu = (n_shards + mesh_size - 1) // mesh_size
    if mesh_idx < len(range(0, n_shards, n_shards_per_gpu)):
        local_start = list(range(0, n_shards, n_shards_per_gpu))[mesh_idx]
    else:
        local_start = n_shards
    # all-gather weights across the TP group and converts to HF format
    # Optimized via a single `all_gather_into_tensor_coalesced` call, which should be
    # faster than plain all_gather.
    non_expert_sd = {}
    _all_gather_specs = []
    all_gather_outputs = {}
    quantization_config = getattr(bridge.hf_config, "quantization_config", None)
    weight_block_size = get_block_size_from_config(quantization_config, strict=True)

    for s in non_expert_specs:
        if s.tensor_model_parallel and mpu.get_tensor_model_parallel_world_size() > 1:
            _all_gather_specs.append(s)
    if _all_gather_specs:
        _all_gather_outputs = all_gather_into_tensor_coalesced(
            [s.param for s in _all_gather_specs],
            group=mpu.get_tensor_model_parallel_group(),
        )
        for s, gathered_param in zip(_all_gather_specs, _all_gather_outputs):
            all_gather_outputs[s.global_name] = gathered_param
    for s in non_expert_specs:
        param = s.param

        if s.tensor_model_parallel:
            # allocate a new tensor with proper size
            if mpu.get_tensor_model_parallel_world_size() <= 1:
                infer_params = [param]
            else:
                infer_params = all_gather_outputs[s.global_name].chunk(
                    mpu.get_tensor_model_parallel_world_size(), dim=0
                )
            # Convert TE FP8 -> torch bf16 -> torch FP8 and finally save the native torch FP8 model
            # First dequantize TE FP8 tensors to bf16, then convert_to_hf will quantize to PyTorch FP8
            # Or if fp8_direct_convert=True, convert TE FP8 -> FP8BlockwiseTensorHelper -> torch FP8
            infer_params = _maybe_convert_from_te_fp8_params(
                infer_params, fp8_direct_convert, weight_block_size
            )
            infer_params = bridge._weight_merge_across_tp(
                s.global_name, infer_params, param
            )
        else:
            infer_params = param
            infer_params = _maybe_convert_from_te_fp8_params(
                infer_params, fp8_direct_convert, weight_block_size
            )
        converted_names, converted_params = bridge._weight_to_hf_format(
            s.global_name, infer_params
        )
        converted_names, converted_params = _maybe_convert_to_torch_fp8_params(
            s.global_name,
            converted_names,
            converted_params,
            quantization_config,
            fp8_direct_convert,
        )
        for n, p in zip(converted_names, converted_params):
            assert n not in non_expert_sd, n
            non_expert_sd[n] = p
    # Split the state dict into shards and save the process's own shard.
    shards = split_state_dict_into_shards(non_expert_sd, n_shards)

    def _save_one_shard(x):
        i, shard = x
        shard_idx = shard_offset + i + local_start
        save_file(
            shard,
            os.path.join(weights_path, output_filename.format(shard=shard_idx + 1)),
        )

    # Multi-threaded save.
    _max_workers = max_workers
    if _max_workers is None:
        _max_workers = min(8, max(1, os.cpu_count() // dist.get_world_size()))
    _max_workers = min(_max_workers, n_shards_per_gpu)
    with ThreadPoolExecutor(max_workers=_max_workers) as executor:
        results = executor.map(
            _save_one_shard,
            list(enumerate(shards[local_start : local_start + n_shards_per_gpu])),
        )
        # consume the result
        for _ in results:
            pass
    # organize metadata
    for i, shard in enumerate(shards):
        shard_idx = shard_offset + i
        for k in shard:
            weight_map[k] = output_filename.format(shard=shard_idx + 1)
    weight_map_list = [None for _ in range(pp_size)]
    dist.all_gather_object(
        weight_map_list,
        weight_map,
        group=pp_group,
    )
    for wm in weight_map_list:
        bin_index["weight_map"].update(wm)

    # 6. Save expert weights.
    if len(expert_param_size) > 0:
        ep_size = mpu.get_expert_model_parallel_world_size()
        ep_rank = mpu.get_expert_model_parallel_rank()
        ep_group = mpu.get_expert_model_parallel_group()
        edp_size = dist.get_world_size(mpu.get_expert_data_parallel_group())
        edp_rank = mpu.get_expert_data_parallel_rank()
        etp_size: int = mpu.get_expert_tensor_parallel_world_size()
        etp_rank: int = mpu.get_expert_tensor_parallel_rank()
        etp_group = mpu.get_expert_tensor_parallel_group()

        # Stacked bridges consolidate all expert writes onto ep_rank==0 (per-PP
        # stage), so the shard offset must NOT include ``ep_rank * pp_size``.
        if stacked_experts:
            shard_offset = sum(pp_stage_non_expert_shards) + sum(
                pp_stage_expert_shards[:pp_rank]
            )
        else:
            shard_offset = sum(pp_stage_non_expert_shards) + sum(
                pp_stage_expert_shards[: ep_rank * pp_size + pp_rank]
            )
        mesh_size = edp_size * etp_size
        mesh_idx = edp_rank * etp_size + etp_rank
        n_shards = int(expert_shards_this_stage)
        n_shards_per_gpu = (n_shards + mesh_size - 1) // mesh_size
        if mesh_idx < len(range(0, n_shards, n_shards_per_gpu)):
            local_start = list(range(0, n_shards, n_shards_per_gpu))[mesh_idx]
        else:
            local_start = n_shards
        # map local expert name to global name if using expert parallel
        for s in expert_specs:
            if ep_size == 1:
                break
            num_experts = bridge.config.num_moe_experts
            num_experts_per_rank = num_experts // ep_size
            name_prefix, local_expert_id = s.global_name.split(".weight")
            local_expert_id = int(local_expert_id)
            global_expert_id = num_experts_per_rank * ep_rank + local_expert_id
            s.global_name = f"{name_prefix}.weight{global_expert_id}"
        # all-gather weights across the TP group and converts to HF format
        _all_gather_specs = []
        all_gather_outputs = {}
        for s in expert_specs:
            if etp_size > 1:
                _all_gather_specs.append(s)
        if _all_gather_specs:
            _all_gather_outputs = all_gather_into_tensor_coalesced(
                [s.param for s in _all_gather_specs],
                group=etp_group,
            )
            for s, gathered_param in zip(_all_gather_specs, _all_gather_outputs):
                all_gather_outputs[s.global_name] = gathered_param
        if stacked_experts and ep_size > 1:
            expert_sd = _emit_stacked_moe_expert_sd(
                bridge,
                expert_specs,
                all_gather_outputs=all_gather_outputs,
                etp_size=etp_size,
                ep_size=ep_size,
                ep_rank=ep_rank,
                ep_group=ep_group,
                quantization_config=quantization_config,
                fp8_direct_convert=fp8_direct_convert,
                weight_block_size=weight_block_size,
            )
        else:
            expert_sd = _emit_per_expert_flat_expert_sd(
                bridge,
                expert_specs,
                all_gather_outputs=all_gather_outputs,
                etp_size=etp_size,
                quantization_config=quantization_config,
                fp8_direct_convert=fp8_direct_convert,
                weight_block_size=weight_block_size,
            )
        # Stacked-MoE: only ep_rank==0 holds the consolidated expert_sd; other
        # EP ranks have empty dicts and must skip shard split/save (they still
        # participate in the weight_map collective below).
        write_shards = not (stacked_experts and ep_rank > 0)
        if write_shards:
            shards = split_state_dict_into_shards(expert_sd, n_shards)
        else:
            shards = []

        def _save_one_shard(x):
            i, shard = x
            shard_idx = shard_offset + i + local_start
            save_file(
                shard,
                os.path.join(weights_path, output_filename.format(shard=shard_idx + 1)),
            )

        if write_shards:
            _max_workers = max_workers
            if _max_workers is None:
                _max_workers = min(8, max(1, os.cpu_count() // dist.get_world_size()))
            _max_workers = min(_max_workers, max(1, n_shards_per_gpu))
            with ThreadPoolExecutor(max_workers=_max_workers) as executor:
                results = executor.map(
                    _save_one_shard,
                    list(
                        enumerate(shards[local_start : local_start + n_shards_per_gpu])
                    ),
                )
                # consume the result
                for _ in results:
                    pass
        # organize metadata
        for i, shard in enumerate(shards):
            shard_idx = shard_offset + i
            for k in shard:
                weight_map[k] = output_filename.format(shard=shard_idx + 1)
        ep_pp_group = mpu.get_expert_tensor_model_pipeline_parallel_group()
        weight_map_list = [None for _ in range(dist.get_world_size(ep_pp_group))]
        dist.all_gather_object(
            weight_map_list,
            weight_map,
            group=ep_pp_group,
        )
        for wm in weight_map_list:
            bin_index["weight_map"].update(wm)

    # 7. save metadata
    if dist.get_rank() == 0:
        bridge.hf_config.save_pretrained(weights_path)
        with open(os.path.join(weights_path, "model.safetensors.index.json"), "w") as f:
            json.dump(bin_index, f, indent=4)
        if base_model_path is not None:
            copy_hf_configs(base_model_path, weights_path)
            _patch_saved_config(base_model_path, weights_path)


def save_critic_value_head(models, weights_path):
    if mpu.is_pipeline_last_stage():
        is_tp_first = mpu.get_tensor_model_parallel_rank() == 0
        is_dp_first = mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        should_save_value_head = is_tp_first and is_dp_first

        if should_save_value_head:
            for model in models:
                _model = unwrap_to_gpt_model(model)
                if hasattr(_model, "output_layer"):
                    value_head_path = os.path.join(weights_path, "value_head.pt")
                    torch.save(_model.output_layer.state_dict(), value_head_path)
                    logger.info(f"Saved ValueHead weights to {value_head_path}")
                    break
