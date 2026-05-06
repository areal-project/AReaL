# SPDX-License-Identifier: Apache-2.0
"""RDT Megatron Adapter for training-side weight update."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from awex.meta.weight_meta import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.sharding.param_sharding import ShardingType
from awex.sharding.rank_info import RankInfo
from awex.transfer.transfer_plan import TransferPlan, TransferPlanBuilder

from areal.experimental.weight_update.rdt import fetch_kv_metadata
from areal.utils import logging

if TYPE_CHECKING:
    from areal.engine.megatron_engine import MegatronEngine

logger = logging.getLogger("RDTMegatronAdapter")


class RDTMegatronAdapter:
    """RDT training adapter for MegatronEngine supporting DP, TP, and PP.

    PP: get_named_parameters already yields only the current stage's layers
    (with globally-correct HF layer indices via get_transformer_layer_offset),
    so each rank naturally reports only its own subset of parameters.

    TP: all_gather_param gathers the full tensor on every TP rank before
    convert_to_hf. IW pulls via Ray RPC, so all TP ranks have identical tensors.

    Unlike awex (NCCL P2P push), RDT uses pull-based Ray RPC:
    - TW stores TransferPlan and knows what each IW needs
    - IW tells TW its infer_rank, TW returns corresponding weights
    """

    def __init__(self, engine: MegatronEngine):
        self._engine = engine
        self._transfer_plans: dict[str, TransferPlan] = {}
        self._transfer_ranks: dict[str, int] = {}

    @property
    def parallelism_strategy(self) -> dict:
        from megatron.core import parallel_state as mpu

        tp_size = mpu.get_tensor_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()
        return {
            "world_size": self._engine.world_size,
            "tp_size": tp_size,
            "pp_size": mpu.get_pipeline_model_parallel_world_size(),
            "dp_size": self._engine.data_parallel_world_size,
            "ep_size": mpu.get_expert_model_parallel_world_size(),
            "dp_replicated": tp_size > 1 or cp_size > 1,
        }

    def get_weight_metadata(self) -> list[ParameterMeta]:
        """Extract parameter metadata for TransferPlan building.

        Returns:
            list[ParameterMeta]: Parameter metadata for all model parameters
        """
        rank_info = self._build_rank_info()
        metadata: list[ParameterMeta] = []

        for hf_name, tensor in self._iter_hf_params():
            shape = tuple(tensor.shape)
            numel = int(tensor.numel())
            shard_meta = ParameterShardMeta(
                tp_rank=rank_info.tp_rank,
                attn_tp_rank=rank_info.attn_tp_rank,
                pp_rank=rank_info.pp_rank,
                ep_rank=rank_info.ep_rank,
                ep_tp_rank=rank_info.ep_tp_rank,
                global_rank=rank_info.global_rank,
                world_size=rank_info.world_size,
                engine_rank=rank_info.engine_rank,
                cp_rank=rank_info.cp_rank,
                cp_size=rank_info.cp_size,
                cp_mode=rank_info.cp_mode,
                name=hf_name,
                shape=shape,
                numel=numel,
                dtype=tensor.dtype,
                global_offset=tuple([0] * len(shape)),
                sharding_type=ShardingType.NO_SHARDING,
                num_shards=1,
                sharding_dim=0,
            )
            replica = ParameterReplicaMeta(shards=[shard_meta])
            metadata.append(
                ParameterMeta(
                    name=hf_name,
                    global_numel=numel,
                    global_shape=shape,
                    dtype=tensor.dtype,
                    shards=[shard_meta],
                    replicas=[replica],
                )
            )

        return metadata

    def get_local_shard_parameters(
        self, required_names: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        """Return local tensors in HF naming.

        Args:
            required_names: Optional filter for specific parameters

        Returns:
            dict[str, torch.Tensor]: Local tensors by HF name
        """
        required = set(required_names) if required_names else None
        result: dict[str, torch.Tensor] = {}
        for hf_name, tensor in self._iter_hf_params():
            if required is not None and hf_name not in required:
                continue
            result[hf_name] = tensor
        return result

    def save_parameters(self, save_path: str, names: list[str] | None = None) -> None:
        """Save local shard parameters to file for debugging.

        Args:
            save_path: File path to save parameters
            names: Optional filter for specific parameters
        """
        params = self.get_local_shard_parameters(names)
        cpu_params = {k: v.detach().cpu().clone() for k, v in params.items()}
        torch.save(cpu_params, save_path)

    def init_weight_update_group(
        self,
        pair_name: str,
        kv_store_url: str,
        infer_world_size: int,
        train_world_size: int,
        num_engines: int,
        transfer_rank: int,
    ) -> None:
        """Initialize RDT weight update group for TW.

        Args:
            pair_name: TW-IW pair identifier
            kv_store_url: Gateway KV store URL
            infer_world_size: Total IW world size
            train_world_size: Total TW world size
            num_engines: Number of IW engines
            transfer_rank: TW's transfer rank
        """
        self._transfer_ranks[pair_name] = transfer_rank

        infer_meta, train_meta = fetch_kv_metadata(kv_store_url, pair_name)

        builder = TransferPlanBuilder(
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            num_infer_engines=num_engines,
        )
        self._transfer_plans[pair_name] = builder.build_local_transfer_plan(
            infer_meta, train_meta, global_transfer_rank=transfer_rank
        )
        logger.info(
            f"RDT TW init: Built TransferPlan for pair '{pair_name}' transfer_rank={transfer_rank}"
        )

    def get_weights_for_infer_rank(
        self, pair_name: str, infer_rank: int, version: int
    ) -> dict[str, torch.Tensor]:
        """Return sliced parameters for a specific infer_rank.

        TW uses its TransferPlan to determine what to send to each IW.

        Args:
            pair_name: TW-IW pair identifier
            infer_rank: IW's global rank (determines which tensors to return)
            version: Weight version number

        Returns:
            dict[str, torch.Tensor]: Sliced tensors for this infer_rank
        """
        if pair_name not in self._transfer_plans:
            raise RuntimeError(f"TransferPlan not initialized for pair '{pair_name}'")

        plan = self._transfer_plans[pair_name]
        local_params = self.get_local_shard_parameters()

        sliced_tensors: dict[str, torch.Tensor] = {}

        for send_rank, operations in plan.inter_operations.items():
            for op in operations:
                if op.recv_shard_meta.global_rank == infer_rank:
                    full_tensor = local_params.get(op.send_shard_meta.name)
                    if full_tensor is None:
                        logger.warning(
                            f"TW tensor not found: {op.send_shard_meta.name}"
                        )
                        continue
                    sliced_tensors[op.recv_shard_meta.name] = full_tensor[
                        op.train_slices
                    ]

        return sliced_tensors

    def teardown_weight_update_group(self, pair_name: str | None = None) -> None:
        """Clear stored TransferPlans.

        Args:
            pair_name: Optional specific pair to teardown; clears all if None
        """
        if pair_name:
            self._transfer_plans.pop(pair_name, None)
            self._transfer_ranks.pop(pair_name, None)
        else:
            self._transfer_plans.clear()
            self._transfer_ranks.clear()

    def _build_rank_info(self) -> RankInfo:
        """Build RankInfo for shard metadata extraction.

        Returns:
            RankInfo: Rank information for current worker
        """
        from megatron.core import parallel_state as mpu

        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        ep_size = mpu.get_expert_model_parallel_world_size()
        ep_rank = mpu.get_expert_model_parallel_rank()
        etp_size = mpu.get_expert_tensor_parallel_world_size()
        etp_rank = mpu.get_expert_tensor_parallel_rank()
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", self._engine.rank))

        return RankInfo(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            dp_size=self._engine.data_parallel_world_size,
            dp_rank=self._engine.data_parallel_rank,
            ep_rank=ep_rank,
            ep_size=ep_size,
            ep_tp_rank=etp_rank,
            ep_tp_size=etp_size,
            attn_tp_rank=tp_rank,
            attn_tp_size=tp_size,
            attn_dp_rank=self._engine.data_parallel_rank,
            world_size=self._engine.world_size,
            global_rank=self._engine.rank,
            local_rank=local_rank,
            engine_rank=0,
            is_infer=False,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_mode="ring" if cp_size > 1 else "none",
        )

    def _iter_hf_params(self):
        """Yield (hf_name, tensor) for every parameter on this rank.

        Uses get_named_parameters + all_gather_param + convert_to_hf to produce
        HF-style per-expert names (e.g. experts.0.gate_proj.weight). The SGLang
        adapter's _unfuse_params converts SGLang's fused w13/w2 format to the
        same per-expert names, so both sides match for the transfer plan.
        """
        from areal.engine.megatron_utils.megatron import (
            all_gather_param,
            convert_to_hf,
            get_named_parameters,
        )

        num_moe_experts = getattr(self._engine.tf_config, "num_moe_experts", None)
        model_name = self._engine.hf_config.model_type
        tie_word_embeddings = getattr(
            self._engine.hf_config, "tie_word_embeddings", False
        )

        for mcore_name, param in get_named_parameters(
            self._engine.model, num_moe_experts
        ):
            gathered = all_gather_param(
                mcore_name,
                param,
                fp8_direct_convert=False,
                quantization_config=None,
                duplicated_param_names=self._engine._duplicated_param_names,
            )
            if not isinstance(gathered, torch.Tensor):
                gathered = gathered.data

            for hf_name, tensor in convert_to_hf(
                self._engine.tf_config,
                model_name,
                mcore_name,
                gathered,
            ):
                if tie_word_embeddings and hf_name == "lm_head.weight":
                    continue
                yield hf_name, tensor.detach()
