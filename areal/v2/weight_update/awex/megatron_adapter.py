# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
import os
import threading
import time
from typing import TYPE_CHECKING

import httpx
import torch
import torch.distributed as dist
from awex.meta.weight_meta import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.sharding.param_sharding import ShardingType
from awex.sharding.rank_info import RankInfo
from awex.transfer.nccl_comm import batch_send_recv, nccl_build_send_ops
from awex.transfer.transfer_plan import TransferPlan, TransferPlanBuilder
from awex.util.tensor_util import cuda_ipc_serialize

from areal.utils import logging
from areal.v2.weight_update.awex import (
    awex_wu_use_group,
    fetch_kv_metadata,
)
from areal.v2.weight_update.awex.delta_config import (
    delta_transfer_enabled,
    make_delta_tracker,
)
from areal.v2.weight_update.awex.delta_detect import (
    build_detector,
    delta_detector_mode,
)
from areal.v2.weight_update.nccl_group import (
    init_weights_update_group,
    setup_batch_isend_irecv,
)
from areal.v2.weight_update.training_adapter import (
    AwexTrainingAdapter,
)

if TYPE_CHECKING:
    from areal.engine.megatron_engine import MegatronEngine

logger = logging.getLogger("AwexMegatronAdapter")


def _group_tensors_for_colocate_ipc(
    tensors: list[torch.Tensor],
    max_group_bytes: int = 512 * 1024 * 1024,
) -> tuple[list[torch.Tensor], list[dict]]:
    """Pack tensors into bounded dtype-homogeneous CUDA IPC groups."""
    buckets: dict[torch.dtype, list[tuple[int, torch.Tensor]]] = {}
    group_tensors: list[torch.Tensor] = []
    metadata: list[dict] = []

    for original_index, tensor in enumerate(tensors):
        source = tensor.detach()
        if source.numel() == 0:
            group_index = len(group_tensors)
            group_tensors.append(
                torch.empty((1,), dtype=source.dtype, device=source.device)
            )
            metadata.append(
                {
                    "original_index": original_index,
                    "shape": source.shape,
                    "dtype": source.dtype,
                    "group_index": group_index,
                    "offset": 0,
                    "size": 0,
                }
            )
            continue
        buckets.setdefault(source.dtype, []).append((original_index, source))

    for entries in buckets.values():
        current: list[tuple[int, torch.Tensor]] = []
        current_bytes = 0

        def finalize_group() -> None:
            nonlocal current, current_bytes
            if not current:
                return
            group_index = len(group_tensors)
            group_tensors.append(
                torch.cat([tensor.reshape(-1) for _, tensor in current]).clone()
            )
            offset = 0
            for original_index, tensor in current:
                metadata.append(
                    {
                        "original_index": original_index,
                        "shape": tensor.shape,
                        "dtype": tensor.dtype,
                        "group_index": group_index,
                        "offset": offset,
                        "size": tensor.numel(),
                    }
                )
                offset += tensor.numel()
            current = []
            current_bytes = 0

        for original_index, source in entries:
            source = source.contiguous()
            tensor_bytes = source.numel() * source.element_size()
            if current and current_bytes + tensor_bytes > max_group_bytes:
                finalize_group()
            current.append((original_index, source))
            current_bytes += tensor_bytes
        finalize_group()

    return group_tensors, metadata


def _install_awex_qwen2_converter() -> None:
    """Keep Qwen2 parameter names aligned with SGLang's HF layout."""
    from awex.converter.mcore_converter import McoreToHFWeightConverter
    from awex.models.registry import ModelRegistry

    class Qwen2McoreToHFWeightConverter(McoreToHFWeightConverter):
        def _fuse_qkv(self, name: str) -> bool:
            del name
            return False

        @staticmethod
        def _normalize_attn_name(name: str) -> str:
            return name

    model_config = ModelRegistry.get_model_config("Qwen2ForCausalLM")
    if isinstance(model_config, dict):
        model_config = dict(model_config)
        model_config["mcore_converter"] = Qwen2McoreToHFWeightConverter
    else:
        model_config.mcore_converter = Qwen2McoreToHFWeightConverter
    ModelRegistry.models["Qwen2ForCausalLM"] = model_config


class AwexMegatronAdapter(AwexTrainingAdapter):
    """Awex training adapter for MegatronEngine supporting DP, TP, and PP.

    PP: get_named_parameters already yields only the current stage's layers
    (with globally-correct HF layer indices via get_transformer_layer_offset),
    so each rank naturally reports and sends only its own subset of parameters.
    The gateway's _merge_training_meta_by_name unions disjoint PP stage params
    by name, so the full model is covered across all PP ranks.

    TP: all_gather_param gathers the full tensor on every TP rank before
    convert_to_hf. dp_replicated=True tells awex that TP ranks within a DP
    group hold identical full tensors and only one needs to send.
    """

    def __init__(self, engine: MegatronEngine):
        self._engine = engine
        setattr(engine, "_awex_weight_update_adapter", self)
        self._transfer_plan: TransferPlan | None = None
        self._weights_update_group = None
        self._transfer_rank: int | None = None
        self._offloaded_optimizer_states: dict = {}
        self._offloaded_weights: dict[str, torch.Tensor] = {}
        self._released_tags: set[str] = set()
        self._colocate_lock = threading.Lock()
        self._colocate_admin_api_key: str = "areal-admin-key"
        self._colocate_http_client: httpx.Client | None = None
        self._colocate_timeout_s: float = 120.0
        # Lazy dte DeltaTracker (sender side); persists across versions to hold
        # the CPU snapshot baseline. Created on first delta-enabled transfer.
        self._delta_tracker = None
        # Lazy change detector: snapshot (default) | inversion (DTE_DELTA_DETECTOR).
        self._delta_detector = None

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
        if "Qwen2ForCausalLM" in getattr(self._engine.hf_config, "architectures", []):
            _install_awex_qwen2_converter()
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
        required = set(required_names) if required_names else None
        result: dict[str, torch.Tensor] = {}
        for hf_name, tensor in self._iter_hf_params():
            if required is not None and hf_name not in required:
                continue
            result[hf_name] = tensor
        return result

    def save_parameters(self, save_path: str, names: list[str] | None = None) -> None:
        weights_offloaded = "weights" in self._released_tags
        if weights_offloaded:
            self.resume_memory(tags=["weights"])
        try:
            params = self.get_local_shard_parameters(names)
            cpu_params = {k: v.detach().cpu().clone() for k, v in params.items()}
            torch.save(cpu_params, save_path)
        finally:
            if weights_offloaded:
                self.release_memory(tags=["weights"])

    def init_weight_update_group(
        self,
        pair_name: str,
        master_addr: str,
        master_port: int,
        transfer_rank: int,
        world_size: int,
        kv_store_url: str,
        infer_world_size: int,
        train_world_size: int,
        num_engines: int,
    ) -> None:
        self._transfer_rank = transfer_rank

        infer_meta, train_meta = fetch_kv_metadata(kv_store_url, pair_name)

        builder = TransferPlanBuilder(
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            num_infer_engines=num_engines,
        )
        self._transfer_plan = builder.build_local_transfer_plan(
            infer_meta, train_meta, global_transfer_rank=transfer_rank
        )

        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        self._weights_update_group = init_weights_update_group(
            master_address=master_addr,
            master_port=master_port,
            rank=transfer_rank,
            world_size=world_size,
            group_name=f"awex_{pair_name}",
            role="training",
        )

    def execute_weight_update(self, version: int) -> None:
        del version
        if self._transfer_plan is None:
            raise RuntimeError("Transfer plan is not initialized")
        if self._weights_update_group is None:
            raise RuntimeError("Weight update group is not initialized")
        if self._transfer_rank is None:
            raise RuntimeError("Transfer rank is not initialized")

        params = self.get_local_shard_parameters()
        send_ops, _, _ = nccl_build_send_ops(
            params,
            self._transfer_plan,
            self._weights_update_group,
            copy_rank=self._transfer_rank,
        )
        batch_send_recv(
            send_ops=send_ops,
            recv_ops=[],
            blocking=True,
            use_group=awex_wu_use_group(),
        )
        dist.barrier(group=self._weights_update_group)

    def batch_isend_irecv(self, **kwargs) -> None:
        setup_kwargs = {k: v for k, v in kwargs.items() if k != "world_size"}
        setup_batch_isend_irecv(
            self._weights_update_group,
            self._transfer_rank,
            kwargs.get("world_size", 0),
            **setup_kwargs,
        )

    def teardown_weight_update_group(self) -> None:
        if self._weights_update_group is not None and dist.is_initialized():
            dist.destroy_process_group(self._weights_update_group)
        self._weights_update_group = None
        self._transfer_plan = None
        self._transfer_rank = None
        if self._colocate_http_client is not None:
            self._colocate_http_client.close()
            self._colocate_http_client = None

    def _build_rank_info(self) -> RankInfo:
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

    def _iter_hf_params(self, theta_by_id: dict[int, torch.Tensor] | None = None):
        """Yield (hf_name, tensor) for every parameter on this rank.

        Uses get_named_parameters + all_gather_param + convert_to_hf to produce
        HF-style per-expert names (e.g. experts.0.gate_proj.weight). The SGLang
        adapter's _unfuse_params converts SGLang's fused w13/w2 format to the
        same per-expert names, so both sides match for the transfer plan.

        ``theta_by_id`` maps ``id(model_param) -> replacement tensor`` (same
        shape/dtype as the mcore param). The AdamW-inversion detector uses it to
        push reconstructed pre-step weights through the EXACT same all_gather +
        convert path as the live payload; a param without an override is
        converted as-is.
        """
        from areal.engine.megatron_utils.megatron import (
            all_gather_param,
            convert_to_hf,
            get_named_parameters,
        )

        overrides = theta_by_id or {}
        num_moe_experts = getattr(self._engine.tf_config, "num_moe_experts", None)
        model_name = self._engine.hf_config.model_type
        tie_word_embeddings = getattr(
            self._engine.hf_config, "tie_word_embeddings", False
        )

        for mcore_name, param in get_named_parameters(
            self._engine.model, num_moe_experts
        ):
            src = overrides.get(id(param), param)
            gathered = all_gather_param(
                mcore_name,
                src,
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

    def _get_inner_optimizers(self):
        """Inner per-chunk optimizers (unwrap Megatron ChainedOptimizer).

        Used by the AdamW-inversion detector to read resident AdamW moments.
        """
        optimizer = self._engine.optimizer
        if optimizer is None:
            return []
        if hasattr(optimizer, "chained_optimizers"):
            return optimizer.chained_optimizers
        if hasattr(optimizer, "optimizers"):
            return optimizer.optimizers
        return [optimizer]

    @torch.no_grad()
    def _sync_model_params_from_optimizer(self) -> None:
        """Make the HF payload read see the latest optimizer main params.

        Megatron's distributed optimizer updates fp32 main-param shards, copies
        them into DDP param buffers, then all-gathers into the model-visible
        params. Colocate weight exchange can run after offload/resume, so do a
        conservative copy + synchronous param sync before reading HF tensors.
        """
        copied = 0
        gathered = 0
        for opt in self._get_inner_optimizers():
            copy_fn = getattr(opt, "_copy_main_params_to_model_params", None)
            if copy_fn is None:
                copy_fn = getattr(opt, "_copy_main_params_to_param_buffer", None)
            if copy_fn is None:
                continue
            copy_fn()
            copied += 1

            gather_fn = getattr(
                opt, "_reset_metadata_and_sync_gather_all_model_params", None
            )
            if gather_fn is not None:
                gather_fn(force_sync=True)
                gathered += 1

        model = self._engine.model
        if model is None:
            return
        model_chunks = model if isinstance(model, (list, tuple)) else [model]
        synced = 0
        if gathered == 0:
            for model_chunk in model_chunks:
                start_param_sync = getattr(model_chunk, "start_param_sync", None)
                if start_param_sync is None:
                    continue
                start_param_sync(force_sync=True)
                synced += 1

        if copied or gathered or synced:
            torch.cuda.synchronize()
            logger.info(
                "Synced Megatron optimizer main params before AWEX payload read "
                "(copied=%d, optimizer_gather=%d, param_sync=%d)",
                copied,
                gathered,
                synced,
            )

    @torch.no_grad()
    def _convert_hf_with_overrides(
        self, theta_by_id: dict[int, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Convert mcore params to HF, substituting reconstructed pre-step
        weights by ``id(model_param)``.

        The inversion detector builds the previous-version HF payload through
        this (the same all_gather + convert path as the live payload), then
        bitwise-compares it against the current payload to get change masks.
        """
        return dict(self._iter_hf_params(theta_by_id))

    # ── Colocated weight transfer methods ─────────────────────────────────

    def init_colocate_weight_update(
        self,
        pair_name: str,
        kv_store_url: str,
        transfer_rank: int,
        infer_world_size: int,
        train_world_size: int,
        num_engines: int,
        master_port: int,
        admin_api_key: str = "areal-admin-key",
        timeout_s: float = 120.0,
    ) -> None:
        self._colocate_pair_name = pair_name
        self._colocate_kv_store_url = kv_store_url
        self._colocate_transfer_rank = transfer_rank
        self._colocate_infer_world_size = infer_world_size
        self._colocate_admin_api_key = admin_api_key
        self._colocate_timeout_s = timeout_s
        if self._colocate_http_client is None:
            self._colocate_http_client = httpx.Client()
        logger.info(
            "Initialized colocate weight update for pair '%s', transfer_rank=%d",
            pair_name,
            transfer_rank,
        )

    def execute_colocate_weight_update(self, version: int) -> None:
        with self._colocate_lock:
            self._execute_colocate_weight_update_locked(version)

    def seed_delta_base(self, version: int = 0) -> None:
        """Virtually seed sender-side delta state from current model weights.

        This is for startup validation where the inference engine has just
        loaded the same checkpoint as the actor. It avoids a real full-weight
        transfer before the first optimizer step, so that first trained update
        can be encoded as an AdamW-inversion delta. In inversion mode this stores
        only names + optimizer watermarks, not a CPU full-model snapshot.
        """
        if not delta_transfer_enabled():
            logger.info("seed_delta_base skipped: delta transfer disabled")
            return
        if self._delta_tracker is None:
            self._delta_tracker = make_delta_tracker()
        if self._delta_detector is None:
            self._delta_detector = build_detector(delta_detector_mode(), self)

        inversion = self._delta_detector.name == "inversion"
        verify_snapshot = inversion and self._delta_verify_snapshot_enabled()
        weights_offloaded = "weights" in self._released_tags
        if weights_offloaded:
            self.resume_memory(tags=["weights"])
        try:
            params_list = list(self.get_local_shard_parameters().items())
            self._delta_tracker.seed(
                params_list,
                version,
                store_snapshot=(not inversion or verify_snapshot),
            )
            if verify_snapshot:
                self._delta_mark_snapshot_names(params_list)
            self._delta_mark_synced(version)
            logger.info(
                "colocate delta virtual seed v%d [%s] (snapshot=%s)",
                version,
                self._delta_detector.name,
                (not inversion or verify_snapshot),
            )
        finally:
            if weights_offloaded:
                self.release_memory(tags=["weights"])

    def _execute_colocate_weight_update_locked(self, version: int) -> None:
        kv_store_url = self._colocate_kv_store_url
        pair_name = self._colocate_pair_name
        transfer_rank = self._colocate_transfer_rank
        assert self._colocate_http_client is not None, (
            "init_colocate_weight_update must be called first"
        )
        client = self._colocate_http_client
        auth_headers = {"Authorization": f"Bearer {self._colocate_admin_api_key}"}
        timeout_s = self._colocate_timeout_s

        weights_offloaded = "weights" in self._released_tags
        self._release_grad_memory()
        if weights_offloaded:
            self.resume_memory(tags=["weights"])

        sync_env = os.environ.get("DTE_SYNC_MODEL_PARAMS_BEFORE_PAYLOAD")
        if sync_env is None or sync_env.strip() == "":
            sync_env = os.environ.get("AWEX_SYNC_MODEL_PARAMS_BEFORE_PAYLOAD")
        if sync_env is None or sync_env.strip() == "":
            # AdamW inversion compares and sends the BF16 HF payload, so the
            # model-visible Megatron buffers must reflect the latest fp32 main
            # params before we read them. This is a GPU-side refresh, not a CPU
            # snapshot.
            sync_before_payload = (
                delta_transfer_enabled() and delta_detector_mode() == "inversion"
            )
        else:
            sync_before_payload = sync_env.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if sync_before_payload:
            self._sync_model_params_from_optimizer()
        params = self.get_local_shard_parameters()
        if delta_transfer_enabled():
            names, tensors = self._delta_encode(params, version)
        else:
            names = list(params.keys())
            tensors = list(params.values())

        group_tensors, metadata = _group_tensors_for_colocate_ipc(tensors)
        torch.cuda.synchronize()

        del tensors

        group_shared = [t.share_memory_() for t in group_tensors]
        serialized_weights = cuda_ipc_serialize((group_shared, metadata, names))
        torch.cuda.synchronize()

        kv_key = f"colocate_weights_rank{transfer_rank}_{version}"

        client.put(
            f"{kv_store_url}/weight_meta/{pair_name}/{kv_key}",
            json={"value": serialized_weights.hex()},
            headers=auth_headers,
            timeout=timeout_s,
        )

        logger.info(
            "Serialized %d params (%d groups) for colocate transfer v%d, rank %d",
            len(names),
            len(group_shared),
            version,
            transfer_rank,
        )

        done_key = f"colocate_done_rank{transfer_rank}_{version}"
        deadline = time.monotonic() + timeout_s
        poll_count = 0
        last_status = -1
        while time.monotonic() < deadline:
            resp = client.get(
                f"{kv_store_url}/weight_meta/{pair_name}/{done_key}",
                timeout=5.0,
            )
            last_status = resp.status_code
            if resp.status_code == 200:
                break
            poll_count += 1
            time.sleep(0.1)
        else:
            raise TimeoutError(
                f"Inference did not signal completion within {timeout_s}s "
                f"(waiting_key={done_key}, put_key={kv_key}, "
                f"polls={poll_count}, last_status={last_status})"
            )

        self._delta_mark_synced(version)

        del group_shared, group_tensors, serialized_weights
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        if weights_offloaded:
            self.release_memory(tags=["weights"])

    def _delta_mark_synced(self, version: int) -> None:
        """Let an external delta detector record post-sync watermarks."""
        if not delta_transfer_enabled() or self._delta_detector is None:
            return
        mark_synced = getattr(self._delta_detector, "mark_synced", None)
        if mark_synced is not None:
            mark_synced(version)

    def _delta_encode(
        self, params: dict[str, torch.Tensor], version: int
    ) -> tuple[list[str], list[torch.Tensor]]:
        """Encode ``params`` as a dte delta/full payload (sender side).

        Mirrors ``dte.engine.DeltaEngine.push`` but ships the payload through the
        existing cuda-IPC colocate channel instead of a dte transport. A full
        sync ships plain tensors (header-less, shapes unchanged); a delta ships
        dte's sparse payload (header + ``w@delta_idx``/``w@delta_val`` + dense
        fallback). The sglang receiver reconstructs full weights before applying,
        so delta stays transparent to the cross-rank NCCL reshard.

        The change mask comes from the configured detector (DTE_DELTA_DETECTOR):
        ``snapshot`` (default) returns None so dte diffs its CPU baseline;
        ``inversion`` reconstructs pre-step weights from the optimizer's resident
        moments (zero snapshot memory). Inversion infeasible this step -> dense.
        """
        if self._delta_tracker is None:
            self._delta_tracker = make_delta_tracker()
        if self._delta_detector is None:
            self._delta_detector = build_detector(delta_detector_mode(), self)
        inversion = self._delta_detector.name == "inversion"
        verify_snapshot = inversion and self._delta_verify_snapshot_enabled()
        names = list(params.keys())
        tensors = list(params.values())
        params_list = list(params.items())

        # Full sync on the first frame (not yet seeded) or a forced anchor.
        reason = self._delta_tracker.full_sync_reason(version)
        reason = self._delta_sync_full_reason(reason, version)
        # Inversion: compute masks only when this rank would otherwise ship a
        # delta; if infeasible this step (precision-aware / step<1 / recover)
        # compute_masks returns None and we fall back to a dense full sync.
        masks = None
        if inversion and reason is None:
            masks = self._delta_detector.compute_masks(names, tensors, version)
            if masks is None:
                reason = "inversion_infeasible"
            elif verify_snapshot and not self._delta_verify_masks_against_snapshot(
                params_list, masks, version
            ):
                reason = "inversion_snapshot_mismatch"

        reason = self._delta_sync_full_reason(reason, version)

        if reason is not None:
            # Full sync: ship plain tensors. Re-seed the snapshot baseline in
            # snapshot mode. Inversion normally keeps no baseline, but the
            # verification mode intentionally stores one for mask cross-checks.
            self._delta_tracker.seed(
                params_list,
                version,
                store_snapshot=(not inversion or verify_snapshot),
            )
            if verify_snapshot:
                self._delta_mark_snapshot_names(params_list)
            logger.info("colocate delta v%d: FULL sync (%s)", version, reason)
            return names, tensors

        encoded = self._delta_tracker.encode(params_list, version, masks=masks)
        if verify_snapshot:
            self._delta_refresh_verification_snapshot(params_list)
        logger.info(
            "colocate delta v%d [%s]: changed %d/%d (%.2f%%) "
            "sparse=%d dense_fallback=%d unchanged=%d "
            "payload=%.1fMB vs dense=%.1fMB",
            version,
            self._delta_detector.name,
            encoded.changed_elements,
            encoded.total_elements,
            100.0 * encoded.changed_elements / max(encoded.total_elements, 1),
            encoded.num_sparse,
            encoded.num_dense_fallback,
            encoded.num_unchanged,
            encoded.payload_bytes / 1e6,
            encoded.dense_bytes / 1e6,
        )
        return encoded.names, encoded.tensors

    def _delta_sync_full_reason(self, reason: str | None, version: int) -> str | None:
        """Promote a rank-local full-sync fallback to all training ranks."""
        if not dist.is_available() or not dist.is_initialized():
            return reason
        try:
            world_size = dist.get_world_size()
        except RuntimeError:
            return reason
        if world_size <= 1:
            return reason

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        needs_full = torch.tensor(
            [1 if reason is not None else 0], dtype=torch.int32, device=device
        )
        dist.all_reduce(needs_full, op=dist.ReduceOp.MAX)
        if int(needs_full.item()) == 0:
            return None
        if reason is not None:
            return reason

        logger.warning(
            "colocate delta v%d: FULL sync (global_full_sync from peer rank)",
            version,
        )
        return "global_full_sync"

    @staticmethod
    def _delta_verify_snapshot_enabled() -> bool:
        env = os.environ.get("DTE_DELTA_VERIFY_SNAPSHOT")
        if env is None or env.strip() == "":
            env = os.environ.get("AWEX_DELTA_VERIFY_SNAPSHOT", "")
        return env.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _delta_mark_snapshot_names(
        self, params_list: list[tuple[str, torch.Tensor]]
    ) -> None:
        names = getattr(self._delta_tracker, "_snapshot_names", None)
        if names is not None:
            names.update(name for name, _ in params_list)

    @torch.no_grad()
    def _delta_refresh_verification_snapshot(
        self, params_list: list[tuple[str, torch.Tensor]]
    ) -> None:
        """Refresh the debug snapshot without resetting delta version counters."""
        snapshot = getattr(self._delta_tracker, "_snapshot", None)
        names = getattr(self._delta_tracker, "_snapshot_names", None)
        if snapshot is None or names is None:
            return

        snapshot.clear()
        names.clear()
        by_storage: dict[tuple[int, int], torch.Tensor] = {}
        pin = torch.cuda.is_available()
        for name, param in params_list:
            data = param.detach().contiguous()
            key = (data.data_ptr(), data.numel())
            cpu_tensor = by_storage.get(key)
            if cpu_tensor is None:
                cpu_tensor = data.cpu().clone()
                if pin:
                    cpu_tensor = cpu_tensor.pin_memory()
                by_storage[key] = cpu_tensor
            snapshot[name] = cpu_tensor
            names.add(name)

    @torch.no_grad()
    def _delta_verify_masks_against_snapshot(
        self,
        params_list: list[tuple[str, torch.Tensor]],
        masks: dict[str, torch.Tensor],
        version: int,
    ) -> bool:
        """Compare AdamW-inversion masks against a resident snapshot baseline."""
        from dte.core import bitwise_changed_mask

        snapshot = getattr(self._delta_tracker, "_snapshot", None)
        if not snapshot:
            logger.error(
                "colocate delta v%d verify snapshot-vs-%s FAILED: "
                "snapshot baseline is missing",
                version,
                self._delta_detector.name,
            )
            return False

        total = 0
        snapshot_changed = 0
        detector_changed = 0
        false_negative = 0
        false_positive = 0
        unverifiable = 0
        fn_examples: list[str] = []
        fp_examples: list[str] = []
        bad_examples: list[str] = []

        for name, cur in params_list:
            cur = cur.detach().contiguous()
            numel = cur.numel()
            total += numel
            snap = snapshot.get(name)
            det_mask = masks.get(name)
            if (
                snap is None
                or snap.dtype != cur.dtype
                or snap.numel() != numel
                or det_mask is None
                or det_mask.numel() != numel
            ):
                unverifiable += 1
                if len(bad_examples) < 5:
                    bad_examples.append(f"{name}:missing_or_bad_mask")
                continue

            snap_mask = bitwise_changed_mask(
                cur, snap.to(cur.device, non_blocking=False)
            ).reshape(-1)
            det_mask = det_mask.to(cur.device, non_blocking=False).reshape(-1).bool()

            snap_count = int(snap_mask.sum().item())
            det_count = int(det_mask.sum().item())
            fn_count = int((snap_mask & ~det_mask).sum().item())
            fp_count = int((det_mask & ~snap_mask).sum().item())

            snapshot_changed += snap_count
            detector_changed += det_count
            false_negative += fn_count
            false_positive += fp_count
            if fn_count and len(fn_examples) < 5:
                fn_examples.append(
                    f"{name}:snapshot={snap_count},detector={det_count},"
                    f"fn={fn_count},fp={fp_count}"
                )
            elif fp_count and len(fp_examples) < 5:
                fp_examples.append(
                    f"{name}:snapshot={snap_count},detector={det_count},"
                    f"fn={fn_count},fp={fp_count}"
                )

        examples = (fn_examples + bad_examples + fp_examples)[:5]
        if false_negative or unverifiable:
            logger.error(
                "colocate delta v%d verify snapshot-vs-%s MISMATCH: "
                "snapshot_changed=%d detector_changed=%d total=%d "
                "false_negative=%d false_positive=%d unverifiable_params=%d "
                "examples=%s",
                version,
                self._delta_detector.name,
                snapshot_changed,
                detector_changed,
                total,
                false_negative,
                false_positive,
                unverifiable,
                examples,
            )
            return False

        if false_positive:
            logger.warning(
                "colocate delta v%d verify snapshot-vs-%s OK conservative: "
                "snapshot_changed=%d detector_changed=%d total=%d "
                "false_positive=%d examples=%s",
                version,
                self._delta_detector.name,
                snapshot_changed,
                detector_changed,
                total,
                false_positive,
                examples,
            )
            return True

        logger.info(
            "colocate delta v%d verify snapshot-vs-%s OK: changed=%d/%d",
            version,
            self._delta_detector.name,
            detector_changed,
            total,
        )
        return True

    def release_memory(self, tags: list[str] | None = None) -> None:
        """Release GPU memory for specified tags by offloading to CPU.

        Supported tags:
            - "optimizer": Offload optimizer state tensors (exp_avg, exp_avg_sq, etc.)
            - "weights": Offload model parameters
        """
        tags = tags or ["optimizer", "weights"]
        tags_to_release = [t for t in tags if t not in self._released_tags]
        if not tags_to_release:
            logger.info("release_memory: tags=%s already released, skipping", tags)
            return

        logger.info("release_memory: offloading tags=%s", tags_to_release)

        if "optimizer" in tags_to_release:
            self._offload_optimizer_states()
            self._released_tags.add("optimizer")

        if "weights" in tags_to_release:
            self._offload_model_weights()
            self._released_tags.add("weights")

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("release_memory: done for tags=%s", tags_to_release)

    def resume_memory(self, tags: list[str] | None = None) -> None:
        """Resume GPU memory for specified tags by reloading from CPU.

        Supported tags:
            - "optimizer": Reload optimizer state tensors to GPU
            - "weights": Reload model parameters to GPU
        """
        tags = tags or ["optimizer", "weights"]
        tags_to_resume = [t for t in tags if t in self._released_tags]
        if not tags_to_resume:
            logger.info("resume_memory: tags=%s not released, skipping", tags)
            return

        logger.info("resume_memory: reloading tags=%s", tags_to_resume)

        if "weights" in tags_to_resume:
            self._reload_model_weights()
            self._released_tags.discard("weights")

        if "optimizer" in tags_to_resume:
            self._reload_optimizer_states()
            self._released_tags.discard("optimizer")

        torch.cuda.synchronize()
        logger.info("resume_memory: done for tags=%s", tags_to_resume)

    def _offload_optimizer_states(self) -> None:
        """Move optimizer state tensors to CPU, keeping references for reload."""
        optimizer = self._engine.optimizer
        if optimizer is None:
            logger.warning("No optimizer found, skipping optimizer offload")
            return

        # Megatron's ChainedOptimizer wraps per-model-chunk optimizers;
        # each in turn wraps a base torch optimizer holding the state dict.
        if hasattr(optimizer, "optimizers"):
            inner_optimizers = optimizer.optimizers
        else:
            inner_optimizers = [optimizer]
            logger.warning(
                "Optimizer does not have 'optimizers' attribute. "
                "Treating it as a single optimizer; offload may be incomplete "
                "for non-standard Megatron optimizer structures."
            )
        for opt in inner_optimizers:
            base_opt = getattr(opt, "optimizer", opt)
            for param, state in base_opt.state.items():
                cpu_state: dict[str, torch.Tensor] = {}
                for key, val in state.items():
                    if isinstance(val, torch.Tensor) and val.is_cuda:
                        cpu_state[key] = val.detach().to("cpu", non_blocking=True)
                        state[key] = torch.empty(0, device="cpu")
                if cpu_state:
                    self._offloaded_optimizer_states[param] = cpu_state

        logger.info(
            "Offloaded optimizer states for %d params",
            len(self._offloaded_optimizer_states),
        )

    def _reload_optimizer_states(self) -> None:
        """Restore optimizer state tensors from CPU back to GPU."""
        if not self._offloaded_optimizer_states:
            return

        optimizer = self._engine.optimizer
        if optimizer is None:
            return

        inner_optimizers = getattr(optimizer, "optimizers", [optimizer])
        for opt in inner_optimizers:
            base_opt = getattr(opt, "optimizer", opt)
            for param, state in base_opt.state.items():
                if param in self._offloaded_optimizer_states:
                    cpu_state = self._offloaded_optimizer_states[param]
                    for key, val in cpu_state.items():
                        state[key] = val.to(param.device, non_blocking=True)

        self._offloaded_optimizer_states.clear()
        logger.info("Reloaded optimizer states to GPU")

    def _offload_model_weights(self) -> None:
        """Move model parameters to CPU, preserving Megatron DDP buffer views."""
        if self._engine.model is None:
            return

        from megatron.core.distributed import DistributedDataParallel as DDP

        model = self._engine.model
        model_chunks = model if isinstance(model, (list, tuple)) else [model]
        count = 0
        for chunk_idx, chunk in enumerate(model_chunks):
            if isinstance(chunk, DDP):
                for buffers in (chunk.buffers, chunk.expert_parallel_buffers):
                    for buf in buffers:
                        offload_to_cpu = getattr(buf, "offload_to_cpu", None)
                        if offload_to_cpu is not None:
                            offload_to_cpu()
                            count += 1
                            continue

                        param_storage = buf.param_data.storage()
                        if param_storage.size() > 0:
                            if not hasattr(buf.param_data, "cpu_data"):
                                try:
                                    buf.param_data.cpu_data = torch.empty(
                                        buf.param_data.data.shape,
                                        dtype=buf.param_data.data.dtype,
                                        pin_memory=torch.cuda.is_available(),
                                        device="cpu",
                                    )
                                except RuntimeError:
                                    buf.param_data.cpu_data = torch.empty(
                                        buf.param_data.data.shape,
                                        dtype=buf.param_data.data.dtype,
                                        device="cpu",
                                    )
                            buf.param_data.cpu_data.copy_(
                                buf.param_data.data, non_blocking=True
                            )
                            buf.param_data_size = param_storage.size()
                            param_storage.resize_(0)
                            count += 1
                        if buf.grad_data.storage().size() > 0:
                            buf.grad_data_size = buf.grad_data.storage().size()
                            buf.grad_data.storage().resize_(0)
                continue

            prefix = f"chunk{chunk_idx}."
            for name, param in chunk.named_parameters():
                if param.is_cuda:
                    self._offloaded_weights[prefix + name] = param.data.detach().to(
                        "cpu", non_blocking=True
                    )
                    param.data = torch.empty(0, device="cpu")
                    count += 1

        logger.info(
            "Offloaded %d model weight buffers/tensors to CPU",
            count,
        )

    def _reload_model_weights(self) -> None:
        """Restore model parameters without breaking Megatron DDP buffer views."""
        if not self._offloaded_weights and self._engine.model is None:
            return
        if self._engine.model is None:
            return

        from megatron.core.distributed import DistributedDataParallel as DDP

        device = self._engine.device
        model = self._engine.model
        model_chunks = model if isinstance(model, (list, tuple)) else [model]
        count = 0
        for chunk_idx, chunk in enumerate(model_chunks):
            if isinstance(chunk, DDP):
                for buffers in (chunk.buffers, chunk.expert_parallel_buffers):
                    for buf in buffers:
                        reload_from_cpu = getattr(buf, "reload_from_cpu", None)
                        if reload_from_cpu is not None:
                            reload_from_cpu(move_grads=False)
                            count += 1
                            continue

                        if (
                            hasattr(buf, "param_data_size")
                            and buf.param_data.storage().size() == 0
                        ):
                            buf.param_data.storage().resize_(buf.param_data_size)
                        cpu_data = getattr(buf.param_data, "cpu_data", None)
                        if cpu_data is not None:
                            buf.param_data.copy_(cpu_data, non_blocking=True)
                            count += 1
                continue

            prefix = f"chunk{chunk_idx}."
            for name, param in chunk.named_parameters():
                key = prefix + name
                if key in self._offloaded_weights:
                    param.data = self._offloaded_weights[key].to(
                        device, non_blocking=True
                    )
                    count += 1

        self._offloaded_weights.clear()
        logger.info("Reloaded %d model weight buffers/tensors to GPU", count)

    def _release_grad_memory(self) -> None:
        """Release Megatron DDP grad buffers until the next training batch."""
        if self._engine.model is None:
            return

        from megatron.core.distributed import DistributedDataParallel as DDP

        model = self._engine.model
        model_chunks = model if isinstance(model, (list, tuple)) else [model]
        count = 0
        for chunk in model_chunks:
            if not isinstance(chunk, DDP):
                continue
            for buffers in (chunk.buffers, chunk.expert_parallel_buffers):
                for buf in buffers:
                    grad_data = getattr(buf, "grad_data", None)
                    if grad_data is None or grad_data.storage().size() == 0:
                        continue
                    buf.grad_data_size = grad_data.storage().size()
                    grad_data.storage().resize_(0)
                    count += 1
        if count:
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Released %d grad buffers", count)

    def ensure_grad_buffers(self) -> None:
        """Reallocate Megatron DDP grad buffers released with train weights."""
        if self._engine.model is None:
            return

        from megatron.core.distributed import DistributedDataParallel as DDP

        model = self._engine.model
        model_chunks = model if isinstance(model, (list, tuple)) else [model]
        count = 0
        for chunk in model_chunks:
            if not isinstance(chunk, DDP):
                continue
            for buffers in (chunk.buffers, chunk.expert_parallel_buffers):
                for buf in buffers:
                    if (
                        hasattr(buf, "grad_data_size")
                        and buf.grad_data.storage().size() == 0
                    ):
                        buf.grad_data.storage().resize_(buf.grad_data_size)
                        buf.grad_data.zero_()
                        count += 1
        if count:
            torch.cuda.synchronize()
            logger.info("Allocated %d grad buffers for training", count)
