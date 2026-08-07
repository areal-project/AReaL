# SPDX-License-Identifier: Apache-2.0

"""AWEX colocate weight reader (native awex worker-reader adapter).

Runs inside the SGLang scheduler process. This is a thin shell around awex's
native ``NCCLWorkerWeightsReader`` that:

1. Eager-registers the inference-side metadata the train writer waits for
   (``infer_conf`` + ``num_infer_engines``), computed via awex's own
   ``InferParamMetaResolver._get_model_param_info`` + ``_build_params_meta``
   (no hand-rolled name normalization or shard merging).
2. Lazily constructs the awex ``NCCLWorkerWeightsReader`` on the first weight
   update (it needs ``training_params_meta``, which only appears after the
   first training step) and delegates the whole IPC-collect + StreamBatch
   transport + writer handshake to it.

Why awex-native instead of the previous hand-written receiver: the community
SGLang scheduler has no ``execute_task_in_model_worker`` driver layer, so we
build the awex *worker* reader directly in-process. The native worker reader
uses ``NcclColocateStreamBatchTransport`` (recursive partition), which is the
transport Asystem actually ships -- replacing the ring-shift transport that
deadlocked on the train-PP=4 / infer-PP=1 mismatch.

The plugin shell still owns the steps awex's *driver* would normally do
(``_pre_update_weights`` wait-for-offload + resume weights, ``_resume_kvcache``
signal-finished); see ``awex_sglang_plugin.process_awex_queue``.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

import torch


def _patch_tms_hook_mode() -> None:
    """Make ``torch_memory_saver.hook_mode`` setter a no-op once initialized.

    ``megatron.core.inference.contexts.dynamic_context`` (pulled in transitively
    by ``awex.converter.mcore_converter`` -> ``megatron.core``) runs a
    module-level ``torch_memory_saver.hook_mode = "torch"``. In the SGLang
    scheduler process the memory_saver singleton is already initialized (sglang
    ran ``_ensure_initialized``, which ``del``s ``_impl_ctor_kwargs``), so that
    late assignment raises ``AttributeError``. awex's model registry swallows the
    import error, the BailingMoe converter never registers, and weight transfer
    later dies with ``Unsupported attention parameter name: attention.g_proj``.
    The singleton's own assert already declares post-init configuration
    unsupported, so dropping the late set is the intended behavior.
    """
    try:
        import torch_memory_saver as _tms
    except Exception:
        return
    inst = getattr(_tms, "torch_memory_saver", None)
    if inst is None:
        return
    cls = type(inst)
    prop = cls.hook_mode
    if getattr(prop.fset, "_awex_safe", False):
        return

    def _safe_setter(self, value):
        if not hasattr(self, "_impl_ctor_kwargs"):
            return  # singleton already initialized; late set is a design no-op
        prop.fset(self, value)

    _safe_setter._awex_safe = True
    cls.hook_mode = property(prop.fget, _safe_setter)


# Must run before any awex import: awex.models.registry auto-imports model
# modules at module load, and the BailingMoe module's transitive megatron import
# trips the hook_mode race above.
_patch_tms_hook_mode()

from awex.meta.infer_meta_resolver import InferParamMetaResolver  # noqa: E402
from awex.meta.meta_resolver import ParamMetaResolver  # noqa: E402
from awex.reader.nccl_reader import (  # noqa: E402
    NCCLWorkerWeightsReader,
    _wait_colocate_write_finished,
)
from awex.sharding import get_sharding_strategy_builder  # noqa: E402
from awex.util.common import simple_hf_config  # noqa: E402

from areal.utils.logging import getLogger  # noqa: E402

logger = getLogger("AwexColocateReader")


class _BailingV3PhysicalKeyNCCLWorkerWeightsReader(NCCLWorkerWeightsReader):
    """Bailing v3 AWEX reader using physical GPU ids for MetaServer keys.

    This override is intentionally specific to the Bailing v3 colocated
    Megatron-to-SGLang path. In addition to fixing logical/physical GPU id
    mapping, it carries Bailing v3 parameter sentinels and transfer workarounds
    for the model's hybrid attention and MoE sharding. It must not be treated as
    a generic AWEX reader without validating those assumptions for a new model.

    In AReaL colocate runs each SGLang process is isolated with a single
    CUDA_VISIBLE_DEVICES entry, so torch/awex see the logical device id as 0.
    The training writer publishes IPC handles under the node-local physical GPU
    id. Keep CUDA operations on the logical device, but use the physical id for
    MetaServer key names and the train/infer device-rank mapping.
    """

    # Sentinel substrings for post-update data validation (incident 15): after
    # every weight update, log shape/norm/first-4 values of a few infer-side
    # tensors. Offline we locate the logged 4-value window inside the
    # train-side HF checkpoint tensor to see WHICH slice actually landed on
    # this rank — a mismatch pattern distinguishes shard permutation from
    # corrupted payloads.
    _AREAL_SENTINELS = (
        "word_embeddings",
        "lm_head",
        "layers.0.attention.q_proj",
        "layers.0.attention.k_proj",
        "layers.0.attention.o_proj",
        "layers.0.attention.dt_bias",
        "layers.0.attention.A_log",
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.down_proj",
        "layers.2.mlp.gate.",
        "layers.2.mlp.experts.0.gate_proj",
        "layers.2.mlp.experts.0.down_proj",
        "layers.2.mlp.experts.100.gate_proj",
        "layers.2.input_layernorm",
    )

    def __init__(self, *args, physical_gpu_id: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._areal_physical_gpu_id = physical_gpu_id

    def update_weights(self, step_id, **kwargs):
        super().update_weights(step_id, **kwargs)
        # Opt-in data-validation probe (set AREAL_AWEX_SENTINEL=1): it costs
        # GPU->CPU syncs per sentinel tensor on every weight update, so it is
        # OFF by default and should be enabled for bring-up/validation runs
        # only (~20 log lines per rank per update).
        if os.environ.get("AREAL_AWEX_SENTINEL", "0") not in ("0", "", "false"):
            self._areal_log_sentinels(step_id)

    def _areal_log_sentinels(self, step_id) -> None:
        logged = 0
        for name, param in getattr(self, "parameters", {}).items():
            if not any(s in name for s in self._AREAL_SENTINELS):
                continue
            try:
                t = param.detach()
                flat = t.reshape(-1)[:4].float().tolist()
                # fp32 accumulation without materializing a full fp32 copy.
                norm = torch.linalg.vector_norm(t, dtype=torch.float32).item()
                logger.info(
                    "[AWEX-SENTINEL] step=%s transfer_rank=%s phys_gpu=%s "
                    "name=%s shape=%s norm=%.6f first4=%s",
                    step_id,
                    self.transfer_rank,
                    self._areal_physical_gpu_id,
                    name,
                    tuple(t.shape),
                    norm,
                    [round(v, 8) for v in flat],
                )
            except Exception as exc:
                logger.warning(
                    "[AWEX-SENTINEL] failed for %s: %s",
                    name,
                    exc,
                )
            logged += 1
            if logged >= 20:
                break

    def _set_device(self):
        """Pin the reader to the correct logical CUDA device.

        The upstream implementation resolves the device via
        ``scheduler.gpu_id -> LOCAL_RANK -> 0``. With TP>1 SGLang servers
        (e.g. flash ``t8``) scheduler processes are NOT isolated via
        CUDA_VISIBLE_DEVICES and none of those sources are set, so every
        rank ends up on device 0 -> NCCL "Duplicate GPU detected" at the
        weights_exchange barrier (942314). Map from the physical GPU id
        instead: with per-process CVD isolation (tiny ``t1``) device_count
        is 1 and the logical id is 0; otherwise the logical id equals the
        node-local physical id.
        """
        import torch

        device_count = torch.cuda.device_count() or 1
        gpu_id = self._areal_physical_gpu_id % device_count
        prev_device = torch.cuda.current_device()
        logger.info(
            "[NCCLWeightsReader] (AReaL override) set device to %d for rank %s "
            "(physical_gpu_id=%d, device_count=%d, previous device=%d)",
            gpu_id,
            self.transfer_rank,
            self._areal_physical_gpu_id,
            device_count,
            prev_device,
        )
        torch.cuda.set_device(gpu_id)
        self.barrier_device = torch.cuda.current_device()
        self.backend = "nccl"
        self.ready_tensor = torch.tensor(1).cuda()

    def _init_reader_in_colocate_mode(self):
        from awex.transfer.transfer_plan import TransferPlanBuilder
        from awex.util import device as device_util
        from awex.util.common import get_ip_address

        ip_address = get_ip_address()
        physical_gpu_id = self._areal_physical_gpu_id
        self.meta_server_client.add_object_to_set(
            "inference_device_rank_entries",
            (ip_address, physical_gpu_id, self.transfer_rank),
        )
        self.meta_server_client.wait_set_until_size(
            "inference_device_rank_entries",
            self.infer_world_size,
            timeout=self.timeout,
        )
        inference_device_entries = self.meta_server_client.get_set(
            "inference_device_rank_entries",
        )
        self.inference_device_mapping = {
            (ip_address, device_id): transfer_rank
            for ip_address, device_id, transfer_rank in inference_device_entries
        }

        self.meta_server_client.wait_set_until_size(
            "training_device_rank_entries",
            self.training_world_size,
            timeout=self.timeout,
        )
        device_rank_entries = self.meta_server_client.get_set(
            "training_device_rank_entries",
        )
        self.training_device_mapping = {
            (ip_address, device_id): transfer_rank
            for ip_address, device_id, transfer_rank in device_rank_entries
        }
        self.train_to_infer_device_mapping = {}
        self.infer_to_train_device_mapping = {}
        for ip_address, device_id, transfer_rank in device_rank_entries:
            infer_rank = self.inference_device_mapping[(ip_address, device_id)]
            self.train_to_infer_device_mapping[transfer_rank] = infer_rank
            self.infer_to_train_device_mapping[infer_rank] = transfer_rank

        plan_builder = TransferPlanBuilder(
            self.infer_world_size,
            self.training_world_size,
            self.num_engines,
            self.enable_debug_mode,
        )
        self.send_transfer_plan = plan_builder.build_local_transfer_plan(
            self.parameters_meta,
            self.training_params_meta,
            self.infer_to_train_device_mapping[self.transfer_rank],
        )
        from awex.transfer.nccl_stream_batch import NcclColocateStreamBatchTransport

        self.colocate_transport = NcclColocateStreamBatchTransport(
            self.transfer_rank,
            self.infer_world_size,
        )
        logger.info(
            "Initialized NCCL weights reader for rank %d in colocate mode "
            "(logical_device=%d, physical_gpu_id=%d)",
            self.transfer_rank,
            device_util.current_device(),
            physical_gpu_id,
        )

    def collect_training_weights(self, step_id, **kwargs):
        if not self.enable_colocate_mode:
            return

        from awex.util import device as device_util
        from awex.util.common import get_ip_address
        from awex.util.gpu import get_gpu_status
        from awex.util.system_util import count_open_fds
        from awex.util.tensor_util import (
            cuda_ipc_deserialize,
            ipc_deserialize,
            reconstruct_tensors_from_groups,
        )

        ip_address = get_ip_address()
        physical_gpu_id = self._areal_physical_gpu_id
        logical_device_id = device_util.current_device()
        key = f"training_serialized_weights_{ip_address}_{physical_gpu_id}_{step_id}"
        logger.info(
            "Start to get serialized ipc weights %s for rank %s (logical_device=%d)",
            key,
            self.rank_coordinate,
            logical_device_id,
        )
        self.send_rank, self.send_rank_info, serialized_weights = (
            self.meta_server_client.get_object(key, timeout=self.timeout)
        )
        logger.info(
            "Finished getting serialized ipc weights %s for rank %s",
            key,
            self.rank_coordinate,
        )
        logger.info(
            "GPU status before deserialization:\n%s for rank %s",
            get_gpu_status(),
            self.rank_coordinate,
        )
        logger.info("Open fds before deserialization: %d", count_open_fds())
        if self.ipc_backend in ("cpu", "npu"):
            group_shared, metadata, names = ipc_deserialize(serialized_weights)
            group_shared = [t.to(logical_device_id) for t in group_shared]
        else:
            group_shared, metadata, names = cuda_ipc_deserialize(serialized_weights)
        device_util.synchronize(device_id=logical_device_id)
        tensors = reconstruct_tensors_from_groups(group_shared, metadata)
        device_util.synchronize(device_id=logical_device_id)
        self.deserialized_weights = dict(zip(names, tensors))
        logger.info(
            "Deserialized %d parameters and %d groups",
            len(self.deserialized_weights),
            len(group_shared),
        )
        logger.info(
            "GPU status after deserialization for rank %s:\n%s",
            self.rank_coordinate,
            get_gpu_status(),
        )
        logger.info("Open fds after deserialization: %d", count_open_fds())

    def _update_weights_in_colocate_mode(self, step_id, **kwargs):
        import gc
        import time

        import torch
        import torch.distributed as dist
        from awex.util import device as device_util
        from awex.util.common import compute_statistics, get_ip_address
        from awex.util.gpu import print_current_gpu_status

        assert self.enable_colocate_mode, "Colocate mode is not enabled"
        self.collect_training_weights(step_id, **kwargs)
        logger.info(
            "Start to update weights using NCCL for step %s from %d ranks(%s) "
            "for rank %s.",
            step_id,
            len(self.transfer_plan.operations),
            self.send_ranks_sample,
            self.rank_coordinate,
        )
        start_time = time.time()
        self.colocate_transport.update_weights_in_colocate_mode(
            self.train_to_infer_device_mapping,
            self.infer_to_train_device_mapping,
            self.transfer_rank,
            self.rank_coordinate,
            self.infer_world_size,
            self.send_transfer_plan,
            self.transfer_plan,
            self.weights_update_group,
            self.deserialized_weights,
            self.parameters,
            step_id=step_id,
        )
        print_current_gpu_status(
            f"after weights update using NCCL for rank {self.rank_coordinate}",
        )
        self.deserialized_weights = None
        gc.collect()
        torch.cuda.synchronize()
        duration = time.time() - start_time
        compute_statistics(
            self._history_update_weights_time,
            step_id,
            duration,
            "Receive weights using NCCL",
        )
        ip_address = get_ip_address()
        physical_gpu_id = self._areal_physical_gpu_id
        key_suffix = f"_{ip_address}_{physical_gpu_id}_{step_id}"
        update_finished_key = f"weights_update_finished{key_suffix}"
        self.meta_server_client.put_object(update_finished_key, True)
        dist.barrier(
            group=self.weights_update_group,
            device_ids=[device_util.current_device()],
        )
        logger.info(
            "Barrier passed for reader step %s with rank %d",
            step_id,
            self.transfer_rank,
        )
        gc.collect()
        if device_util.get_device_type() == "cuda":
            torch.cuda.empty_cache()
        write_finished_key = f"write_finished{key_suffix}"
        _wait_colocate_write_finished(
            self.meta_server_client,
            write_finished_key,
            self.weights_update_group,
            self.transfer_rank,
        )
        logger.info(
            "Finished updating weights in colocate mode for rank %d",
            self.transfer_rank,
        )


def _ensure_awex_models_registered() -> None:
    """Rebuild awex's model registry in case it cached a failed auto-import.

    ``import_model_configs`` is ``lru_cache``-d and ``ModelRegistry`` is built
    once at module load. If anything imported the registry before our hook_mode
    patch took effect, the BailingMoe converter would be silently missing. Clear
    the cache and rebuild now that the patch is in place.
    """
    try:
        from awex.models import registry as _reg

        _reg.import_model_configs.cache_clear()
        _reg.ModelRegistry.models = _reg.import_model_configs()
        missing = [
            m
            for m in ("BailingMoeV2_5ForCausalLM", "BailingMoeV2ForCausalLM")
            if m not in _reg.ModelRegistry.models
        ]
        if missing:
            logger.warning(f"awex model registry still missing converters: {missing}")
    except Exception as e:  # pragma: no cover - diagnostics only
        logger.warning(f"Failed to rebuild awex model registry: {e}")


_ensure_awex_models_registered()


class _SingleInstanceMetaResolver(ParamMetaResolver):
    """Aggregate per-rank raw meta of ONE inference instance into ParameterMeta.

    awex's ``InferParamMetaResolver`` normally drives this via
    ``execute_task_in_model_worker`` (a driver fan-out we do not have). We
    instead exchange the per-rank raw meta dicts through the MetaServer
    (see ``_build_instance_params_meta``) and reuse awex's ``_build_params_meta``
    for the aggregation, plus awex's own sharding strategy builder for
    ``_get_sharding_info``. This yields the exact same ``parameters_meta`` the
    native reader expects, with awex converter parameter names (no hand-rolled
    normalization).
    """

    def __init__(self, hf_config, engine_name, infer_engine_config, raw_meta_list):
        super().__init__(hf_config)
        self._raw_meta_list = raw_meta_list
        rank0 = self._select_rank0(raw_meta_list)
        self._model_arch_name = rank0["model_arch_name"]
        self._sharding_strategy = get_sharding_strategy_builder(engine_name)(
            self._model_arch_name,
            infer_engine_config,
            rank0["rank_info"],
        )

    @staticmethod
    def _select_rank0(raw_meta_list):
        for info in raw_meta_list:
            if info["rank_info"].global_rank == 0:
                return info
        return raw_meta_list[0]

    def get_model_arch_name(self) -> str:
        return self._model_arch_name

    def get_parameters_meta(self):
        return self._build_params_meta()

    def _get_params_raw_meta(self):
        return self._raw_meta_list

    def _get_sharding_info(self, name, rank_info, param_meta):
        return self._sharding_strategy.get_sharding_strategy(
            name, rank_info=rank_info, param_meta=param_meta
        )


class AwexColocateReader:
    """Thin adapter binding awex's native worker reader into a SGLang scheduler."""

    def __init__(self, scheduler: Any):
        self._scheduler = scheduler
        self._meta_server_client = None
        self._reader: NCCLWorkerWeightsReader | None = None
        self._released_tags: set[str] = set()

        self._transfer_rank: int | None = None
        self._local_gpu_id: int | None = None
        self._infer_world_size: int | None = None
        self._train_world_size: int | None = None
        self._meta_server_addr: str | None = None

        # External-instance decomposition (computed in initialize()).
        self._infer_instance_world_size: int | None = None
        self._num_infer_engines: int | None = None
        self._engine_rank: int | None = None
        self._instance_local_rank: int | None = None

        # Inference-side parameters_meta for ONE engine instance, computed via
        # awex resolver + MetaServer raw-meta exchange. Reused as the native
        # reader's ``parameters_meta`` constructor arg.
        self._infer_params_meta = None
        self._infer_conf: dict | None = None
        self._initialized = False

    # ── model / context helpers ───────────────────────────────────────

    def _get_model(self) -> torch.nn.Module:
        return self._scheduler.tp_worker.model_runner.model

    def _build_model_context(self) -> dict[str, Any]:
        """awex model_context describing ONE inference engine instance.

        ``world_size`` is the single-server tp*pp; ``global_rank`` is the
        instance-local rank (= tp_rank for pp=1). The cross-server NCCL identity
        (engine_rank / global transfer_rank) is tracked separately by the awex
        reader. ``infer_engine_config`` (== server_args) is required by
        ``WorkerWeightsReader.__init__`` and the backport's model_context omits
        it, so we add it here.
        """
        scheduler = self._scheduler
        server_args = scheduler.server_args
        tp_size = int(getattr(server_args, "tp_size", 1))
        pp_size = int(getattr(server_args, "pp_size", 1))
        dp_size = int(getattr(server_args, "dp_size", 1))

        # incident 15: the v3 fork keeps tp_rank on scheduler.tp_worker, NOT on the
        # Scheduler itself. `getattr(scheduler, "tp_rank", 0)` silently
        # returned 0 on EVERY rank, so every reader's rank_info claimed
        # tp_rank=0, the sharding strategy computed offset-0 slices for all 64
        # ranks, and the whole engine ended up with shard 0 of every tensor
        # (942397 sentinel: identical norm/first4 across all ranks; MetaServer
        # raw meta: gr=0..7 but tp=0 everywhere). Resolve through tp_worker
        # and fall back to the instance-local rank (== tp rank for pp=1);
        # never silently default to 0.
        def _rank_attr(name: str) -> int | None:
            value = getattr(scheduler, name, None)
            if value is None:
                value = getattr(getattr(scheduler, "tp_worker", None), name, None)
            return None if value is None else int(value)

        tp_rank = _rank_attr("tp_rank")
        if tp_rank is None and self._instance_local_rank is not None:
            tp_rank = int(self._instance_local_rank) % max(tp_size, 1)
        if tp_rank is None:
            raise RuntimeError(
                "Cannot resolve tp_rank from scheduler/tp_worker and "
                "instance_local_rank is unset; refusing to default to 0 "
                "(would silently corrupt the AWEX transfer plan, incident 15)"
            )

        if self._infer_instance_world_size is not None:
            world_size = self._infer_instance_world_size
            global_rank = self._instance_local_rank
        else:
            world_size = tp_size * pp_size
            global_rank = tp_rank

        pp_rank = _rank_attr("pp_rank")
        attn_tp_rank = _rank_attr("attn_tp_rank")
        attn_tp_size = _rank_attr("attn_tp_size")
        attn_dp_rank = _rank_attr("attn_dp_rank")

        return {
            "scheduler": scheduler,
            "infer_engine_config": server_args,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "pp_rank": 0 if pp_rank is None else pp_rank,
            "pp_size": pp_size,
            "dp_size": dp_size,
            "world_size": world_size,
            "global_rank": global_rank,
            "local_rank": tp_rank,
            "attn_tp_rank": tp_rank if attn_tp_rank is None else attn_tp_rank,
            "attn_tp_size": tp_size if attn_tp_size is None else attn_tp_size,
            "attn_dp_rank": 0 if attn_dp_rank is None else attn_dp_rank,
        }

    def get_parallelism(self) -> dict:
        ctx = self._build_model_context()
        server_args = self._scheduler.server_args
        return {
            "world_size": ctx["world_size"],
            "tp_size": int(getattr(server_args, "tp_size", ctx["tp_size"])),
            "pp_size": int(getattr(server_args, "pp_size", ctx["pp_size"])),
            "dp_size": int(getattr(server_args, "dp_size", ctx["dp_size"])),
            "ep_size": int(getattr(server_args, "ep_size", 1)),
            "num_engines": self._num_infer_engines or 1,
        }

    # ── metadata (awex-native, no hand-rolled normalization) ──────────

    def _compute_local_raw_meta(self) -> dict:
        """Per-rank raw meta via awex's own staticmethod (HF-converted names)."""
        server_args = self._scheduler.server_args
        model_context = self._build_model_context()
        return InferParamMetaResolver._get_model_param_info(
            "sglang",
            server_args,
            convert_params=True,
            engine_rank=self._engine_rank or 0,
            model=self._get_model(),
            model_context=model_context,
        )

    def _build_instance_params_meta(self):
        """Gather single-instance raw meta via the MetaServer, then aggregate.

        Returns the awex ``parameters_meta`` (list[ParameterMeta]) for ONE
        inference engine instance (the ``instance_world`` instance-local ranks).

        We exchange per-rank raw meta through the MetaServer instead of an
        ``all_gather`` over ``tp_cpu_group``: that group is sglang's TP
        request-broadcast group, driven by the scheduler MainThread's
        ``recv_requests`` -> ``broadcast_pyobj``. This method runs on the
        plugin's background thread, so a collective on the shared group races
        the MainThread broadcast and deadlocks (two ops in flight on one
        non-thread-safe group). The MetaServer exchange needs no process-group
        collective, is isolated per engine instance by ``engine_rank``, and also
        sidesteps the ``dist.new_group`` collective-ordering trap (train + infer
        share the default world in colocate mode).
        """
        local_raw = self._compute_local_raw_meta()

        instance_world = self._infer_instance_world_size or 1
        if instance_world > 1:
            client = self._meta_server_client
            prefix = f"infer_instance_raw_meta_{self._engine_rank}"
            client.put_object(f"{prefix}_{self._instance_local_rank}", local_raw)
            raw_meta_list = [
                client.get_object(f"{prefix}_{r}", timeout=300.0)
                for r in range(instance_world)
            ]
        else:
            raw_meta_list = [local_raw]

        # MetaServer serializes RankInfo to a dict on the wire (as did the
        # legacy all_gather); rebuild the object before awex's resolver reads it.
        from awex.sharding.rank_info import RankInfo

        for info in raw_meta_list:
            ri = info.get("rank_info")
            if isinstance(ri, dict):
                info["rank_info"] = RankInfo(**ri)

        resolver = _SingleInstanceMetaResolver(
            self._get_model().config,
            "sglang",
            self._scheduler.server_args,
            raw_meta_list,
        )
        return resolver.get_parameters_meta()

    def get_weight_metadata(self):
        """Inference-side parameters_meta for ONE engine instance."""
        if self._infer_params_meta is None:
            self._infer_params_meta = self._build_instance_params_meta()
        return self._infer_params_meta

    # ── eager init: register infer_conf + num_infer_engines ───────────

    def initialize(
        self,
        meta_server_addr: str,
        transfer_rank: int,
        infer_world_size: int,
        train_world_size: int,
        local_gpu_id: int,
        timeout_s: float = 300.0,
    ) -> None:
        """Eager init: publish the metadata the train writer waits for.

        Must NOT block on the training side (runs before the first training step
        finishes). The native ``NCCLWorkerWeightsReader`` is built lazily in
        ``update_weights`` once ``training_params_meta`` is available. Device
        entry registration (``inference_device_rank_entries``) is left to the
        native reader's ``_init_reader_in_colocate_mode``.
        """
        from awex.meta.meta_server import MetaServerClient

        if infer_world_size != train_world_size:
            raise ValueError(
                f"Colocate mode requires equal total rank counts "
                f"(same physical GPUs), got infer={infer_world_size} "
                f"vs train={train_world_size}"
            )

        self._transfer_rank = transfer_rank
        self._local_gpu_id = local_gpu_id
        self._infer_world_size = infer_world_size
        self._train_world_size = train_world_size
        self._meta_server_addr = meta_server_addr

        server_args = self._scheduler.server_args
        tp_size = int(getattr(server_args, "tp_size", 1))
        pp_size = int(getattr(server_args, "pp_size", 1))
        instance_world = max(1, tp_size * pp_size)
        if infer_world_size % instance_world != 0:
            raise ValueError(
                f"infer_world_size ({infer_world_size}) must be divisible by the "
                f"per-instance world tp*pp ({instance_world})"
            )
        self._infer_instance_world_size = instance_world
        self._num_infer_engines = infer_world_size // instance_world
        self._engine_rank = transfer_rank // instance_world
        self._instance_local_rank = transfer_rank % instance_world
        logger.info(
            "AWEX instance decomposition: transfer_rank=%d -> engine_rank=%d, "
            "instance_local_rank=%d (instance_world=%d, num_engines=%d)",
            transfer_rank,
            self._engine_rank,
            self._instance_local_rank,
            instance_world,
            self._num_infer_engines,
        )

        host, port = meta_server_addr.rsplit(":", 1)
        self._meta_server_client = MetaServerClient(host, int(port))

        # Compute single-instance parameters_meta (also reused as the native
        # reader's constructor arg later).
        self.get_weight_metadata()

        par = self.get_parallelism()
        infer_conf = {
            "engine_name": "sglang",
            "infer_atten_tp_size": par["tp_size"],
            "infer_world_size": infer_world_size,
            "hf_config": simple_hf_config(self._get_model().config),
            # P69 root cause: upstream awex's reader publishes router_dtype so
            # the train-side converter casts mlp.gate.weight to the dtype the
            # inference engine actually holds (fp32 for BailingMoe). This
            # hand-rolled infer_conf omitted it, so the converter fell back to
            # its bf16 default and gate shards went out 2N bytes against a 4N
            # irecv — the deterministic chunk-7 wedge of trial-0610-1509. The
            # wire-level dtype reconciliation (0877ea4) papers over any such
            # mismatch generically, but keep the semantic path whole so new
            # models behave identically to native awex.
            "router_dtype": getattr(self._get_model().config, "router_dtype", "bf16"),
        }
        self._infer_conf = infer_conf

        # Only one rank publishes the engine-instance-wide info the writer waits
        # for. transfer_rank 0 is engine_rank 0, instance_local_rank 0.
        if transfer_rank == 0:
            self._meta_server_client.put_object("infer_conf", infer_conf)
            self._meta_server_client.put_object(
                "num_infer_engines", self._num_infer_engines
            )
            logger.info(
                "Registered infer_conf + num_infer_engines=%d with MetaServer",
                self._num_infer_engines,
            )

        self._initialized = True
        logger.info(
            "Eager init done: transfer_rank=%d, local_gpu_id=%d, infer_world_size=%d "
            "(native worker reader construction deferred to first update_weights)",
            transfer_rank,
            local_gpu_id,
            infer_world_size,
        )

    # ── lazy native-reader construction + weight update ───────────────

    def _ensure_reader(self) -> NCCLWorkerWeightsReader:
        if self._reader is not None:
            return self._reader

        client = self._meta_server_client
        training_params_meta = client.get_object(
            "training_params_meta", timeout=10000.0
        )
        logger.info("Got training_params_meta from MetaServer")

        model_context = self._build_model_context()
        reader = _BailingV3PhysicalKeyNCCLWorkerWeightsReader(
            engine_name="sglang",
            model=self._get_model(),
            model_context=model_context,
            infer_conf=self._infer_conf,
            engine_rank=self._engine_rank,
            num_engines=self._num_infer_engines,
            meta_server_addr=self._meta_server_addr,
            parameters_meta=self._infer_params_meta,
            training_params_meta=training_params_meta,
            enable_colocate_mode=True,
            ipc_backend="cuda",
            enable_debug_mode=False,
            physical_gpu_id=self._local_gpu_id,
        )
        reader.initialize()
        self._reader = reader
        logger.info(
            "Constructed native NCCLWorkerWeightsReader (transfer_rank=%d, "
            "engine_rank=%d, num_engines=%d)",
            reader.transfer_rank,
            self._engine_rank,
            self._num_infer_engines,
        )
        return reader

    def update_weights(self, version: int) -> None:
        """Run one colocate weight update via the native awex worker reader.

        The native reader internally does: IPC collect -> StreamBatch transport
        -> put ``weights_update_finished`` -> barrier -> get_then_delete
        ``write_finished`` -> flush_cache. The plugin only needs to wrap this
        with the driver-equivalent wait-for-offload + resume + signal steps.
        """
        if not self._initialized:
            raise RuntimeError("AwexColocateReader not initialized")
        reader = self._ensure_reader()
        self._pre_process_model_weights()
        reader.update_weights(step_id=version)
        self._rebuild_derived_weights()
        logger.info("Colocate weight update completed: version=%d", version)

    def _iter_model_parts(self) -> list[tuple[Any, bool]]:
        models = self._get_model()
        if isinstance(models, (list, tuple)):
            if len(models) == 2:
                return [(models[0], False), (models[1], True)]
            return [(model, idx > 0) for idx, model in enumerate(models)]
        return [(models, False)]

    @staticmethod
    def _call_model_hook(model: Any, hook_name: str, **kwargs: Any) -> bool:
        hook = getattr(model, hook_name, None)
        if hook is None:
            return False

        try:
            params = inspect.signature(hook).parameters
        except (TypeError, ValueError):
            hook()
            return True

        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            hook(**kwargs)
        else:
            accepted = {
                name
                for name, p in params.items()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
            hook(**{k: v for k, v in kwargs.items() if k in accepted})
        return True

    def _pre_process_model_weights(self) -> None:
        """Run model-side pre-load hooks before AWEX writes params in-place."""
        for model, _ in self._iter_model_parts():
            if self._call_model_hook(model, "pre_process_weights_if_quant"):
                logger.info("pre_process_weights_if_quant() prepared model weights")

    def _rebuild_derived_weights(self) -> None:
        """Re-derive non-parameter tensors after an in-place AWEX weight write.

        P75 root cause: sglang's ``load_model`` ends with
        ``post_load_weights()``, which splits each MLA layer's
        ``kv_b_proj.weight`` into the absorbed-path tensors ``w_kc``/``w_vc``
        — ``.contiguous()`` copies stored as plain attributes, in neither
        ``named_parameters`` nor ``named_buffers``. The memory-saver
        release/resume cycle remaps their pages to zeros, and the AWEX reader
        rewrites only named parameters via in-place ``copy_`` (bypassing
        ``model.load_weights``), so nothing ever rebuilds them: decode's
        forward_absorb then consumes zeros and the 4 MLA layers degenerate
        while the 28 Lightning layers stay healthy (reward 0.77 -> ~0 within
        5 steps). Rebuild after EVERY transfer — train weights move each
        version, so a one-time fix would go stale. ``bind_or_assign`` copies
        into the existing tensors in place, which keeps captured CUDA-graph
        addresses valid.
        """
        did_post_load = False
        for model, is_nextn in self._iter_model_parts():
            did_post_load = (
                self._call_model_hook(
                    model,
                    "post_load_weights",
                    is_nextn=is_nextn,
                    weight_names=None,
                )
                or did_post_load
            )
            if self._call_model_hook(model, "post_process_weights_if_quant"):
                logger.info("post_process_weights_if_quant() finalized model weights")

        torch.cuda.synchronize()
        if did_post_load:
            logger.info("post_load_weights() re-derived absorbed MLA weights")

    # ── memory release/resume (delegate to SGLang native) ─────────────

    def release_memory(self, tags: list[str] | None = None) -> None:
        from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput

        tags = tags or ["kv_cache"]
        native_tags = [t for t in tags if t not in self._released_tags]
        if native_tags:
            req = ReleaseMemoryOccupationReqInput(tags=native_tags)
            self._scheduler.release_memory_occupation(req)
            self._released_tags.update(native_tags)
        logger.info("release_memory: tags=%s", tags)

    def resume_memory(self, tags: list[str] | None = None) -> None:
        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput

        tags = tags or ["kv_cache"]
        resume_tags = [t for t in tags if t in self._released_tags]
        if resume_tags:
            req = ResumeMemoryOccupationReqInput(tags=resume_tags)
            self._scheduler.resume_memory_occupation(req)
            self._released_tags.difference_update(resume_tags)
        logger.info("resume_memory: tags=%s", tags)

    # ── writer-coordination handshake (driver-equivalent shell steps) ──

    def wait_for_training_offloaded(self, version: int) -> None:
        """Wait for the writer to offload its model weights (avoid 2x weights).

        Equivalent to awex driver ``_pre_update_weights``'s wait on
        ``all_training_offloaded_weights``.
        """
        from areal.engine.awex_colocate import awex_colocate_timeout_s

        self._meta_server_client.wait_set_until_size(
            "all_training_offloaded_weights",
            self._train_world_size,
            timeout=awex_colocate_timeout_s(),
        )

    def wait_for_weights_ready(
        self, version: int, timeout_s: float | None = None
    ) -> None:
        """Block until the writer has published THIS version's IPC handles.

        Used by the plugin's background thread as the per-version trigger to
        enqueue a weight-update marker. We probe the per-version
        ``training_serialized_weights_{ip}_{gpu}_{version}`` key with MetaServer
        ``wait_key`` (existence-only, NO deserialization), for two reasons:

        1. Per-version gating. The unversioned ``all_training_offloaded_weights``
           set is only deleted by the writer's rank0 in ``finish_colocate_weight_update``
           (a later phase than the engine's signal_finished), so gating on it
           lets the background thread fire v+1 off a *stale* satisfied set while
           the writer is still in v's finish phase. The collected v+1 IPC then
           blocks waiting for a not-yet-published key, hogging the scheduler main
           loop so it cannot serve rollout -> train waits on rollout -> deadlock.
           The writer only puts the v+1 serialized key in the NEXT training cycle,
           so gating on it cannot fire early.
        2. No double-attach. ``get_object`` would deserialize the CUDA IPC handle
           in the background thread, racing the worker reader's own collect inside
           update_weights. ``wait_key`` only checks presence (``_has_key``).
        """
        from awex.util.common import get_ip_address

        from areal.engine.awex_colocate import awex_colocate_timeout_s

        ip = get_ip_address()
        key = f"training_serialized_weights_{ip}_{self._local_gpu_id}_{version}"
        self._meta_server_client.wait_key(
            key,
            timeout=awex_colocate_timeout_s() if timeout_s is None else timeout_s,
        )

    def signal_finished_weights_update(self) -> None:
        """Signal this engine finished, so the writer can resume kv_cache.

        Equivalent to awex driver ``_resume_kvcache``'s add to
        ``finished_weights_update_engines``. Only one rank per engine instance
        (instance_local_rank == 0) signals, with its real engine_rank, so the
        set collects exactly num_infer_engines unique entries.
        """
        if self._instance_local_rank != 0:
            return
        self._meta_server_client.add_object_to_set(
            "finished_weights_update_engines", self._engine_rank
        )

    def teardown(self) -> None:
        self._reader = None


__all__ = ["AwexColocateReader"]
