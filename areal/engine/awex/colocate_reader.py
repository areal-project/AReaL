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

Why the awex-native reader instead of a hand-rolled receiver: the community
SGLang scheduler has no ``execute_task_in_model_worker`` driver layer, so we
build the awex *worker* reader directly in-process. The native worker reader
uses ``NcclColocateStreamBatchTransport`` (recursive partition), the transport
AWEX ships -- a hand-rolled ring-shift transport deadlocks on mismatched
train/infer pipeline layouts (e.g. train PP=4 vs infer PP=1).

The plugin shell still owns the steps awex's *driver* would normally do
(``_pre_update_weights`` wait-for-offload + resume weights, ``_resume_kvcache``
signal-finished); see ``awex_sglang_plugin.process_awex_queue``.
"""

from __future__ import annotations

import gc
import inspect
import os
import time
from typing import Any

import torch
import torch.distributed as dist

from areal.engine.awex.memory_saver import patch_tms_hook_mode  # noqa: E402

# Keep direct imports of this module safe as well. The rollout entry point calls
# this helper earlier, before SGLang starts model initialization.
patch_tms_hook_mode()

from awex.meta.infer_meta_resolver import InferParamMetaResolver  # noqa: E402
from awex.meta.meta_resolver import ParamMetaResolver  # noqa: E402
from awex.reader.nccl_reader import NCCLWorkerWeightsReader  # noqa: E402

try:
    from awex.reader.nccl_reader import (  # noqa: E402
        _wait_colocate_write_finished as _awex_wait_colocate_write_finished,
    )
except ImportError:  # AWEX 0.7.0 uses one completion key per physical GPU.
    _awex_wait_colocate_write_finished = None
from awex.sharding import get_sharding_strategy_builder  # noqa: E402
from awex.transfer.nccl_stream_batch import (  # noqa: E402
    NcclColocateStreamBatchTransport,
)
from awex.util.common import simple_hf_config  # noqa: E402

from areal.engine.weight_finite import (  # noqa: E402
    check_named_tensors_finite,
    iter_module_named_tensors,
)
from areal.utils.logging import getLogger  # noqa: E402

logger = getLogger("AwexColocateReader")


def _wait_colocate_write_finished(
    meta_server_client: Any,
    write_finished_key: str,
    weights_update_group: Any,
    transfer_rank: int,
) -> None:
    """Wait for the writer across both deployed AWEX handshake APIs."""
    if _awex_wait_colocate_write_finished is not None:
        _awex_wait_colocate_write_finished(
            meta_server_client,
            write_finished_key,
            weights_update_group,
            transfer_rank,
        )
        return

    meta_server_client.get_object_then_delete(write_finished_key)


class _BoundedMemoryNcclColocateStreamBatchTransport(NcclColocateStreamBatchTransport):
    """Run AWEX recursive P2P without retaining every send clone at once.

    Upstream AWEX clones every remote send slice while constructing the transfer
    plan.  A Qwen3-30B 8-way colocate update consequently retains roughly 7/8
    of the model (about 53 GiB per GPU) before NCCL starts.  Keep source views
    in the plan and materialize only one operation per active peer at a time.
    The temporary clones stay alive until their sends complete, then become
    reusable by the CUDA allocator before the next operation index.
    """

    def update_weights_in_colocate_mode(
        self,
        train_to_infer_device_mapping,
        infer_to_train_device_mapping,
        transfer_rank,
        rank_coordinate,
        world_size,
        send_transfer_plan,
        recv_transfer_plan,
        weights_update_group,
        send_parameters,
        recv_parameters,
        *,
        step_id=-1,
        async_op=True,
        **kwargs,
    ):
        import os
        from concurrent.futures import Future

        from awex.transfer.nccl_comm import (
            detect_hang,
            execute_tensors_to_copy,
            validate_rank_mappings,
        )
        from awex.transfer.nccl_stream_batch import hang_detector
        from awex.transfer.transfer_plan import slice_tensor
        from awex.util import device as device_util

        logger.info(
            "Using bounded-memory RECURSIVE PARTITION P2P for rank %s",
            rank_coordinate,
        )
        task_id = f"{rank_coordinate}-{step_id}"
        validate_rank_mappings(
            train_to_infer_device_mapping, infer_to_train_device_mapping
        )
        start_time = time.time()

        send_ops = dict(send_transfer_plan.operations)
        recv_ops = dict(recv_transfer_plan.operations)
        num_sends = sum(len(ops) for ops in send_ops.values())
        num_recvs = sum(len(ops) for ops in recv_ops.values())
        logger.info(
            "Start bounded-memory weights update for %s, num_sends=%d, num_recvs=%d",
            task_id,
            num_sends,
            num_recvs,
        )

        all_send_p2p_ops = {}
        all_recv_p2p_ops = {}
        tensors_to_copy = []
        train_slice_context = {}
        non_contiguous_tensor_pairs = []

        for peer_rank, ops in send_ops.items():
            mapped_peer_rank = train_to_infer_device_mapping.get(peer_rank, peer_rank)
            if mapped_peer_rank == transfer_rank:
                for op in ops:
                    send_tensor = send_parameters[op.send_shard_meta.name]
                    tensor_sliced = slice_tensor(
                        send_tensor,
                        op,
                        True,
                        slice_context=train_slice_context,
                    )
                    tensors_to_copy.append(tensor_sliced)
                continue

            p2p_ops = []
            for op in ops:
                send_tensor = send_parameters[op.send_shard_meta.name]
                tensor_sliced = slice_tensor(
                    send_tensor,
                    op,
                    True,
                    slice_context=train_slice_context,
                )
                recv_rank = train_to_infer_device_mapping.get(
                    op.recv_rank, op.recv_rank
                )
                # Deliberately retain the source view.  _execute_ops_concurrent
                # clones a bounded batch immediately before enqueueing sends.
                p2p_op = dist.P2POp(
                    dist.isend if async_op else dist.send,
                    tensor_sliced,
                    recv_rank,
                    group=weights_update_group,
                )
                p2p_ops.append((op, p2p_op))
            all_send_p2p_ops[mapped_peer_rank] = p2p_ops

        for send_rank, ops in recv_ops.items():
            recv_from_rank = train_to_infer_device_mapping[send_rank]
            if recv_from_rank == transfer_rank:
                continue
            p2p_ops = []
            for op in ops:
                recv_tensor = recv_parameters[op.recv_shard_meta.name]
                tensor_sliced = slice_tensor(recv_tensor, op, False)
                if not tensor_sliced.is_contiguous():
                    original_tensor = tensor_sliced
                    tensor_sliced = tensor_sliced.contiguous()
                    non_contiguous_tensor_pairs.append((original_tensor, tensor_sliced))
                p2p_op = dist.P2POp(
                    dist.irecv if async_op else dist.recv,
                    tensor_sliced,
                    recv_from_rank,
                    group=weights_update_group,
                )
                p2p_ops.append((op, p2p_op))
            all_recv_p2p_ops[recv_from_rank] = p2p_ops

        if tensors_to_copy:
            send_rank = infer_to_train_device_mapping[transfer_rank]
            execute_tensors_to_copy(
                tensors_to_copy,
                recv_transfer_plan.operations[send_rank],
                recv_parameters,
                f"tensor copy for {task_id}",
            )
        else:
            logger.info("No tensors to copy for %s", task_id)

        future = Future()
        total_send_ops = sum(len(ops) for ops in all_send_p2p_ops.values())
        total_recv_ops = sum(len(ops) for ops in all_recv_p2p_ops.values())
        message = (
            f"[{os.getpid()}] execute {total_send_ops} sends "
            f"{total_recv_ops} recvs with bounded recursive partition for {task_id}"
        )
        hang_detector.submit(detect_hang, future, message, [], timeout=60)

        self.execute_recursive_partition_stream_transfer(
            transfer_rank,
            world_size,
            all_send_p2p_ops,
            all_recv_p2p_ops,
            weights_update_group,
            rank_coordinate,
            step_id,
        )
        if non_contiguous_tensor_pairs:
            with torch.no_grad():
                for original_tensor, recv_tensor in non_contiguous_tensor_pairs:
                    original_tensor.copy_(recv_tensor)
            non_contiguous_tensor_pairs.clear()
        device_util.synchronize()
        future.set_result(True)
        logger.info(
            "Finished bounded-memory weights update for %s, took %.4f seconds",
            task_id,
            time.time() - start_time,
        )

    def _execute_ops_concurrent(self, ops_dict, peer_ranks):
        """Execute one tensor per active peer and release send clones promptly."""
        from awex.util import device as device_util

        peer_ops_with_rank = [
            (peer_rank, ops_dict[peer_rank])
            for peer_rank in peer_ranks
            if peer_rank in ops_dict
        ]
        if not peer_ops_with_rank:
            return 0

        peer_to_stream_idx = {
            peer_rank: index % len(self._stream_pool)
            for index, (peer_rank, _) in enumerate(peer_ops_with_rank)
        }
        max_ops = max(len(ops) for _, ops in peer_ops_with_rank)
        total_ops = 0

        for op_idx in range(max_ops):
            work_handles = []
            owned_send_tensors = []
            for peer_rank, ops in peer_ops_with_rank:
                if op_idx >= len(ops):
                    continue
                plan_op, p2p_op = ops[op_idx]
                is_send = p2p_op.op is dist.isend or p2p_op.op is dist.send
                stream = self._stream_pool[peer_to_stream_idx[peer_rank]]
                with device_util.stream(stream):
                    # Prepare the payload on the same stream that consumes it.
                    # clone()/to() on the caller's default stream followed by
                    # isend() on this dedicated stream has no ordering edge;
                    # NCCL can otherwise read a partially written clone and
                    # silently deliver sparse NaN/Inf values.
                    tensor_for_transfer = (
                        p2p_op.tensor.clone() if is_send else p2p_op.tensor
                    )
                    if is_send:
                        # NCCL send/recv counts are expressed in elements of
                        # each side's dtype. A dtype mismatch therefore changes
                        # the wire size. Match the inference shard's dtype.
                        recv_dtype = getattr(plan_op.recv_shard_meta, "dtype", None)
                        if (
                            recv_dtype is not None
                            and tensor_for_transfer.dtype != recv_dtype
                        ):
                            tensor_for_transfer = tensor_for_transfer.to(recv_dtype)
                        owned_send_tensors.append(tensor_for_transfer)
                    result = p2p_op.op(
                        tensor_for_transfer,
                        p2p_op.peer,
                        group=p2p_op.group,
                    )
                if p2p_op.op is dist.isend or p2p_op.op is dist.irecv:
                    work_handles.append(result)
                total_ops += 1

            for work in work_handles:
                work.wait()
            # ProcessGroupNCCL Work.wait() only guarantees that the CUDA work
            # has been enqueued.  The send clones must remain alive until NCCL
            # has actually consumed them; otherwise the caching allocator can
            # reuse their storage for the next batch and silently corrupt the
            # transferred model.  Drain this bounded batch before releasing it.
            device_util.synchronize()
            work_handles.clear()
            owned_send_tensors.clear()
            tensor_for_transfer = None
            result = None

        return total_ops


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
        self.colocate_transport = _BoundedMemoryNcclColocateStreamBatchTransport(
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
        ip_address = get_ip_address()
        physical_gpu_id = self._areal_physical_gpu_id
        key_suffix = f"_{ip_address}_{physical_gpu_id}_{step_id}"
        update_finished_key = f"weights_update_finished{key_suffix}"
        try:
            # Check the local CUDA IPC import before any redistribution. This
            # separates imported-mapping corruption from the subsequent P2P
            # path in both diagnostics and failure handling.
            check_named_tensors_finite(
                self.deserialized_weights.items(),
                stage="awex_reader_ipc_imported",
                version=step_id,
                logger=logger,
                process_group=self.weights_update_group,
            )
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
            # Validate while the imported CUDA IPC mappings are still alive.
            # Acknowledging first lets the writer release/reuse their storage,
            # which both widens the lifetime race and strands the actor if a
            # later validation rejects the received model.
            check_named_tensors_finite(
                self.parameters.items(),
                stage="awex_reader_pre_ack",
                version=step_id,
                logger=logger,
                process_group=self.weights_update_group,
            )
            self._run_pre_ack_callback(step_id)
            named_tensors_factory = getattr(
                self, "_areal_pre_ack_named_tensors_factory", None
            )
            if named_tensors_factory is not None:
                check_named_tensors_finite(
                    named_tensors_factory(),
                    stage="awex_reader_derived_pre_ack",
                    version=step_id,
                    logger=logger,
                    process_group=self.weights_update_group,
                )
        except Exception as exc:
            # Reuse the versioned completion key as a failure result. Every
            # writer rank already blocks on its paired physical GPU key, so
            # this wakes it immediately without a second polling protocol.
            self.meta_server_client.put_object(
                update_finished_key,
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            logger.exception(
                "Rejected AWEX weights before writer IPC release: step=%s rank=%s",
                step_id,
                self.rank_coordinate,
            )
            raise
        print_current_gpu_status(
            f"after weights update using NCCL for rank {self.rank_coordinate}",
        )
        self.deserialized_weights = None
        gc.collect()
        torch.cuda.synchronize()
        # Flush unused importer mappings before the early acknowledgement. The
        # writer retains its exporter until the later all-engine completion,
        # so that final signal has an unambiguous no-reader-owns-IPC meaning.
        if device_util.get_device_type() == "cuda":
            torch.cuda.empty_cache()
        duration = time.time() - start_time
        compute_statistics(
            self._history_update_weights_time,
            step_id,
            duration,
            "Receive weights using NCCL",
        )
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

    def _run_pre_ack_callback(self, step_id: int) -> None:
        """Run local derived-weight rebuild and fail every rank consistently."""
        callback = getattr(self, "_areal_pre_ack_callback", None)
        if callback is None:
            return

        local_error: BaseException | None = None
        try:
            callback(step_id)
        except BaseException as exc:  # noqa: BLE001
            local_error = exc

        any_error = local_error is not None
        if dist.is_initialized():
            backend = str(dist.get_backend(self.weights_update_group)).lower()
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if "nccl" in backend
                else torch.device("cpu")
            )
            failed = torch.tensor(int(any_error), dtype=torch.int32, device=device)
            dist.all_reduce(
                failed,
                op=dist.ReduceOp.MAX,
                group=self.weights_update_group,
            )
            any_error = bool(failed.item())

        if not any_error:
            return
        if local_error is not None:
            raise RuntimeError(
                "AWEX derived-weight rebuild failed before writer ACK: "
                f"{type(local_error).__name__}: {local_error}"
            ) from local_error
        raise RuntimeError(
            "AWEX derived-weight rebuild failed on another rank before writer ACK"
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
        tp_rank = int(getattr(scheduler, "tp_rank", 0))

        if self._infer_instance_world_size is not None:
            world_size = self._infer_instance_world_size
            global_rank = self._instance_local_rank
        else:
            world_size = tp_size * pp_size
            global_rank = tp_rank

        return {
            "scheduler": scheduler,
            "infer_engine_config": server_args,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "pp_rank": int(getattr(scheduler, "pp_rank", 0)),
            "pp_size": pp_size,
            "dp_size": dp_size,
            "world_size": world_size,
            "global_rank": global_rank,
            "local_rank": tp_rank,
            "attn_tp_rank": int(getattr(scheduler, "attn_tp_rank", tp_rank)),
            "attn_tp_size": int(getattr(scheduler, "attn_tp_size", tp_size)),
            "attn_dp_rank": int(getattr(scheduler, "attn_dp_rank", 0)),
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
        raw_meta = InferParamMetaResolver._get_model_param_info(
            "sglang",
            server_args,
            convert_params=True,
            engine_rank=self._engine_rank or 0,
            model=self._get_model(),
            model_context=model_context,
        )

        # AWEX's worker reader exposes ``lm_head.weight`` as an alias of the
        # embedding tensor for tied models, and its MCore metadata resolver
        # publishes the same alias on the training side.  The pinned AWEX
        # version only mirrors that alias into *vLLM* inference metadata,
        # leaving SGLang with train=255 / infer=254 and no transfer plan.  Keep
        # the SGLang metadata aligned with the parameter dictionary that the
        # native reader constructs during ``initialize()``.
        hf_config = getattr(self._get_model(), "config", None)
        pp_rank = int(model_context.get("pp_rank", 0))
        pp_size = int(model_context.get("pp_size", 1))
        params_meta = raw_meta.get("params_meta", [])
        names = {param_meta["name"] for param_meta in params_meta}
        if (
            getattr(hf_config, "tie_word_embeddings", False)
            and pp_rank == pp_size - 1
            and "lm_head.weight" not in names
            and "model.embed_tokens.weight" in names
        ):
            embedding_meta = next(
                param_meta
                for param_meta in params_meta
                if param_meta["name"] == "model.embed_tokens.weight"
            )
            lm_head_meta = dict(embedding_meta)
            lm_head_meta["name"] = "lm_head.weight"
            params_meta.append(lm_head_meta)
            logger.info(
                "Infer meta: added lm_head.weight alias for tied embeddings in SGLang"
            )
        return raw_meta

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
        if self._engine_rank is None:
            raise RuntimeError(
                "AwexColocateReader must be initialized before getting weight metadata"
            )
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
            # AWEX's native reader publishes router_dtype so the train-side
            # converter casts mlp.gate.weight to the dtype the inference
            # engine actually holds (fp32 for BailingMoe). Omitting it makes
            # the converter fall back to its bf16 default: gate shards go out
            # as 2N bytes against a 4N irecv and the transfer wedges
            # deterministically. The wire-level dtype reconciliation below
            # papers over any such mismatch generically, but keep the
            # semantic path whole so new models behave identically to native
            # awex.
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

    @torch.no_grad()
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
        models = [model for model, _ in self._iter_model_parts()]
        reader._areal_pre_ack_callback = (
            lambda _step_id: self._rebuild_derived_weights()
        )
        reader._areal_pre_ack_named_tensors_factory = lambda: iter_module_named_tensors(
            models,
            extra_tensor_attrs=(
                "w_kc",
                "w_vc",
                "w_scale",
                "w_scale_k",
                "w_scale_v",
            ),
        )
        reader.update_weights(step_id=version)
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

        Root cause: sglang's ``load_model`` ends with
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
        from areal.engine.awex.colocate_writer import awex_colocate_timeout_s

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

        from areal.engine.awex.colocate_writer import awex_colocate_timeout_s

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
