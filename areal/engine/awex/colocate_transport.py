# SPDX-License-Identifier: Apache-2.0

"""Bounded-memory AWEX transport shared with the validated training recipe."""

import math
import os
import time
from typing import Any

import torch
import torch.distributed as dist

from areal.engine.awex.memory_saver import patch_tms_hook_mode
from areal.utils.logging import getLogger

patch_tms_hook_mode()

from awex.transfer.nccl_stream_batch import (  # noqa: E402
    NcclColocateStreamBatchTransport,
)

logger = getLogger("AwexColocateReader")
_DEFAULT_EXPERT_PACK_OPS = 64
_DEFAULT_EXPERT_PACK_MB = 64


def _read_positive_env_int(name: str, default: int) -> int:
    """Read a positive integer env value, preserving the safe default on errors."""
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        return max(1, int(raw_value))
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, raw_value, default)
        return default


def _expert_pack_limits() -> tuple[int, int]:
    max_ops = _read_positive_env_int("AWEX_EXPERT_PACK_OPS", _DEFAULT_EXPERT_PACK_OPS)
    max_bytes = (
        _read_positive_env_int("AWEX_EXPERT_PACK_MB", _DEFAULT_EXPERT_PACK_MB)
        * 1024
        * 1024
    )
    return max_ops, max_bytes


class _BoundedMemoryNcclColocateStreamBatchTransport(NcclColocateStreamBatchTransport):
    """Run AWEX recursive P2P without retaining every send clone at once.

    Upstream AWEX clones every remote send slice while constructing the transfer
    plan.  A Qwen3-30B 8-way colocate update consequently retains roughly 7/8
    of the model (about 53 GiB per GPU) before NCCL starts.  Keep source views
    in the plan and materialize only one operation per active peer at a time.
    The temporary clones stay alive until their sends complete, then become
    reusable by the CUDA allocator before the next operation index.

    ``AWEX_EXPERT_PACK_OPS`` combines up to 64 consecutive routed-expert
    operations for the same peer into one flat wire tensor by default. Set it
    to one to restore the unpacked path. ``AWEX_EXPERT_PACK_MB`` bounds each
    flat tensor and defaults to 64 MiB.
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

        self._expert_pack_config = _expert_pack_limits()
        expert_pack_ops, expert_pack_bytes = self._expert_pack_config
        self._expert_pack_stats = None
        if expert_pack_ops > 1:
            self._expert_pack_stats = {
                "logical_ops": 0,
                "wire_ops": 0,
                "packed_wire_ops": 0,
            }
            logger.info(
                "Expert P2P packing enabled for %s: max_ops=%d, max_mb=%.1f",
                task_id,
                expert_pack_ops,
                expert_pack_bytes / 1024 / 1024,
            )

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
                    tensor_sliced = torch.empty_like(
                        tensor_sliced, memory_format=torch.contiguous_format
                    )
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

        # slice_tensor may materialize send slices on the caller stream.
        # Finish planning copies before independent transfer streams consume
        # them, including ranks with no local copy to synchronize implicitly.
        device_util.synchronize()

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
        if self._expert_pack_stats is not None:
            stats = self._expert_pack_stats
            logger.info(
                "Expert P2P packing finished for %s: logical_ops=%d, wire_ops=%d, "
                "packed_wire_ops=%d",
                task_id,
                stats["logical_ops"],
                stats["wire_ops"],
                stats["packed_wire_ops"],
            )
        logger.info(
            "Finished bounded-memory weights update for %s, took %.4f seconds",
            task_id,
            time.time() - start_time,
        )

    @staticmethod
    def _is_expert_operation(plan_op: Any) -> bool:
        if getattr(plan_op, "param_class", None) == "expert":
            return True
        for meta_name in ("send_shard_meta", "recv_shard_meta"):
            name = getattr(getattr(plan_op, meta_name, None), "name", "")
            if ".experts." in name:
                return True
        return False

    @staticmethod
    def _operation_wire_dtype(plan_op: Any, p2p_op: Any) -> torch.dtype:
        recv_dtype = getattr(getattr(plan_op, "recv_shard_meta", None), "dtype", None)
        return recv_dtype if recv_dtype is not None else p2p_op.tensor.dtype

    @staticmethod
    def _operation_wire_numel(plan_op: Any, p2p_op: Any) -> int:
        overlap_shape = getattr(plan_op, "overlap_shape", None)
        if overlap_shape is not None:
            return math.prod(overlap_shape)
        return p2p_op.tensor.numel()

    @classmethod
    def _partition_expert_operations(
        cls,
        operations: list[tuple[Any, Any]],
        max_pack_ops: int,
        max_pack_bytes: int,
    ) -> list[list[tuple[Any, Any]]]:
        """Pack consecutive compatible expert ops without changing FIFO order."""
        batches: list[list[tuple[Any, Any]]] = []
        current_batch: list[tuple[Any, Any]] = []
        current_signature = None
        current_bytes = 0

        def flush_current() -> None:
            nonlocal current_batch, current_signature, current_bytes
            if current_batch:
                batches.append(current_batch)
            current_batch = []
            current_signature = None
            current_bytes = 0

        for plan_op, p2p_op in operations:
            if not cls._is_expert_operation(plan_op):
                flush_current()
                batches.append([(plan_op, p2p_op)])
                continue

            wire_dtype = cls._operation_wire_dtype(plan_op, p2p_op)
            wire_bytes = (
                cls._operation_wire_numel(plan_op, p2p_op) * wire_dtype.itemsize
            )
            signature = (p2p_op.op, p2p_op.peer, id(p2p_op.group), wire_dtype)
            exceeds_limit = current_batch and (
                len(current_batch) >= max_pack_ops
                or current_bytes + wire_bytes > max_pack_bytes
                or signature != current_signature
            )
            if exceeds_limit:
                flush_current()

            if wire_bytes > max_pack_bytes:
                batches.append([(plan_op, p2p_op)])
                continue

            current_batch.append((plan_op, p2p_op))
            current_signature = signature
            current_bytes += wire_bytes

        flush_current()
        return batches

    @classmethod
    def _pack_send_batch(cls, batch: list[tuple[Any, Any]]) -> torch.Tensor:
        wire_dtype = cls._operation_wire_dtype(*batch[0])
        source_tensors = [p2p_op.tensor for _, p2p_op in batch]
        numels = [tensor.numel() for tensor in source_tensors]
        packed = torch.empty(
            sum(numels),
            dtype=wire_dtype,
            device=source_tensors[0].device,
        )
        packed_views = [
            flat_view.view_as(source)
            for flat_view, source in zip(packed.split(numels), source_tensors)
        ]
        with torch.no_grad():
            torch._foreach_copy_(packed_views, source_tensors)
        return packed

    @staticmethod
    def _allocate_packed_recv_batch(
        batch: list[tuple[Any, Any]], wire_dtype: torch.dtype
    ) -> torch.Tensor:
        destination_tensors = [p2p_op.tensor for _, p2p_op in batch]
        return torch.empty(
            sum(tensor.numel() for tensor in destination_tensors),
            dtype=wire_dtype,
            device=destination_tensors[0].device,
        )

    @staticmethod
    def _unpack_recv_batch(packed: torch.Tensor, batch: list[tuple[Any, Any]]) -> None:
        destination_tensors = [p2p_op.tensor for _, p2p_op in batch]
        numels = [tensor.numel() for tensor in destination_tensors]
        packed_views = [
            flat_view.view_as(destination)
            for flat_view, destination in zip(packed.split(numels), destination_tensors)
        ]
        with torch.no_grad():
            torch._foreach_copy_(destination_tensors, packed_views)

    def _execute_ops_concurrent(self, ops_dict, peer_ranks):
        expert_pack_config = getattr(self, "_expert_pack_config", None)
        if expert_pack_config is None:
            expert_pack_config = _expert_pack_limits()
        max_pack_ops, max_pack_bytes = expert_pack_config
        if max_pack_ops <= 1:
            return self._execute_ops_concurrent_unpacked(ops_dict, peer_ranks)
        return self._execute_ops_concurrent_packed(
            ops_dict,
            peer_ranks,
            max_pack_ops,
            max_pack_bytes,
        )

    def _execute_ops_concurrent_unpacked(self, ops_dict, peer_ranks):
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

    def _execute_ops_concurrent_packed(
        self,
        ops_dict,
        peer_ranks,
        max_pack_ops: int,
        max_pack_bytes: int,
    ) -> int:
        """Execute one bounded expert pack per active peer and FIFO position."""
        from awex.util import device as device_util

        peer_batches_with_rank = [
            (
                peer_rank,
                self._partition_expert_operations(
                    ops_dict[peer_rank], max_pack_ops, max_pack_bytes
                ),
            )
            for peer_rank in peer_ranks
            if peer_rank in ops_dict
        ]
        if not peer_batches_with_rank:
            return 0

        peer_to_stream_idx = {
            peer_rank: index % len(self._stream_pool)
            for index, (peer_rank, _) in enumerate(peer_batches_with_rank)
        }
        max_batches = max(len(batches) for _, batches in peer_batches_with_rank)
        logical_ops = 0
        wire_ops = 0
        packed_wire_ops = 0

        for batch_idx in range(max_batches):
            work_handles = []
            owned_send_tensors = []
            owned_recv_tensors = []
            pending_recv_unpacks = []
            for peer_rank, batches in peer_batches_with_rank:
                if batch_idx >= len(batches):
                    continue
                batch = batches[batch_idx]
                plan_op, p2p_op = batch[0]
                is_send = p2p_op.op is dist.isend or p2p_op.op is dist.send
                is_recv = p2p_op.op is dist.irecv or p2p_op.op is dist.recv
                stream = self._stream_pool[peer_to_stream_idx[peer_rank]]
                with device_util.stream(stream):
                    if len(batch) == 1:
                        tensor_for_transfer = (
                            p2p_op.tensor.clone() if is_send else p2p_op.tensor
                        )
                        if is_send:
                            recv_dtype = self._operation_wire_dtype(plan_op, p2p_op)
                            if tensor_for_transfer.dtype != recv_dtype:
                                tensor_for_transfer = tensor_for_transfer.to(recv_dtype)
                            owned_send_tensors.append(tensor_for_transfer)
                    elif is_send:
                        tensor_for_transfer = self._pack_send_batch(batch)
                        owned_send_tensors.append(tensor_for_transfer)
                    elif is_recv:
                        recv_dtype = self._operation_wire_dtype(plan_op, p2p_op)
                        tensor_for_transfer = self._allocate_packed_recv_batch(
                            batch, recv_dtype
                        )
                        owned_recv_tensors.append(tensor_for_transfer)
                    else:
                        raise RuntimeError(
                            "Expert packing only supports torch.distributed P2P ops"
                        )

                    result = p2p_op.op(
                        tensor_for_transfer,
                        p2p_op.peer,
                        group=p2p_op.group,
                    )
                    if len(batch) > 1 and is_recv:
                        pending_recv_unpacks.append(
                            (stream, tensor_for_transfer, batch)
                        )

                if p2p_op.op is dist.isend or p2p_op.op is dist.irecv:
                    work_handles.append((result, stream))
                logical_ops += len(batch)
                wire_ops += 1
                packed_wire_ops += int(len(batch) > 1)

            for work, stream in work_handles:
                # ProcessGroupNCCL wait establishes the completion dependency
                # on the current stream.  Use the same transfer stream that
                # will consume a packed receive below.
                with device_util.stream(stream):
                    work.wait()
            for stream, packed, batch in pending_recv_unpacks:
                with device_util.stream(stream):
                    self._unpack_recv_batch(packed, batch)
            # Keep both packed send and receive buffers alive until their NCCL
            # and foreach-copy work has drained from every active peer stream.
            device_util.synchronize()
            work_handles.clear()
            owned_send_tensors.clear()
            owned_recv_tensors.clear()
            pending_recv_unpacks.clear()
            tensor_for_transfer = None
            result = None

        if getattr(self, "_expert_pack_stats", None) is not None:
            self._expert_pack_stats["logical_ops"] += logical_ops
            self._expert_pack_stats["wire_ops"] += wire_ops
            self._expert_pack_stats["packed_wire_ops"] += packed_wire_ops
        return wire_ops
