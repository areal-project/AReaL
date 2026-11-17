# SPDX-License-Identifier: Apache-2.0
"""Compare a real MindSpeed GDN layer under CP with its CP=1 equivalent.

Run with torchrun --nproc-per-node=2 (or 4). No model checkpoint is required.
"""

import argparse
import importlib
import json
import os
import sys
from datetime import timedelta
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401
import torch_npu.contrib.transfer_to_npu  # noqa: F401

from areal.utils import logging

logger = logging.getLogger("MegatronEngine")


def local_indices(lengths, cp, rank, device):
    """Select the two zigzag chunks owned by a CP rank in every sequence."""
    parts = []
    offset = 0
    for length in lengths:
        assert length % (2 * cp) == 0
        width = length // (2 * cp)
        for chunk in (rank, 2 * cp - rank - 1):
            parts.extend(range(offset + chunk * width, offset + (chunk + 1) * width))
        offset += length
    return torch.tensor(parts, device=device, dtype=torch.long)


def run_checks(args):
    # Keep pytest/torchrun options out of the training argument parser.
    with patch.object(sys, "argv", [sys.argv[0]]):
        _run_checks(args)


def _run_checks(args):
    import mindspeed.megatron_adaptor  # noqa: F401 isort: skip

    import megatron.core.ssm.gated_delta_net as megatron_gdn
    from megatron.core import parallel_state as mpu
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_net import GatedDeltaNetSubmodules
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    from mindspeed.core.fusions.fused_rms_norm import RMSNorm
    from mindspeed.core.ssm.gated_delta_net import GatedDeltaNet

    class Norm(RMSNorm):
        def __init__(self, config, hidden_size, eps):
            super().__init__(
                dim=hidden_size, eps=eps, sequence_parallel=False, config=config
            )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    torch.set_num_threads(1)
    device = torch.device("npu", local_rank)
    owns_distributed = not dist.is_initialized()
    if owns_distributed:
        dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    cp = dist.get_world_size()
    rank = dist.get_rank()
    assert cp in (2, 4, 8), "Run with 2, 4 or 8 processes to exercise CP collectives"
    key_heads = max(4, cp)
    value_heads = 2 * key_heads
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp,
    )
    try:
        model_parallel_cuda_manual_seed(123)
        tp_group = mpu.get_tensor_model_parallel_group()
        cp_group = mpu.get_context_parallel_group()
        pg_parallel = ProcessGroupCollection(tp=tp_group, cp=cp_group)
        pg_reference = ProcessGroupCollection(tp=tp_group, cp=tp_group)
        adapter = importlib.import_module(
            "mindspeed.core.ssm.npu_chunk_gated_delta_rule"
        )
        from areal.engine.megatron_utils.mindspeed_gdn_patch import (
            ensure_mindspeed_gdn_native_kkt,
        )

        assert not getattr(adapter, "_NATIVE_KKT", False), (
            "Test requires MindSpeed without the direct native migration"
        )
        assert ensure_mindspeed_gdn_native_kkt()
        assert (
            adapter._chunk_gated_delta_rule_fwd.__module__
            == "areal.engine.megatron_utils.mindspeed_gdn_native"
        )
        native = importlib.import_module("fla_npu.ops.ascendc")

        def reject_triton(*args, **kwargs):
            raise AssertionError("Triton KKT was called")

        adapter.chunk_scaled_dot_kkt_fwd = reject_triton

        def layer(dtype, groups, cp_size, normalize):
            config = TransformerConfig(
                num_layers=1,
                hidden_size=256,
                num_attention_heads=8,
                linear_conv_kernel_dim=4,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_num_key_heads=key_heads,
                linear_num_value_heads=value_heads,
                normalization="RMSNorm",
                activation_func=F.silu,
                bf16=dtype == torch.bfloat16,
                fp16=dtype == torch.float16,
                params_dtype=dtype,
                pipeline_dtype=dtype,
                tensor_model_parallel_size=1,
                context_parallel_size=cp_size,
                sequence_parallel=False,
                use_cpu_initialization=True,
                gradient_accumulation_fusion=False,
                deterministic_mode=False,
                layernorm_zero_centered_gamma=False,
                experimental_attention_variant="gated_delta_net",
                linear_attention_freq=[1],
            )
            spec = GatedDeltaNetSubmodules(
                in_proj=ColumnParallelLinear, out_norm=Norm, out_proj=RowParallelLinear
            )
            return GatedDeltaNet(
                config,
                submodules=spec,
                layer_number=1,
                bias=False,
                conv_bias=False,
                use_qk_l2norm=normalize,
                pg_collection=groups,
            ).to(device=device, dtype=dtype)

        records = []
        failures = []
        for dtype in (torch.bfloat16, torch.float16):
            for packed, lengths, batch, normalize in (
                (False, [208], 2, False),
                (True, [80, 144, 208], 1, False),
                (False, [256], 2, True),
                (True, [128, 256], 1, True),
            ):
                if args.layout != "all" and packed != (args.layout == "packed"):
                    continue
                reference = layer(dtype, pg_reference, 1, normalize)
                parallel = layer(dtype, pg_parallel, cp, normalize)
                for parameter in reference.parameters():
                    dist.broadcast(parameter.data, src=0, group=cp_group)
                parallel.load_state_dict(reference.state_dict())
                reference.gated_delta_rule = partial(
                    reference.gated_delta_rule, chunk_size=args.chunk_size
                )
                parallel.gated_delta_rule = partial(
                    parallel.gated_delta_rule, chunk_size=args.chunk_size
                )
                if not packed and args.dense_torch_conv:
                    from mindspeed.core.ssm.torch_causal_conv1d import causal_conv1d

                    megatron_gdn.causal_conv1d = causal_conv1d
                conv_impl = megatron_gdn.causal_conv1d.__module__
                generator = torch.Generator(device="cpu").manual_seed(321)
                full = torch.randn(
                    (sum(lengths), batch, 256), generator=generator, dtype=torch.float32
                ).to(device, dtype)
                upstream = (
                    torch.randn(full.shape, generator=generator, dtype=torch.float32)
                    .mul_(0.1)
                    .to(device, dtype)
                )
                indices = local_indices(lengths, cp, rank, device)
                x_ref = full.detach().requires_grad_()
                x_cp = full.index_select(0, indices).detach().requires_grad_()
                params = None
                cu_list = [0]
                for length in lengths:
                    cu_list.append(cu_list[-1] + length)
                if packed:
                    cu = torch.tensor(cu_list, dtype=torch.int32, device=device)
                    params = PackedSeqParams(
                        qkv_format="thd",
                        cu_seqlens_q=cu,
                        cu_seqlens_kv=cu,
                        max_seqlen_q=max(lengths),
                        max_seqlen_kv=max(lengths),
                    )
                native_calls = []
                native_kkt = native.npu_chunk_scaled_dot_kkt

                def record_native(k, g, beta, **kwargs):
                    native_calls.append(
                        {"shape": list(k.shape), "cu_seqlens": kwargs.get("cu_seqlens")}
                    )
                    return native_kkt(k, g, beta, **kwargs)

                with patch.object(
                    native, "npu_chunk_scaled_dot_kkt", side_effect=record_native
                ):
                    out_ref, _ = reference(
                        x_ref, attention_mask=None, packed_seq_params=params
                    )
                # FLA's first call may replace its lazy export, including the spy.
                with patch.object(
                    native, "npu_chunk_scaled_dot_kkt", side_effect=record_native
                ):
                    out_cp, _ = parallel(
                        x_cp, attention_mask=None, packed_seq_params=params
                    )
                out_ref.backward(upstream)
                out_cp.backward(upstream.index_select(0, indices))
                record = {
                    "rank": rank,
                    "cp": cp,
                    "dtype": str(dtype),
                    "packed": packed,
                    "lengths": lengths,
                    "normalize_qk": normalize,
                    "chunk_size": args.chunk_size,
                    "convolution": conv_impl,
                    "native_calls": native_calls,
                    "checks": {},
                }
                expected_shapes = [
                    [batch, value_heads, sum(lengths), 128],
                    [batch, value_heads // cp, sum(lengths), 128],
                ]
                assert [call["shape"] for call in native_calls] == expected_shapes, (
                    native_calls
                )
                assert all(
                    call["cu_seqlens"] == (cu_list if packed else None)
                    for call in native_calls
                )

                def compare(name, actual, expected):
                    actual, expected = (
                        actual.detach().float(),
                        expected.detach().float(),
                    )
                    finite = bool(
                        torch.isfinite(actual).all() and torch.isfinite(expected).all()
                    )
                    rel = float(
                        (actual - expected).norm() / expected.norm().clamp_min(1e-12)
                    )
                    max_abs = float((actual - expected).abs().max())
                    record["checks"][name] = {
                        "finite": finite,
                        "relative_l2": rel,
                        "max_abs": max_abs,
                    }
                    limit = 0.02 if dtype == torch.bfloat16 else 0.006
                    atol = (0.025 if dtype == torch.bfloat16 else 0.008) * float(
                        expected.abs().max()
                    ) + 1e-6
                    rtol = 0.04 if dtype == torch.bfloat16 else 0.012
                    elementwise_close = bool(
                        torch.all(
                            (actual - expected).abs() <= atol + rtol * expected.abs()
                        )
                    )
                    record["checks"][name]["elementwise_close"] = elementwise_close
                    if not finite or rel > limit or not elementwise_close:
                        failures.append((record["dtype"], packed, normalize, name, rel))

                compare("output", out_cp, out_ref.index_select(0, indices))
                compare(
                    "input_gradient", x_cp.grad, x_ref.grad.index_select(0, indices)
                )
                for (name, p_cp), (ref_name, p_ref) in zip(
                    parallel.named_parameters(), reference.named_parameters()
                ):
                    assert name == ref_name
                    assert (p_cp.grad is None) == (p_ref.grad is None), name
                    if p_cp.grad is not None:
                        total = p_cp.grad.detach().float()
                        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=cp_group)
                        compare("parameter_gradient/" + name, total, p_ref.grad)
                records.append(record)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.with_name(
                    args.output.stem + f"-rank{rank}.json"
                ).write_text(json.dumps(records, indent=2) + "\n")
                if rank == 0:
                    logger.info(
                        json.dumps(
                            {
                                "event": "case",
                                "cp": cp,
                                "dtype": str(dtype),
                                "packed": packed,
                                "normalize_qk": normalize,
                                "chunk_size": args.chunk_size,
                                "convolution": conv_impl,
                                "max_relative_l2": max(
                                    c["relative_l2"] for c in record["checks"].values()
                                ),
                                "failures": failures,
                            }
                        ),
                    )
                del reference, parallel, full, upstream, x_ref, x_cp, out_ref, out_cp
        failed = torch.tensor([len(failures)], device=device, dtype=torch.int32)
        dist.all_reduce(failed, op=dist.ReduceOp.SUM, group=cp_group)
        any_failed = bool(failed.item())
        assert not any_failed, failures
    finally:
        mpu.destroy_model_parallel()
        if owns_distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layout", choices=("packed", "dense", "all"), default="packed"
    )
    parser.add_argument(
        "--dense-torch-conv",
        action="store_true",
        help="Isolate dense KKT/CP from the legacy native convolution adapter",
    )
    parser.add_argument("--chunk-size", type=int, choices=(64, 128), default=64)
    args = parser.parse_args()
    run_checks(args)


if __name__ == "__main__":
    main()
