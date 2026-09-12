# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from areal.api.cli_args import MicroBatchSpec
from areal.engine.fsdp_engine import FSDPEngine, FSDPTrainContext
from areal.utils.data import (
    pack_tensor_dict,
    pad_mb_list,
    split_padded_tensor_dict_into_mb_list,
    unsqueeze_mb_list,
)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.calls = 0

    def forward(self, input_ids):
        self.calls += 1
        return SimpleNamespace(logits=input_ids.float() * self.weight)


class _LoopHarness:
    enable_tree_training = False

    def __init__(self, model):
        self.model = model
        self.cpu_group = dist.group.WORLD

    def _prepare_mb_inputs(self, item):
        inputs = {"input_ids": item.padded_mb["input_ids"]}
        return inputs, FSDPTrainContext(inputs, item.orig_mb)


def _run_uneven_microbatches(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        # Rank 0 cannot be repartitioned into rank 1's three nonempty batches.
        ids = torch.arange(1, (1 if rank == 0 else 3) + 1).reshape(-1, 1)
        data = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        mb_list = split_padded_tensor_dict_into_mb_list(
            data, MicroBatchSpec(n_mbs=1, max_tokens_per_mb=1), sync_mbs=False
        )
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
        mb_list = unsqueeze_mb_list(pad_mb_list(mb_list))
        model = DistributedDataParallel(_TinyModel())
        engine = _LoopHarness(model)

        for forward_only in (True, False):
            outputs = []
            model.module.calls = 0
            model.zero_grad()

            def process_output(logits, ctx):
                outputs.append(logits[0].detach())
                return logits.sum()

            with torch.set_grad_enabled(not forward_only):
                FSDPEngine.forward_backward_batch(
                    engine, mb_list, process_output, forward_only=forward_only
                )

            assert model.module.calls == 3
            assert len(outputs) == len(ids)
            torch.testing.assert_close(
                torch.stack(outputs), ids.flatten().float() * 2, rtol=0, atol=0
            )
            if not forward_only:
                # DDP average of real gradients: (1 + (1 + 2 + 3)) / 2.
                torch.testing.assert_close(
                    model.module.weight.grad, torch.tensor(3.5), rtol=0, atol=0
                )
            assert len(mb_list) == len(ids)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
def test_uneven_microbatches_preserve_outputs_and_gradients(tmp_path):
    """Exercise the FSDP execution loop with real CPU gradient collectives."""
    mp.spawn(
        _run_uneven_microbatches,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        join=True,
    )
