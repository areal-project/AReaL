# SPDX-License-Identifier: Apache-2.0
"""Distributed engine-loss parity, including sequence and vocabulary partitioning."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _parallel_worker(rank, tp, cp, rendezvous):
    world = tp * cp
    from tests.test_score_centering_vocab_parallel import _dense

    from areal.engine.fsdp_engine import FSDPEngine, FSDPTrainContext

    # PyTorch's Gloo all-gather backward uses global scatter ranks and cannot
    # handle noncontiguous subgroups. Exercise mixed TP/CP with real NCCL.
    if world > 2:
        torch.cuda.set_device(rank)
        torch.set_default_device(torch.device("cuda", rank))
    dist.init_process_group(
        "nccl" if world > 2 else "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        tp_groups = [
            dist.new_group(list(range(c * tp, (c + 1) * tp))) for c in range(cp)
        ]
        cp_groups = [dist.new_group([t + c * tp for c in range(cp)]) for t in range(tp)]
        t, c = rank % tp, rank // tp
        tp_group, cp_group = tp_groups[c], cp_groups[t]
        for cap in (None, 2.0):
            for sparse_mask in (False, True):
                torch.manual_seed(718)
                n, vocab = 13, 24
                logits = torch.randn(n, vocab, dtype=torch.float64)
                tokens = torch.arange(n) % vocab
                ids = torch.stack([tokens, (tokens + 11) % vocab], -1)
                q = torch.full((n, 2), -3.0, dtype=torch.float64)
                mask = torch.zeros(n, dtype=torch.bool)
                mask[: 1 if sparse_mask else 11] = True
                inputs = dict(
                    input_ids=tokens,
                    loss_mask=mask,
                    sampler_head_ids=ids,
                    sampler_head_logprobs=q,
                    cu_seqlens=torch.tensor([0, 5, 13], dtype=torch.int32),
                    max_seqlen=8,
                )
                oracle = logits.clone().requires_grad_()
                lp, correction = _dense(oracle, tokens.roll(-1), ids, q, cap)
                # Vary sampled weights independently of the head correction.
                weight = 1.0 if cap is None else (lp.detach() + 4).exp().clamp_max(cap)
                expected = ((correction - weight * lp) * mask).sum() / mask.sum()
                expected.backward()
                config = SimpleNamespace(
                    is_critic=False,
                    use_score_centering=True,
                    logprobs_chunk_size=3,
                    rejection_sampling=None,
                )
                if cap is not None:
                    from areal.api.cli_args import RejectionSamplingConfig

                    config.rejection_sampling = RejectionSamplingConfig(
                        action="clamp", upper=cap
                    )
                full = torch.cat([logits, torch.zeros(3, vocab, dtype=logits.dtype)])
                local = full.chunk(cp)[c]
                labels = torch.cat(
                    [tokens.roll(-1), torch.zeros(3, dtype=torch.long)]
                ).chunk(cp)[c]
                engine = FSDPEngine.__new__(FSDPEngine)
                engine.config, engine.enable_tree_training = config, False
                engine.model_config = SimpleNamespace(vocab_size=vocab)
                engine.parallel_helper = SimpleNamespace(
                    tp_size=tp, sp_size=cp, tp_group=tp_group
                )
                engine.sp_group = cp_group
                context = FSDPTrainContext(
                    model_inputs=dict(input_ids=labels, rolled_input_ids=labels),
                    mb_input=inputs,
                    pad_length=2,
                    ulysses_pad_size=1,
                )
                expected_grad = torch.cat(
                    [oracle.grad, torch.zeros(3, vocab, dtype=logits.dtype)]
                ).chunk(cp)[c]
                sl = slice(t * vocab // tp, (t + 1) * vocab // tp)
                local = local[:, sl].clone().requires_grad_()

                def loss_fn(lp, entropy, data, *, sc_correction, **kwargs):
                    torch.testing.assert_close(
                        lp[mask],
                        oracle.log_softmax(-1)
                        .gather(-1, tokens.roll(-1)[:, None])
                        .squeeze(-1)[mask],
                        atol=1e-10,
                        rtol=1e-10,
                    )
                    w = 1.0 if cap is None else (lp.detach() + 4).exp().clamp_max(cap)
                    return (
                        (sc_correction - w * lp) * data["loss_mask"]
                    ).sum() / mask.sum()

                loss = engine._compute_logprobs_and_loss(
                    local, context, loss_fn, lambda _: mask.sum(), mask.sum() * cp
                )
                torch.testing.assert_close(loss, expected / cp, atol=1e-10, rtol=1e-10)
                loss.backward()
                torch.testing.assert_close(
                    local.grad, expected_grad[:, sl], atol=1e-10, rtol=1e-10
                )
                # Adam state/update parity catches errors beyond finite gradients.
                ref = local.detach().clone().requires_grad_()
                ref.grad = expected_grad[:, sl].clone()
                torch.optim.Adam([local], lr=0.01).step()
                torch.optim.Adam([ref], lr=0.01).step()
                torch.testing.assert_close(local, ref, atol=1e-10, rtol=1e-10)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.parametrize("tp,cp", [(2, 1), (1, 2), (2, 2)])
def test_score_centering_parallel_engine_gradients_and_adam(tmp_path, tp, cp):
    """TP2, CP2 and TP2*CP2 reproduce dense gradients and Adam updates."""
    world = tp * cp
    if world > 2 and torch.cuda.device_count() < world:
        pytest.skip("Mixed TP/CP collectives require four CUDA devices")
    mp.spawn(
        _parallel_worker,
        args=(tp, cp, str(tmp_path / "rendezvous")),
        nprocs=world,
        join=True,
    )


@pytest.mark.slow
@pytest.mark.parametrize("tp,cp", [(2, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("tis_cap", [None, 2.0])
def test_fsdp_score_centering_local_checkpoint_update(tp, cp, tis_cap):
    """Opt-in real-model update using AREAL_TEST_MODEL, without downloading weights."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    model = os.environ.get("AREAL_TEST_MODEL")
    if not model:
        pytest.skip("Set AREAL_TEST_MODEL to a local Qwen3 checkpoint")
    if torch.cuda.device_count() < tp * cp:
        pytest.skip("Insufficient CUDA devices for this TP/CP layout")
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={tp * cp}",
        str(root / "tests/torchrun/run_fsdp_score_centering.py"),
        "--model",
        model,
        "--tp",
        str(tp),
        "--cp",
        str(cp),
    ]
    if tis_cap is not None:
        cmd += ["--tis-cap", str(tis_cap)]
    result = subprocess.run(cmd, cwd=root, text=True, capture_output=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS fsdp:" in result.stdout + result.stderr
