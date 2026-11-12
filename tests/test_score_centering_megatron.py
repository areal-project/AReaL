# SPDX-License-Identifier: Apache-2.0
"""Megatron SC configuration, packed engine loss, and distributed gradient checks."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_score_centering_vocab_parallel import _dense

from areal.api.cli_args import MegatronEngineConfig, PPOActorConfig


def test_megatron_score_centering_config_accepts_tp():
    PPOActorConfig(
        backend="megatron:d1t2", use_score_centering=True, recompute_logprob=False
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(enable_chunked_logits=True, lm_head_loss_chunk_size=16),
        dict(enable_mtp_training=True, enable_mtp=True),
        dict(entropy_requires_grad=True),
    ],
)
def test_megatron_score_centering_config_rejects_unsupported_modes(options):
    with pytest.raises(ValueError, match="score centering"):
        PPOActorConfig(
            backend="megatron:d1t2",
            use_score_centering=True,
            recompute_logprob=False,
            megatron=MegatronEngineConfig(**options),
        )


@pytest.mark.parametrize("dummy", [False, True])
def test_megatron_score_centering_loss_hook(dummy, monkeypatch):
    from types import SimpleNamespace

    from tests.test_named_loss_engines import _load_methods, _namespace

    namespace = _namespace(monkeypatch)
    namespace["mpu"].get_tensor_model_parallel_world_size = lambda: 1
    namespace["lang_config"] = lambda config: config
    method = _load_methods(
        "areal/engine/megatron_engine.py",
        "MegatronEngine",
        {"_compute_logprobs_and_loss"},
        namespace,
    )["_compute_logprobs_and_loss"]
    engine = SimpleNamespace(
        config=SimpleNamespace(
            is_critic=False, use_score_centering=True, logprobs_chunk_size=2
        ),
        enable_tree_training=False,
        hf_config=SimpleNamespace(vocab_size=17),
    )
    x = torch.randn(5, 20, dtype=torch.float32, requires_grad=True)
    oracle = x.detach().clone().requires_grad_()
    inputs = dict(
        input_ids=torch.tensor([2, 5, 7, 3, 8]),
        sampler_head_ids=torch.tensor([[0, 6, 12]] * 5),
        sampler_head_logprobs=torch.full((5, 3), -3.0),
        loss_mask=torch.tensor([0, 1, 1, 0, 1]),
    )
    if dummy:
        inputs[namespace["TRANSPORT_DUMMY_KEY"]] = True
    calls = []

    def loss_fn(lp, entropy, data, *, sc_correction, **kwargs):
        calls.append(True)
        return ((sc_correction - lp) * data["loss_mask"]).sum() / 3

    loss = method(
        engine, x, inputs, loss_fn, lambda _: torch.tensor(3.0), torch.tensor(6.0)
    )
    if dummy:
        assert not calls
        loss.backward()
        torch.testing.assert_close(x.grad, torch.zeros_like(x), atol=0, rtol=0)
    else:
        lp, c = _dense(
            oracle[:, :17],
            inputs["input_ids"].roll(-1),
            inputs["sampler_head_ids"],
            inputs["sampler_head_logprobs"],
        )
        expected = ((c - lp) * inputs["loss_mask"]).sum() / 6
        torch.testing.assert_close(loss, expected, atol=2e-06, rtol=2e-06)
        loss.backward()
        expected.backward()
        torch.testing.assert_close(x.grad, oracle.grad, atol=2e-06, rtol=2e-06)


def _parallel_worker(rank, tp, cp, rendezvous):
    world = tp * cp
    from megatron.core import parallel_state as mpu

    from tests.test_score_centering_vocab_parallel import _dense

    from areal.engine.megatron_engine import MegatronEngine
    from areal.engine.megatron_utils.packed_context_parallel import (
        split_packed_seqs_for_context_parallel,
    )
    from areal.utils.data import pad_packed_tensor_dict

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
        t, c = (rank % tp, rank // tp)
        tp_group, cp_group = (tp_groups[c], cp_groups[t])
        mpu.get_tensor_model_parallel_group = lambda: tp_group
        mpu.get_tensor_model_parallel_world_size = lambda: tp
        mpu.get_context_parallel_group = lambda: cp_group
        mpu.get_context_parallel_world_size = lambda: cp
        mpu.get_context_parallel_rank = lambda: c
        for cap in (None, 2.0):
            for sparse_mask in (False, True):
                torch.manual_seed(718)
                n, vocab = (13, 24)
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
                padded, padding, old_cu, _ = pad_packed_tensor_dict(
                    inputs, None, seq_align_to=8
                )
                full = torch.zeros(16, vocab, dtype=torch.float64)
                full[:5], full[8:16] = (logits[:5], logits[5:])
                labels = torch.zeros(16, dtype=torch.long)
                labels[:5], labels[8:16] = (tokens.roll(-1)[:5], tokens.roll(-1)[5:])
                cu = padded["cu_seqlens"]
                local = split_packed_seqs_for_context_parallel(full, cu)
                engine = MegatronEngine.__new__(MegatronEngine)
                engine.config, engine.enable_tree_training = (config, False)
                engine.hf_config = SimpleNamespace(vocab_size=vocab)
                cp_inputs = dict(
                    inputs,
                    _cp_local_labels=split_packed_seqs_for_context_parallel(labels, cu),
                    _cp_local_sampler_head_ids=split_packed_seqs_for_context_parallel(
                        padded["sampler_head_ids"], cu
                    ),
                    _cp_local_sampler_head_logprobs=split_packed_seqs_for_context_parallel(
                        padded["sampler_head_logprobs"], cu
                    ),
                    _cp_padded_cu_seqlens=cu,
                    _cp_padding_length=padding,
                    _cp_old_cu_seqlens=old_cu,
                )
                grad_full = torch.zeros_like(full)
                grad_full[:5], grad_full[8:16] = (oracle.grad[:5], oracle.grad[5:])
                expected_grad = split_packed_seqs_for_context_parallel(grad_full, cu)
                context = cp_inputs
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
                native_logits = local.detach().clone().requires_grad_()
                numerator, count = engine._compute_logprobs_and_loss(
                    native_logits,
                    context,
                    loss_fn,
                    lambda _: mask.sum(),
                    None,
                    per_token_loss=True,
                )
                dist.all_reduce(count, group=cp_group)
                torch.testing.assert_close(count, mask.sum(), atol=0, rtol=0)
                (numerator / count).backward()
                torch.testing.assert_close(
                    native_logits.grad, expected_grad[:, sl], atol=1e-10, rtol=1e-10
                )
                ref = local.detach().clone().requires_grad_()
                ref.grad = expected_grad[:, sl].clone()
                torch.optim.Adam([local], lr=0.01).step()
                torch.optim.Adam([ref], lr=0.01).step()
                torch.testing.assert_close(local, ref, atol=1e-10, rtol=1e-10)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.parametrize("tp,cp", [(2, 1), (1, 2), (1, 4), (2, 2)])
def test_megatron_score_centering_parallel_gradients_and_adam(tmp_path, tp, cp):
    """TP2, CP2, CP4 and TP2*CP2 reproduce dense loss, logits gradients and Adam updates."""
    world = tp * cp
    pytest.importorskip("megatron.core")
    if world > 2 and torch.cuda.device_count() < world:
        pytest.skip("Mixed TP/CP collectives require four CUDA devices")
    mp.spawn(
        _parallel_worker,
        args=(tp, cp, str(tmp_path / "rendezvous")),
        nprocs=world,
        join=True,
    )


@pytest.mark.parametrize("with_heads", [False, True])
def test_score_centering_cp_forward_with_optional_sampler_heads(
    monkeypatch, with_heads
):
    """CP splits available heads but keeps ordinary head-free forwards usable."""
    pytest.importorskip("megatron.core")
    from areal.engine import megatron_engine as module
    from areal.utils.data import MicroBatchList, MicroBatchSpec

    row = {
        "input_ids": torch.tensor([10, 11, 12, 13]),
        "loss_mask": torch.tensor([False, True, True, False]),
        "policy_version_mask": torch.tensor([False, False, True, False]),
        "mtp_version_mask": torch.tensor([False, False, True, True]),
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "max_seqlen": 4,
    }
    if with_heads:
        row["sampler_head_ids"] = torch.arange(8).reshape(4, 2)
        row["sampler_head_logprobs"] = torch.full((4, 2), -3.0)
    batch = MicroBatchList(
        data={},
        mb_spec=MicroBatchSpec(),
        mbs=[row],
        padded_mbs=[dict(row)],
        group_lens=[4],
        padding_lengths=[0],
        padded_to_lengths=[4],
    )
    engine = module.MegatronEngine.__new__(module.MegatronEngine)
    engine._ensure_ready = lambda: None
    engine._r3_enabled = False
    engine._r3_pending_routed_experts = None
    engine._r3_pending_valid = None
    engine.tf_config = SimpleNamespace(num_moe_experts=None)
    engine.mcore_config = SimpleNamespace(
        enable_mtp_training=False,
        enable_chunked_logits=False,
        lm_head_loss_chunk_size=0,
    )
    engine.config = SimpleNamespace(is_critic=False, use_score_centering=True)
    engine.process_group_initialized = True
    engine._cpu_group = object()
    engine.is_vision_model = False
    engine.enable_tree_training = False
    engine.device = "cpu"
    engine.dtype = torch.bfloat16
    engine.model = [object()]
    engine.bridge_cls = "megatron-bridge"
    engine.use_padded_seq = False
    engine.use_model_packed_seq = False

    seen = []

    def schedule(**kwargs):
        output, callback = kwargs["forward_step_func"](
            kwargs["data_iterator"], kwargs["model"]
        )
        callback(output)

    def forward(_model, input_, **_kwargs):
        assert "policy_version_mask" in input_
        return torch.zeros(2, 17)

    def process_output(output, inputs):
        seen.append(True)
        torch.testing.assert_close(
            inputs["_cp_local_labels"], torch.tensor([11, 10]), atol=0, rtol=0
        )
        torch.testing.assert_close(
            inputs["policy_version_mask"], row["policy_version_mask"], atol=0, rtol=0
        )
        for key in ("sampler_head_ids", "sampler_head_logprobs"):
            assert (f"_cp_local_{key}" in inputs) is with_heads
            if with_heads:
                torch.testing.assert_close(
                    inputs[f"_cp_local_{key}"], row[key][[0, 3]], atol=0, rtol=0
                )

    monkeypatch.setattr(module, "get_forward_backward_func", lambda: schedule)
    monkeypatch.setattr(
        module, "validate_transport_padding", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(module, "packed_context_parallel_forward", forward)
    monkeypatch.setattr(module.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(module.mpu, "is_pipeline_last_stage", lambda **_kwargs: True)

    monkeypatch.setattr(module.mpu, "get_context_parallel_rank", lambda: 0)
    engine.forward_backward_batch(
        batch, process_output, forward_only=True, gather_cp_output=False
    )
    assert seen == [True]


@pytest.mark.slow
@pytest.mark.parametrize(
    "tp,cp,pp,tis_cap",
    [
        (1, 1, 1, None),
        (2, 1, 1, 2.0),
        (1, 4, 1, 2.0),
        (1, 1, 2, None),
        (2, 2, 2, None),
        (2, 2, 2, 2.0),
    ],
)
def test_megatron_score_centering_local_checkpoint_update(tp, cp, pp, tis_cap):
    """Opt-in packed Qwen3 forward and actual optimizer update, without downloads."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    model = os.environ.get("AREAL_TEST_MODEL")
    if not model:
        pytest.skip("Set AREAL_TEST_MODEL to a local Qwen3 checkpoint")
    world = tp * cp * pp
    if torch.cuda.device_count() < world:
        pytest.skip("Insufficient CUDA devices for this TP/CP/PP layout")
    pytest.importorskip("megatron.core")
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world}",
        str(root / "tests/torchrun/run_megatron_score_centering.py"),
        "--model",
        model,
        "--tp",
        str(tp),
        "--cp",
        str(cp),
        "--pp",
        str(pp),
    ]
    if tis_cap is not None:
        cmd += ["--tis-cap", str(tis_cap)]
    env = dict(
        os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        cmd, cwd=root, env=env, text=True, capture_output=True, timeout=600
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "'result': 'PASS'" in result.stdout


@pytest.mark.parametrize("cap", [None, 2.0])
@pytest.mark.parametrize("all_expired", [False, True])
@pytest.mark.parametrize("mode", ["token", "sequence", "prompt", "constant"])
def test_score_centering_stale_tokens_keep_original_normalization(
    cap, all_expired, mode, monkeypatch
):
    """New main's stale-token filter masks both SC terms without renormalizing."""
    from areal.api.cli_args import RejectionSamplingConfig
    from areal.trainer.ppo.actor import grpo_loss_fn
    from areal.utils.functional.loss_aggregation import (
        ConstantLength,
        PromptMean,
        SequenceMean,
        TokenMean,
    )
    from areal.utils.stats_tracker import DistributedStatsTracker

    monkeypatch.setattr(
        "areal.trainer.ppo.actor.stats_tracker", DistributedStatsTracker()
    )
    lp = torch.tensor([[-1.0, -2.0, -3.0], [-2.0, -3.0, -4.0]], requires_grad=True)
    correction = torch.full_like(lp, 0.3, requires_grad=True)
    original = torch.tensor([[True, True, True], [True, True, False]])
    active = torch.tensor([[True, False, True], [False, True, False]])
    if all_expired:
        active.zero_()
    advantages = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 2.0, 0.0]])
    behavior = lp.detach() - 1.0
    reductions = {
        "token": TokenMean(original),
        "sequence": SequenceMean(original),
        "prompt": PromptMean(original.float() / 5, step_scale=1.0, engine_weight=2.0),
        "constant": ConstantLength(3.0, original),
    }
    reduction = reductions[mode]
    expected_normalizer = reduction.normalizer().clone()
    loss = grpo_loss_fn(
        lp,
        torch.zeros_like(lp),
        dict(
            logprobs=behavior,
            prox_logp=lp.detach(),
            advantages=advantages,
            loss_mask=original,
            policy_version_mask=active,
        ),
        eps_clip=0.2,
        eps_clip_higher=None,
        c_clip=None,
        use_score_centering=True,
        sc_correction=correction,
        rejection_sampling=None
        if cap is None
        else RejectionSamplingConfig(action="clamp", upper=cap),
        loss_step=SimpleNamespace(bind=lambda _: reduction),
    )
    weight = 1.0 if cap is None else (lp.detach() - behavior).exp().clamp_max(cap)
    expected = reduction.aggregate(
        advantages * (correction - weight * lp), original & active
    )
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-6)
    actual_grads = torch.autograd.grad(loss, (lp, correction), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (lp, correction))
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            actual[~active], torch.zeros_like(actual[~active]), atol=0, rtol=0
        )
    torch.testing.assert_close(
        reduction.normalizer(), expected_normalizer, atol=0, rtol=0
    )
    assert torch.isfinite(loss)
    if all_expired:
        assert loss.item() == 0


@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("alignment,target", [(None, 16), (8, None), (8, 32)])
def test_megatron_sampler_heads_survive_packed_padding(width, alignment, target):
    """Head rows keep their token alignment under sequence and tail padding."""
    from areal.utils.data import pad_packed_tensor_dict, unpad_logits

    tokens = torch.arange(13)
    ids = torch.arange(13 * width).reshape(13, width)
    logprobs = -(ids.float() + 1)
    inputs = dict(
        input_ids=tokens,
        sampler_head_ids=ids,
        sampler_head_logprobs=logprobs,
        cu_seqlens=torch.tensor([0, 5, 13], dtype=torch.int32),
        max_seqlen=8,
    )
    padded, tail_length, old_cu, _ = pad_packed_tensor_dict(
        inputs, target, seq_align_to=alignment
    )
    cu = padded["cu_seqlens"]
    valid = torch.zeros(padded["input_ids"].shape[0], dtype=torch.bool)
    for i in range(2):
        valid[cu[i] : cu[i] + old_cu[i + 1] - old_cu[i]] = True
    for key in ("sampler_head_ids", "sampler_head_logprobs"):
        actual = padded[key]
        assert actual.shape == (len(valid), width)
        torch.testing.assert_close(actual[valid], inputs[key], atol=0, rtol=0)
        fill = 0 if key == "sampler_head_ids" else float("-inf")
        torch.testing.assert_close(
            actual[~valid], torch.full_like(actual[~valid], fill), atol=0, rtol=0
        )
        restored = unpad_logits(actual, tail_length, cu, old_cu)
        torch.testing.assert_close(restored, inputs[key], atol=0, rtol=0)
