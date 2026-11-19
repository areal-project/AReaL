# SPDX-License-Identifier: Apache-2.0

"""CPU regression tests for the engine's Megatron loss callback contracts."""

import functools
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tests.test_megatron_bridge_config import _load_function

from areal.engine.core.train_engine import _gradient_accumulation

ENGINE_SOURCE = "areal/engine/megatron_engine.py"


class _MicroBatches(list):
    @property
    def mbs(self):
        return self

    def to(self, device):
        return self


def _bind_per_token_loss_methods(engine, namespace):
    """Bind the scalar normalization helpers on the lightweight engine fixture."""
    namespace.setdefault("ChunkedLMHeadOutput", type("ChunkedLMHeadOutput", (), {}))
    engine._per_token_count = _load_function(
        ENGINE_SOURCE, "_per_token_count", namespace
    ).__func__
    engine._compute_loss_totals = functools.partial(
        _load_function(ENGINE_SOURCE, "_compute_loss_totals", namespace), engine
    )


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("cp_size", [1, 8])
@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("n_mbs", [1, 3])
@pytest.mark.parametrize("loss_scale", [1.0, 128.0])
@pytest.mark.parametrize("aggregation", ["token-mean", "traj-mean", "prompt-mean"])
def test_train_gradient_matches_global_mean(
    per_token, cp_size, dp_size, n_mbs, loss_scale, aggregation
):
    """Unequal and empty microbatches retain one global normalization."""
    parameter = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    batches = [
        _MicroBatches(
            {
                "weight": torch.tensor(
                    float(w)
                    if aggregation == "token-mean"
                    else (i + 1) / (16 + rank)
                    if w
                    else 0.0
                ),
                "loss_mask": torch.ones(w, dtype=torch.bool),
                "target": float(rank + i),
            }
            for i, w in enumerate([7] if n_mbs == 1 else [3, 0, 4 + rank])
        )
        for rank in range(dp_size)
    ]

    def weight_fn(inputs):
        return inputs["weight"]

    total_weight = sum(weight_fn(mb) for batch in batches for mb in batch)
    total_tokens = sum(
        mb["loss_mask"].count_nonzero() for batch in batches for mb in batch
    )
    reference = (
        sum(
            (parameter - mb["target"]).square() * weight_fn(mb)
            for batch in batches
            for mb in batch
        )
        / total_weight
    )
    reference_grad = torch.autograd.grad(reference, parameter)[0]
    scheduled_losses, counts = [], []

    for batch in batches:
        for cp_rank in range(cp_size):
            configs = [
                SimpleNamespace(
                    calculate_per_token_loss=per_token, grad_scale_func=None
                )
                for _ in range(2)
            ]
            namespace = {
                "torch": torch,
                "_gradient_accumulation": _gradient_accumulation,
                "nullcontext": nullcontext,
                "dist": SimpleNamespace(
                    all_reduce=lambda totals, group: totals.copy_(
                        torch.stack((total_weight.float(), total_tokens.float()))
                        * cp_size
                    )
                ),
                "get_model_config": lambda model: model.config,
                "mpu": SimpleNamespace(
                    get_data_parallel_group=lambda **kw: None,
                    get_data_parallel_world_size=lambda: dp_size,
                    get_context_parallel_world_size=lambda: cp_size,
                    get_context_parallel_rank=lambda: cp_rank,
                ),
                "compute_total_loss_weight": lambda mbs, fn, group: sum(
                    fn(mb) for batch in batches for mb in batch
                )
                * cp_size,
            }
            train = _load_function(ENGINE_SOURCE, "train_batch", namespace)
            compute = _load_function(
                ENGINE_SOURCE, "_compute_logprobs_and_loss", namespace
            )
            build_output = _load_function(
                ENGINE_SOURCE, "_build_per_token_loss_output", namespace
            )
            set_scale = _load_function(
                ENGINE_SOURCE, "_set_optimizer_grad_scale_func", namespace
            )
            engine = SimpleNamespace(
                mcore_config=SimpleNamespace(enable_mtp_training=False),
                _ensure_ready=lambda: None,
                optimizer_zero_grad=lambda: None,
                optimizer_step=lambda: {},
                _normalize_batch_input=lambda inputs: (inputs, None),
                _prepare_mb_list=lambda inputs: batch,
                model=[SimpleNamespace(config=config) for config in configs],
                # Bridge can mutate the model config without changing tf_config.
                tf_config=SimpleNamespace(calculate_per_token_loss=not per_token),
                config=SimpleNamespace(is_critic=True, loss_aggregation=aggregation),
                enable_tree_training=False,
                device="cpu",
                optimizer=SimpleNamespace(
                    get_loss_scale=lambda: torch.tensor(loss_scale),
                    scale_loss=lambda loss: loss * loss_scale,
                ),
            )
            engine._compute_logprobs_and_loss = lambda *a, **kw: compute(
                engine, *a, **kw
            )
            engine._build_per_token_loss_output = lambda *a, **kw: build_output(
                engine, *a, **kw
            )
            _bind_per_token_loss_methods(engine, namespace)
            set_scale(engine)

            def forward_backward(mbs, process_output, forward_only):
                assert not forward_only
                for mb in mbs:
                    result = process_output(parameter.reshape(1), mb)
                    assert all(c.grad_scale_func is not None for c in configs)
                    if per_token:
                        loss, count = result
                        assert count.dtype == torch.int64
                        counts.append(count)
                        scheduled_losses.append(configs[0].grad_scale_func(loss))
                    else:
                        # MCore two-value callback and DDP averaging.
                        scheduled_losses.append(
                            configs[0].grad_scale_func(result)
                            * cp_size
                            / n_mbs
                            / (dp_size * cp_size)
                        )

            engine.forward_backward_batch = forward_backward
            train(
                engine,
                {},
                lambda output, mb: (output - mb["target"]).square().mean(),
                weight_fn,
            )

    loss = sum(scheduled_losses)
    if per_token:
        count = sum(counts)
        torch.testing.assert_close(count.float(), total_tokens.float(), rtol=0, atol=0)
        loss = loss / count
    gradient = torch.autograd.grad(loss / loss_scale, parameter)[0]
    torch.testing.assert_close(gradient, reference_grad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("weight", [0, 1, 7, 8, 10, 16011])
@pytest.mark.parametrize("cp_size", [1, 8])
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_per_token_output_preserves_weight_and_gradient(weight, cp_size, dtype):
    """Integer partitioning preserves odd counts, including fewer tokens than ranks."""
    loss = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
    build_output = _load_function(
        ENGINE_SOURCE,
        "_build_per_token_loss_output",
        {
            "torch": torch,
            "_gradient_accumulation": _gradient_accumulation,
            "nullcontext": nullcontext,
            "mpu": SimpleNamespace(
                get_context_parallel_world_size=lambda: cp_size,
                get_context_parallel_rank=lambda: rank,
            ),
        },
    )
    outputs = []
    for rank in range(cp_size):
        outputs.append(
            build_output(
                None,
                loss,
                torch.tensor(weight, dtype=dtype),
                1.0,
                token_count=torch.tensor(weight, dtype=dtype),
            )
        )
    numerators, counts = zip(*outputs)
    assert all(count.dtype == torch.int64 for count in counts)
    assert sum(counts) == weight
    assert max(counts) - min(counts) <= 1
    torch.testing.assert_close(sum(numerators), loss * weight, rtol=0, atol=0)
    sum(numerators).backward()
    torch.testing.assert_close(
        loss.grad, torch.tensor(float(weight), dtype=loss.dtype), rtol=0, atol=0
    )


def test_optimizer_scale_hook_handles_shared_configs_and_dynamic_scale():
    """All model chunks use the current optimizer scale without duplicate setup."""

    class ModelConfig:
        def __init__(self):
            self.assignments = 0

        @property
        def grad_scale_func(self):
            return self._scale

        @grad_scale_func.setter
        def grad_scale_func(self, value):
            self.assignments += 1
            self._scale = value

    configs = [ModelConfig(), ModelConfig()]
    scale = torch.tensor(128.0)
    optimizer = SimpleNamespace(scale_loss=lambda loss: loss * scale)
    engine = SimpleNamespace(
        mcore_config=SimpleNamespace(enable_mtp_training=False),
        optimizer=optimizer,
        model=[SimpleNamespace(config=c) for c in [configs[0], configs[1], configs[0]]],
    )
    set_scale = _load_function(
        ENGINE_SOURCE,
        "_set_optimizer_grad_scale_func",
        {"get_model_config": lambda model: model.config},
    )
    set_scale(engine)
    assert all(config.assignments == 1 for config in configs)
    scale.fill_(256)
    for config in configs:
        torch.testing.assert_close(
            config.grad_scale_func(torch.tensor(2.0)),
            torch.tensor(512.0),
            rtol=0,
            atol=0,
        )


def test_optimizer_scale_hook_without_optimizer_is_noop():
    """Reference engines can initialize without training or loss scaling."""
    set_scale = _load_function(ENGINE_SOURCE, "_set_optimizer_grad_scale_func", {})
    set_scale(SimpleNamespace(optimizer=None, model=None))


class _AuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, main_loss, aux_loss, scale):
        ctx.save_for_backward(aux_loss, scale)
        return main_loss

    @staticmethod
    def backward(ctx, grad_output):
        aux_loss, scale = ctx.saved_tensors
        return grad_output, torch.ones_like(aux_loss) * scale, None


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("loss_scale", [1.0, 128.0])
def test_main_and_auxiliary_gradients_share_optimizer_scale(per_token, loss_scale):
    """MCore seeds auxiliary gradients independently of the main-loss backward."""
    config = SimpleNamespace(grad_scale_func=None)
    engine = SimpleNamespace(
        mcore_config=SimpleNamespace(enable_mtp_training=False),
        optimizer=SimpleNamespace(scale_loss=lambda loss: loss * loss_scale),
        model=[SimpleNamespace(config=config)],
    )
    set_scale = _load_function(
        ENGINE_SOURCE,
        "_set_optimizer_grad_scale_func",
        {"get_model_config": lambda model: model.config},
    )
    set_scale(engine)
    main = torch.tensor(2.0, requires_grad=True)
    aux = torch.tensor(3.0, requires_grad=True)
    tokens = 5 if per_token else 1
    loss = _AuxiliaryLoss.apply(
        main.square() * tokens,
        0.25 * aux.square() * tokens,
        config.grad_scale_func(torch.tensor(1.0)),
    )
    config.grad_scale_func(loss).backward()
    torch.testing.assert_close(
        main.grad / (loss_scale * tokens), 2 * main.detach(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        aux.grad / (loss_scale * tokens), 0.5 * aux.detach(), rtol=0, atol=0
    )


@pytest.mark.parametrize("per_token", [False, True])
def test_train_all_zero_weight_batch_is_rejected(per_token):
    """Retain the positive global-weight guard before any forward/backward work."""
    namespace = {
        "torch": torch,
        "_gradient_accumulation": _gradient_accumulation,
        "nullcontext": nullcontext,
        "dist": SimpleNamespace(all_reduce=lambda *args, **kwargs: None),
        "mpu": SimpleNamespace(get_data_parallel_group=lambda **kwargs: None),
        "get_model_config": lambda model: model.config,
    }
    train = _load_function(ENGINE_SOURCE, "train_batch", namespace)
    engine = SimpleNamespace(
        mcore_config=SimpleNamespace(enable_mtp_training=False),
        _ensure_ready=lambda: None,
        optimizer_zero_grad=lambda: None,
        _normalize_batch_input=lambda inputs: (inputs, None),
        _prepare_mb_list=lambda inputs: _MicroBatches([{"weight": torch.tensor(0)}]),
        model=[
            SimpleNamespace(config=SimpleNamespace(calculate_per_token_loss=per_token))
        ],
        device="cpu",
    )
    _bind_per_token_loss_methods(engine, namespace)
    with pytest.raises(
        AssertionError, match="Global total loss weight must be positive"
    ):
        train(engine, {}, None, lambda mb: mb["weight"])


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("aggregation", ["traj-mean", "prompt-mean"])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("global_tokens", [0, 5])
def test_train_all_masked_rank_uses_global_valid_token_guard(
    per_token, aggregation, cp_size, global_tokens
):
    """Reject globally empty steps before routing, but allow an empty DP rank."""
    dp_group = object()
    dp_cp_group = object()
    reductions = []
    calls = []
    batches = _MicroBatches(
        [{"loss_mask": torch.zeros(3, dtype=torch.bool), "weight": torch.tensor(1.0)}]
    )

    def all_reduce(totals, group):
        assert group is dp_cp_group
        reductions.append(group)
        totals.copy_(torch.tensor([2.0, float(global_tokens)]) * cp_size)

    namespace = {
        "torch": torch,
        "_gradient_accumulation": _gradient_accumulation,
        "nullcontext": nullcontext,
        "dist": SimpleNamespace(all_reduce=all_reduce),
        "mpu": SimpleNamespace(
            get_data_parallel_group=lambda with_context_parallel=False: (
                dp_cp_group if with_context_parallel else dp_group
            ),
            get_data_parallel_world_size=lambda: 2,
            get_context_parallel_world_size=lambda: cp_size,
        ),
        "compute_total_loss_weight": lambda *args: torch.tensor(2.0 * cp_size),
        "get_model_config": lambda model: model.config,
    }
    train = _load_function(ENGINE_SOURCE, "train_batch", namespace)
    engine = SimpleNamespace(
        mcore_config=SimpleNamespace(enable_mtp_training=False),
        _ensure_ready=lambda: None,
        optimizer_zero_grad=lambda: None,
        _normalize_batch_input=lambda inputs: (inputs, None),
        _prepare_mb_list=lambda inputs: batches,
        model=[
            SimpleNamespace(config=SimpleNamespace(calculate_per_token_loss=per_token))
        ],
        config=SimpleNamespace(loss_aggregation=aggregation),
        # Real routers see input tokens even when every action mask is zero.
        forward_backward_batch=lambda *args, **kwargs: calls.append("router"),
        optimizer_step=lambda: calls.append("optimizer") or {},
        device="cpu",
    )
    _bind_per_token_loss_methods(engine, namespace)

    if global_tokens == 0:
        with pytest.raises(
            ValueError, match="Global valid token count must be positive"
        ):
            train(engine, {}, None, lambda mb: mb["weight"])
        assert calls == []
    else:
        train(engine, {}, None, lambda mb: mb["weight"])
        assert calls == ["router", "optimizer"]
    assert reductions == [dp_cp_group]


def test_valid_token_guard_keeps_reward_model_count_contract():
    """No-mask reward batches retain their callback's weight units."""
    namespace = {
        "torch": torch,
        "_gradient_accumulation": _gradient_accumulation,
        "nullcontext": nullcontext,
        "dist": SimpleNamespace(all_reduce=lambda *args, **kwargs: None),
        "mpu": SimpleNamespace(get_data_parallel_group=lambda **kwargs: None),
    }
    engine = SimpleNamespace(device="cpu")
    _bind_per_token_loss_methods(engine, namespace)
    weight, tokens = engine._compute_loss_totals(
        _MicroBatches([{}]), lambda mb: torch.tensor(2.0)
    )
    torch.testing.assert_close(tokens / weight, torch.tensor(1.0), rtol=0, atol=0)


@pytest.mark.parametrize("with_count", [False, True])
def test_forward_callback_preserves_loss_contract(with_count):
    """Per-token training emits three values; eval and forward keep two."""
    output = torch.tensor([2.0])
    mb = SimpleNamespace(
        orig_mb={}, padded_mb={}, padding_length=0, old_cu_seqlens=None
    )
    batches = _MicroBatches([mb])
    batches.max_seqlen = 1
    captured = []

    def schedule(**kwargs):
        result, callback = kwargs["forward_step_func"](
            kwargs["data_iterator"], kwargs["model"]
        )
        captured.append(callback(result))

    forward = _load_function(
        ENGINE_SOURCE,
        "forward_backward_batch",
        {
            "torch": torch,
            "_gradient_accumulation": _gradient_accumulation,
            "nullcontext": nullcontext,
            "functools": functools,
            "mpu": SimpleNamespace(
                get_context_parallel_world_size=lambda: 1,
                is_pipeline_last_stage=lambda **kw: True,
            ),
            "packed_context_parallel_forward": lambda *a, **kw: output,
            "unpad_logits": lambda output, **kw: output,
            "get_forward_backward_func": lambda: schedule,
            "trace_scope": lambda *args: nullcontext(),
            "_float16_wrapper_fp32_output": _load_function(
                ENGINE_SOURCE, "_float16_wrapper_fp32_output", {"torch": torch}
            ),
        },
    )
    engine = SimpleNamespace(
        _ensure_ready=lambda: None,
        enable_tree_training=False,
        is_vision_model=False,
        use_padded_seq=False,
        use_model_packed_seq=False,
        mcore_config=SimpleNamespace(
            enable_mtp_training=False,
            lm_head_loss_chunk_size=0,
            enable_chunked_logits=False,
        ),
        config=SimpleNamespace(is_critic=False),
        dtype=torch.float32,
        model=[SimpleNamespace()],
    )
    count = torch.tensor(3)
    forward(
        engine,
        batches,
        lambda output, inputs: (output.sum(), count) if with_count else output.sum(),
        forward_only=not with_count,
    )
    result = captured[0]
    assert len(result) == (3 if with_count else 2)
    assert result[-1] == {}
    torch.testing.assert_close(result[0], output.sum(), rtol=0, atol=0)
    if with_count:
        assert result[1] is count


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("vpp_size", [1, 2])
def test_mtp_scale_uses_physical_tokens_with_sequence_loss(cp_size, vpp_size):
    """Every VPP chunk retains MTP targets and physical-token normalization."""
    captured = []

    def packed_forward(model, inputs, **kwargs):
        captured.append(inputs["mtp_kwargs"])
        return torch.ones(1, 5)

    def schedule(*, forward_step_func, data_iterator, model, **kwargs):
        iterators = data_iterator if vpp_size > 1 else [data_iterator]
        models = model if vpp_size > 1 else [model]
        for iterator, chunk in zip(iterators, models):
            forward_step_func(iterator, chunk)

    namespace = {
        "torch": torch,
        "_gradient_accumulation": _gradient_accumulation,
        "nullcontext": nullcontext,
        "functools": functools,
        "dist": SimpleNamespace(all_reduce=lambda *args, **kwargs: None),
        "mpu": SimpleNamespace(
            get_data_parallel_group=lambda **kwargs: None,
            get_pipeline_model_parallel_group=lambda: None,
            get_context_parallel_world_size=lambda: cp_size,
            get_data_parallel_world_size=lambda **kwargs: cp_size,
            is_pipeline_last_stage=lambda **kwargs: True,
        ),
        "get_model_config": lambda model: model.config,
        "packed_context_parallel_forward": packed_forward,
        "_is_multi_modal_payload_key": lambda key: False,
        "unpad_logits": lambda output, **kwargs: output,
        "get_forward_backward_func": lambda: schedule,
        "trace_scope": lambda *args: nullcontext(),
        "_float16_wrapper_fp32_output": _load_function(
            ENGINE_SOURCE, "_float16_wrapper_fp32_output", {"torch": torch}
        ),
    }
    train = _load_function(ENGINE_SOURCE, "train_batch", namespace)
    batch = {
        "input_ids": torch.ones(1, 5, dtype=torch.long),
        "loss_mask": torch.ones(1, 5),
    }
    batches = _MicroBatches(
        [
            SimpleNamespace(
                padded_mb=batch, orig_mb=batch, old_cu_seqlens=None, padding_length=0
            )
        ]
    )
    batches.max_seqlen = 5

    engine = SimpleNamespace(
        mcore_config=SimpleNamespace(
            enable_mtp_training=True,
            lm_head_loss_chunk_size=0,
            enable_chunked_logits=False,
        ),
        config=SimpleNamespace(is_critic=False),
        dtype=torch.float32,
        _ensure_ready=lambda: None,
        optimizer_zero_grad=lambda: None,
        optimizer_step=lambda: {},
        _normalize_batch_input=lambda inputs: (inputs, None),
        _prepare_mb_list=lambda inputs: batches,
        _compute_loss_totals=lambda *args: (
            torch.tensor(float(cp_size)),
            torch.tensor(5.0 * cp_size),
        ),
        model=[
            SimpleNamespace(config=SimpleNamespace(calculate_per_token_loss=True))
            for _ in range(vpp_size)
        ],
        enable_tree_training=False,
        is_vision_model=False,
        use_padded_seq=False,
        use_model_packed_seq=False,
        device="cpu",
    )
    engine.forward_backward_batch = functools.partial(
        _load_function(ENGINE_SOURCE, "forward_backward_batch", namespace), engine
    )
    train(engine, {}, None, None)
    assert len(captured) == vpp_size
    assert all(kwargs is batch["mtp_kwargs"] for kwargs in captured)
    assert captured[0]["mtp_labels"] is batch["input_ids"]
    torch.testing.assert_close(
        captured[0]["mtp_loss_mask"], torch.tensor([[1, 1, 1, 1, 0.0]]), rtol=0, atol=0
    )
    # Three valid two-token-ahead targets, normalized by five physical tokens.
    torch.testing.assert_close(
        captured[0]["batch"]["scale"], torch.tensor(5.0 / 3), rtol=1e-6, atol=1e-7
    )
