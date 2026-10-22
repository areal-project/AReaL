# SPDX-License-Identifier: Apache-2.0

"""Execute engine methods on CPU without importing CUDA/MCore implementations.

Production methods are compiled from their AST unchanged; scheduling and device
kernels are stand-ins. This verifies callback contracts and normalization, not
real pipeline communication, fused kernels or MCore gradient finalization.
"""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from areal.api.loss import is_named_loss_normalizer, named_loss_normalizer
from areal.engine.core import (
    compute_microbatch_loss_weight,
    compute_total_loss_weight,
    normalize_loss,
)
from areal.engine.megatron_utils.loss import (
    compute_global_token_count,
    scale_named_loss_for_native,
)
from areal.utils.data import TRANSPORT_DUMMY_KEY


def _load_methods(path, class_name, names, namespace):
    module = ast.parse((Path(__file__).resolve().parents[1] / path).read_text())
    cls = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == names
    # Annotations are irrelevant to this CPU harness and must not import CUDA.
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), path, "exec"), namespace)
    return {name: namespace[name] for name in names}


def _namespace(monkeypatch, cp_size=1, cp_rank=0):
    # Collective math is covered separately using real CPU/Gloo processes.
    distributed = SimpleNamespace(all_reduce=lambda tensor, group: None)
    return {
        "nullcontext": nullcontext,
        "torch": torch,
        "dist": distributed,
        "TRANSPORT_DUMMY_KEY": TRANSPORT_DUMMY_KEY,
        "ChunkedLMHeadOutput": SimpleNamespace,
        "compute_microbatch_loss_weight": compute_microbatch_loss_weight,
        "compute_total_loss_weight": compute_total_loss_weight,
        "normalize_loss": normalize_loss,
        "is_named_loss_normalizer": is_named_loss_normalizer,
        "compute_global_token_count": compute_global_token_count,
        "scale_named_loss_for_native": scale_named_loss_for_native,
        "aggregate_eval_losses": lambda losses, group: torch.stack(losses).sum(),
        "tensor_container_to": lambda data, device: data,
        "record_training_batch": lambda *args: nullcontext(),
        "get_model_config": lambda model: model.config,
        "mpu": SimpleNamespace(
            get_data_parallel_group=lambda **kwargs: object(),
            get_data_parallel_world_size=lambda **kwargs: 1,
            get_context_parallel_world_size=lambda: cp_size,
            get_context_parallel_rank=lambda: cp_rank,
            is_pipeline_last_stage=lambda: True,
        ),
    }


@pytest.mark.parametrize(
    "backend,native",
    [("fsdp", False), ("archon", False), ("megatron", False), ("megatron", True)],
)
def test_engine_transport_dummy_skips_both_callbacks(backend, native, monkeypatch):
    path = (
        f"areal/engine/{backend}_engine.py"
        if backend != "archon"
        else "areal/experimental/engine/archon_engine.py"
    )
    class_name = {
        "fsdp": "FSDPEngine",
        "archon": "ArchonEngine",
        "megatron": "MegatronEngine",
    }[backend]
    methods = _load_methods(
        path, class_name, {"_compute_logprobs_and_loss"}, _namespace(monkeypatch)
    )

    @named_loss_normalizer(names=("policy",))
    def normalizer(batch):
        pytest.fail("Normalizer called on transport dummy")

    def loss_fn(*args, **kwargs):
        pytest.fail("Loss called on transport dummy")

    logits = torch.tensor([1.0], dtype=torch.float16, requires_grad=True)
    data = {TRANSPORT_DUMMY_KEY: True, "input_ids": torch.ones(1)}
    method = methods["_compute_logprobs_and_loss"]
    if backend == "megatron":
        # Chunked outputs must retain their graph connection too.
        output = SimpleNamespace(logprobs=logits)
        result = method(
            SimpleNamespace(),
            output,
            data,
            loss_fn,
            normalizer,
            {"policy": torch.tensor(0.0)},
            per_token_loss=native,
            total_token_count=torch.tensor(100000) if native else None,
        )
        loss = result[0] if native else result
        if native:
            assert result[1] == 0
    else:
        loss = method(
            SimpleNamespace(),
            logits,
            SimpleNamespace(mb_input=data),
            loss_fn,
            normalizer,
            {"policy": torch.tensor(0.0)},
        )
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=0)


@pytest.mark.parametrize(
    "backend,is_critic",
    [
        ("fsdp", True),
        ("fsdp", False),
        ("archon", True),
        ("archon", False),
        ("megatron", True),
    ],
)
@pytest.mark.parametrize("multiplier", [1, 2])
def test_engine_named_terms_use_independent_denominators(
    backend, is_critic, multiplier, monkeypatch
):
    path = (
        f"areal/engine/{backend}_engine.py"
        if backend != "archon"
        else "areal/experimental/engine/archon_engine.py"
    )
    class_name = {
        "fsdp": "FSDPEngine",
        "archon": "ArchonEngine",
        "megatron": "MegatronEngine",
    }[backend]
    method = _load_methods(
        path, class_name, {"_compute_logprobs_and_loss"}, _namespace(monkeypatch)
    )["_compute_logprobs_and_loss"]
    engine = SimpleNamespace(
        config=SimpleNamespace(is_critic=is_critic),
        enable_tree_training=False,
        _compute_values=lambda x, _: x,
        _gather_critic_output=lambda x, _: x.squeeze(-1),
    )

    engine._compute_logprobs_entropy = lambda x, *_: (x.squeeze(-1), x.squeeze(-1) * 0)
    engine._get_vocab_min_max_logits = lambda x, *_: (
        x.squeeze(-1) * 0,
        x.squeeze(-1) * 0,
    )
    engine._gather_actor_train_outputs = lambda x, _: (
        x.squeeze(-1),
        x.squeeze(-1) * 0,
        x.squeeze(-1) * 0,
        x.squeeze(-1) * 0,
    )

    @named_loss_normalizer(names=("policy", "aux"))
    def normalizer(batch):
        return {"policy": torch.tensor(0.25), "aux": torch.tensor(2.0)}

    def loss_fn(values, *args, **kwargs):
        return {"policy": values.square().sum(), "aux": (values - 1).square().sum()}

    logits = torch.tensor([[0.3], [0.7]], requires_grad=True)
    totals = {"policy": torch.tensor(0.5), "aux": torch.tensor(4.0)}
    ctx = SimpleNamespace(
        mb_input={}, model_inputs={}, ulysses_pad_size=0, pad_length=0
    )
    actual = method(
        engine,
        logits,
        {} if backend == "megatron" else ctx,
        loss_fn,
        normalizer,
        totals,
        loss_multiplier=multiplier,
    )
    expected = multiplier * (
        logits.square().sum() / 0.5 + (logits - 1).square().sum() / 4
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("count", [0, 1, 3, 17])
def test_native_named_cp_counts_and_gradient_preserve_independent_mass(cp_size, count):
    theta = torch.tensor(0.7, requires_grad=True)
    total = torch.tensor(count)
    # Replicated full-sequence numerator; masses were reduced over DP+CP.
    per_rank = theta.square() / cp_size
    outputs = [
        scale_named_loss_for_native(per_rank, total, total, cp_size, rank)
        for rank in range(cp_size)
    ]
    assert sum(tokens for _, tokens in outputs) == count
    actual = sum(value for value, _ in outputs) / max(count, 1)
    torch.testing.assert_close(actual, theta.square(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        torch.autograd.grad(actual, theta)[0], 2 * theta.detach(), rtol=1e-6, atol=1e-7
    )


def test_native_named_missing_token_mask_is_explicit_error():
    with pytest.raises(ValueError, match="explicit loss_mask"):
        compute_global_token_count([{}], object(), "cpu")


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("dummy_count", [1, 3])
def test_megatron_train_and_eval_preserve_scalar_and_named_contracts(
    native, named, cp_size, dummy_count, monkeypatch
):
    def all_reduce(tensor, group):
        if group == "dp-cp":
            tensor.mul_(cp_size)

    monkeypatch.setattr("areal.engine.core.train_engine.dist.all_reduce", all_reduce)
    monkeypatch.setattr("areal.engine.megatron_utils.loss.dist.all_reduce", all_reduce)
    namespace = _namespace(monkeypatch, cp_size=cp_size)
    namespace["mpu"].get_data_parallel_group = lambda **kwargs: (
        "dp-cp" if kwargs.get("with_context_parallel") else "dp"
    )
    namespace["aggregate_eval_losses"] = (
        lambda losses, group: torch.stack(losses).sum() * cp_size
    )
    methods = _load_methods(
        "areal/engine/megatron_engine.py",
        "MegatronEngine",
        {
            "train_batch",
            "eval_batch",
            "_compute_logprobs_and_loss",
            "_per_token_count",
            "_build_per_token_loss_output",
            "_per_token_loss_multiplier",
        },
        namespace,
    )

    class Microbatches:
        mbs = [
            {
                "x": torch.tensor([[1.0]]),
                "loss_mask": torch.ones(1),
                "mass": torch.tensor(0.25),
            },
            *[
                {TRANSPORT_DUMMY_KEY: True, "input_ids": torch.ones(1)}
                for _ in range(dummy_count)
            ],
            {
                "x": torch.tensor([[2.0]]),
                "loss_mask": torch.ones(1),
                "mass": torch.tensor(0.75),
            },
        ]

        def __len__(self):
            return len(self.mbs)

    engine = SimpleNamespace(
        _weight_residency=None,
        hf_config=None,
        device="cpu",
        model=[
            SimpleNamespace(config=SimpleNamespace(calculate_per_token_loss=native))
        ],
        config=SimpleNamespace(is_critic=True),
        enable_tree_training=False,
    )
    for name, method in methods.items():
        setattr(
            engine,
            name,
            method if name == "_per_token_count" else MethodType(method, engine),
        )
    parameter = torch.nn.Parameter(torch.tensor(0.7))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    engine._ensure_ready = lambda: None
    engine._normalize_batch_input = lambda data: (data, None)
    engine._prepare_mb_list = lambda *args, **kwargs: Microbatches()
    engine.optimizer_zero_grad = optimizer.zero_grad
    engine._collect_mtp_loss = lambda _: None
    engine.optimizer_step = lambda: (optimizer.step() or {})

    def forward_backward(mbs, callback, forward_only):
        losses = []
        counts = []
        for mb in mbs.mbs:
            output = parameter * mb.get("x", torch.ones(1, 1))
            result = callback(output, mb)
            if isinstance(result, tuple):
                value, count = result
                losses.append(value)
                counts.append(count)
                if forward_only:
                    # Native schedule may mutate its tensor; eval metric must be cloned.
                    value.mul_(7)
            else:
                if forward_only:
                    # MCore's two-element callback output is scaled in-place
                    # regardless of calculate_per_token_loss.
                    result.mul_(cp_size).div_(len(mbs))
                losses.append(result)
        if not forward_only:
            # Simulate CP SUM of identical full-sequence objectives. For these
            # one-token rows rank 0 owns all count remainders; other CP ranks
            # contribute zero counts, but the same numerator and gradient.
            value = sum(losses) * cp_size
            value = value / max(sum(counts), 1) if native else value / len(mbs)
            value.backward()

    engine.forward_backward_batch = forward_backward

    def normalizer(batch):
        return {"policy": batch["mass"]} if named else batch["mass"]

    if named:
        normalizer = named_loss_normalizer(names=("policy",))(normalizer)

    def loss_fn(values, batch):
        value = values.square().sum()
        return {"policy": value * batch["mass"]} if named else value

    expected_before = 0.7**2 * (0.25 + 4 * 0.75)
    actual_eval = engine.eval_batch({}, loss_fn, normalizer)
    torch.testing.assert_close(
        actual_eval, torch.tensor(expected_before), rtol=1e-6, atol=1e-7
    )
    engine.train_batch({}, loss_fn, normalizer)
    expected_after = 0.7 - 0.1 * (2 * 0.7 * (0.25 + 4 * 0.75))
    torch.testing.assert_close(
        parameter.detach(), torch.tensor(expected_after), rtol=1e-6, atol=1e-7
    )
