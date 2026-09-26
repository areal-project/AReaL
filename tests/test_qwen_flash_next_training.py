# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from transformers import GenerationConfig, PretrainedConfig

from areal.engine.core.model import SequencePackingMode, resolve_sequence_packing_mode
from areal.models.mcore.mcore_bridge_adapter import (
    MCoreBridgeAdapter,
    _configure_qwen4_exp_parameters,
    qwen4_exp_optimizer_overrides,
)
from areal.models.mcore.mcore_bridge_checkpoint import (
    finalize_mcore_bridge_checkpoint,
)

LAYERS_PREFIX = "model.language_model.layers"


@pytest.mark.parametrize("topk", [1, 3, 7])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stable_qsa_topk_preserves_ties_empty_rows_and_padding(topk, device):
    from examples.swe.qwen38_flash_next.patch_sglang_qsa_topk import stable_qsa_topk

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    logits = torch.tensor(
        [[float("nan"), 4, 4, 2, float("inf")], [1, 2, 3, 4, 5], [0, 7, 2, 9, 1]],
        dtype=torch.float32,
    )
    starts, ends = [1, 2, 0], [4, 2, 5]
    expected = []
    for row, start, end in zip(logits.tolist(), starts, ends):
        selected = sorted(range(start, end), key=lambda i: (-row[i], i))[:topk]
        relative = sorted(i - start for i in selected)
        expected.append(relative + [-1] * (topk - len(relative)))

    actual = stable_qsa_topk(
        logits.to(device),
        torch.tensor(starts, device=device),
        torch.tensor(ends, device=device),
        topk,
    ).cpu()

    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0
    )


def test_stable_qsa_topk_avoids_scalar_extraction_and_nonzero():
    from examples.swe.qwen38_flash_next.patch_sglang_qsa_topk import stable_qsa_topk

    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA to profile asynchronous assertions")
    logits = torch.tensor([[3.0, 3.0, 1.0]], device="cuda")
    starts, ends = torch.tensor([0], device="cuda"), torch.tensor([3], device="cuda")
    stable_qsa_topk(logits, starts, ends, 2)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        result = stable_qsa_topk(logits, starts, ends, 2)
    operations = {event.key for event in prof.key_averages()}
    assert not operations.intersection(
        {"aten::item", "aten::_local_scalar_dense", "aten::nonzero", "aten::is_nonzero"}
    )
    torch.testing.assert_close(
        result.cpu(), torch.tensor([[0, 1]], dtype=torch.int32), rtol=0, atol=0
    )


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda_graph"])
@pytest.mark.parametrize(
    "start,end,score,error",
    [
        (-1, 1, 0.0, "Invalid row bounds"),
        (1, 0, 0.0, "Invalid row bounds"),
        (0, 2, 0.0, "Invalid row bounds"),
        (0, 1, float("nan"), "Nonfinite valid scores"),
        (0, 1, float("inf"), "Nonfinite valid scores"),
        (0, 1, -float("inf"), "Nonfinite valid scores"),
    ],
)
def test_stable_qsa_topk_rejects_invalid_inputs(device, start, end, score, error):
    from examples.swe.qwen38_flash_next.patch_sglang_qsa_topk import stable_qsa_topk

    if device == "cpu":
        with pytest.raises(ValueError, match=error):
            stable_qsa_topk(
                torch.tensor([[score]]), torch.tensor([start]), torch.tensor([end]), 1
            )
        return
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    # A device assertion poisons its CUDA context: isolate each failure.
    code = f"""
import torch
from examples.swe.qwen38_flash_next.patch_sglang_qsa_topk import stable_qsa_topk
logits = torch.zeros((1, 1), device="cuda")
starts = torch.tensor([0], device="cuda")
ends = torch.tensor([1], device="cuda")
if {device == "cuda_graph"}:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        stable_qsa_topk(logits, starts, ends, 1)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = stable_qsa_topk(logits, starts, ends, 1)
    graph.replay()
    torch.cuda.synchronize()
    assert result.cpu().tolist() == [[0]]
logits.fill_(float({str(score)!r}))
starts.fill_({start})
ends.fill_({end})
if {device == "cuda_graph"}:
    graph.replay()
else:
    stable_qsa_topk(logits, starts, ends, 1)
torch.cuda.synchronize()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        timeout=60,
    )
    assert result.returncode != 0
    assert "device-side assert" in result.stderr, result.stderr
    assert error in result.stderr, result.stderr


PLE_PREFIX = f"{LAYERS_PREFIX}.1.ple.ple_embedding."


@pytest.mark.parametrize("profile", ["swe", "swe-eval"])
@pytest.mark.parametrize("explicit_timeout", [None, "120"])
def test_recipe_task_timeout_preserves_eval_defaults_and_explicit_overrides(
    profile, explicit_timeout, monkeypatch, tmp_path
):
    import yaml

    from examples.swe import train_swe_rl
    from examples.swe.qwen38_flash_next import train_rl
    from examples.swe.utils import SWEPPOConfig

    from areal.api import cli_args

    recipe = yaml.safe_load(
        Path(train_rl.__file__).with_name("swe_mm_rl.yaml").read_text()
    )
    config = SWEPPOConfig()
    config.actor.optimizer = cli_args.OptimizerConfig(**recipe["actor"]["optimizer"])
    config.econfig.arena_task_envs = recipe["econfig"].get("arena_task_envs", {})
    config.econfig.arena_task_envs["UNRELATED_SETTING"] = "preserved"
    if explicit_timeout is not None:
        config.econfig.arena_task_envs["DSH_LLM_REQUEST_TIMEOUT_SECONDS"] = (
            explicit_timeout
        )
    selection = tmp_path / "tasks.json"
    selection.write_text(json.dumps(["env:example@1"]))
    monkeypatch.setenv("QWEN_ARENA_TASK_IDS_FILE", str(selection))
    monkeypatch.delenv("QWEN_BATCH_REPLAY_PATH", raising=False)
    monkeypatch.delenv("QWEN_BATCH_REPLAY_PATHS", raising=False)
    monkeypatch.setattr(cli_args, "load_expr_config", lambda *_: (config, None))

    class DatasetBoundaryReached(Exception):
        pass

    def check_task_envs(econfig, **kwargs):
        expected = explicit_timeout or ("7200" if profile == "swe" else None)
        assert (
            econfig.arena_task_envs.get("DSH_LLM_REQUEST_TIMEOUT_SECONDS") == expected
        )
        assert econfig.arena_task_envs["UNRELATED_SETTING"] == "preserved"
        raise DatasetBoundaryReached

    monkeypatch.setattr(train_swe_rl, "get_arena_mixture_dataset", check_task_envs)
    with pytest.raises(DatasetBoundaryReached):
        train_rl.main(profile, [])


@pytest.fixture
def ple_checkpoint():
    config = SimpleNamespace(
        hf_model_type="qwen4_exp",
        ngram_size=3,
        heads_per_ngram=1,
        ple_embed_dim=4,
        split_ngram_parts=3,
        make_ngram_vocab_size_divisible_by=4,
        ple_layer_ids=[2],
    )
    tensors = {
        PLE_PREFIX + "layer_multipliers": torch.tensor([11, 13, 17]),
        PLE_PREFIX + "ngram_heads_offsets": torch.tensor([0, 5]),
        PLE_PREFIX + "ngram_heads_vocab_sizes": torch.tensor([5, 7]),
        PLE_PREFIX + "ngram_embedding.weight_scale": torch.tensor(0.25),
    }
    for part in range(config.split_ngram_parts):
        tensors[f"{PLE_PREFIX}ngram_embedding.shard_{part}.weight"] = (
            torch.arange(8, dtype=torch.float32).reshape(4, 2).to(torch.float8_e4m3fn)
        )
    return config, tensors


def test_load_missing_ple_shard_fails_before_bridge_io(tmp_path, ple_checkpoint):
    pytest.importorskip("mcore_bridge")
    config, tensors = ple_checkpoint
    del tensors[PLE_PREFIX + "ngram_embedding.shard_0.weight"]
    save_file(tensors, tmp_path / "model.safetensors")
    adapter = MCoreBridgeAdapter.__new__(MCoreBridgeAdapter)
    adapter.config = config
    adapter.bridge = SimpleNamespace(
        hf_layers_prefix=LAYERS_PREFIX, load_weights=Mock()
    )

    with pytest.raises(ValueError, match="Missing required PLE"):
        adapter.load_weights([], str(tmp_path))

    adapter.bridge.load_weights.assert_not_called()


@pytest.fixture
def qwen_model_with_embeddings():
    model = torch.nn.Module()
    model.visual = torch.nn.Linear(2, 2)
    model.language_model = torch.nn.Module()
    model.language_model.embedding = torch.nn.Module()
    model.language_model.embedding.word_embeddings = torch.nn.Embedding(8, 2)
    model.language_model.decoder = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attention = torch.nn.Module()
    layer.self_attention.indexer = torch.nn.Linear(2, 2)
    layer.self_attention.linear_qkv = torch.nn.Linear(2, 2)
    layer.ple = torch.nn.Module()
    layer.ple.ple_embedding = torch.nn.Module()
    layer.ple.ple_embedding.cpu_offload = False
    layer.ple.ple_embedding.ngram_embedding = torch.nn.Embedding(8, 2)
    layer.ple.value_proj = torch.nn.Linear(2, 2)
    model.language_model.decoder.layers = torch.nn.ModuleList([layer])
    return model, layer


def test_qwen_training_rejects_runtime_host_ple_even_when_env_disabled(
    qwen_model_with_embeddings, monkeypatch
):
    model, layer = qwen_model_with_embeddings
    monkeypatch.setenv("PLE_CPU_OFFLOAD", "0")
    layer.ple.ple_embedding.cpu_offload = True
    layer.ple.ple_embedding.host_table = torch.ones(8, 2, requires_grad=True)

    with pytest.raises(NotImplementedError, match="host table has no backward path"):
        _configure_qwen4_exp_parameters(model)


def test_qwen_freeze_ple_table_keeps_small_parameters_trainable(
    qwen_model_with_embeddings,
):
    model, layer = qwen_model_with_embeddings
    frozen = _configure_qwen4_exp_parameters(model, freeze_ple_table=True)

    assert not layer.ple.ple_embedding.ngram_embedding.weight.requires_grad
    assert not hasattr(
        layer.ple.ple_embedding.ngram_embedding.weight, "no_weight_decay"
    )
    assert model.language_model.embedding.word_embeddings.weight.requires_grad
    assert all(
        parameter.requires_grad for parameter in layer.ple.value_proj.parameters()
    )
    assert (
        "language_model.decoder.layers.0.ple.ple_embedding.ngram_embedding.weight"
        in frozen
    )


def test_qwen_training_rejects_unregistered_plain_tensor_table(
    qwen_model_with_embeddings,
):
    model, layer = qwen_model_with_embeddings
    table = layer.ple.ple_embedding.ngram_embedding
    del table.weight
    table.weight = torch.ones(8, 2, requires_grad=True)

    with pytest.raises(ValueError, match="registered trainable Parameter"):
        _configure_qwen4_exp_parameters(model)


def test_qwen_mcore_optimizer_groups_train_embeddings_with_ple_zero_decay(
    qwen_model_with_embeddings, tmp_path
):
    if importlib.util.find_spec("megatron") is None:
        pytest.skip("MCore optimizer grouping requires the pinned training runtime")
    from megatron.core.optimizer import OptimizerConfig, _get_param_groups

    model, layer = qwen_model_with_embeddings
    _configure_qwen4_exp_parameters(model, freeze_ple_table=False)
    config = OptimizerConfig(optimizer="adam", lr=0.01, min_lr=0.0, weight_decay=0.1)
    overrides = qwen4_exp_optimizer_overrides(config)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_path / 'optimizer-group-store'}",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=20),
    )
    try:
        groups = _get_param_groups([model], config, overrides)
    finally:
        dist.destroy_process_group()
    grouped_parameters = {
        id(parameter): group for group in groups for parameter in group["params"]
    }
    token_embedding = model.language_model.embedding.word_embeddings
    ple_table = layer.ple.ple_embedding.ngram_embedding
    assert grouped_parameters[id(token_embedding.weight)]["wd_mult"] == 1.0
    assert grouped_parameters[id(ple_table.weight)]["wd_mult"] == 0.0
    assert grouped_parameters[id(layer.ple.value_proj.bias)]["wd_mult"] == 0.0
    assert id(layer.self_attention.indexer.weight) not in grouped_parameters
    assert id(model.visual.weight) not in grouped_parameters
    optimizer = torch.optim.AdamW(
        [
            {**group, "weight_decay": config.weight_decay * group["wd_mult"]}
            for group in groups
        ],
        lr=config.lr,
    )
    token_ids = torch.tensor([1, 2, 1])
    ngram_ids = torch.tensor([3, 5, 3])
    before_tokens = token_embedding.weight.detach().clone()
    before_ple = ple_table.weight.detach().clone()
    (
        token_embedding(token_ids).square().mean()
        + ple_table(ngram_ids).square().mean()
    ).backward()
    optimizer.step()
    assert not torch.equal(token_embedding.weight[token_ids], before_tokens[token_ids])
    assert not torch.equal(ple_table.weight[ngram_ids], before_ple[ngram_ids])
    torch.testing.assert_close(ple_table.weight[0], before_ple[0], atol=0, rtol=0)


@pytest.fixture
def packed_forward(monkeypatch):
    mpu = SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: 1,
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        is_pipeline_last_stage=lambda **_kwargs: True,
    )
    core = ModuleType("megatron.core")
    core.parallel_state = mpu
    metadata = ModuleType("megatron.core.packed_seq_params")
    metadata.PackedSeqParams = SimpleNamespace
    for name, module in (
        ("megatron", ModuleType("megatron")),
        ("megatron.core", core),
        ("megatron.core.packed_seq_params", metadata),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    path = (
        Path(__file__).resolve().parents[1]
        / "areal/engine/megatron_utils/packed_context_parallel.py"
    )
    spec = importlib.util.spec_from_file_location("_test_packed_inputs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("language_model_only", [False, True])
def test_qwen_text_packing_partitions_ids_and_positions_once(
    packed_forward, monkeypatch, cp_size, language_model_only
):
    module = packed_forward
    assert (
        resolve_sequence_packing_mode("qwen4_exp", "mcore-bridge")
        == SequencePackingMode.WRAPPER_THD
    )
    ids = torch.arange(24, dtype=torch.long) + 10
    cu_seqlens = torch.tensor([0, 8, 24], dtype=torch.int32)
    positions = torch.cat([torch.arange(8), torch.arange(16)])
    monkeypatch.setattr(module.mpu, "get_context_parallel_world_size", lambda: cp_size)
    outputs = []
    for cp_rank in range(cp_size):
        monkeypatch.setattr(
            module.mpu, "get_context_parallel_rank", lambda rank=cp_rank: rank
        )
        model = MagicMock(
            side_effect=lambda **inputs: (
                inputs["input_ids"] * 7 + inputs["position_ids"]
            ).unsqueeze(-1)
        )
        output = module.packed_context_parallel_forward(
            model,
            {"input_ids": ids, "cu_seqlens": cu_seqlens},
            gather_cp_output=False,
            is_vision_model=True,
            use_wrapper_packed_seq=True,
            language_model_only=language_model_only,
        )
        inputs = model.call_args.kwargs
        if cp_size == 1:
            expected_indices = torch.arange(24)
        else:
            # Megatron's balanced CP layout assigns chunk r and its mirror
            # independently within each packed document.
            expected_indices = torch.cat(
                [
                    doc.reshape(2 * cp_size, -1)[
                        [cp_rank, 2 * cp_size - cp_rank - 1]
                    ].flatten()
                    for doc in (torch.arange(8), torch.arange(8, 24))
                ]
            )
        assert inputs["input_ids"].shape == (1, 24 // cp_size)
        assert inputs["position_ids"].shape == inputs["input_ids"].shape
        assert inputs["attention_mask"] is None
        torch.testing.assert_close(
            inputs["input_ids"][0], ids[expected_indices], atol=0, rtol=0
        )
        torch.testing.assert_close(
            inputs["position_ids"][0], positions[expected_indices], atol=0, rtol=0
        )
        metadata = inputs["packed_seq_params"]
        assert metadata.qkv_format == "thd"
        assert metadata.max_seqlen_q == 16
        torch.testing.assert_close(metadata.cu_seqlens_q, cu_seqlens, atol=0, rtol=0)
        outputs.append(output.squeeze(-1))

    indices = module._build_cp_reassemble_indices(cu_seqlens, cp_size)
    reconstructed = torch.cat(outputs)[indices]
    torch.testing.assert_close(reconstructed, ids * 7 + positions, atol=0, rtol=0)


def test_model_owned_thd_keeps_full_ids_and_video_payload(packed_forward):
    module = packed_forward
    model = MagicMock(return_value=torch.ones(1, 8, 2))
    pixels = torch.ones(2, 4)
    module.packed_context_parallel_forward(
        model,
        {
            "input_ids": torch.arange(8),
            "cu_seqlens": torch.tensor([0, 4, 8], dtype=torch.int32),
            "pixel_values_videos": pixels,
        },
        gather_cp_output=False,
        is_vision_model=True,
        use_model_packed_seq=True,
    )
    inputs = model.call_args.kwargs
    torch.testing.assert_close(inputs["input_ids"], torch.arange(8).reshape(2, 4))
    assert inputs["position_ids"] is None
    assert inputs["attention_mask"].all()
    assert inputs["pixel_values_videos"] is pixels


def test_wrapper_mtp_preserves_thd_supervision_without_padding_mask(packed_forward):
    model = MagicMock(return_value=torch.zeros(1, 8, 1))
    mask = torch.tensor([1, 1, 1, 0, 1, 1, 1, 0], dtype=torch.float32)
    packed_forward.packed_context_parallel_forward(
        model,
        {
            "input_ids": torch.arange(8),
            "cu_seqlens": torch.tensor([0, 4, 8], dtype=torch.int32),
            "mtp_loss_mask": mask,
        },
        gather_cp_output=False,
        is_vision_model=True,
        use_wrapper_packed_seq=True,
    )
    kwargs = model.call_args.kwargs
    assert kwargs["attention_mask"] is None
    torch.testing.assert_close(kwargs["loss_mask"].reshape(-1), mask)


class _SourceConfig(PretrainedConfig):
    model_type = "qwen4_exp"


def _source_config():
    return _SourceConfig(
        text_config=PretrainedConfig(
            mtp={
                "hybrid": True,
                "num_hidden_layers": 1,
                "layer_types": ["full_attention"],
            },
            mtp_num_hidden_layers=1,
            mtp_use_dedicated_embeddings=False,
        )
    )


class _TokenizerWriter:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def save_pretrained(self, path):
        if self.fail:
            raise OSError("tokenizer save failed")
        (Path(path) / "tokenizer_config.json").write_text("{}")


class _TensorSaveAlreadyFinished:
    hf_layers_prefix = "model.language_model.layers"

    def save_weights(self, models, path):
        # The real bridge has returned from its own tensor-save collectives.
        # This double exercises only the following CPU validation contract.
        pass


def _checkpoint_collective_worker(rank: int, directory: str):
    root = Path(directory)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{root / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    group = dist.group.WORLD
    try:
        config = _source_config()
        before = config.to_dict()
        for name, expected_error in (
            ("success", None),
            ("missing_text", "missing required non-restorable"),
            ("tokenizer_error", "tokenizer save failed"),
        ):
            try:
                report = finalize_mcore_bridge_checkpoint(
                    str(root / "source"),
                    str(root / name),
                    hf_config=config,
                    language_model_only=True,
                    mtp_enabled=False,
                    cpu_group=group,
                    tokenizer=_TokenizerWriter(fail=name == "tokenizer_error"),
                )
            except RuntimeError as exc:
                assert expected_error is not None and expected_error in str(exc)
            else:
                assert expected_error is None
                assert report["restored_keys"] == ["model.visual.patch_embed.weight"]
                assert report["omitted_mtp_keys"] == ["mtp.fc_hidden.weight"]
                saved = json.loads((root / name / "config.json").read_text())
                assert saved["text_config"]["mtp"] is None
                assert saved["text_config"]["mtp_num_hidden_layers"] == 0
                assert (root / name / "tokenizer_config.json").is_file()
                assert (
                    root / name / "chat_template.jinja"
                ).read_text() == "{{ messages[0]['content'] }}"
                assert (
                    GenerationConfig.from_pretrained(root / name).max_new_tokens == 16
                )
            assert config.to_dict() == before
            # Both success and error paths leave peers able to do another collective.
            dist.barrier(group=group)

        adapter = MCoreBridgeAdapter.__new__(MCoreBridgeAdapter)
        adapter.config = SimpleNamespace(
            hf_model_type="qwen4_exp",
            ngram_size=3,
            heads_per_ngram=1,
            ple_embed_dim=4,
            split_ngram_parts=1,
            make_ngram_vocab_size_divisible_by=4,
            ple_layer_ids=[] if rank == 0 else [1],
        )
        adapter.bridge = _TensorSaveAlreadyFinished()
        try:
            adapter.save_weights([], str(root / "success"), cpu_group=group)
        except RuntimeError as exc:
            assert "rank 1" in str(exc)
            assert "Missing required PLE checkpoint tensor" in str(exc)
        else:
            raise AssertionError("A single-rank PLE failure must reach every rank")
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
def test_checkpoint_finalization_errors_reach_all_cpu_ranks(tmp_path):
    pytest.importorskip("mcore_bridge.utils.qwen4_exp_checkpoint")
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    GenerationConfig(eos_token_id=2, max_new_tokens=16).save_pretrained(source)
    (source / "chat_template.jinja").write_text("{{ messages[0]['content'] }}")
    save_file(
        {
            "lm_head.weight": torch.ones(2),
            "model.visual.patch_embed.weight": torch.ones(3),
            "mtp.fc_hidden.weight": torch.ones(2),
        },
        source / "model.safetensors",
    )
    for name in ("success", "missing_text", "tokenizer_error"):
        output = tmp_path / name
        output.mkdir()
        weights = (
            {} if name == "missing_text" else {"lm_head.weight": torch.full((2,), 7.0)}
        )
        save_file(weights, output / "model.safetensors")
    context = mp.spawn(
        _checkpoint_collective_worker, args=(str(tmp_path),), nprocs=2, join=False
    )
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("CPU checkpoint collective timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def _check_cp_causal_gradients(rank, init_file):
    import torch.nn.functional as functional
    from mcore_bridge.model.modules import ple
    from mcore_bridge.utils import megatron_utils

    from areal.engine.megatron_utils.qwen4_exp_cp import (
        install_qwen4_exp_ple_cp_autograd,
    )

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        megatron_utils.mpu.get_context_parallel_world_size = lambda: 2
        megatron_utils.mpu.get_context_parallel_rank = lambda: rank
        megatron_utils.mpu.get_context_parallel_group = lambda: dist.group.WORLD
        original = ple.reconstruct_tensor_cp
        install_qwen4_exp_ple_cp_autograd()
        installed = ple.reconstruct_tensor_cp
        install_qwen4_exp_ple_cp_autograd()
        assert ple.reconstruct_tensor_cp is installed
        assert megatron_utils.reconstruct_tensor_cp is original
        full = torch.arange(16, dtype=torch.float64).reshape(8, 1, 2) / 10
        indices = torch.tensor([0, 1, 6, 7] if rank == 0 else [2, 3, 4, 5])
        weights = torch.arange(1, 17, dtype=torch.float64).reshape_as(full)
        kernel = torch.tensor(
            [[[0.2, 0.3, 0.5]], [[0.4, 0.1, 0.7]]], dtype=torch.float64
        )

        def causal(hidden, conv_weight):
            x = hidden.permute(1, 2, 0)
            return functional.conv1d(
                functional.pad(x, (2, 0)), conv_weight, groups=2
            ).permute(2, 0, 1)

        reference = full.clone().requires_grad_()
        ref_kernel = kernel.clone().requires_grad_()
        expected = causal(reference, ref_kernel)
        (expected * weights).sum().backward()

        # IDs and no-grad/recompute forwards retain the pinned bridge behavior.
        ids = indices.unsqueeze(0)
        torch.testing.assert_close(
            installed(ids, None, dim=1), torch.arange(8).unsqueeze(0), rtol=0, atol=0
        )
        with torch.no_grad():
            torch.testing.assert_close(
                installed(full[indices], None, dim=0), full, rtol=0, atol=0
            )
        for differentiable in (False, True):
            local = full[indices].clone().requires_grad_()
            local_kernel = kernel.clone().requires_grad_()
            reconstruct = installed if differentiable else original
            restored = reconstruct(local, None, dim=0)
            output = causal(restored, local_kernel)[indices]
            torch.testing.assert_close(
                output, expected.detach()[indices], rtol=1e-12, atol=1e-12
            )
            (output * weights[indices]).sum().backward()
            if differentiable:
                torch.testing.assert_close(
                    local.grad, reference.grad[indices], rtol=1e-12, atol=1e-12
                )
                dist.all_reduce(local_kernel.grad, group=dist.group.WORLD)
                torch.testing.assert_close(
                    local_kernel.grad, ref_kernel.grad, rtol=1e-12, atol=1e-12
                )
            else:
                assert not torch.allclose(local.grad, reference.grad[indices])
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
def test_cp_causal_convolution_preserves_remote_input_gradients(tmp_path):
    """Two real CP ranks match CP1 across both zigzag boundaries, including backward."""
    pytest.importorskip("mcore_bridge.model.modules.ple")
    mp.spawn(
        _check_cp_causal_gradients,
        args=(str(tmp_path / "cp-init"),),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("device", ["cpu", "meta", "cuda"])
def test_bridge_live_sync_stages_cpu_exports_after_draining_bucket(monkeypatch, device):
    from areal.engine.megatron_engine import MegatronEngine

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    tensors = [
        torch.arange(4, dtype=torch.bfloat16).reshape(2, 2).t(),
        torch.tensor([7], dtype=torch.int64),
        torch.tensor([1.25, 2.5, 3.75]),
    ]
    exports = [(f"ple.shard_{i}", tensor) for i, tensor in enumerate(tensors)]
    engine = MegatronEngine.__new__(MegatronEngine)
    engine.device = torch.device(device)
    engine.model = []
    engine.process_group_initialized = True
    engine._cpu_group = object()
    engine.bridge = SimpleNamespace(export_hf_weights=Mock(return_value=iter(exports)))
    engine.is_pipeline_parallel_head = lambda: True
    events = []
    original_to = torch.Tensor.to
    names = {id(tensor): name for name, tensor in exports}

    def stage(tensor, *args, **kwargs):
        if id(tensor) in names:
            events.append(("stage", names[id(tensor)]))
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", stage)
    received = []

    def send(meta, bucket):
        # The normal sender waits for broadcasts and then clears the bucket.
        assert len(bucket) == 1
        name, tensor = bucket[0]
        source = dict(exports)[name]
        assert tensor.device.type == device
        assert tensor.dtype == source.dtype
        assert tensor.shape == source.shape
        assert tensor.is_contiguous()
        if device != "meta":
            torch.testing.assert_close(tensor.cpu(), source, rtol=0, atol=0)
        events.append(("send", name))
        received.append(name)
        bucket.clear()

    engine._update_bucket_weights_from_distributed = send
    barrier = Mock()
    monkeypatch.setattr(dist, "barrier", barrier)
    engine._update_weights_via_bridge(
        SimpleNamespace(weight_chunked_mem_mb=8 / 1024**2)
    )
    assert received == [name for name, _ in exports]
    assert events == [
        (action, name) for name, _ in exports for action in ("stage", "send")
    ]
    engine.bridge.export_hf_weights.assert_called_once_with(
        engine.model, cpu=False, show_progress=False
    )
    barrier.assert_called_once_with(group=engine.cpu_group)


def test_bridge_live_sync_non_sender_does_not_stage_exports(monkeypatch):
    from areal.engine.megatron_engine import MegatronEngine

    engine = MegatronEngine.__new__(MegatronEngine)
    engine.model = []
    engine.process_group_initialized = True
    engine._cpu_group = object()
    engine.bridge = SimpleNamespace(
        export_hf_weights=lambda *args, **kwargs: iter(
            [("ple.shard", torch.ones(4)), ("other", None)]
        )
    )
    engine.is_pipeline_parallel_head = lambda: False
    engine._update_bucket_weights_from_distributed = Mock()
    stage = Mock(side_effect=AssertionError("Non-sender must not stage tensors"))
    monkeypatch.setattr(torch.Tensor, "to", stage)
    barrier = Mock()
    monkeypatch.setattr(dist, "barrier", barrier)
    engine._update_weights_via_bridge(SimpleNamespace(weight_chunked_mem_mb=1))
    stage.assert_not_called()
    engine._update_bucket_weights_from_distributed.assert_not_called()
    barrier.assert_called_once_with(group=engine.cpu_group)


def test_mcore_bridge_rejects_unsupported_mtp_before_model_construction():
    from areal.engine.megatron_engine import MegatronEngine

    engine = MegatronEngine.__new__(MegatronEngine)
    engine.bridge_cls = "mcore-bridge"
    engine.mcore_config = SimpleNamespace(enable_mtp_training=True)
    with pytest.raises(NotImplementedError, match="MCore 0.19 MTP supervision"):
        engine._build_hf_mcore_bridge()
