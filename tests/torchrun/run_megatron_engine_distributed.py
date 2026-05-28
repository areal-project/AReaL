import argparse
import copy
import os
import tempfile
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from transformers import AutoTokenizer

from tests.utils import get_model_path

from areal.api import FinetuneSpec, SaveLoadMeta
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.engine import FSDPEngine, MegatronEngine
from areal.infra.platforms import current_platform
from areal.utils import seeding
from areal.utils.data import broadcast_tensor_container

MODEL_PATHS = {
    "qwen3": get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
    ),
    "qwen3moe": get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-30B-A3B/", "Qwen/Qwen3-30B-A3B"
    ),
}


def write_result(out: str, succ: bool):
    with open(out, "w") as f:
        if succ:
            f.write("Passed")
        else:
            f.write("Failed")


def mock_input(
    batch_size=128,
    min_seqlen=1,
    max_seqlen=1024,
    device=current_platform.device_type,
) -> dict[str, Any]:
    """Create mock padded input data (same format for huggingface) for testing.
    Returns a dict with input_ids, attention_mask, and position_ids.
    """
    pad_token_id = 0
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (batch_size,), dtype=torch.int, device=device
    )
    max_seqlen = int(max(seqlens))
    input_ids = torch.randint(
        10000, 50000, (batch_size, max_seqlen), dtype=torch.long, device=device
    )
    attn_mask = torch.zeros((batch_size, max_seqlen), dtype=torch.bool, device=device)

    attn_mask[
        torch.arange(0, max_seqlen, device=device).unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1
    input_ids.masked_fill_(~attn_mask, pad_token_id)

    return dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )


def make_engine(model_type, backend, mb_spec, vpp_size=1, init_optimizer=False):
    config = TrainEngineConfig(
        backend=backend,
        experiment_name="test",
        trial_name="test",
        path=MODEL_PATHS[model_type],
        mb_spec=mb_spec,
        optimizer=OptimizerConfig() if init_optimizer else None,
        megatron=MegatronEngineConfig(virtual_pipeline_parallel_size=vpp_size),
    )
    alloc_mode = ModelAllocation.from_str(backend)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine = MegatronEngine(config)
    engine.create_process_group(parallel_strategy=alloc_mode.parallel)
    engine.initialize(addr=None, ft_spec=ft_spec)
    return engine


def make_fsdp_engine(model_type, backend, mb_spec, init_optimizer=False):
    engine_config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test",
        trial_name="test",
        mb_spec=mb_spec,
        path=MODEL_PATHS[model_type],
        optimizer=OptimizerConfig() if init_optimizer else None,
    )
    alloc_mode = ModelAllocation.from_str(backend)
    # ignore parallel strategy for a stable forward output
    alloc_mode.parallel.data_parallel_size *= alloc_mode.parallel.world_size
    alloc_mode.parallel.pipeline_parallel_size = 1
    alloc_mode.parallel.tensor_parallel_size = 1
    alloc_mode.parallel.context_parallel_size = 1
    engine = FSDPEngine(engine_config)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine.create_process_group(parallel_strategy=alloc_mode.parallel)
    engine.initialize(None, ft_spec)
    return engine


def test_forward(
    model_type: str, alloc_mode: str, output: str | None = None, vpp_size: int = 1
):
    rank = int(os.environ["RANK"])

    mb_spec = MicroBatchSpec(max_tokens_per_mb=256)
    engine = make_engine(model_type, alloc_mode, mb_spec, vpp_size=vpp_size)
    seeding.set_random_seed(0, key=f"trainer{rank}")

    input_ = mock_input(batch_size=16, max_seqlen=128, device=engine.device)
    print(f"rank {rank} is_data_parallel_head()={engine.is_data_parallel_head()}")
    bcasted_input = broadcast_tensor_container(
        input_,
        src_rank=engine.current_data_parallel_head(),
        group=engine.context_and_model_parallel_group,
    )
    logprobs = engine.forward(
        input_=bcasted_input,
        aggregate_fn=lambda xs: torch.cat(xs, dim=0),
    )

    print(f"final rank {rank} result: shape: {logprobs.shape}, logprobs: {logprobs}")

    # All ranks in the model parallel group should have the same logprobs
    dist.barrier()
    model_parallel_group = mpu.get_model_parallel_group()
    model_parallel_world_size = len(dist.get_process_group_ranks(model_parallel_group))
    logprobs_list = [
        torch.empty_like(logprobs) for _ in range(model_parallel_world_size)
    ]
    dist.all_gather(logprobs_list, logprobs, group=model_parallel_group)

    is_equal = all(
        torch.equal(logprobs, logprobs_list[0]) for logprobs in logprobs_list
    )
    assert is_equal, "Logprobs should be the same across all model parallel ranks."

    # make FSDP engine, and check if the difference between FSDP and megatron engine
    fsdp_engine = make_fsdp_engine("qwen3", alloc_mode, mb_spec)
    fsdp_logprobs = fsdp_engine.forward(
        input_=input_,
        aggregate_fn=lambda xs: torch.cat(xs, dim=0),
    )
    print(
        f"rank {rank} logprobs.shape={logprobs.shape} fsdp_logprobs.shape={fsdp_logprobs.shape}"
    )
    # only compare results on data parallel head
    failed = False
    if engine.is_data_parallel_head():
        diff = torch.abs(logprobs - fsdp_logprobs)
        print(
            f"rank {rank} diff between megatron and fsdp logprobs: {diff}, max(diff)={torch.max(diff)} avg(diff)={torch.mean(diff)}"
        )

        cosine_sim = torch.nn.functional.cosine_similarity(
            logprobs.flatten().to(torch.float32),
            fsdp_logprobs.flatten().to(torch.float32),
            dim=0,
        )
        print(f"Cosine Similarity: {cosine_sim.item()}")

        if cosine_sim < 0.99:
            raise AssertionError(
                f"Cosine similarity {cosine_sim.item()} is less than 0.99"
            )

    current_platform.synchronize()
    dist.barrier()
    fsdp_engine.destroy()
    engine.destroy()

    print(f"Test: test_forward(model_type={model_type}, alloc_mode={alloc_mode}) Done.")
    if rank == 0 and output is not None:
        write_result(output, not failed)


def mock_loss_fn(
    logprobs: torch.Tensor, entropy: torch.Tensor, input_data: dict, **kwargs
) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logprobs)


def test_train(
    model_type: str, alloc_mode: str, output: str | None = None, vpp_size: int = 1
):
    print(f"running train test: model_type={model_type} alloc_mode={alloc_mode}")
    rank = int(os.environ["RANK"])

    mb_spec = MicroBatchSpec(max_tokens_per_mb=256)
    engine = make_engine(
        model_type, alloc_mode, mb_spec, init_optimizer=True, vpp_size=vpp_size
    )
    seeding.set_random_seed(0, key=f"trainer{rank}")

    input_ = mock_input(batch_size=16, max_seqlen=128, device=engine.device)
    print(f"rank {rank} is_data_parallel_head()={engine.is_data_parallel_head()}")
    bcasted_input = broadcast_tensor_container(
        input_,
        src_rank=engine.current_data_parallel_head(),
        group=engine.context_and_model_parallel_group,
    )

    train_result = engine.train_batch(
        input_=bcasted_input,
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: x["cu_seqlens"][-1],
    )

    print(f"final rank {rank} train_result: {train_result}")
    current_platform.synchronize()
    dist.barrier()
    engine.destroy()

    if rank == 0 and output is not None:
        write_result(output, True)
    print(f"Test: test_train(model_type={model_type}, alloc_mode={alloc_mode}) Done.")


def test_grad_norm_mb_invariance(
    model_type: str, alloc_mode: str, output: str | None = None, vpp_size: int = 1
):
    """Regression guard for the `loss_multiplier` fix in `train_batch`.

    Megatron Core's pipeline schedule applies `loss /= num_microbatches` on the
    2-tuple loss-func return path. Since AReaL already normalizes the per-mb
    loss globally via `w_i / W_total`, that extra division must be compensated
    in `loss_multiplier`. If the compensation is missing (or wrong), the
    resulting gradient — and therefore the grad_norm reported by the optimizer
    — scales inversely with `num_microbatches`.

    This test runs `train_batch` on the SAME input and seeded-identical weights
    with two different `max_tokens_per_mb` values that yield a different number
    of micro-batches. The reported grad_norm must match within a tight
    tolerance. Pre-fix the values differ by exactly the ratio of mb counts.
    """
    print(
        f"running grad_norm_mb_invariance: model_type={model_type} "
        f"alloc_mode={alloc_mode}"
    )
    rank = int(os.environ["RANK"])
    batch_size = 16
    max_seqlen = 128

    grad_norms: list[float] = []
    engines = []
    # Two configs that yield different `num_microbatches` for the same total
    # batch. Values chosen to keep at least one mb for the smallest PP chunk
    # while still producing distinct mb counts between the two runs.
    for max_tokens_per_mb in (4096, 256):
        mb_spec = MicroBatchSpec(max_tokens_per_mb=max_tokens_per_mb)
        # Reset seed before creating each engine so parameter init is identical.
        seeding.set_random_seed(0, key=f"engine{rank}")
        engine = make_engine(
            model_type, alloc_mode, mb_spec, init_optimizer=True, vpp_size=vpp_size
        )
        # Reset seed again before building the batch so input is identical.
        seeding.set_random_seed(0, key=f"data{rank}")
        input_ = mock_input(
            batch_size=batch_size, max_seqlen=max_seqlen, device=engine.device
        )
        bcasted_input = broadcast_tensor_container(
            input_,
            src_rank=engine.current_data_parallel_head(),
            group=engine.context_and_model_parallel_group,
        )

        result = engine.train_batch(
            input_=bcasted_input,
            loss_fn=mock_loss_fn,
            loss_weight_fn=lambda x: x["cu_seqlens"][-1],
        )
        print(
            f"rank {rank} max_tokens_per_mb={max_tokens_per_mb} train_result={result}"
        )
        grad_norms.append(float(result["grad_norm"]))

        current_platform.synchronize()
        dist.barrier()
        engines.append(engine)

    for engine in engines:
        engine.destroy()
    # grad_norm is reported only on the DP head; other ranks may see NaN/0 but
    # they all agree by virtue of the Megatron optimizer's internal all-reduce.
    # Tolerance: 1e-3 relative — small enough to catch the num_microbatches
    # ratio (>=2x) while permitting benign non-associativity of fp16/bf16 sums
    # across a different mb grouping.
    g0, g1 = grad_norms
    ok = abs(g0 - g1) <= 1e-3 * max(abs(g0), abs(g1), 1e-12)
    if not ok:
        print(
            f"FAIL rank {rank}: grad_norm differs across num_microbatches: {g0} vs {g1}"
        )

    if rank == 0 and output is not None:
        write_result(output, ok)
    print(
        f"Test: test_grad_norm_mb_invariance(model_type={model_type}, "
        f"alloc_mode={alloc_mode}) Done."
    )


def test_train_dcp_save_load(
    model_type: str, alloc_mode: str, output: str | None = None, vpp_size: int = 1
):
    print(
        f"running test_train_dcp_save_load(model_type={model_type} alloc_mode={alloc_mode})"
    )
    rank = int(os.environ["RANK"])

    base_dir = tempfile.gettempdir()
    path = os.path.join(base_dir, "megatron_engine_train_dcp_test")
    if rank == 0:
        os.makedirs(path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[model_type])

    mb_spec = MicroBatchSpec(max_tokens_per_mb=256)
    engine = make_engine(
        model_type, alloc_mode, mb_spec, init_optimizer=True, vpp_size=vpp_size
    )

    seeding.set_random_seed(0, key=f"trainer{rank}")

    input_ = mock_input(batch_size=16, max_seqlen=128, device=engine.device)
    print(f"rank {rank} is_data_parallel_head()={engine.is_data_parallel_head()}")
    bcasted_input = broadcast_tensor_container(
        input_,
        src_rank=engine.current_data_parallel_head(),
        group=engine.context_and_model_parallel_group,
    )

    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="dcp",
        tokenizer=tokenizer,
        with_optim=True,
        base_model_path=None,
    )

    # train step 1
    train_result = engine.train_batch(
        input_=bcasted_input,
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: x["cu_seqlens"][-1],
    )

    print(f"final rank {rank} train_result: {train_result}")

    current_platform.synchronize()
    dist.barrier()

    # save checkpoint for recover
    engine.save(save_load_meta)

    # train step 2
    engine.train_batch(
        input_=bcasted_input,
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: x["cu_seqlens"][-1],
    )

    with torch.no_grad():
        engine.eval()
        params = copy.deepcopy(dict(engine.model.named_parameters()))

        for p in engine.model.parameters():
            p.data.zero_()

        # recover
        engine.load(save_load_meta)

    engine.train()
    # train step 2 after recover
    engine.train_batch(
        input_=bcasted_input,
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: x["cu_seqlens"][-1],
    )

    current_platform.synchronize()
    dist.barrier()

    with torch.no_grad():
        engine.eval()
        succ = True
        for name, param in engine.model.named_parameters():
            if not torch.allclose(param, params[name]):
                diff = torch.abs(params[name] - param)
                print(
                    f"rank {rank} diff of {name}: {diff}, max(diff)={torch.max(diff)} avg(diff)={torch.mean(diff)}, count(diff)={torch.count_nonzero(diff)}"
                )
                succ = False
        assert succ, "Weights should be same after recover"

    current_platform.synchronize()
    dist.barrier()

    engine.destroy()

    if output:
        write_result(output, True)

    print(
        f"Test: test_train_dcp_save_load(model_type={model_type}, alloc_mode={alloc_mode}) Done."
    )


def test_simple_dcp_save_load(
    model_type: str, alloc_mode: str, output: str | None = None, vpp_size: int = 1
):
    print(
        f"running test_simple_dcp_save_load(model_type={model_type} alloc_mode={alloc_mode})"
    )
    rank = int(os.environ["RANK"])

    base_dir = tempfile.gettempdir()
    path = os.path.join(base_dir, "megatron_engine_simple_dcp_test")
    if rank == 0:
        os.makedirs(path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[model_type])

    mb_spec = MicroBatchSpec(max_tokens_per_mb=256)
    engine = make_engine(
        model_type, alloc_mode, mb_spec, init_optimizer=True, vpp_size=vpp_size
    )

    seeding.set_random_seed(0, key=f"trainer{rank}")
    print(f"rank {rank} is_data_parallel_head()={engine.is_data_parallel_head()}")

    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="dcp",
        tokenizer=tokenizer,
        with_optim=False,
        base_model_path=None,
    )

    with torch.no_grad():
        engine.eval()
        params = copy.deepcopy(dict(engine.model.named_parameters()))
        engine.save(save_load_meta)

        for p in engine.model.parameters():
            p.data.zero_()

        engine.load(save_load_meta)

        succ = True
        for name, param in engine.model.named_parameters():
            if not torch.allclose(param, params[name]):
                diff = torch.abs(params[name] - param)
                print(
                    f"rank {rank} diff of {name}: {diff}, max(diff)={torch.max(diff)} avg(diff)={torch.mean(diff)}, count(diff)={torch.count_nonzero(diff)}"
                )
                succ = False
        assert succ, "Weights should be same after recover"

    current_platform.synchronize()
    dist.barrier()

    engine.destroy()

    if output:
        write_result(output, True)

    print(
        f"Test: test_simple_dcp_save_load(model_type={model_type}, alloc_mode={alloc_mode}) Done."
    )


def main():
    parser = argparse.ArgumentParser(description="Run Megatron Engine Distributed Test")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["qwen3", "qwen3moe"],
        default="qwen3",
        help="Type of model to test",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="megatron:d1p2t2c2",
        help="Backend allocation string for the model (e.g., 'megatron:d1p2t2c2')",
    )
    parser.add_argument(
        "--vpp_size",
        type=int,
        default=1,
        help="Virtual pipeline parallel size",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the output result",
    )
    parser.add_argument(
        "--test_type",
        type=str,
        choices=[
            "forward",
            "train",
            "grad_norm_mb_invariance",
            "simple_dcp_save_load",
            "train_dcp_save_load",
        ],
        default="forward",
        help="Type of test to run: 'forward' or 'train'",
    )
    args = parser.parse_args()

    print(args)
    if args.test_type == "train":
        test_train(
            args.model_type,
            args.backend,
            output=args.output,
            vpp_size=args.vpp_size,
        )
    elif args.test_type == "forward":
        test_forward(
            args.model_type,
            args.backend,
            output=args.output,
            vpp_size=args.vpp_size,
        )
    elif args.test_type == "grad_norm_mb_invariance":
        test_grad_norm_mb_invariance(
            args.model_type,
            args.backend,
            output=args.output,
            vpp_size=args.vpp_size,
        )
    elif args.test_type == "simple_dcp_save_load":
        test_simple_dcp_save_load(
            args.model_type,
            args.backend,
            output=args.output,
            vpp_size=args.vpp_size,
        )
    elif args.test_type == "train_dcp_save_load":
        test_train_dcp_save_load(
            args.model_type,
            args.backend,
            output=args.output,
            vpp_size=args.vpp_size,
        )
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    # run with `torchrun` to test with multiple GPUs & multiple nodes
    main()
