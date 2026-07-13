"""Shared utilities for Archon Engine tests."""

import os
import subprocess
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from areal.api import FinetuneSpec, ParallelStrategy
from areal.api.cli_args import MicroBatchSpec, OptimizerConfig, TrainEngineConfig
from areal.engine import FSDPLMEngine
from areal.experimental.engine.archon_engine import ArchonLMEngine
from areal.infra.platforms import current_platform
from areal.utils.data import pad_sequences_to_tensors
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.network import find_free_ports
from areal.utils.testing_utils import (
    DENSE_MODEL_PATHS,
    MODEL_PATHS,
    MOE_MODEL_PATHS,
    get_dataset_path,
    get_model_path,
)
from areal.utils.testing_utils import (
    load_archon_model as _load_archon_model_impl,
)

# Re-export constants for backward compatibility
__all__ = [
    "DENSE_MODEL_PATHS",
    "MOE_MODEL_PATHS",
    "MODEL_PATHS",
    "DATASET_PATH",
    "get_model_path",
    "get_dataset_path",
    "get_model_path_for_type",
    "ComparisonMetrics",
    "compare_tensors",
    "setup_environment",
    "load_hf_model",
    "load_archon_model",
    "create_test_input",
    "load_gsm8k_samples",
    "compare_logprobs",
    "compare_outputs",
    "run_torchrun_test",
    "setup_distributed_environment",
    "create_engine_config",
    "create_grpo_batch",
    "DualEngineFixture",
    "dual_engines",
]


def get_model_path_for_type(model_type: str) -> str | None:
    """Get model path for a given model type.

    Args:
        model_type: HF model_type (e.g., "qwen2", "qwen3").

    Returns:
        Model path if configured, None otherwise.
    """
    return MODEL_PATHS.get(model_type)


DATASET_PATH = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")


@dataclass
class ComparisonMetrics:
    """Metrics for comparing two tensors."""

    max_diff: float
    mean_diff: float
    std_diff: float
    allclose: bool
    shape_match: bool

    def __str__(self) -> str:
        return (
            f"max_diff={self.max_diff:.6f}, mean_diff={self.mean_diff:.6f}, "
            f"std_diff={self.std_diff:.6f}, allclose={self.allclose}, "
            f"shape_match={self.shape_match}"
        )


def compare_tensors(
    t1: torch.Tensor,
    t2: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> ComparisonMetrics:
    """Compare two tensors and return detailed metrics."""
    shape_match = t1.shape == t2.shape
    if not shape_match:
        return ComparisonMetrics(
            max_diff=float("inf"),
            mean_diff=float("inf"),
            std_diff=float("inf"),
            allclose=False,
            shape_match=False,
        )

    diff = (t1.float() - t2.float()).abs()
    return ComparisonMetrics(
        max_diff=diff.max().item(),
        mean_diff=diff.mean().item(),
        std_diff=diff.std().item(),
        allclose=torch.allclose(t1.float(), t2.float(), atol=atol, rtol=rtol),
        shape_match=True,
    )


def setup_environment():
    """Set up environment for tests."""
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        current_platform.set_device(rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    else:
        current_platform.set_device(0)


def load_hf_model(model_path: str, dtype: torch.dtype = torch.bfloat16):
    """Load HuggingFace model with SDPA attention."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = model.to(current_platform.device_type)
    model.eval()
    return model


def load_archon_model(model_path: str, dtype: torch.dtype = torch.bfloat16):
    """Load Archon model with same weights.

    Wraps the production load_archon_model to handle unsupported models
    with pytest.skip instead of raising an exception.
    """
    try:
        return _load_archon_model_impl(model_path, dtype, skip_unsupported=False)
    except ValueError as e:
        if "not supported by Archon" in str(e):
            pytest.skip(str(e))
        raise


def create_test_input(
    batch_size: int = 2,
    seq_len: int = 32,
    vocab_size: int = 151936,
    device: torch.device = None,
    packed: bool = False,
) -> dict:
    """Create deterministic test input.

    Args:
        batch_size: Number of sequences.
        seq_len: Length of each sequence.
        vocab_size: Vocabulary size.
        device: Target device.
        packed: If True, pack sequences into single batch with cu_seqlens.

    Returns:
        dict with input_ids, and optionally cu_seqlens/max_seqlen if packed.
    """
    torch.manual_seed(42)
    if device is None:
        device = torch.device(current_platform.device_type)

    if packed:
        # Generate packed input: [1, batch_size * seq_len]
        total_len = batch_size * seq_len
        input_ids = torch.randint(100, vocab_size - 100, (1, total_len), device=device)
        cu_seqlens = torch.tensor(
            [i * seq_len for i in range(batch_size + 1)],
            dtype=torch.int32,
            device=device,
        )
        return {
            "input_ids": input_ids,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": seq_len,
        }
    else:
        input_ids = torch.randint(
            100, vocab_size - 100, (batch_size, seq_len), device=device
        )
        return {"input_ids": input_ids}


def load_gsm8k_samples(model_path: str, num_samples: int = 5):
    """Load real samples from GSM8K dataset."""
    tokenizer = load_hf_tokenizer(model_path)
    gsm8k_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")
    dataset = load_dataset(path=gsm8k_path, name="main", split="train")

    samples = []
    for sample in dataset:
        if len(samples) >= num_samples:
            break
        prompt = sample["question"]
        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(prompt + sample["answer"] + tokenizer.eos_token)

        if len(full_ids) <= 512:
            samples.append(
                {
                    "prompt": prompt,
                    "answer": sample["answer"],
                    "prompt_ids": prompt_ids,
                    "full_ids": full_ids,
                }
            )
    return samples, tokenizer


def compare_logprobs(
    logits_a: torch.Tensor, logits_b: torch.Tensor, target_ids: torch.Tensor
) -> dict:
    """Compare logprobs for target tokens."""
    logprobs_a = torch.log_softmax(logits_a.float(), dim=-1)
    logprobs_b = torch.log_softmax(logits_b.float(), dim=-1)

    # Get logprobs for target tokens
    target_lp_a = logprobs_a.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    target_lp_b = logprobs_b.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    diff = (target_lp_a - target_lp_b).abs()
    return {
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
    }


def compare_outputs(
    archon_out: torch.Tensor, hf_out: torch.Tensor, name: str = "output"
) -> dict:
    """Compare two tensor outputs with precision metrics."""
    diff = (archon_out.float() - hf_out.float()).abs()
    return {
        "name": name,
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
    }


def run_torchrun_test(script_path: str, n_gpus: int, extra_args: list[str] = None):
    """Run a test script with torchrun."""
    port = find_free_ports(1)[0]
    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        script_path,
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        pytest.fail(f"Test failed with error: {e.stderr}")


# =============================================================================
# GRPO Engine Testing Utilities
# =============================================================================


def setup_distributed_environment():
    """Set up distributed environment for single GPU engine tests."""
    if dist.is_initialized():
        return

    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(find_free_ports(1)[0]))

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        world_size=world_size,
        rank=rank,
    )
    current_platform.set_device(rank)


def create_engine_config(
    model_path: str,
    engine_type: str = "archon",
    lr: float = 1.7e-5,
    weight_decay: float = 0.017,
) -> TrainEngineConfig:
    """Create engine configuration for testing.

    Args:
        model_path: Path to the model.
        engine_type: Engine type for experiment naming.
        lr: Learning rate.
        weight_decay: Weight decay.

    Returns:
        TrainEngineConfig configured for testing.
    """
    return TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name=f"test_{engine_type}_grpo",
        trial_name="test",
        path=model_path,
        mb_spec=MicroBatchSpec(n_mbs=1),
        optimizer=OptimizerConfig(
            type="adam",
            lr=lr,
            weight_decay=weight_decay,
            warmup_steps_proportion=0.001,
            lr_scheduler_type="constant",
            gradient_clipping=1.0,
        ),
        temperature=1.0,
    )


def create_grpo_batch(
    model_path: str,
    batch_size: int = 4,
    max_seq_len: int = 256,
) -> dict[str, Any]:
    """Create a batch that mimics GRPO training inputs.

    Args:
        model_path: Path to the model for tokenizer.
        batch_size: Number of samples in the batch.
        max_seq_len: Maximum sequence length.

    Returns:
        dict with input_ids, attention_mask, loss_mask, and GRPO fields.
    """
    tokenizer = load_hf_tokenizer(model_path)
    dataset = load_dataset(path=DATASET_PATH, name="main", split="train")

    samples = []
    for sample in dataset:
        if len(samples) >= batch_size:
            break
        text = sample["question"] + " " + sample["answer"] + tokenizer.eos_token
        tokens = tokenizer.encode(text)

        # Truncate if too long instead of skipping
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        prompt_tokens = tokenizer.encode(sample["question"])
        prompt_len = min(len(prompt_tokens), len(tokens))
        loss_mask = [0] * prompt_len + [1] * (len(tokens) - prompt_len)
        samples.append(
            {
                "input_ids": tokens,
                "loss_mask": loss_mask,
            }
        )

    if len(samples) < batch_size:
        raise ValueError(
            f"Could not find enough samples. Got {len(samples)}, need {batch_size}. "
            f"Try increasing max_seq_len (current: {max_seq_len})."
        )

    batch = pad_sequences_to_tensors(samples)

    # Add attention_mask
    batch["attention_mask"] = (batch["input_ids"] != tokenizer.pad_token_id).long()

    # Add GRPO-specific fields with realistic values
    seq_len = batch["input_ids"].shape[1]
    device = torch.device(current_platform.device_type)

    # Simulate logprobs from inference engine
    batch["logprobs"] = torch.randn(batch_size, seq_len, device=device) * 0.5 - 2.0
    batch["old_logprobs"] = batch["logprobs"].clone()

    # Simulate rewards (binary for GSM8K)
    batch["rewards"] = torch.randint(0, 2, (batch_size,), device=device).float()

    # Simulate values (from critic, or zeros if no critic)
    batch["values"] = torch.zeros(batch_size, seq_len, device=device)

    # Move all tensors to device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    return batch


class DualEngineFixture:
    """Fixture to manage Archon and FSDP engine lifecycle for comparison tests."""

    def __init__(self, model_path: str = None):
        """Initialize fixture.

        Args:
            model_path: Path to model. Uses MODEL_PATHS["qwen2"] if not specified.
        """
        self.model_path = model_path or MODEL_PATHS["qwen2"]
        self.archon_engine: ArchonLMEngine | None = None
        self.fsdp_engine: FSDPLMEngine | None = None

    def setup(self):
        """Initialize both engines."""
        setup_distributed_environment()

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        parallel_strategy = ParallelStrategy(data_parallel_size=world_size)
        ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=4, train_batch_size=4)

        # Initialize Archon engine
        self.archon_engine = ArchonLMEngine(
            create_engine_config(self.model_path, "archon")
        )
        self.archon_engine.create_process_group(parallel_strategy=parallel_strategy)
        self.archon_engine.initialize(addr=None, ft_spec=ft_spec)

        # Initialize FSDP engine
        self.fsdp_engine = FSDPLMEngine(create_engine_config(self.model_path, "fsdp"))
        self.fsdp_engine.create_process_group(parallel_strategy=parallel_strategy)
        self.fsdp_engine.initialize(addr=None, ft_spec=ft_spec)

    def teardown(self):
        """Cleanup engines."""
        if self.archon_engine is not None:
            self.archon_engine.destroy()
        if self.fsdp_engine is not None:
            self.fsdp_engine.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture(scope="module")
def dual_engines():
    """Pytest fixture to provide initialized Archon and FSDP engines."""
    fixture = DualEngineFixture()
    fixture.setup()
    yield fixture
    fixture.teardown()
