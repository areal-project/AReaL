"""End-to-end checkpoint tests for State Dict Adapter functionality.

Tests cover:
1. Multi-file safetensors loading (fqn_to_index_mapping)
2. ArchonEngine.save() and load() with HF format using DCP infrastructure

These tests require:
- Real HF checkpoint files (Qwen3-30B-A3B MoE model)
- Multiple GPUs for distributed tests
- Sufficient GPU memory

Run tests:
    # Quick tests (single GPU, small model)
    pytest tests/experimental/archon/test_checkpoint_e2e.py -v -k "not moe"

    # Full MoE tests (multi-GPU, large model)
    pytest tests/experimental/archon/test_checkpoint_e2e.py -v -m multi_gpu
"""

import json
import os
import subprocess

import pytest
import torch

from tests.experimental.archon.utils import (
    DENSE_MODEL_PATHS,
    MOE_MODEL_PATHS,
)

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# =============================================================================
# Multi-file Safetensors Index Tests
# =============================================================================


class TestMultiFileSafetensorsIndex:
    """Tests for multi-file safetensors index loading."""

    @pytest.fixture
    def moe_model_path(self):
        """Get path to Qwen3-30B-A3B MoE model."""
        path = MOE_MODEL_PATHS.get("qwen3_moe")
        if path is None:
            pytest.skip("Qwen3-30B-A3B model path not configured")
        if not os.path.exists(path):
            pytest.skip(f"Qwen3-30B-A3B model not found at {path}")
        return path

    @pytest.fixture
    def dense_model_path(self):
        """Get path to Qwen3-0.6B dense model."""
        path = DENSE_MODEL_PATHS.get("qwen3")
        if path is None:
            pytest.skip("Qwen3-0.6B model path not configured")
        if not os.path.exists(path):
            pytest.skip(f"Qwen3-0.6B model not found at {path}")
        return path

    def test_load_moe_safetensors_index(self, moe_model_path):
        """Test loading multi-file safetensors index from real MoE checkpoint."""
        from transformers import AutoConfig

        from areal.experimental.models.archon.qwen3 import Qwen3StateDictAdapter

        # Load config
        config = AutoConfig.from_pretrained(moe_model_path, trust_remote_code=True)

        # Create adapter with hf_assets_path
        adapter = Qwen3StateDictAdapter(config, hf_assets_path=moe_model_path)

        # Verify index was loaded
        assert adapter.fqn_to_index_mapping is not None
        assert len(adapter.fqn_to_index_mapping) > 0

        # Verify index file structure
        index_path = os.path.join(moe_model_path, "model.safetensors.index.json")
        assert os.path.exists(index_path)

        with open(index_path) as f:
            index_data = json.load(f)

        # Verify all keys in weight_map are in fqn_to_index_mapping
        for hf_key in index_data["weight_map"]:
            assert hf_key in adapter.fqn_to_index_mapping, (
                f"Missing key in fqn_to_index_mapping: {hf_key}"
            )

        # Print summary
        print(f"\nLoaded index with {len(adapter.fqn_to_index_mapping)} weights")
        print("Index maps to files 1-16 (16 safetensors files)")

        # Verify file indices are in expected range
        indices = set(adapter.fqn_to_index_mapping.values())
        assert min(indices) == 1
        assert max(indices) == 16  # 16 files for Qwen3-30B-A3B

    def test_get_hf_storage_reader_moe(self, moe_model_path):
        """Test creating HuggingFaceStorageReader for multi-file checkpoint."""
        from torch.distributed.checkpoint import HuggingFaceStorageReader
        from transformers import AutoConfig

        from areal.experimental.models.archon.qwen3 import Qwen3StateDictAdapter

        config = AutoConfig.from_pretrained(moe_model_path, trust_remote_code=True)
        adapter = Qwen3StateDictAdapter(config, hf_assets_path=moe_model_path)

        # Get reader
        reader = adapter.get_hf_storage_reader(moe_model_path)
        assert isinstance(reader, HuggingFaceStorageReader)

        print(f"\nCreated HuggingFaceStorageReader for {moe_model_path}")

    def test_moe_config_parsing(self, moe_model_path):
        """Test that MoE config is correctly parsed for expert count."""
        from transformers import AutoConfig

        from areal.experimental.models.archon.qwen3 import Qwen3StateDictAdapter

        config = AutoConfig.from_pretrained(moe_model_path, trust_remote_code=True)
        adapter = Qwen3StateDictAdapter(config, hf_assets_path=moe_model_path)

        # Qwen3-30B-A3B has 128 experts
        assert adapter.moe_enabled is True
        assert adapter.num_experts == 128

        print(f"\nMoE config: {adapter.num_experts} experts")


# =============================================================================
# Engine Integration Tests (Multi-GPU)
# =============================================================================


def _run_engine_checkpoint_test(
    n_gpus: int, test_type: str, model_path: str, output_file: str
):
    """Run engine checkpoint test with torchrun."""
    port = find_free_ports(1)[0]
    script_path = "tests/experimental/archon/torchrun/run_checkpoint_tests.py"

    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        script_path,
        f"--test_type={test_type}",
        f"--model_path={model_path}",
        f"--output={output_file}",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")

    # Verify result
    with open(output_file) as f:
        result = f.read().strip()
    assert result == "Passed", f"Test failed: {result}"


@pytest.mark.multi_gpu
@pytest.mark.slow
class TestEngineCheckpointIntegration:
    """End-to-end tests for ArchonEngine checkpoint methods.

    These tests require multiple GPUs and use real model checkpoints.
    Tests use the unified save()/load() interface with HF format.
    """

    @pytest.fixture
    def dense_model_path(self):
        """Get path to Qwen3-0.6B dense model."""
        path = DENSE_MODEL_PATHS.get("qwen3")
        if path is None:
            pytest.skip("Qwen3-0.6B model path not configured")
        if not os.path.exists(path):
            pytest.skip(f"Qwen3-0.6B model not found at {path}")
        return path

    @pytest.fixture
    def moe_model_path(self):
        """Get path to Qwen3-30B-A3B MoE model."""
        path = MOE_MODEL_PATHS.get("qwen3_moe")
        if path is None:
            pytest.skip("Qwen3-30B-A3B model path not configured")
        if not os.path.exists(path):
            pytest.skip(f"Qwen3-30B-A3B model not found at {path}")
        return path

    def test_engine_save_hf_dense_2gpu(self, dense_model_path, tmp_path_factory):
        """Test ArchonEngine.save() with weight_format="hf" on 2 GPUs."""
        if current_platform.device_count() < 2:
            pytest.skip("This test requires 2 GPUs")

        output = tmp_path_factory.mktemp("test_output") / "result.out"
        _run_engine_checkpoint_test(
            n_gpus=2,
            test_type="save_hf_dense",
            model_path=dense_model_path,
            output_file=str(output),
        )

    def test_engine_moe_checkpoint_4gpu(self, moe_model_path, tmp_path_factory):
        """Test MoE checkpoint save/load on 4 GPUs with EP=4."""
        if current_platform.device_count() < 4:
            pytest.skip("This test requires 4 GPUs")

        output = tmp_path_factory.mktemp("test_output") / "result.out"
        _run_engine_checkpoint_test(
            n_gpus=4,
            test_type="moe_checkpoint",
            model_path=moe_model_path,
            output_file=str(output),
        )

    def test_engine_save_load_forward_match_2gpu(
        self, dense_model_path, tmp_path_factory
    ):
        """Test forward output matches before save and after load (2 GPUs).

        This is the most critical test for checkpoint correctness:
        1. Initialize engine and run forward
        2. Save checkpoint
        3. Load checkpoint
        4. Run forward again
        5. Verify outputs match
        """
        if current_platform.device_count() < 2:
            pytest.skip("This test requires 2 GPUs")

        output = tmp_path_factory.mktemp("test_output") / "result.out"
        _run_engine_checkpoint_test(
            n_gpus=2,
            test_type="save_load_forward_match",
            model_path=dense_model_path,
            output_file=str(output),
        )

    def test_engine_save_load_forward_match_1gpu(
        self, dense_model_path, tmp_path_factory
    ):
        """Test forward output matches before save and after load (1 GPU).

        Single GPU version of forward match test for faster CI.
        """
        if current_platform.device_count() < 1:
            pytest.skip("This test requires at least 1 GPU")

        output = tmp_path_factory.mktemp("test_output") / "result.out"
        _run_engine_checkpoint_test(
            n_gpus=1,
            test_type="save_load_forward_match",
            model_path=dense_model_path,
            output_file=str(output),
        )

    def test_engine_save_load_forward_match_with_compile_ac(
        self, dense_model_path, tmp_path_factory
    ):
        """Test forward match with torch.compile + activation checkpointing.

        This test verifies that checkpoint save/load works correctly when the model
        has wrapper prefixes in parameter names (e.g., _orig_mod from torch.compile,
        _checkpoint_wrapped_module from activation checkpointing).
        """
        if current_platform.device_count() < 1:
            pytest.skip("This test requires at least 1 GPU")

        output = tmp_path_factory.mktemp("test_output") / "result.out"
        _run_engine_checkpoint_test(
            n_gpus=1,
            test_type="save_load_forward_match_with_compile_ac",
            model_path=dense_model_path,
            output_file=str(output),
        )


# =============================================================================
# Async Checkpoint Tests (Multi-GPU)
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
class TestAsyncCheckpointIntegration:
    """Tests for async HF checkpoint save with ArchonEngine.

    Verifies that async save produces identical output to sync save.
    """

    @pytest.fixture
    def dense_model_path(self):
        path = DENSE_MODEL_PATHS.get("qwen3")
        if path is None:
            pytest.skip("Qwen3-0.6B model path not configured")
        if not os.path.exists(path):
            pytest.skip(f"Qwen3-0.6B model not found at {path}")
        return path

    def test_async_save_hf_dense_2gpu(self, dense_model_path, tmp_path_factory):
        """Test async HF save matches sync save on 2 GPUs."""
        if current_platform.device_count() < 2:
            pytest.skip("This test requires 2 GPUs")

        output = tmp_path_factory.mktemp("test_output") / "result.out"
        _run_engine_checkpoint_test(
            n_gpus=2,
            test_type="async_save_hf_dense",
            model_path=dense_model_path,
            output_file=str(output),
        )

    def test_async_save_hf_dense_1gpu(self, dense_model_path, tmp_path_factory):
        """Test async HF save matches sync save on 1 GPU."""
        if current_platform.device_count() < 1:
            pytest.skip("This test requires at least 1 GPU")

        output = tmp_path_factory.mktemp("test_output") / "result.out"
        _run_engine_checkpoint_test(
            n_gpus=1,
            test_type="async_save_hf_dense",
            model_path=dense_model_path,
            output_file=str(output),
        )


# =============================================================================
# Weight Comparison Tests
# =============================================================================


class TestWeightComparisonAfterConversion:
    """Tests that verify weight values are preserved after conversion."""

    @pytest.fixture
    def dense_model_path(self):
        """Get path to Qwen3-0.6B dense model."""
        path = DENSE_MODEL_PATHS.get("qwen3")
        if path is None:
            pytest.skip("Qwen3-0.6B model path not configured")
        if not os.path.exists(path):
            pytest.skip(f"Qwen3-0.6B model not found at {path}")
        return path

    @pytest.mark.slow
    def test_adapter_roundtrip_preserves_weights(self, dense_model_path):
        """Test that adapter roundtrip preserves exact weight values."""
        from safetensors.torch import load_file
        from transformers import AutoConfig

        from areal.experimental.models.archon.qwen3 import Qwen3StateDictAdapter

        config = AutoConfig.from_pretrained(dense_model_path, trust_remote_code=True)
        adapter = Qwen3StateDictAdapter(config, hf_assets_path=dense_model_path)

        # Load original weights
        safetensors_path = os.path.join(dense_model_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            original_state = load_file(safetensors_path)
        else:
            # Multi-file checkpoint
            index_path = os.path.join(dense_model_path, "model.safetensors.index.json")
            with open(index_path) as f:
                index = json.load(f)

            original_state = {}
            loaded_files = set()
            for key, filename in index["weight_map"].items():
                if filename not in loaded_files:
                    file_path = os.path.join(dense_model_path, filename)
                    file_state = load_file(file_path)
                    original_state.update(file_state)
                    loaded_files.add(filename)

        # Roundtrip: HF -> Archon -> HF
        archon_state = adapter.from_hf(original_state)
        roundtrip_state = adapter.to_hf(archon_state)

        # Check if weight tying is enabled
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

        # Compare weights
        mismatches = []
        for key in original_state:
            if "rotary_emb" in key:
                continue  # Skip rotary embeddings (not stored)

            # Skip lm_head.weight when tie_word_embeddings is True
            # HF save_pretrained() doesn't save it either (reconstructed from embeddings)
            if tie_word_embeddings and key == "lm_head.weight":
                continue

            if key not in roundtrip_state:
                mismatches.append(f"Missing key: {key}")
                continue

            original = original_state[key]
            roundtrip = roundtrip_state[key]

            if original.shape != roundtrip.shape:
                mismatches.append(
                    f"Shape mismatch for {key}: {original.shape} vs {roundtrip.shape}"
                )
                continue

            if not torch.allclose(original, roundtrip, rtol=1e-5, atol=1e-5):
                max_diff = (original.float() - roundtrip.float()).abs().max().item()
                mismatches.append(f"Value mismatch for {key}: max_diff={max_diff}")

        if mismatches:
            for m in mismatches[:10]:  # Show first 10 mismatches
                print(m)
            pytest.fail(f"Found {len(mismatches)} mismatches in roundtrip")

        skipped_msg = (
            " (lm_head.weight skipped due to tie_word_embeddings)"
            if tie_word_embeddings
            else ""
        )
        print(
            f"\nRoundtrip verified: {len(roundtrip_state)} weights preserved exactly{skipped_msg}"
        )
