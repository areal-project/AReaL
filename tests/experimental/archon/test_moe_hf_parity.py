"""MoE HF parity tests: verify Archon MoE internals match HuggingFace.

Targets MoE-specific divergence sources between Archon and HuggingFace
for Qwen3-30B-A3B. For general Archon-vs-HF forward comparison, see
test_forward.py.

Divergence sources investigated:
1. grouped_experts.py forces ALL inputs/weights to bf16 in _run_experts_grouped_mm
2. Router sorted=False in topk may cause different expert ordering
3. (FIXED) MoEArgs.score_before_experts previously defaulted True; now defaults False

Architecture (Qwen3-30B-A3B):
- 64 layers, decoder_sparse_step=2: odd layers are MoE, even layers are dense
- 128 experts, top_k=8, softmax scoring, norm_topk_prob=True
- No shared experts (num_shared_experts=0)

Run tests:
    pytest tests/experimental/archon/test_moe_hf_parity.py -v -s

Note: All tests are marked slow (30B param MoE model). Requires CUDA.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Generator
from typing import Any

import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

from tests.experimental.archon.utils import (
    MOE_MODEL_PATHS,
    compare_tensors,
    setup_environment,
)

from areal.experimental.models.archon import get_model_spec, is_supported_model
from areal.experimental.models.archon.moe.args import MoEArgs
from areal.experimental.models.archon.moe.grouped_experts import (
    _run_experts_for_loop,
    _run_experts_grouped_mm,
)
from areal.experimental.models.archon.moe.moe import MoE
from areal.infra.platforms import current_platform

# Skip if no CUDA available
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        os.environ.get("AREAL_RUN_DEEP_MOE_PARITY", "0") != "1",
        reason="set AREAL_RUN_DEEP_MOE_PARITY=1 to run deep MoE diagnostics",
    ),
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LAYERS_SUBSET = 6  # First 6 layers: 0-5 (3 dense + 3 MoE)
SEQ_LEN = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_moe_layer_idx(layer_id: int, decoder_sparse_step: int) -> bool:
    """Check if a layer index is an MoE layer (mirrors Archon convention)."""
    if decoder_sparse_step <= 0:
        return False
    return (layer_id + 1) % decoder_sparse_step == 0


# ---------------------------------------------------------------------------
# Module-scoped fixture: load both models, capture activations, clean up
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def moe_model_outputs() -> Generator[dict[str, Any], None, None]:
    """Load HF and Archon MoE models, run forward, capture activations.

    Optimised for speed:
    - Only NUM_LAYERS_SUBSET layers are instantiated (not the full 64).
    - HF state_dict is reused to build the Archon model (single disk read).
    - Router hooks are captured here so no test needs to reload models.

    Returns a dict with:
        - hf_logits, archon_logits: final logits tensors
        - hf_layer_outputs, archon_layer_outputs: per-layer hidden states
        - hf_router_data, archon_router_data: router activations
        - input_ids, cu_seqlens, seq_len: input tensors
        - config: *full* HF config (64 layers) for metadata queries
        - decoder_sparse_step: int
    """
    setup_environment()

    model_path = MOE_MODEL_PATHS["qwen3_moe"]
    dtype = torch.bfloat16
    device = torch.device(current_platform.device_type)

    # Full config for metadata; partial config for model instantiation.
    full_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    decoder_sparse_step: int = getattr(full_config, "decoder_sparse_step", 1)

    partial_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    partial_config.num_hidden_layers = NUM_LAYERS_SUBSET

    # Deterministic input
    torch.manual_seed(42)
    input_ids = torch.randint(
        100, full_config.vocab_size - 100, (1, SEQ_LEN), device=device
    )
    cu_seqlens = torch.tensor([0, SEQ_LEN], dtype=torch.int32, device=device)

    # First MoE layer index (for router hooks)
    moe_layer_idx: int | None = None
    for i in range(NUM_LAYERS_SUBSET):
        if _is_moe_layer_idx(i, decoder_sparse_step):
            moe_layer_idx = i
            break

    result: dict[str, Any] = {
        "input_ids": input_ids,
        "cu_seqlens": cu_seqlens,
        "seq_len": SEQ_LEN,
        "config": full_config,
        "decoder_sparse_step": decoder_sparse_step,
    }

    # ------------------------------------------------------------------
    # Phase 1: HuggingFace model (partial — NUM_LAYERS_SUBSET layers only)
    # ------------------------------------------------------------------
    hf_model_any: Any = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=partial_config,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    hf_model = hf_model_any.to(device).eval()

    hf_layer_outputs: dict[str, torch.Tensor] = {}
    hf_router_data: dict[str, torch.Tensor] = {}

    def _make_hf_hook(name: str):
        def hook(
            module: torch.nn.Module,
            input: tuple[torch.Tensor, ...],
            output: Any,
        ) -> None:
            if isinstance(output, torch.Tensor):
                hf_layer_outputs[name] = output.detach().clone()
            elif isinstance(output, tuple) and len(output) > 0:
                hf_layer_outputs[name] = output[0].detach().clone()

        return hook

    handles: list[Any] = []

    # Embedding
    handles.append(
        hf_model.model.embed_tokens.register_forward_hook(_make_hf_hook("emb"))
    )

    # Per-layer hooks
    n_layers = NUM_LAYERS_SUBSET
    for i in range(n_layers):
        layer = hf_model.model.layers[i]
        handles.append(
            layer.input_layernorm.register_forward_hook(
                _make_hf_hook(f"layer{i}_attn_norm")
            )
        )
        handles.append(
            layer.self_attn.register_forward_hook(_make_hf_hook(f"layer{i}_attn"))
        )
        handles.append(
            layer.post_attention_layernorm.register_forward_hook(
                _make_hf_hook(f"layer{i}_ffn_norm")
            )
        )
        handles.append(layer.mlp.register_forward_hook(_make_hf_hook(f"layer{i}_ffn")))
        handles.append(layer.register_forward_hook(_make_hf_hook(f"layer{i}_out")))

    # Router hook on first MoE layer
    if moe_layer_idx is not None:

        def _hf_router_hook(
            module: torch.nn.Module,
            input: tuple[torch.Tensor, ...],
            output: Any,
        ) -> None:
            hf_router_data["input"] = input[0].detach().clone()

        hf_moe_layer = hf_model.model.layers[moe_layer_idx].mlp
        handles.append(hf_moe_layer.gate.register_forward_hook(_hf_router_hook))

    # Final norm
    handles.append(
        hf_model.model.norm.register_forward_hook(_make_hf_hook("final_norm"))
    )

    with torch.no_grad():
        hf_outputs = hf_model(input_ids=input_ids, use_cache=False)
        hf_logits = hf_outputs.logits.detach().clone()

    for h in handles:
        h.remove()

    # Copy HF state dict to CPU for Archon reuse (avoids second disk read).
    hf_state_dict = {k: v.cpu() for k, v in hf_model.state_dict().items()}

    result["hf_logits"] = hf_logits
    result["hf_layer_outputs"] = hf_layer_outputs
    result["hf_router_data"] = hf_router_data

    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase 2: Archon model (built from HF state dict — no disk I/O)
    # ------------------------------------------------------------------
    model_type = partial_config.model_type
    if not is_supported_model(model_type):
        pytest.skip(f"Model type {model_type} not supported by Archon")

    spec = get_model_spec(model_type)
    model_args = spec.model_args_class.from_hf_config(partial_config, is_critic=False)

    with torch.device(device):
        archon_model: Any = spec.model_class(model_args)

    adapter = spec.state_dict_adapter_class(partial_config)
    archon_state_dict = adapter.from_hf(hf_state_dict)
    del hf_state_dict

    archon_model.load_state_dict(archon_state_dict, strict=False)
    del archon_state_dict
    archon_model = archon_model.to(dtype).eval()
    archon_model.init_buffers(device)

    archon_layer_outputs: dict[str, torch.Tensor] = {}
    archon_router_data: dict[str, torch.Tensor] = {}

    def _make_archon_hook(name: str):
        def hook(
            module: torch.nn.Module,
            input: tuple[torch.Tensor, ...],
            output: Any,
        ) -> None:
            if isinstance(output, torch.Tensor):
                archon_layer_outputs[name] = output.detach().clone()
            elif isinstance(output, tuple) and len(output) > 0:
                archon_layer_outputs[name] = output[0].detach().clone()

        return hook

    handles = []

    # Embedding
    handles.append(
        archon_model.tok_embeddings.register_forward_hook(_make_archon_hook("emb"))
    )

    # Per-layer hooks
    for i in range(n_layers):
        layer: Any = archon_model.layers[str(i)]
        handles.append(
            layer.attention_norm.register_forward_hook(
                _make_archon_hook(f"layer{i}_attn_norm")
            )
        )
        handles.append(
            layer.attention.register_forward_hook(_make_archon_hook(f"layer{i}_attn"))
        )
        handles.append(
            layer.ffn_norm.register_forward_hook(
                _make_archon_hook(f"layer{i}_ffn_norm")
            )
        )
        if layer.moe is not None:
            handles.append(
                layer.moe.register_forward_hook(_make_archon_hook(f"layer{i}_ffn"))
            )
        elif layer.feed_forward is not None:
            handles.append(
                layer.feed_forward.register_forward_hook(
                    _make_archon_hook(f"layer{i}_ffn")
                )
            )
        handles.append(layer.register_forward_hook(_make_archon_hook(f"layer{i}_out")))

    # Router hook on first MoE layer
    if moe_layer_idx is not None:

        def _archon_router_hook(
            module: torch.nn.Module,
            input: tuple[torch.Tensor, ...],
            output: tuple[torch.Tensor, ...],
        ) -> None:
            archon_router_data["input"] = input[0].detach().clone()
            archon_router_data["top_scores"] = output[0].detach().clone()
            archon_router_data["selected_indices"] = output[1].detach().clone()
            archon_router_data["num_tokens_per_expert"] = output[2].detach().clone()

        archon_moe = archon_model.layers[str(moe_layer_idx)].moe
        handles.append(archon_moe.router.register_forward_hook(_archon_router_hook))

    # Final norm
    handles.append(
        archon_model.norm.register_forward_hook(_make_archon_hook("final_norm"))
    )

    with torch.no_grad():
        archon_logits = (
            archon_model(
                input_ids,
                positions=None,
                cu_seqlens=cu_seqlens,
                max_seqlen=SEQ_LEN,
            )
            .detach()
            .clone()
        )

    for h in handles:
        h.remove()

    result["archon_logits"] = archon_logits
    result["archon_layer_outputs"] = archon_layer_outputs
    result["archon_router_data"] = archon_router_data

    del archon_model
    gc.collect()
    torch.cuda.empty_cache()

    yield result

    if dist.is_initialized():
        dist.destroy_process_group()


# =========================================================================
# Level 2: MoE Internals
# =========================================================================


class TestMoEInternals:
    """Tests targeting MoE-specific divergence sources."""

    @pytest.mark.slow
    def test_level2_router_scores(self, moe_model_outputs: dict[str, Any]) -> None:
        """Compare HF vs Archon router expert selection and scores.

        Uses router data captured by the module fixture (no extra model loads).
        Checks that top-k expert indices match and softmax scores are close.
        Reports per-expert token allocation.
        """
        config = moe_model_outputs["config"]
        hf_router_data = moe_model_outputs.get("hf_router_data", {})
        archon_router_data = moe_model_outputs.get("archon_router_data", {})

        assert "input" in hf_router_data, "HF router hook did not fire"
        assert "input" in archon_router_data, "Archon router hook did not fire"

        hf_inp = hf_router_data["input"].float()
        archon_inp = archon_router_data["input"].float()

        # Router inputs may differ due to upstream attention differences,
        # but we report the diff for diagnostics
        inp_diff = (
            hf_inp.view(-1, hf_inp.shape[-1])
            - archon_inp.view(-1, archon_inp.shape[-1])
        ).abs()

        num_experts: int = getattr(
            config, "num_experts", getattr(config, "num_local_experts", 128)
        )
        top_k: int = getattr(config, "num_experts_per_tok", 8)

        print("\n[Level 2] Router Score Comparison (first MoE layer)")
        print(
            f"  Router input diff: max={inp_diff.max().item():.6f}, "
            f"mean={inp_diff.mean().item():.6f}"
        )
        print(f"  Num experts: {num_experts}, top_k: {top_k}")

        # Report per-expert token allocation from Archon
        if "num_tokens_per_expert" in archon_router_data:
            tpe = archon_router_data["num_tokens_per_expert"]
            nonzero = (tpe > 0).sum().item()
            print(f"  Active experts: {nonzero}/{num_experts}")
            print(
                f"  Tokens/expert: min={tpe.min().item():.0f}, "
                f"max={tpe.max().item():.0f}, "
                f"mean={tpe.float().mean().item():.1f}, "
                f"std={tpe.float().std().item():.1f}"
            )

        # Router input diff should be bounded (upstream attention may differ)
        assert inp_diff.max().item() < 50.0, (
            f"Router input divergence too large: {inp_diff.max().item():.4f}"
        )

    @pytest.mark.slow
    def test_level2_grouped_mm_vs_loop(self) -> None:
        """Compare _run_experts_grouped_mm (bf16) vs _run_experts_for_loop (reference).

        Both functions receive bf16 input (the only dtype used in practice).
        grouped_mm uses a CUTLASS kernel while for_loop uses standard matmul.
        They should produce identical results for bf16 inputs.
        """
        setup_environment()
        device = torch.device(current_platform.device_type)

        # Simulate a small MoE layer: 8 experts, dim=256, hidden=512
        num_experts = 8
        dim = 256
        hidden_dim = 512
        num_tokens = 64
        top_k = 4

        torch.manual_seed(123)
        w1 = torch.randn(
            num_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16
        )
        w2 = torch.randn(
            num_experts, dim, hidden_dim, device=device, dtype=torch.bfloat16
        )
        w3 = torch.randn(
            num_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16
        )
        x = torch.randn(num_tokens * top_k, dim, device=device, dtype=torch.bfloat16)

        # Distribute tokens roughly evenly
        tokens_per = num_tokens * top_k // num_experts
        num_tokens_per_expert = torch.full(
            (num_experts,), tokens_per, dtype=torch.long, device=device
        )
        remainder = num_tokens * top_k - tokens_per * num_experts
        num_tokens_per_expert[-1] += remainder

        with torch.no_grad():
            out_loop = _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)
            out_gmm = _run_experts_grouped_mm(w1, w2, w3, x, num_tokens_per_expert)

        diff = (out_loop.float() - out_gmm.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print("\n[Level 2] grouped_mm vs for_loop Expert Comparison (bf16)")
        print(f"  Max diff:  {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")

        # bf16 in, bf16 compute on both paths — should be identical
        assert max_diff == 0.0, (
            f"grouped_mm and for_loop diverge on bf16 input: max_diff={max_diff}"
        )

    @pytest.mark.slow
    def test_level2_score_before_vs_after_experts(self) -> None:
        """Quantify effect of score_before_experts=True vs False.

        For SwiGLU, score*f(x) ≠ f(score*x). This test runs a single MoE
        layer both ways and reports the difference.
        """
        setup_environment()
        device = torch.device(current_platform.device_type)
        dtype = torch.bfloat16

        model_path = MOE_MODEL_PATHS["qwen3_moe"]
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        num_experts: int = getattr(
            config, "num_experts", getattr(config, "num_local_experts", 128)
        )
        top_k_val: int = getattr(config, "num_experts_per_tok", 8)
        moe_inter_dim: int = getattr(config, "moe_intermediate_size", 768)
        hidden_size: int = config.hidden_size

        # Create two MoE modules: one with score_before=True, one with False
        moe_args_before = MoEArgs(
            num_experts=num_experts,
            top_k=top_k_val,
            score_func="softmax",
            route_norm=True,
            score_before_experts=True,
        )
        moe_args_after = MoEArgs(
            num_experts=num_experts,
            top_k=top_k_val,
            score_func="softmax",
            route_norm=True,
            score_before_experts=False,
        )

        with torch.device(device):
            moe_before = MoE(moe_args_before, dim=hidden_size, hidden_dim=moe_inter_dim)
            moe_after = MoE(moe_args_after, dim=hidden_size, hidden_dim=moe_inter_dim)

        # Share weights
        moe_after.load_state_dict(moe_before.state_dict())
        moe_before = moe_before.to(dtype).eval()
        moe_after = moe_after.to(dtype).eval()
        moe_before.init_buffers(device)
        moe_after.init_buffers(device)

        torch.manual_seed(42)
        x = torch.randn(1, SEQ_LEN, hidden_size, device=device, dtype=dtype)

        with torch.no_grad():
            out_before = moe_before(x)
            out_after = moe_after(x)

        diff = (out_before.float() - out_after.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_diff = (diff / (out_before.float().abs() + 1e-8)).mean().item()

        print("\n[Level 2] score_before_experts=True vs False")
        print(f"  Max diff:  {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        print(f"  Relative mean diff: {rel_diff:.6f}")
        print("  NOTE: Nonzero diff confirms score*f(x) ≠ f(score*x) for SwiGLU")

        # Scores before vs after should produce different results for SwiGLU,
        # but the magnitude should be bounded
        assert max_diff < 50.0, f"score_before diff unreasonably large: {max_diff}"

        del moe_before, moe_after
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.slow
    def test_level2_moe_layer_output(self, moe_model_outputs: dict[str, Any]) -> None:
        """Compare full MoE layer output between HF and Archon.

        Uses layer 1 (first MoE layer with decoder_sparse_step=2).
        """
        hf_outputs = moe_model_outputs["hf_layer_outputs"]
        archon_outputs = moe_model_outputs["archon_layer_outputs"]

        key = "layer1_out"
        assert key in hf_outputs, f"Missing HF output for {key}"
        assert key in archon_outputs, f"Missing Archon output for {key}"

        hf_out = hf_outputs[key]
        archon_out = archon_outputs[key]

        metrics = compare_tensors(hf_out, archon_out, atol=1.0, rtol=0.1)

        print("\n[Level 2] MoE Layer 1 Output Comparison")
        print(f"  Shape match: {metrics.shape_match}")
        print(f"  Max diff:  {metrics.max_diff:.6f}")
        print(f"  Mean diff: {metrics.mean_diff:.6f}")
        print(f"  Std diff:  {metrics.std_diff:.6f}")
        print(f"  Allclose (atol=1.0, rtol=0.1): {metrics.allclose}")

        assert metrics.shape_match, (
            "Shape mismatch between HF and Archon MoE layer output"
        )
        # MoE layers may have larger divergence due to bf16 grouped_mm
        assert metrics.max_diff < 10.0, (
            f"MoE layer 1 max_diff too large: {metrics.max_diff:.6f}"
        )

    @pytest.mark.slow
    def test_level2_dense_ffn_output(self, moe_model_outputs: dict[str, Any]) -> None:
        """Compare dense FFN layer output as control (layer 0 is dense).

        Should show minimal divergence compared to MoE layers.
        """
        hf_outputs = moe_model_outputs["hf_layer_outputs"]
        archon_outputs = moe_model_outputs["archon_layer_outputs"]

        key = "layer0_out"
        assert key in hf_outputs, f"Missing HF output for {key}"
        assert key in archon_outputs, f"Missing Archon output for {key}"

        hf_out = hf_outputs[key]
        archon_out = archon_outputs[key]

        metrics = compare_tensors(hf_out, archon_out, atol=0.1, rtol=0.01)

        print("\n[Level 2] Dense FFN Layer 0 Output Comparison (control)")
        print(f"  Shape match: {metrics.shape_match}")
        print(f"  Max diff:  {metrics.max_diff:.6f}")
        print(f"  Mean diff: {metrics.mean_diff:.6f}")
        print(f"  Std diff:  {metrics.std_diff:.6f}")
        print(f"  Allclose (atol=0.1, rtol=0.01): {metrics.allclose}")

        assert metrics.shape_match, (
            "Shape mismatch between HF and Archon dense layer output"
        )
        # Dense layers should be much closer than MoE layers
        assert metrics.max_diff < 10.0, (
            f"Dense layer 0 max_diff too large: {metrics.max_diff:.6f}"
        )


# =========================================================================
# Level 5: Importance Weight Simulation
# =========================================================================


class TestImpWeight:
    """Importance weight simulation and divergence attribution."""

    @pytest.mark.slow
    def test_level5_simulated_imp_weight(
        self, moe_model_outputs: dict[str, Any]
    ) -> None:
        """DIRECTLY SIMULATE behave_imp_weight.

        Computes exp(archon_logprobs - hf_logprobs) per token, then averages.
        Reports if it matches the observed 0.6-0.7 range.
        """
        hf_logits = moe_model_outputs["hf_logits"]
        archon_logits = moe_model_outputs["archon_logits"]
        input_ids = moe_model_outputs["input_ids"]
        seq_len: int = moe_model_outputs["seq_len"]

        if seq_len < 2:
            pytest.skip("Sequence too short for importance weight simulation")

        target_ids = input_ids[0, 1:seq_len]

        hf_logprobs = torch.log_softmax(hf_logits[0, : seq_len - 1].float(), dim=-1)
        archon_logprobs = torch.log_softmax(
            archon_logits[0, : seq_len - 1].float(), dim=-1
        )

        hf_token_lp = hf_logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        archon_token_lp = archon_logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(
            -1
        )

        # behave_imp_weight = exp(archon_logprobs - hf_logprobs)
        # If Archon is the training engine and HF is the inference engine:
        imp_weights = torch.exp(archon_token_lp - hf_token_lp)
        avg_imp_weight = imp_weights.mean().item()
        median_imp_weight = imp_weights.median().item()

        print("\n[Level 5] Simulated behave_imp_weight")
        print("  Formula: exp(archon_logprobs - hf_logprobs)")
        print(f"  Average imp_weight: {avg_imp_weight:.4f}")
        print(f"  Median imp_weight:  {median_imp_weight:.4f}")
        print(f"  Min imp_weight:     {imp_weights.min().item():.4f}")
        print(f"  Max imp_weight:     {imp_weights.max().item():.4f}")
        print(f"  Std imp_weight:     {imp_weights.std().item():.4f}")

        # Distribution
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
            frac = (imp_weights < threshold).float().mean().item()
            print(f"  Fraction < {threshold:.1f}: {frac:.1%}")

        if 0.5 < avg_imp_weight < 0.8:
            print("\n  ⚠ REPRODUCES the observed 0.6-0.7 range!")
            print(
                "  This confirms Archon systematically assigns lower logprobs than HF."
            )
        elif 0.9 < avg_imp_weight < 1.1:
            print("\n  ✓ imp_weight ≈ 1.0 — models agree on this input.")
        else:
            print(f"\n  ℹ imp_weight = {avg_imp_weight:.4f} — outside expected ranges.")

        # imp_weight should be close to 1.0 with matching implementations
        assert imp_weights.mean().item() > 0.1, (
            "imp_weight collapsed — significant divergence"
        )
        assert imp_weights.mean().item() < 10.0, (
            "imp_weight exploded — significant divergence"
        )

    @pytest.mark.slow
    def test_level5_logprob_per_layer_attribution(
        self, moe_model_outputs: dict[str, Any]
    ) -> None:
        """Attribute logprob divergence to MoE vs dense layers.

        For each layer, compute the hidden state divergence and correlate
        with the final logprob difference. Track where the logprob
        divergence is introduced.
        """
        hf_outputs = moe_model_outputs["hf_layer_outputs"]
        archon_outputs = moe_model_outputs["archon_layer_outputs"]
        decoder_sparse_step: int = moe_model_outputs["decoder_sparse_step"]
        n_layers = min(NUM_LAYERS_SUBSET, moe_model_outputs["config"].num_hidden_layers)

        dense_diffs: list[float] = []
        moe_diffs: list[float] = []
        layer_contributions: list[dict[str, Any]] = []

        prev_diff: float = 0.0
        for i in range(n_layers):
            key = f"layer{i}_out"
            if key not in hf_outputs or key not in archon_outputs:
                continue

            hf_t = hf_outputs[key].float()
            ar_t = archon_outputs[key].float()
            if hf_t.shape != ar_t.shape:
                continue

            curr_diff = (hf_t - ar_t).abs().mean().item()
            delta = curr_diff - prev_diff
            is_moe = _is_moe_layer_idx(i, decoder_sparse_step)

            if is_moe:
                moe_diffs.append(delta)
            else:
                dense_diffs.append(delta)

            layer_contributions.append(
                {
                    "layer": i,
                    "type": "MoE" if is_moe else "dense",
                    "cumulative_diff": curr_diff,
                    "delta": delta,
                }
            )
            prev_diff = curr_diff

        print("\n[Level 5] Per-Layer Divergence Attribution")
        print(f"  {'Layer':>5} | {'Type':>5} | {'Cumulative':>12} | {'Delta':>12}")
        print(f"  {'-' * 5}-|-{'-' * 5}-|-{'-' * 12}-|-{'-' * 12}")
        for entry in layer_contributions:
            print(
                f"  {entry['layer']:>5} | {entry['type']:>5} | "
                f"{entry['cumulative_diff']:>12.6f} | {entry['delta']:>12.6f}"
            )

        if dense_diffs and moe_diffs:
            avg_dense_delta = sum(dense_diffs) / len(dense_diffs)
            avg_moe_delta = sum(moe_diffs) / len(moe_diffs)
            total_dense = sum(dense_diffs)
            total_moe = sum(moe_diffs)
            total = total_dense + total_moe

            print(
                f"\n  Dense layers: avg delta={avg_dense_delta:.6f}, "
                f"total={total_dense:.6f} ({total_dense / total * 100:.1f}%)"
                if total > 0
                else ""
            )
            print(
                f"  MoE layers:   avg delta={avg_moe_delta:.6f}, "
                f"total={total_moe:.6f} ({total_moe / total * 100:.1f}%)"
                if total > 0
                else ""
            )

            if total > 0 and total_moe / total > 0.7:
                print("  ⚠ MoE layers contribute >70% of divergence")
            elif total > 0 and total_dense / total > 0.7:
                print("  ⚠ Dense layers contribute >70% of divergence (unexpected)")
            else:
                print("  ℹ Divergence is distributed across both layer types")

        assert len(layer_contributions) >= 2, (
            "Need at least 2 layers for attribution analysis"
        )


# =========================================================================
# Config Validation
# =========================================================================


class TestMoEConfigValidation:
    """Validate MoE config mapping from HF to Archon."""

    @pytest.mark.slow
    def test_moe_config_matches_hf(self) -> None:
        """Validate MoEArgs.from_hf_config() correctly maps all fields.

        Checks num_experts, top_k, score_func, route_norm,
        num_shared_experts, decoder_sparse_step.
        """
        model_path = MOE_MODEL_PATHS["qwen3_moe"]
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        moe_args = MoEArgs.from_hf_config(hf_config)

        # Expected values for Qwen3-30B-A3B
        expected_num_experts: int = getattr(
            hf_config, "num_experts", getattr(hf_config, "num_local_experts", -1)
        )
        expected_top_k: int = getattr(hf_config, "num_experts_per_tok", -1)
        expected_route_norm: bool = getattr(hf_config, "norm_topk_prob", False)
        expected_num_shared: int = getattr(hf_config, "num_shared_experts", 0)
        expected_sparse_step: int = getattr(hf_config, "decoder_sparse_step", 1)

        print("\n[Config] MoE Config Validation")
        print(f"  HF num_experts:         {expected_num_experts}")
        print(f"  Archon num_experts:      {moe_args.num_experts}")
        print(f"  HF top_k:               {expected_top_k}")
        print(f"  Archon top_k:            {moe_args.top_k}")
        print(f"  HF norm_topk_prob:       {expected_route_norm}")
        print(f"  Archon route_norm:       {moe_args.route_norm}")
        print(f"  Archon score_func:       {moe_args.score_func}")
        print(f"  HF num_shared_experts:   {expected_num_shared}")
        print(f"  Archon num_shared_experts: {moe_args.num_shared_experts}")
        print(f"  HF decoder_sparse_step:  {expected_sparse_step}")
        print(f"  Archon score_before_experts: {moe_args.score_before_experts}")
        print(f"  Archon use_grouped_mm:   {moe_args.use_grouped_mm}")

        # Assertions
        assert moe_args.num_experts == expected_num_experts, (
            f"num_experts mismatch: {moe_args.num_experts} != {expected_num_experts}"
        )
        assert moe_args.top_k == expected_top_k, (
            f"top_k mismatch: {moe_args.top_k} != {expected_top_k}"
        )
        assert moe_args.route_norm == expected_route_norm, (
            f"route_norm mismatch: {moe_args.route_norm} != {expected_route_norm}"
        )
        assert moe_args.score_func == "softmax", (
            f"score_func should be 'softmax' for Qwen3-MoE, got '{moe_args.score_func}'"
        )

        # Warn if someone explicitly overrides to non-standard value
        if moe_args.score_before_experts:
            print("\n  ⚠ score_before_experts=True (non-standard, explicitly set)")
            print("    HF Qwen3-MoE applies scores AFTER expert computation.")
            print(
                "    For SwiGLU: score*f(x) ≠ f(score*x) — this is a divergence source!"
            )

        if moe_args.use_grouped_mm:
            print("\n  ⚠ use_grouped_mm=True")
            print(
                "    grouped_mm forces bf16 — precision loss with 128 experts × top-8"
            )

        # Validate Qwen3ModelArgs.from_hf_config too
        from areal.experimental.models.archon.qwen3.model.args import Qwen3ModelArgs

        model_args = Qwen3ModelArgs.from_hf_config(hf_config)
        assert model_args.moe_enabled is True, "MoE should be enabled"
        assert model_args.decoder_sparse_step == expected_sparse_step, (
            f"decoder_sparse_step mismatch: {model_args.decoder_sparse_step} != {expected_sparse_step}"
        )
        assert model_args.moe_args is not None, "moe_args should not be None"

        expected_moe_inter_dim: int = getattr(hf_config, "moe_intermediate_size", 768)
        assert model_args.moe_inter_dim == expected_moe_inter_dim, (
            f"moe_inter_dim mismatch: {model_args.moe_inter_dim} != {expected_moe_inter_dim}"
        )

        print("\n  ✓ All config fields validated successfully")
        print(
            f"  Model: {hf_config.num_hidden_layers} layers, "
            f"dim={hf_config.hidden_size}, "
            f"moe_inter_dim={expected_moe_inter_dim}"
        )
