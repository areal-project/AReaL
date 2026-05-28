# BailingMoeV2.5 Adaptation Guide for AReaL

## 1. Architecture Overview

BailingMoeV2.5 is a **heterogeneous** Transformer with 4 types of layer combinations:

```
BailingMoeV2_5ForCausalLM
  Embedding → N TransformerLayers (heterogeneous) → Final LayerNorm → lm_head
```

Each layer = **Attention** (Lightning or MLA) + **MLP** (Dense or MoE).

### Layer Assignment Rules

- **Attention**: Every `layer_group_size`-th layer uses MLA; others use Lightning
  Attention. Formula: `is_lightning = (layer_number + 1) % layer_group_size != 0`
- **MLP**: First `first_k_dense_replace` layers use Dense; rest use MoE (256 experts,
  sigmoid routing, grouped TopK with `n_group=8`, `topk_group=4`, shared experts, expert
  bias).

### Attention Mechanisms

**Lightning Attention** (linear, O(n)):

- Fused QKV → Q/K LayerNorm → Partial RoPE (first half of head_dim only) →
  `chunk_simple_gla` kernel (fla library) → Gate (sigmoid gating with `g_proj` +
  `g_norm` on attn output) → Output proj
- Per-head decay via ALiBi slopes, scaled by layer depth:
  `g_gamma = -alibi_slopes * (1 - (layer_idx - 1) / (N - 1) + 1e-5)`

**MLA (Multi-Latent Attention)** (standard softmax):

- Low-rank KV compression via `q_a_proj → q_a_layernorm → q_b_proj` and
  `kv_a_proj → kv_a_layernorm → kv_b_proj`
- Reuses megatron-core `MLASelfAttention` directly

### Model Variants

| Model | model_type         | Layers | Hidden | q_lora_rank | rope_theta | layer_group_size |
| ----- | ------------------ | ------ | ------ | ----------- | ---------- | ---------------- |
| Mini  | bailing_moe_linear | 20     | 2048   | null        | 10000      | 5                |
| Flash | bailing_hybrid     | 32     | 4096   | 1536        | 6000000    | 8                |

______________________________________________________________________

## 2. Implementation: Key Files & Design

### File Map

| File                                        | Purpose                                                               |
| ------------------------------------------- | --------------------------------------------------------------------- |
| `areal/models/mcore/lightning_attention.py` | Custom Lightning Attention module for megatron-core                   |
| `areal/models/mcore/bailing_moe.py`         | HF config → `MLATransformerConfig` + heterogeneous layer spec builder |
| `areal/models/mcore/bailing_moe_bridge.py`  | mbridge weight mapping (HF ↔ mcore)                                   |
| `areal/models/mcore/registry.py`            | Architecture dispatch (`BailingMoeV2_5ForCausalLM` branch)            |
| `areal/engine/core/model.py`                | `VALID_MOE_MODELS` list                                               |
| `areal/engine/megatron_utils/megatron.py`   | `convert_bailingmoe_to_hf()` for NCCL weight push                     |
| `areal/models/mcore/hf_load.py`             | `_load_fused_qkv_weight()` for HF→megatron QKV format                 |

### Heterogeneous Layer Spec Construction

Cannot use single `get_gpt_decoder_block_spec()` (assumes uniform layers). Build 4 spec
variants:

1. Generate base specs with `get_gpt_layer_with_transformer_engine_spec()` for
   MLA+Dense, MLA+MoE, Standard+Dense, Standard+MoE
1. Deep-copy Standard specs and replace `self_attention` with custom
   `LightningSelfAttention`
1. Per-layer: select spec based on `is_lightning_layer()` and `is_moe`

### QKV Format Conversion

HF stores `[Q_all, K_all, V_all]` concatenated `[3, H, D, hidden]`. Megatron expects
per-head interleaved `[H, 3, D, hidden]`. Conversion: `permute(1, 0, 2, 3)`. Inverse for
NCCL push is the same permute in reverse direction. Both load and save paths verified as
precise inverses.

### Key Design Decisions

- `rotary_percent=1.0` in AReaL (not 0.5 as in HybridEngine) because megatron-core MLA
  passes `qk_pos_emb_head_dim=64`, so `64 * 1.0 = 64` gives correct RoPE dim
- Lightning layers create their own `RotaryEmbedding` (different RoPE dim from MLA)
- Uses `chunk_simple_gla` (not `chunk_lightning_attn`) to pass custom ALiBi g_gamma

______________________________________________________________________

## 3. Critical Bug Fixes

### Bug A: `layer_idx` Offset Error (root cause of loss=7.8)

**Symptom**: Initial loss 7.8 instead of expected \<=2.

**Root cause**: HF reference uses `(self.layer_idx - 1)` in slope formula (0-indexed
minus 1), which gives layer 0 a scale of `1.0526`. AReaL intentionally uses
`self.layer_idx` directly (layer 0 scale = `1.0`), matching SGLang and HybridEngine
behavior. The original bug was that the slope formula was missing entirely.

**Fix**: `layer_scale = 1.0 - self.layer_idx / max(self.num_layers - 1, 1) + 1e-5`

### Bug B: `g` vs `g_gamma` Parameter

`g` and `g_gamma` are mathematically equivalent for constant decay. But the `g` path
computes backward gradients on the constant buffer via
`chunk_local_cumsum(dg, reverse=True)`, introducing gradient noise that raised loss from
7.8 to 14.42. `g_gamma` is the correct choice.

### Bug C: `gate_norm` Applied Twice

Gate norm was applied to both gate and attn_output using the same weight. HF reference
norms only `attn_output`, then multiplies by `sigmoid(raw_gate)`.

**Fix**: Remove `gate = self.gate_norm(gate)`, keep only
`attn_output = self.gate_norm(attn_output)`.

### Bug D: Missing QKV Format Conversion

TP>1 weight loading silently produced garbage without format conversion.

**Fix**: Added `_load_fused_qkv_weight()` implementing `[3,H,D,hidden] → [H,3,D,hidden]`
permutation + per-head TP slicing.

### Bug E: `q_lora_rank` Default Inconsistency

`bailing_moe.py` defaulted to 512, bridge defaulted to None. Unified to None.

______________________________________________________________________

## 4. Context Parallelism (CP) Support

### Strategy: All-to-All Head Parallelism

For Lightning Attention layers (linear attention is head-independent):

```
Input: [S/CP, B, H_local, D]
  → all_to_all CP→HP → [S, B, H_local/CP, D]  (full seq, fewer heads)
  → zigzag undo (restore sequential order)
  → RoPE (BSHD path, continuous positions)
  → chunk_simple_gla (g_gamma CP-sliced)
  → zigzag redo
  → all_to_all HP→CP → [S/CP, B, H_local, D]
```

For MLA layers: megatron-core/TE ring attention (built-in CP).

### MLA RoPE Monkey-Patch

**Problem**: megatron-core 0.13.1 truncates `rotary_pos_emb` to `[0:total_tokens/CP]`,
but `_get_thd_freqs_on_this_cp_rank` needs the full-length freq table for zigzag
position indexing.

**Solution**: Detect truncated freqs, rebuild to full length using
`freqs[p] = p * freqs[1]` (valid for standard RoPE where `outer(positions, inv_freq)` is
linear in position).

### Sequence Alignment Constraint

All CP paths require: `each sub-sequence length ≡ 0 (mod tp_size × cp_size × 2)`.

### Gradient Propagation

`postprocess_packed_seqs_context_parallel` uses `output.detach()` for all-gather, then
replaces the local rank's slice with the original (gradient-carrying) tensor. This is a
correct gradient-partitioning strategy: each CP rank backpropagates only through its own
tokens.

______________________________________________________________________

## 5. CP=8 NCCL SIGSEGV Crash

**Symptom**: CP=8 with `megatron:(attn:d2p2t2c8|ffn:d2p2e16)` on 64 GPUs crashes at step
110-112 with SIGSEGV.

**Root cause**: Ray scheduler does not set `NCCL_ASYNC_ERROR_HANDLING=1` (unlike torch
elastic agent). Without this, NCCL communicators run in **blocking mode**. With 5+
communicators (TP, PP, CP, EP, DP) competing on the same GPU, blocking `ncclGroupEnd()`
calls create resource contention that gradually corrupts NCCL internal state.

**Fix**: Add `NCCL_ASYNC_ERROR_HANDLING: "1"` to worker env_vars. Recommended as global
default.

**Why CP=4 didn't crash**: Fewer communicators (3-4), insufficient contention.

**Why `NCCL_DEBUG=INFO` also prevented crash**: Extra logging I/O shifted timing
(Heisenbug).

______________________________________________________________________

## 6. Configuration Examples

### SFT (no CP)

```yaml
allocation_mode: megatron:d8p4t2e8   # 64 GPUs
actor:
  dtype: bfloat16
  gradient_checkpointing: true
```

### CP=8

```yaml
allocation_mode: megatron:(attn:d2p2t2c8|ffn:d2p2e16)
env_vars:
  NCCL_ASYNC_ERROR_HANDLING: "1"   # Required for CP>=8 with Ray
```

### Performance Results

| Metric              | CP=4           | CP=8                      |
| ------------------- | -------------- | ------------------------- |
| Step time           | 106.66s        | **47.28s** (2.25x faster) |
| GPU mem (allocated) | 57.88 GB (73%) | **41.00 GB (52%)**        |
| Loss convergence    | Normal         | Normal (0.696 → 0.380)    |

### Environment Dependencies

| Component                    | Version     |
| ---------------------------- | ----------- |
| flash-linear-attention (fla) | >= 0.3.0    |
| SGLang                       | >= 0.5.9    |
| TransformerEngine            | 2.11.0      |
| torch                        | 2.6.0+cu126 |

______________________________________________________________________

## 7. Validation: AReaL vs HybridEngine

Mini model comparison (real text, seq_len=512, EP=8):

| Metric | AReaL | HybridEngine | Diff  |
| ------ | ----- | ------------ | ----- |
| Loss   | 1.512 | 1.491        | 1.33% |
| PPL    | 4.54  | 4.44         | 2.2%  |

Per-layer: Lightning layers cosine 0.999+, MLA layers 0.92-0.95 (from TE/flash_attn
version differences, not a bug).

______________________________________________________________________

## 8. Known Open Issues

1. ~~**expert_bias HF name**~~: Fixed — NCCL path now uses `expert_bias` (matching HF
   checkpoint).
1. **kv_channels may be None**: `hf_to_mcore_base_args()` reads `head_dim` which
   BailingMoe doesn't have. Should explicitly set `kv_channels`.
1. **BailingHybridForCausalLM not in registry.py**: Bridge registers `bailing_hybrid`
   but `registry.py` only checks `BailingMoeV2_5ForCausalLM` and
   `BailingMoeLinearForCausalLM`.
