---
name: add-archon-model
description: Guide for adding a new model to the Archon engine. Use when user wants to add support for a new HuggingFace model architecture in ArchonEngine.
---

# Add Archon Model

Add support for a new HuggingFace model architecture in the Archon training engine.

## When to Use

This skill is triggered when:

- User asks "how do I add a model to Archon?"
- User wants to support a new model family (e.g., Llama, Mistral, DeepSeek) in
  ArchonEngine
- User mentions adding a new `ModelSpec` or model type for Archon

## Prerequisites

Before starting, ensure:

- The target model is available on HuggingFace (has `config.json` with `model_type`)
- You know the HuggingFace model ID (e.g., `meta-llama/Llama-3-8B`)
- The model uses a standard transformer architecture (decoder-only)

## Step-by-Step Guide

### Step 1: Analyze the Target Model Architecture

Read the HuggingFace model's source code to extract key architecture information.

**Action**: Fetch and analyze the model's HuggingFace configuration and modeling files.

1. Read the model's `config.json` (via `AutoConfig.from_pretrained`) to identify:

   - `model_type` string (this is the key used for registry lookup)
   - All architecture hyperparameters (hidden_size, num_layers, etc.)
   - Any model-specific fields (e.g., `qk_norm`, `attention_bias`, MoE fields)

1. Read the HuggingFace `modeling_*.py` source to identify:

   - **Attention variant**: Does it have Q/K norm? Attention bias? Sliding window?
     Multi-latent attention?
   - **FFN variant**: SwiGLU (gate_proj + up_proj + down_proj)? GeGLU? Standard MLP?
   - **MoE support**: Does it have MoE layers? What router type? Shared experts?
   - **RoPE variant**: Standard RoPE? YaRN? NTK-aware scaling? What is the inv_freq
     formula?
   - **Normalization**: RMSNorm or LayerNorm? Pre-norm or post-norm? Elementwise affine?
   - **Weight tying**: Does `tie_word_embeddings` appear in config?
   - **State dict key names**: What are the HF weight key naming conventions?

1. Summarize findings in a checklist like:

```
Target model: <name>
HF model_type: "<model_type>" (and variants like "<model_type>_moe" if applicable)
Attention: [standard GQA / with QK norm / with bias / sliding window / ...]
FFN: [SwiGLU / GeGLU / standard MLP / ...]
MoE: [no / yes - num_experts, top_k, shared_experts]
RoPE: [standard / YaRN / NTK-aware / ...]
Norm: [RMSNorm / LayerNorm] with [pre-norm / post-norm]
Weight tying: [yes / no]
```

### Step 2: Select the Reference Model

Choose the closest existing implementation as a starting point:

| Target characteristics               | Reference | Why                                     |
| ------------------------------------ | --------- | --------------------------------------- |
| Dense-only, standard GQA, no QK norm | `qwen2`   | Simplest baseline, pure dense           |
| Has QK norm, or has MoE support      | `qwen3`   | Supports QK norm + MoE + shared experts |

**Action**: Copy the reference model directory as the starting point:

```
areal/experimental/models/archon/<model>/
  __init__.py
  spec.py
  model/
    args.py
    model.py
    rope.py
    state_dict_adapter.py
  infra/
    parallelize.py
```

### Step 3: Implement `args.py`

Adapt `<Model>ModelArgs` to match the target model's HuggingFace config fields.

**Key changes from reference**:

1. Update the `@dataclass` fields to match the target model's hyperparameters:

   - Field names should use Archon conventions (`dim`, `n_layers`, `n_heads`,
     `n_kv_heads`, `vocab_size`, `head_dim`, `hidden_dim`, `norm_eps`, `rope_theta`,
     etc.)
   - Default values should match the smallest variant of the target model
   - Add model-specific fields (e.g., `attention_bias`, `qk_norm`, `sliding_window`)

1. Update `from_hf_config()` to correctly map HuggingFace config attributes:

   - Use `getattr(hf_config, "field_name", default)` for optional fields
   - Handle variant-specific fields (e.g., MoE fields only present in MoE variants)
   - The method must return an instance of the model args class

**Critical**: Verify every field mapping against the HF model's `config.json`. Incorrect
mappings here cause silent errors downstream.

**Base class contract** (`BaseModelArgs`):

```python
@dataclass
class <Model>ModelArgs(BaseModelArgs):
    # ... model-specific fields ...

    @classmethod
    def from_hf_config(
        cls,
        hf_config: PretrainedConfig,
        is_critic: bool = False,
        **kwargs,
    ) -> <Model>ModelArgs:
        # Map HF config fields to Archon model args
        ...
```

### Step 4: Implement `model.py`

Adapt the model architecture to match the target model.

**Key components to adapt**:

1. **Normalization** (`RMSNorm` or similar):

   - Check if `elementwise_affine` is configurable
   - Check the epsilon default value
   - If the model uses `LayerNorm`, implement accordingly

1. **Attention** module:

   - Q/K/V projection: Check bias presence (`nn.Linear(..., bias=True/False)`)
   - QK norm: Add `q_norm`/`k_norm` if the model has them, remove if it doesn't
   - GQA: `n_kv_heads` \< `n_heads` for grouped-query attention
   - Ulysses SP: Keep the `set_cp_group` / `_sp_enabled` pattern from the reference
   - Output projection: Check bias presence

1. **FeedForward** module:

   - SwiGLU: `w2(silu(w1(x)) * w3(x))` -- most common for modern LLMs
   - Check bias in linear layers
   - For MoE models: `MoE` module replaces `FeedForward` on designated layers

1. **TransformerBlock**: Pre-norm (most modern LLMs) vs post-norm

   - MoE layer detection via `_is_moe_layer()` if applicable

1. **Top-level Model** (`<Model>Model(BaseArchonModel)`):

   - `tok_embeddings`, `layers` (as `ModuleDict`), `norm`, `output`/`score`
   - `init_weights()`: Match initialization scheme from HF
   - `init_buffers()`: RoPE cache + MoE buffers
   - `forward()`: Must follow `BaseArchonModel` signature:
     `(tokens, positions, cu_seqlens, max_seqlen, tree_attn_meta=None) -> Tensor`

**Base class contract** (`BaseArchonModel`):

```python
class <Model>Model(BaseArchonModel):
    def forward(self, tokens, positions, cu_seqlens, max_seqlen, tree_attn_meta=None) -> torch.Tensor: ...
    def init_weights(self) -> None: ...
    def init_buffers(self, buffer_device) -> None: ...
```

### Step 5: Implement `rope.py`

Handle the rotary position embedding variant.

**Options**:

1. **Standard RoPE** (same as qwen2/qwen3): Re-export from qwen2:

   ```python
   from areal.experimental.models.archon.qwen2.model.rope import (
       apply_rotary_emb,
       precompute_rope_cache,
       repeat_kv,
       reshape_for_broadcast,
       rotate_half,
   )
   ```

1. **Custom RoPE** (YaRN, NTK-aware, etc.): Implement custom `precompute_rope_cache()`
   and `apply_rotary_emb()` functions. The key difference is usually in how `inv_freq`
   is computed (scaling factors, interpolation, etc.).

### Step 6: Implement `state_dict_adapter.py`

Map between HuggingFace and Archon weight key names.

**This is the most error-prone step.** The adapter must correctly handle:

1. **Key name mapping** (`from_hf_map` dict):

   - Embedding: `model.embed_tokens.weight` -> `tok_embeddings.weight`
   - Attention: `model.layers.{}.self_attn.q_proj.weight` ->
     `layers.{}.attention.wq.weight`
   - FFN: `model.layers.{}.mlp.gate_proj.weight` -> `layers.{}.feed_forward.w1.weight`
   - Norms: `model.layers.{}.input_layernorm.weight` ->
     `layers.{}.attention_norm.weight`
   - Output: `lm_head.weight` -> `output.weight`
   - Skip keys (set to `None`): `rotary_emb.inv_freq` (computed at runtime)
   - Model-specific keys: bias terms, QK norm weights, etc.

1. **Reverse mapping** (`to_hf_map`): Auto-generated from `from_hf_map`

1. **MoE expert weights** (if applicable): 3D\<->2D conversion for expert weights. Copy
   the MoE handling from qwen3 if the model has MoE.

1. **Weight tying**: Skip `output.weight` during `to_hf()` if `tie_word_embeddings=True`

**Verification approach**: After implementation, the adapter should satisfy:

```python
# Roundtrip: archon -> hf -> archon preserves all keys
hf_sd = adapter.to_hf(archon_sd)
roundtrip_sd = adapter.from_hf(hf_sd)
assert set(roundtrip_sd.keys()) == set(archon_sd.keys())
```

**Base class contract** (`BaseStateDictAdapter`):

```python
class <Model>StateDictAdapter(BaseStateDictAdapter):
    def from_hf(self, hf_state_dict) -> dict[str, Any]: ...
    def to_hf(self, archon_state_dict) -> dict[str, Any]: ...
    def convert_single_to_hf(self, name, tensor) -> list[tuple[str, torch.Tensor]]: ...
```

### Step 7: Implement `parallelize.py`

Define the parallelization strategy for the model.

**The parallelize function** applies parallelism in this order:

1. TP (Tensor Parallelism) -- shard attention/FFN across devices
1. EP (Expert Parallelism) -- for MoE models only
1. CP (Context Parallelism / Ulysses SP) -- sequence parallelism
1. AC (Activation Checkpointing) -- memory optimization
1. torch.compile -- compilation optimization
1. FSDP (Fully Sharded Data Parallelism) -- data parallelism

**Key adaptations by model architecture**:

- **Attention with QK norm**: wq/wk use `use_local_output=False` (DTensor output for
  norm), add `SequenceParallel(sequence_dim=2)` for q_norm/k_norm
- **Attention without QK norm**: wq/wk/wv all use `use_local_output=True`
- **Attention with bias**: Bias terms follow the same parallel plan as their weights
- **MoE layers**: Separate TP plan for MoE input/output, router gate, and expert
  weights. Copy from qwen3's `apply_moe_ep_tp()` and `apply_non_moe_tp()`
- **Dense-only models**: Simpler plan without MoE handling. Copy from qwen2

**Function signature** (must match `ParallelizeFn` protocol):

```python
def parallelize_<model>(
    model: nn.Module,
    parallel_dims: ArchonParallelDims,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    loss_parallel: bool = True,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ac_config: ActivationCheckpointConfig | None = None,
    enable_compile: bool = True,
) -> nn.Module:
```

### Step 8: Create `spec.py` and Register

Assemble the `ModelSpec` and register it.

```python
from areal.experimental.models.archon.model_spec import ModelSpec, register_model_spec
from areal.experimental.models.archon.pipeline_parallel import pipeline_llm
from areal.experimental.models.archon.<model>.infra.parallelize import parallelize_<model>
from areal.experimental.models.archon.<model>.model.args import <Model>ModelArgs
from areal.experimental.models.archon.<model>.model.model import <Model>Model
from areal.experimental.models.archon.<model>.model.state_dict_adapter import (
    <Model>StateDictAdapter,
)

<MODEL>_SPEC = ModelSpec(
    name="<Model>",
    model_class=<Model>Model,
    model_args_class=<Model>ModelArgs,
    state_dict_adapter_class=<Model>StateDictAdapter,
    parallelize_fn=parallelize_<model>,
    supported_model_types=frozenset({"<model_type>"}),  # From HF config.json
    pipelining_fn=pipeline_llm,
)

# Auto-register when module is imported
register_model_spec(<MODEL>_SPEC)

__all__ = ["<MODEL>_SPEC"]
```

**Note**: `supported_model_types` should include all HF `model_type` strings that this
implementation handles (e.g., `{"qwen3", "qwen3_moe"}` for Qwen3).

### Step 9: Register in `__init__.py`

Add the import to `areal/experimental/models/archon/__init__.py`:

```python
from areal.experimental.models.archon.<model> import spec as <model>_spec  # noqa: F401
```

This triggers auto-registration when the module is imported.

### Step 10: Verify and Test

Verification should be done in stages, adapting based on available hardware and the test
patterns in `tests/experimental/archon/`.

**Before writing tests**, examine the existing test files to understand current
patterns:

```
tests/experimental/archon/
  conftest.py             -- Pytest configuration (version checks)
  utils.py                -- Shared utilities (model loading, comparison)
  test_qwen3_args.py      -- Args unit tests (CPU-only)
  test_state_dict_adapter.py  -- State dict roundtrip tests
  test_weight_sync.py     -- Weight completeness tests (meta device)
  test_forward.py         -- Forward precision comparison (single GPU)
  ...
```

**Test stages** (write tests appropriate for the model's complexity):

#### Stage 1: Args Tests (CPU-only, always write these)

Test `from_hf_config()` with mock HuggingFace configs:

```python
# Pattern: Create mock PretrainedConfig, verify args mapping
from unittest.mock import MagicMock

def test_args_from_hf_config():
    hf_config = MagicMock()
    hf_config.hidden_size = 4096
    hf_config.num_hidden_layers = 32
    # ... set all required fields
    args = <Model>ModelArgs.from_hf_config(hf_config)
    assert args.dim == 4096
    assert args.n_layers == 32
```

#### Stage 2: State Dict Adapter Tests (CPU-only)

Test key mapping roundtrip:

```python
def test_state_dict_roundtrip():
    # Create adapter with mock config
    adapter = <Model>StateDictAdapter(mock_config)
    # Create fake archon state dict with expected keys
    archon_sd = {"tok_embeddings.weight": torch.randn(vocab, dim), ...}
    # Roundtrip
    hf_sd = adapter.to_hf(archon_sd)
    roundtrip = adapter.from_hf(hf_sd)
    assert set(roundtrip.keys()) == set(archon_sd.keys())
```

#### Stage 3: Weight Completeness (meta device, CPU-only)

Verify all model parameters have HF mappings:

```python
def test_weight_completeness():
    # Create model on meta device
    with torch.device("meta"):
        model = <Model>Model(args)
    adapter = <Model>StateDictAdapter(hf_config)
    # Check every archon param has a HF mapping
    for name, _ in model.named_parameters():
        hf_pairs = adapter.convert_single_to_hf(name, torch.empty(0))
        assert len(hf_pairs) > 0, f"No HF mapping for {name}"
```

#### Stage 4: Forward Precision (single GPU, if available)

Compare Archon model output against HuggingFace reference:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_forward_matches_hf():
    # Load both HF and Archon models
    # Run forward on same input
    # Compare logits within tolerance
```

**Important**: Do NOT hardcode the test categories. Inspect the existing test files in
`tests/experimental/archon/` and follow the same patterns, fixtures, and markers. Adapt
test scope to the model's specific features (e.g., add MoE-specific tests only if the
model has MoE).

## Reference Implementations

| Model | Directory                                 | Features                                                |
| ----- | ----------------------------------------- | ------------------------------------------------------- |
| Qwen2 | `areal/experimental/models/archon/qwen2/` | Dense, attention bias, no QK norm                       |
| Qwen3 | `areal/experimental/models/archon/qwen3/` | Dense + MoE, QK norm, no attention bias, shared experts |

## Architecture Decision Map

| Feature             | qwen2    | qwen3                      | What to check in target model                            |
| ------------------- | -------- | -------------------------- | -------------------------------------------------------- |
| Attention bias      | Yes      | No                         | `attention_bias` in HF config                            |
| QK norm             | No       | Yes                        | `qk_norm` in HF config or QKNorm module in modeling file |
| MoE                 | No       | Yes                        | `num_experts`/`num_local_experts` in HF config           |
| Shared experts      | No       | Yes                        | `num_shared_experts` in HF config                        |
| Decoder sparse step | No       | Yes                        | `decoder_sparse_step` in HF config                       |
| Weight tying        | Both     | Both                       | `tie_word_embeddings` in HF config                       |
| RoPE                | Standard | Standard (re-export qwen2) | Check inv_freq formula in HF modeling code               |

## Common Mistakes

- Not mapping all HF keys in `state_dict_adapter.py` (causes silent weight drops)
- Wrong `from_hf_config()` field mapping (uses wrong HF config attribute name)
- Forgetting to handle `None` keys in `from_hf_map` (keys to skip like
  `rotary_emb.inv_freq`)
- Missing MoE expert weight 3D\<->2D conversion when model has MoE
- Wrong TP plan for attention with/without QK norm (`use_local_output` must match)
- Forgetting to add import line in `areal/experimental/models/archon/__init__.py`
- Not including all `model_type` variants in `supported_model_types` frozenset
- Using `print` instead of `areal.utils.logging.getLogger()`

## File Checklist

After completion, verify all files exist and are consistent:

- [ ] `areal/experimental/models/archon/<model>/__init__.py`
- [ ] `areal/experimental/models/archon/<model>/spec.py` -- ModelSpec + register
- [ ] `areal/experimental/models/archon/<model>/model/args.py` -- ModelArgs +
  from_hf_config
- [ ] `areal/experimental/models/archon/<model>/model/model.py` -- Model + Attention +
  FFN
- [ ] `areal/experimental/models/archon/<model>/model/rope.py` -- RoPE (or re-export)
- [ ] `areal/experimental/models/archon/<model>/model/state_dict_adapter.py` -- Key
  mapping
- [ ] `areal/experimental/models/archon/<model>/infra/parallelize.py` -- Parallel
  strategy
- [ ] `areal/experimental/models/archon/__init__.py` -- Import line added
- [ ] `tests/experimental/archon/test_<model>_*.py` -- Tests

______________________________________________________________________

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .opencode/skills/add-archon-model/SKILL.md
Invocation: /add-archon-model <model_name>

## Purpose

Semi-automated guide for adding new model architectures to the Archon training engine.
Unlike simpler skills (add-reward, add-dataset), this skill actively guides Claude to:
1. Analyze HuggingFace source code to extract architecture details
2. Select the closest reference implementation (qwen2 or qwen3)
3. Generate code skeletons adapted to the target architecture
4. Create appropriate tests based on existing test patterns

## How to Update

### When New Reference Models Are Added
1. Add to "Reference Implementations" table
2. Update "Architecture Decision Map" with new feature columns
3. Update Step 2 (reference selection) with new options

### When Base Classes Change
1. Update contract signatures in Steps 3, 4, 6, 7
2. Update file checklist if new files are required

### When ModelSpec Changes
1. Update Step 8 with new ModelSpec fields
2. Update spec.py template

### When Test Patterns Change
1. Update Step 10 with new test patterns
2. Do NOT hardcode test categories -- keep it flexible

### Important Design Decisions
- This skill is SEMI-AUTOMATED: Claude should read HF source and generate code,
  not just provide templates for the user to fill in manually
- The skill references existing test files rather than hardcoding test categories,
  ensuring it stays current as the test suite evolves
- Reference model selection (qwen2 vs qwen3) is based on MoE and QK norm presence

================================================================================
-->
