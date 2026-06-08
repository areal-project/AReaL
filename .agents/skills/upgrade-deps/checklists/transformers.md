---
package: transformers
github: huggingface/transformers
branch_template: v${VERSION}
upstream_paths:
  - src/transformers/models/auto/auto_factory.py
  - src/transformers/models/auto/configuration_auto.py
  - src/transformers/configuration_utils.py
  - src/transformers/tokenization_utils_fast.py
  - src/transformers/tokenization_utils_base.py
  - src/transformers/processing_utils.py
  - src/transformers/optimization.py
  - src/transformers/modeling_flash_attention_utils.py
  - src/transformers/integrations/flash_attention.py
  - src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
  - src/transformers/models/qwen2_5_vl/
  - src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
  - src/transformers/utils/import_utils.py
---

## Affected Files

### Primary (engine layer — most likely to break)

| File                                         | Imports / Usage                                                                                                                                                                                                                                 |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `areal/engine/fsdp_engine.py`                | `AutoConfig`, `AutoModelForCausalLM`, `AutoModelForImageTextToText`, `AutoModelForTokenClassification`, `PretrainedConfig`, `PreTrainedTokenizerFast`, `ProcessorMixin`, `get_linear_schedule_with_warmup`, `get_constant_schedule_with_warmup` |
| `areal/engine/megatron_engine.py`            | `PretrainedConfig`                                                                                                                                                                                                                              |
| `areal/engine/fsdp_utils/__init__.py`        | `PreTrainedModel`                                                                                                                                                                                                                               |
| `areal/engine/fsdp_utils/parallel.py`        | `PretrainedConfig`                                                                                                                                                                                                                              |
| `areal/experimental/engine/archon_engine.py` | `AutoConfig`, `PretrainedConfig`, `PreTrainedTokenizerFast`                                                                                                                                                                                     |

### Secondary (model / infra layer — HIGH RISK: monkey-patching)

| File                                                                       | Imports / Usage                                                                                                                                                                                                                                                    |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `areal/models/transformers/ulyssess_patch.py`                              | `transformers.modeling_flash_attention_utils._flash_attention_forward` (direct import); `transformers.integrations.flash_attention._flash_attention_forward` (monkey-patch); dynamic import of `transformers.models.{qwen2_vl,qwen2_5_vl,qwen3_vl}` module strings |
| `areal/models/transformers/qwen3_vl.py`                                    | `transformers.integrations.flash_attention.flash_attention_forward`; `transformers.models.qwen3_vl.modeling_qwen3_vl.{apply_rotary_pos_emb, repeat_kv}`                                                                                                            |
| `areal/models/transformers/qwen2_vl.py`                                    | `transformers.integrations.flash_attention.flash_attention_forward`; `transformers.models.qwen2_vl.modeling_qwen2_vl.{apply_multimodal_rotary_pos_emb, repeat_kv}`                                                                                                 |
| `areal/models/transformers/vision_sp_shard.py`                             | monkey-patches Qwen VL vision modules (internal submodule access)                                                                                                                                                                                                  |
| `areal/models/tree_attn/module_fsdp.py`                                    | `transformers.integrations.flash_attention._flash_attention_forward` (monkey-patch save/restore)                                                                                                                                                                   |
| `areal/workflow/rlvr.py`                                                   | `PreTrainedTokenizerFast` (`decode`, `apply_chat_template`, `eos_token_id`)                                                                                                                                                                                        |
| `areal/workflow/vision_rlvr.py`                                            | `AutoProcessor`, `PreTrainedTokenizerFast`                                                                                                                                                                                                                         |
| `areal/workflow/multi_turn.py`                                             | `PreTrainedTokenizerFast`                                                                                                                                                                                                                                          |
| `areal/utils/hf_utils.py`                                                  | `AutoTokenizer.from_pretrained()`, `AutoProcessor.from_pretrained()`                                                                                                                                                                                               |
| `areal/utils/seeding.py`                                                   | `transformers.set_seed()`                                                                                                                                                                                                                                          |
| `areal/infra/platforms/__init__.py`                                        | `transformers.utils.import_utils.is_torch_npu_available`                                                                                                                                                                                                           |
| `areal/infra/rpc/serialization.py`                                         | `AutoTokenizer`, `PreTrainedTokenizer`, `PreTrainedTokenizerFast`, `AutoProcessor`, `ProcessorMixin`                                                                                                                                                               |
| `areal/models/mcore/registry.py`                                           | `AutoConfig`, `PretrainedConfig`                                                                                                                                                                                                                                   |
| `areal/models/mcore/bailing_moe.py`                                        | `PretrainedConfig`                                                                                                                                                                                                                                                 |
| `areal/experimental/models/archon/*/args.py` and `*/state_dict_adapter.py` | `PretrainedConfig`                                                                                                                                                                                                                                                 |

### Tertiary (tests and examples)

| File                                     | Imports / Usage                                                                                                                    |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 30+ files under `tests/` and `examples/` | `AutoTokenizer`, `AutoConfig`, `AutoModelForCausalLM`, `PreTrainedTokenizerFast` — standard load patterns only, no monkey-patching |

______________________________________________________________________

## API Usage Catalog

For each function/class below, verify the call signature against the upstream source at
the target version. Focus on: **missing new required parameters**, **removed old
parameters**, **renamed parameters**, **changed return types**, **changed method
signatures on returned objects**, and **moved/renamed modules**.

### 1. Auto\* model and config classes

**Source:** `src/transformers/models/auto/auto_factory.py`,
`src/transformers/models/auto/configuration_auto.py`

Called in `areal/engine/fsdp_engine.py`:

```python
# Config loading (line 196)
self.model_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=self.config.path,
    trust_remote_code=True,
)

# VLM path (line 899)
model = AutoModelForImageTextToText.from_pretrained(
    pretrained_model_name_or_path=self.config.path,
    trust_remote_code=True,
    dtype=dtype,
    attn_implementation=self.config.attn_impl,
)

# LLM path — _create_llm_actor_or_critic (lines 829–856)
model_class = AutoModelForTokenClassification if is_critic else AutoModelForCausalLM
model_kwargs = {"num_labels": 1} if is_critic else {}
model_kwargs.update({"dtype": dtype, "attn_implementation": self.config.attn_impl})

# Branch: from_config (meta-device, memory-efficient) or from_pretrained
model = model_class.from_config(self.model_config, **model_kwargs)
# or:
model = model_class.from_pretrained(
    pretrained_model_name_or_path=self.config.path,
    trust_remote_code=True,
    **model_kwargs,
)
```

Also called in `areal/utils/hf_utils.py`, `areal/models/mcore/registry.py`,
`areal/experimental/engine/archon_engine.py`, and
`areal/experimental/models/archon/*/args.py`.

**Check:** Verify `from_pretrained` / `from_config` keyword arguments (`dtype`,
`attn_implementation`, `trust_remote_code`, `num_labels`). Check whether
`AutoModelForImageTextToText` still exists or was renamed/merged. Confirm `from_config`
still accepts `num_labels` for `AutoModelForTokenClassification`. Look for new required
kwargs introduced in the target version.

______________________________________________________________________

### 2. `PretrainedConfig`

**Source:** `src/transformers/configuration_utils.py`

Called in `areal/engine/fsdp_engine.py`, `areal/engine/megatron_engine.py`,
`areal/engine/fsdp_utils/parallel.py`, `areal/experimental/engine/archon_engine.py`,
`areal/models/mcore/registry.py`, and various
`areal/experimental/models/archon/*/args.py`:

```python
# Type annotation / isinstance check only
config: PretrainedConfig

# Attribute access (common patterns)
config.model_type
config.num_attention_heads
config.num_key_value_heads
config.text_config.num_attention_heads   # VLM sub-config
```

**Check:** Confirm `PretrainedConfig` is still the base class at this path (not moved).
Check whether `text_config` sub-config attribute access pattern for VLMs is still valid.
Verify no new required fields were added to `__init__`.

______________________________________________________________________

### 3. `PreTrainedTokenizerFast` and tokenizer methods

**Source:** `src/transformers/tokenization_utils_fast.py`,
`src/transformers/tokenization_utils_base.py`

Called in `areal/utils/hf_utils.py` (`load_hf_tokenizer`, lines 12–30):

```python
@lru_cache(maxsize=8)
def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True, padding_side=None):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        fast_tokenizer=fast_tokenizer,
        trust_remote_code=True,
        force_download=True,
        **kwargs,  # padding_side if not None
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
```

Called in `areal/workflow/rlvr.py`, `areal/workflow/multi_turn.py`, and
`areal/workflow/vision_rlvr.py`:

```python
tokenizer.decode(token_ids, ...)
tokenizer.apply_chat_template(messages, ...)
tokenizer.eos_token_id
```

**Check:** Verify `fast_tokenizer` parameter still exists in
`AutoTokenizer.from_pretrained` (it was renamed to `use_fast` in some versions — confirm
which name the current version uses). Confirm `apply_chat_template` signature is
unchanged (especially `tokenize`, `add_generation_prompt`, `return_tensors` kwargs).
Check `force_download` is still accepted. Confirm `eos_token_id` and `pad_token_id` are
still plain attributes. Note: the function is `@lru_cache`'d — if the return type
changes, the cache must be invalidated.

______________________________________________________________________

### 4. `AutoProcessor` / `ProcessorMixin`

**Source:** `src/transformers/processing_utils.py`

Called in `areal/utils/hf_utils.py` (`load_hf_processor_and_tokenizer`, lines 33–55):

```python
processor = transformers.AutoProcessor.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    force_download=True,
    use_fast=True,
)
```

Also used as type annotation in `areal/infra/rpc/serialization.py` and
`areal/engine/fsdp_engine.py` (`ProcessorMixin`).

**Check:** Confirm `AutoProcessor.from_pretrained` still accepts `use_fast`. Verify
`ProcessorMixin` is still the importable base class at `transformers.ProcessorMixin`.
Check for signature changes on processor call methods (image/text preprocessing) if
`vision_rlvr.py` calls them directly.

______________________________________________________________________

### 5. LR scheduler helpers

**Source:** `src/transformers/optimization.py`

Called in `areal/engine/fsdp_engine.py` (lines 1017–1026):

```python
# "linear" scheduler (line 1017)
self.lr_scheduler = get_linear_schedule_with_warmup(
    self.optimizer,
    num_warmup_steps,
    total_train_steps,
)

# "constant" scheduler (line 1023)
self.lr_scheduler = get_constant_schedule_with_warmup(
    self.optimizer,
    num_warmup_steps,
)
```

**Check:** Confirm both functions still exist in `transformers.optimization` (they have
occasionally been soft-deprecated in favor of `get_scheduler`). Verify positional
argument order (`optimizer`, `num_warmup_steps`, `num_training_steps`) is unchanged.
Check return type is still a `LambdaLR`-compatible scheduler.

______________________________________________________________________

### 6. `transformers.set_seed`

**Source:** `src/transformers/trainer_utils.py` (re-exported from top-level)

Called in `areal/utils/seeding.py` (line 28):

```python
import transformers
transformers.set_seed(seed)
```

**Check:** Confirm `set_seed` is still exported at `transformers.set_seed` (top-level
re-export). Verify signature is `set_seed(seed: int)` with no new required parameters.

______________________________________________________________________

### 7. `transformers.modeling_flash_attention_utils._flash_attention_forward` (HIGH RISK)

**Source:** `src/transformers/modeling_flash_attention_utils.py`

Called in `areal/models/transformers/ulyssess_patch.py` (import line 6, usage line 57):

```python
from transformers.modeling_flash_attention_utils import _flash_attention_forward

# Called inside _ulysses_flash_attention_forward as a pass-through (line 57):
attn_output = _flash_attention_forward(
    query_states, key_states, value_states, *args, **kwargs,
)
```

**Check:** This is a **private** function (leading underscore). Verify it still exists
at this exact import path. Confirm its positional signature
`(query_states, key_states, value_states, ...)` is unchanged. Any argument reorder or
rename breaks all Ulysses sequence parallel attention paths. Note: this is used as the
_real_ flash attention call inside the Ulysses wrapper — it's the only place the actual
flash attention kernel is invoked.

______________________________________________________________________

### 8. `transformers.integrations.flash_attention._flash_attention_forward` (HIGH RISK — monkey-patched)

**Source:** `src/transformers/integrations/flash_attention.py`

Monkey-patched in `areal/models/transformers/ulyssess_patch.py` (lines 244–246) and
`areal/models/tree_attn/module_fsdp.py` (lines 163–166, 182–185):

```python
# ulyssess_patch.py (line 244) — for non-VLM models
from transformers.integrations import flash_attention
flash_attention._flash_attention_forward = _ulysses_flash_attention_forward

# module_fsdp.py (lines 163-166) — save and replace for tree attention
ORIGINAL_FLASH_ATTENTION_FORWARD = flash_attention._flash_attention_forward
flash_attention._flash_attention_forward = _tree_attn_fwd_func

# module_fsdp.py (lines 182-185) — restore original after use
flash_attention._flash_attention_forward = ORIGINAL_FLASH_ATTENTION_FORWARD
ORIGINAL_FLASH_ATTENTION_FORWARD = None
```

**Check:** Confirm `transformers.integrations.flash_attention` module still exists and
still has a `_flash_attention_forward` attribute. If the module was renamed, moved, or
the function removed, ALL of the Ulysses and tree-attention patches silently break. Also
verify `flash_attention_forward` (public, **no** underscore) still exists — it is
imported directly in `areal/models/transformers/qwen2_vl.py` (line 7) and
`areal/models/transformers/qwen3_vl.py` (line 12) via
`from transformers.integrations.flash_attention import flash_attention_forward`.

______________________________________________________________________

### 9. Qwen2-VL internal symbols (HIGH RISK — imported from private modules)

**Source:** `src/transformers/models/qwen2_vl/modeling_qwen2_vl.py`

Imported in `areal/models/transformers/qwen2_vl.py` (lines 8–11):

```python
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)
```

Also accessed by module string in `areal/models/transformers/ulyssess_patch.py` (lines
187–193):

```python
"module": "transformers.models.qwen2_vl.modeling_qwen2_vl",
"attn_class": "Qwen2VLAttention",
"model_class": "Qwen2VLTextModel",
"patch_module": "areal.models.transformers.qwen2_vl",
"patch_attn_func": "ulysses_flash_attn_forward",
```

The patch replaces `Qwen2VLAttention.forward` at line 229:
`attn_class.forward = patch_attn_func`.

**Check:** Confirm `apply_multimodal_rotary_pos_emb` and `repeat_kv` are still exported
from this submodule (they are internal helpers, not public API). Confirm
`Qwen2VLAttention` and `Qwen2VLTextModel` class names are unchanged. Verify the patch
target `Qwen2VLAttention.forward` still accepts the same signature as
`areal/models/transformers/qwen2_vl.py:ulysses_flash_attn_forward`.

______________________________________________________________________

### 10. Qwen2.5-VL internal symbols (HIGH RISK — imported by module string)

**Source:** `src/transformers/models/qwen2_5_vl/`

Accessed by module string in `areal/models/transformers/ulyssess_patch.py` (lines
180–186):

```python
"module": "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
"attn_class": "Qwen2_5_VLAttention",
"model_class": "Qwen2_5_VLTextModel",
"patch_module": "areal.models.transformers.qwen2_vl",
"patch_attn_func": "ulysses_flash_attn_forward",
```

Note: Qwen2.5-VL reuses the **same** patch function from `qwen2_vl.py` (not a separate
file).

**Check:** Confirm the submodule path
`transformers.models.qwen2_5_vl.modeling_qwen2_5_vl` is unchanged. Confirm class names
`Qwen2_5_VLAttention` and `Qwen2_5_VLTextModel` still exist. Verify
`Qwen2_5_VLAttention.forward` has the same signature as `Qwen2VLAttention.forward`
(since they share the same patch function).

______________________________________________________________________

### 11. Qwen3-VL internal symbols (HIGH RISK — imported from private modules)

**Source:** `src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`

Imported in `areal/models/transformers/qwen3_vl.py` (lines 13–16):

```python
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
    repeat_kv,
)
```

Also imports `flash_attention_forward` (public, no underscore) at line 12:

```python
from transformers.integrations.flash_attention import flash_attention_forward
```

Accessed by module string in `areal/models/transformers/ulyssess_patch.py` (lines
194–200):

```python
"module": "transformers.models.qwen3_vl.modeling_qwen3_vl",
"attn_class": "Qwen3VLTextAttention",
"model_class": "Qwen3VLTextModel",
"patch_module": "areal.models.transformers.qwen3_vl",
"patch_attn_func": "ulysses_flash_attn_forward",
```

Note: Qwen3-VL has its **own** patch function (different from Qwen2-VL) with this
signature (line 25):

```python
def ulysses_flash_attn_forward(
    self, hidden_states, position_embeddings, attention_mask=None,
    past_key_values=None, cache_position=None, **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
```

**Check:** Confirm `apply_rotary_pos_emb` and `repeat_kv` still exist at this path.
Confirm `Qwen3VLTextAttention` and `Qwen3VLTextModel` class names are unchanged. Verify
`Qwen3VLTextAttention.forward` signature matches the patch:
`(self, hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kwargs)`.
Any signature change breaks the Ulysses SP for Qwen3-VL.

______________________________________________________________________

### 12. `transformers.utils.import_utils.is_torch_npu_available`

**Source:** `src/transformers/utils/import_utils.py`

Called in `areal/infra/platforms/__init__.py` (lines 6, 19):

```python
from transformers.utils.import_utils import is_torch_npu_available
is_npu_available = is_torch_npu_available()
```

**Check:** Confirm the function is still at this exact submodule path (private utils
module). Check whether it was moved to a different location or replaced with a different
NPU detection mechanism.

______________________________________________________________________

## Version-Guarded Code

No known version-guarded code exists in AReaL for `transformers`.
