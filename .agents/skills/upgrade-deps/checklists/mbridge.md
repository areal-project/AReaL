---
package: mbridge
github: ISEEKYAN/mbridge
branch_template: ${VERSION}
upstream_paths:
  - mbridge/__init__.py
  - mbridge/core/__init__.py
  - mbridge/core/auto_bridge.py
  - mbridge/core/bridge.py
  - mbridge/core/llm_bridge.py
  - mbridge/core/util.py
---

## Affected Files

### Primary (engine layer — most likely to break)

| File                                        | Imports / Usage                                                                                                                                                                                 |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `areal/engine/megatron_engine.py`           | `import mbridge` → `mbridge.AutoBridge.from_pretrained`, `.dtype`, `.set_extra_args`                                                                                                            |
| `areal/models/mcore/hf_load.py`             | `from mbridge.core.bridge import Bridge` → `bridge._get_actual_hf_path`, `._weight_name_mapping_*`, `.hf_config`, `.dtype`, `.config`                                                           |
| `areal/models/mcore/hf_save.py`             | `from mbridge.core import Bridge`, `from mbridge.core.util import unwrap_model` → `bridge._weight_name_mapping_*`, `._weight_merge_across_tp`, `._weight_to_hf_format`, `.hf_config`, `.config` |
| `areal/models/mcore/bailing_moe_bridge.py`  | `from mbridge.core import LLMBridge, register_model` → subclasses `LLMBridge`, uses `@register_model` decorator                                                                                 |
| `areal/models/tree_attn/module_megatron.py` | `from mbridge.core import LLMBridge` → monkey-patches `LLMBridge._get_transformer_layer_spec`                                                                                                   |

### Secondary (model / infra layer)

| File                             | Imports / Usage                                                                          |
| -------------------------------- | ---------------------------------------------------------------------------------------- |
| `areal/models/mcore/registry.py` | `from mbridge.core.bridge import Bridge` → `bridge.hf_config`, `.config`, `.get_model()` |

### Tertiary (tests, config)

| File                                          | Imports / Usage                                         |
| --------------------------------------------- | ------------------------------------------------------- |
| `tests/test_estimate_num_params.py`           | `import mbridge` → `mbridge.AutoBridge.from_pretrained` |
| `areal/tools/validation_base.py`              | `"mbridge"` in `CRITICAL_PACKAGES` (metadata only)      |
| `areal/tools/validate_docker_installation.py` | `"mbridge"` in package validation list (metadata only)  |

______________________________________________________________________

## API Usage Catalog

For each function/class below, verify the call signature against the upstream source at
the target version. Focus on: **missing new required parameters**, **removed old
parameters**, **renamed parameters**, **changed return types**, **changed method
signatures on returned objects**, and **moved/renamed modules**.

### 1. `mbridge.AutoBridge.from_pretrained`

**Source:** `mbridge/core/auto_bridge.py`

Called in `areal/engine/megatron_engine.py` (line 507):

```python
self.bridge = mbridge.AutoBridge.from_pretrained(
    self.config.path, trust_remote_code=True
)
```

Also called in `tests/test_estimate_num_params.py` (line 43):

```python
bridge = mbridge.AutoBridge.from_pretrained(model_name_or_path)
```

**Check:** Confirm `trust_remote_code` is still accepted as a keyword argument. Verify
the first positional arg is still the model path (str). Verify the return type is still
a `Bridge` (or subclass like `LLMBridge`) instance with `.hf_config`, `.config`, and
`.dtype` attributes.

______________________________________________________________________

### 2. `Bridge.dtype` (property/attribute)

**Source:** `mbridge/core/bridge.py`

Set in `areal/engine/megatron_engine.py` (line 510):

```python
self.bridge.dtype = self.dtype
```

**Check:** Confirm `dtype` is still a settable attribute (not read-only property).
Verify it accepts `torch.dtype` values. This controls the precision used for weight
loading and conversion throughout the bridge.

______________________________________________________________________

### 3. `Bridge.set_extra_args`

**Source:** `mbridge/core/bridge.py` or `mbridge/core/llm_bridge.py`

Called in `areal/engine/megatron_engine.py` (lines 512–518):

```python
self.bridge.set_extra_args(
    recompute_granularity=self.mcore_config.recompute_granularity,
    recompute_method=self.mcore_config.recompute_method,
    recompute_num_layers=self.mcore_config.recompute_num_layers,
    distribute_saved_activations=self.mcore_config.distribute_saved_activations,
    recompute_modules=self.mcore_config.recompute_modules,
)
```

**Check:** Confirm all five keyword arguments are still accepted:
`recompute_granularity`, `recompute_method`, `recompute_num_layers`,
`distribute_saved_activations`, `recompute_modules`. Check if new required arguments
were added.

______________________________________________________________________

### 4. `Bridge.get_model`

**Source:** `mbridge/core/bridge.py` or `mbridge/core/llm_bridge.py`

Called in `areal/models/mcore/registry.py` (lines 178–188):

```python
models = bridge.get_model(
    wrap_with_ddp=mcore_config.wrap_with_ddp,
    ddp_config=dataclasses.asdict(mcore_config.ddp),
    use_torch_fsdp2=mcore_config.use_torch_fsdp2,
    use_custom_fsdp=mcore_config.use_custom_fsdp,
    fp16=tf_config.fp16,
    bf16=tf_config.bf16,
    use_precision_aware_optimizer=mcore_config.use_precision_aware_optimizer,
    overlap_param_gather_with_optimizer_step=mcore_config.overlap_param_gather_with_optimizer_step,
)
```

**Check:** Confirm all keyword arguments are still accepted: `wrap_with_ddp`,
`ddp_config`, `use_torch_fsdp2`, `use_custom_fsdp`, `fp16`, `bf16`,
`use_precision_aware_optimizer`, `overlap_param_gather_with_optimizer_step`. Verify the
return type is still an iterable of model instances (GPTModel or DDP-wrapped). Check for
newly added required parameters.

______________________________________________________________________

### 5. `Bridge._get_actual_hf_path`

**Source:** `mbridge/core/bridge.py`

Called in `areal/models/mcore/hf_load.py` (line 539):

```python
weights_path = bridge._get_actual_hf_path(weights_path)
```

**Check:** Confirm this private method still exists and accepts a single string argument
(path). Verify it returns a string (resolved path to the actual HF weights directory).
Since this is a private API, check if it was renamed or replaced.

______________________________________________________________________

### 6. `Bridge._weight_name_mapping_mcore_local_to_global`

**Source:** `mbridge/core/bridge.py` or `mbridge/core/llm_bridge.py`

Called in `areal/models/mcore/hf_load.py` (line 570):

```python
local_to_global_map = bridge._weight_name_mapping_mcore_local_to_global(model)
```

Also called in `areal/models/mcore/hf_save.py` (line 243):

```python
bridge._weight_name_mapping_mcore_local_to_global(model, consider_ep=False)
```

**Check:** Confirm the method still accepts a model as the first positional argument.
Verify `consider_ep` keyword argument is still accepted (used in hf_save.py). Verify the
return type is still `dict[str, str]` mapping local parameter names to global names.
Check if the method was renamed or its signature changed.

______________________________________________________________________

### 7. `Bridge._weight_name_mapping_mcore_to_hf`

**Source:** `mbridge/core/bridge.py` or `mbridge/core/llm_bridge.py`

Called in `areal/models/mcore/hf_load.py` (line 573):

```python
local_to_hf_map = {
    k: bridge._weight_name_mapping_mcore_to_hf(v)
    for k, v in local_to_global_map.items()
    if "_extra_state" not in k
}
```

**Check:** Confirm the method accepts a single string argument (mcore global weight
name). Verify the return type is still `list[str]` (list of HF weight names). This is
also overridden in `BailingMoeBridge` — check that the base class signature is stable.

______________________________________________________________________

### 8. `Bridge._weight_merge_across_tp`

**Source:** `mbridge/core/bridge.py` or `mbridge/core/llm_bridge.py`

Called in `areal/models/mcore/hf_save.py` (lines 414–415):

```python
infer_params = bridge._weight_merge_across_tp(
    s.global_name, infer_params, param
)
```

**Check:** Confirm the method accepts three positional arguments: `mcore_weights_name`
(str), `tp_shards` (list of tensors), `param` (tensor). Verify the return type is still
a `torch.Tensor`. This is overridden in `BailingMoeBridge` — check base class stability.

______________________________________________________________________

### 9. `Bridge._weight_to_hf_format`

**Source:** `mbridge/core/bridge.py` or `mbridge/core/llm_bridge.py`

Called in `areal/models/mcore/hf_save.py` (lines 422–423):

```python
converted_names, converted_params = bridge._weight_to_hf_format(
    s.global_name, infer_params
)
```

**Check:** Confirm the method accepts two positional arguments: `mcore_weights_name`
(str) and `mcore_weights` (tensor). Verify the return type is still
`tuple[list[str], list[torch.Tensor]]`. This is overridden in `BailingMoeBridge` — check
base class stability.

______________________________________________________________________

### 10. `mbridge.core.util.unwrap_model`

**Source:** `mbridge/core/util.py`

Called in `areal/models/mcore/hf_save.py` (line 237):

```python
models = [unwrap_model(model) for model in models]
```

**Check:** Confirm the function still accepts a single model argument. Verify it returns
the unwrapped model (strips DDP/FSDP wrappers). Check if the function was moved to a
different module.

______________________________________________________________________

### 11. `LLMBridge` (subclassing contract)

**Source:** `mbridge/core/llm_bridge.py`

Subclassed in `areal/models/mcore/bailing_moe_bridge.py` (line 87):

```python
class BailingMoeBridge(LLMBridge):
    TransformerConfigClass = MLATransformerConfig

    def _build_config(self): ...
    def _get_gptmodel_args(self) -> dict: ...
    def _get_transformer_layer_spec(self, vp_stage: int | None = None): ...
    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]: ...
    def _weight_merge_across_tp(self, mcore_weights_name, tp_shards, param): ...
    def _weight_to_mcore_format(self, mcore_weights_name, hf_weights): ...
    def _weight_to_hf_format(self, mcore_weights_name, mcore_weights): ...
```

**Check:** Confirm `LLMBridge` still exposes these overridable methods with compatible
signatures. Verify `TransformerConfigClass` class attribute is still supported. Check if
any abstract methods were added that subclasses must now implement. Verify
`_build_base_config(...)` still accepts the keyword arguments used in
`BailingMoeBridge._build_config()` (especially MoE and MLA parameters). Check if
`has_vp_stage` attribute is still read by the base class.

______________________________________________________________________

### 12. `register_model` decorator

**Source:** `mbridge/core/__init__.py` (re-exported from `mbridge/core/llm_bridge.py` or
similar)

Used in `areal/models/mcore/bailing_moe_bridge.py` (lines 84–86):

```python
@register_model("bailing_moe_v2")
@register_model("bailing_moe_linear")
@register_model("bailing_hybrid")
class BailingMoeBridge(LLMBridge):
    ...
```

**Check:** Confirm `register_model` is still importable from `mbridge.core`. Verify it
still accepts a single string argument (model type name). Verify it still works as a
class decorator that registers the bridge class for `AutoBridge.from_pretrained`
dispatch. Check if the registration mechanism changed (e.g., from decorator to explicit
registry call).

______________________________________________________________________

### 13. `LLMBridge._get_transformer_layer_spec` (monkey-patch target)

**Source:** `mbridge/core/llm_bridge.py`

Monkey-patched in `areal/models/tree_attn/module_megatron.py` (lines 195–213):

```python
original_layer_spec_getter = LLMBridge._get_transformer_layer_spec

def _patched_getter(self, vp_stage: int | None = None):
    spec: TransformerBlockSubmodules = original_layer_spec_getter(self, vp_stage)
    for layer_spec in spec.layer_specs:
        ...
    return spec

LLMBridge._get_transformer_layer_spec = _patched_getter
```

**Check:** Confirm `_get_transformer_layer_spec` still exists on `LLMBridge` with
signature `(self, vp_stage: int | None = None)`. Verify the return type is still
`TransformerBlockSubmodules` with a `.layer_specs` attribute (list of layer specs). Each
layer spec must still have `.module`, `.submodules`, and
`.submodules.self_attention.submodules.core_attention` that can be reassigned. If the
method was renamed or its return structure changed, the monkey-patch will silently fail.

______________________________________________________________________

### 14. `Bridge.hf_config` and `Bridge.config` (properties)

**Source:** `mbridge/core/bridge.py`

Accessed throughout:

- `bridge.hf_config` → HuggingFace `PretrainedConfig` instance (used in hf_load.py,
  hf_save.py, registry.py, bailing_moe_bridge.py)
- `bridge.config` → Megatron `TransformerConfig` instance (used in hf_save.py,
  registry.py)

```python
# hf_load.py
bridge.hf_config  # .num_key_value_heads, .hidden_size, .num_attention_heads, .quantization_config
bridge.config     # .fp8, .fp8_param
bridge.dtype      # torch.dtype

# registry.py
bridge.hf_config  # ._name_or_path
bridge.config     # returned as TransformerConfig
```

**Check:** Confirm `hf_config` still returns a HuggingFace `PretrainedConfig` (or
subclass). Confirm `config` still returns a Megatron `TransformerConfig` with at least
`.fp8`, `.fp8_param`, `.num_moe_experts` attributes. These are the most widely accessed
properties — any type change will cascade across all Primary files.

______________________________________________________________________

## Version-Guarded Code

_None currently._ The `mbridge` package is pinned to a git commit (`310e8fb`) rather
than a versioned release, so no version-conditional code exists.
