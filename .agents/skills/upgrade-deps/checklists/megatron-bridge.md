---
package: megatron-bridge
github: NVIDIA-NeMo/Megatron-Bridge
branch_template: v${VERSION}
upstream_paths:
  - megatron/bridge/__init__.py
  - megatron/bridge/auto_bridge.py
  - megatron/bridge/peft/lora.py
  - megatron/bridge/models/gpt_provider.py
  - megatron/bridge/models/qwen_vl/qwen35_vl_provider.py
---

## Affected Files

### Primary (engine layer — most likely to break)

| File                                                     | Imports / Usage                                                |
| -------------------------------------------------------- | -------------------------------------------------------------- |
| `areal/engine/megatron_engine.py`                        | `megatron.bridge.AutoBridge`, `megatron.bridge.peft.lora.LoRA` |
| `areal/engine/megatron_utils/megatron_bridge_patches.py` | Qwen3.5 VLM provider spec builder                              |

### Secondary (model / infra layer)

| File                             | Imports / Usage                                     |
| -------------------------------- | --------------------------------------------------- |
| `areal/models/mcore/registry.py` | Qwen3.5 and GPT provider transformer specifications |

### Tertiary (tests, config)

| File                             | Imports / Usage                                                                   |
| -------------------------------- | --------------------------------------------------------------------------------- |
| `areal/tools/validation_base.py` | `"megatron-bridge"` → `"megatron.bridge"` in `PACKAGE_IMPORT_MAP` (metadata only) |

______________________________________________________________________

## API Usage Catalog

For each function/class below, verify the call signature against the upstream source at
the target version. Focus on: **missing new required parameters**, **removed old
parameters**, **renamed parameters**, **changed return types**, **changed method
signatures on returned objects**, and **moved/renamed modules**.

### 1. `megatron.bridge.AutoBridge.from_hf_pretrained`

**Source:** `megatron/bridge/auto_bridge.py`

Called in `areal/engine/megatron_engine.py` (lines 795-801):

```python
self.bridge = MegatronBridgeAutoBridge.from_hf_pretrained(
    self.config.path,
    trust_remote_code=True,
    dtype=self.config.dtype,
)
```

**Check:** Confirm `trust_remote_code` and `dtype` are still accepted keyword arguments.
Verify the first positional arg is still the model path. Verify the method still returns
a bridge object that exposes `hf_pretrained`, `transformer_config`,
`to_megatron_provider`, `export_hf_weights`, `export_adapter_weights`,
`save_hf_pretrained`, `save_hf_adapter`, and `load_hf_weights`. Check for any new
required parameters.

______________________________________________________________________

### 2. `megatron.bridge.AutoBridge.save_hf_pretrained`

**Source:** `megatron/bridge/auto_bridge.py`

Called in `areal/engine/megatron_engine.py` (lines 2784-2789):

```python
self.bridge.save_hf_pretrained(
    self.model,
    path,
    source_path=base_model_path,
    strict=not self._mtp_head_dropped,
)
```

**Check:** Confirm `source_path` and `strict` are still valid keyword arguments. Verify
the positional order of `model` and `path` has not changed. Check the return type
(currently void/`None`).

______________________________________________________________________

### 3. `megatron.bridge.AutoBridge.load_hf_weights`

**Source:** `megatron/bridge/auto_bridge.py`

Called in `areal/engine/megatron_engine.py` (lines 3126-3127):

```python
with torch.device("cpu"):
    self.bridge.load_hf_weights(self.model, hf_path=path)
```

**Check:** Confirm `hf_path` is still the correct keyword name. Verify `model` is still
the first positional argument. Check for newly added required arguments.

______________________________________________________________________

### 4. `megatron.bridge.AutoBridge.save_hf_adapter`

**Source:** `megatron/bridge/auto_bridge.py`

Called in `areal/engine/megatron_engine.py` (lines 2771-2776):

```python
self.bridge.save_hf_adapter(
    self.model,
    path=path,
    peft_config=self.bridge_lora,
    base_model_name_or_path=base_model_path or self.config.path,
)
```

**Check:** Confirm the native method accepts the arguments used by AReaL. Its current
signature is:

```python
def save_hf_adapter(
    self, model, path, peft_config, base_model_name_or_path=None, show_progress=True
)
```

Any mismatch in parameter names or order will break adapter saving. Note that
`peft_config` receives a `megatron.bridge.peft.lora.LoRA` instance, not a dictionary.

______________________________________________________________________

### 5. `megatron.bridge.AutoBridge.export_hf_weights`

**Source:** `megatron/bridge/auto_bridge.py`

Called in `areal/engine/megatron_engine.py` (lines 2551-2555):

```python
for hf_name, hf_tensor in self.bridge.export_hf_weights(
    self.model,
    cpu=False,
    show_progress=False,
):
    ...
```

**Check:** Confirm `cpu` and `show_progress` remain accepted and the method yields
`(name, tensor)` pairs on every distributed rank while performing any required model
parallel collectives.

______________________________________________________________________

### 6. `megatron.bridge.AutoBridge.export_adapter_weights`

**Source:** `megatron/bridge/auto_bridge.py`

Called in `areal/engine/megatron_engine.py` (lines 2590-2594):

```python
for hf_name, hf_tensor in export_adapter_weights(
    self.model,
    cpu=False,
    show_progress=False,
):
    ...
```

**Check:** Confirm `cpu` and `show_progress` remain accepted and the method yields
`(name, tensor)` pairs on every distributed rank. The exported names must retain the
LoRA parameter suffixes normalized by `normalize_bridge_lora_name`.

______________________________________________________________________

### 7. `megatron.bridge.peft.lora.LoRA`

**Source:** `megatron/bridge/peft/lora.py`

Called in `areal/engine/megatron_engine.py` (lines 334-339):

```python
self.bridge_lora = MegatronBridgeLoRA(
    target_modules=target_modules,
    dim=self.config.lora_rank,
    alpha=self.config.lora_alpha,
    dropout=self.config.lora_dropout,
)
```

**Check:** Confirm `dim` is still the rank parameter (not renamed to `r` or `rank`).
Verify `alpha` and `dropout` are still accepted. Check the `target_modules` accepted
type (list of strings vs. regex).

______________________________________________________________________

### 8. `LoRA.__call__` (apply to model)

**Source:** `megatron/bridge/peft/lora.py`

Called in `areal/engine/megatron_engine.py` (lines 341-343):

```python
model = self.bridge_lora(model, training=True)
self.bridge_lora.set_params_to_save(model)
```

**Check:** Confirm `LoRA` instances are still callable with `(model, training=...)`.
Verify the return type remains an iterable of model chunks. Confirm
`set_params_to_save(model)` still exists and marks LoRA parameters for checkpoint
saving. Check if `training=True` is still the correct keyword to enable gradients on
LoRA parameters.

______________________________________________________________________

### 9. `megatron.bridge.AutoBridge.to_megatron_provider`

**Source:** `megatron/bridge/auto_bridge.py`

Called in `areal/models/mcore/registry.py` (line 338):

```python
provider = bridge.to_megatron_provider(load_weights=False)
```

**Check:** Confirm `load_weights` is still accepted and that the returned provider
exposes the parallelism, recompute, model-freezing, and transformer-spec attributes
configured by AReaL before model construction.

______________________________________________________________________

### 10. `megatron.bridge.models.qwen_vl.qwen35_vl_provider`

**Source:** `megatron/bridge/models/qwen_vl/qwen35_vl_provider.py`

Wrapped in `areal/engine/megatron_utils/megatron_bridge_patches.py` (lines 25-31):

```python
original = qwen35_vl_provider.get_transformer_block_with_experimental_attention_variant_spec
```

The same module attribute is assigned in `areal/models/mcore/registry.py` (lines
340-346):

```python
provider.transformer_layer_spec = qwen35_vl_provider.get_transformer_block_with_experimental_attention_variant_spec
```

**Check:** Confirm the module and spec-builder name still exist and the builder still
returns a block spec with iterable `layer_specs` whose attention and MLP submodules can
be replaced before model construction.

______________________________________________________________________

### 11. `megatron.bridge.models.gpt_provider` layer specs

**Source:** `megatron/bridge/models/gpt_provider.py`

Imported in `areal/models/mcore/registry.py` (lines 375-378):

```python
from megatron.bridge.models.gpt_provider import (
    default_layer_spec,
    local_layer_spec,
)
```

**Check:** Confirm both spec callables remain available and provider instances continue
to store the selected callable in `transformer_layer_spec`; AReaL compares by identity
before substituting NPU-compatible LoRA specifications.

______________________________________________________________________

## Version-Guarded Code

_None._
