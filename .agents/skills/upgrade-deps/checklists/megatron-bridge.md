---
package: megatron-bridge
github: NVIDIA-NeMo/Megatron-Bridge
branch_template: v${VERSION}
upstream_paths:
  - megatron/bridge/__init__.py
  - megatron/bridge/auto_bridge.py
  - megatron/bridge/peft/lora.py
---

## Affected Files

### Primary (engine layer — most likely to break)

| File                                           | Imports / Usage                                                |
| ---------------------------------------------- | -------------------------------------------------------------- |
| `areal/engine/megatron_engine.py`              | `megatron.bridge.AutoBridge`, `megatron.bridge.peft.lora.LoRA` |
| `areal/engine/megatron_utils/megatron_lora.py` | `megatron.bridge.AutoBridge` (inside function, monkey-patched) |

### Secondary (model / infra layer)

_None._

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

Called in `areal/engine/megatron_engine.py` (line 430):

```python
self.bridge = MegatronBridgeAutoBridge.from_hf_pretrained(
    self.config.path,
    trust_remote_code=True,
    dtype=self.config.dtype,
)
```

**Check:** Confirm `trust_remote_code` and `dtype` are still accepted keyword arguments.
Verify the first positional arg is still the model path. Verify the method still returns
a bridge object that exposes `save_hf_pretrained`, `load_hf_weights`, and (depending on
version) `save_hf_adapter`. Check for any new required parameters.

______________________________________________________________________

### 2. `megatron.bridge.AutoBridge.save_hf_pretrained`

**Source:** `megatron/bridge/auto_bridge.py`

Called in `areal/engine/megatron_engine.py` (line 1561):

```python
bridge.save_hf_pretrained(model, path, source_path=base_model_path)
```

**Check:** Confirm `source_path` is still a valid keyword argument. Verify positional
order of `model` and `path` hasn't changed. Check return type (currently void/`None`).

______________________________________________________________________

### 3. `megatron.bridge.AutoBridge.load_hf_weights`

**Source:** `megatron/bridge/auto_bridge.py`

Called in `areal/engine/megatron_engine.py` (line 1595):

```python
bridge.load_hf_weights(model, hf_path=path)
```

**Check:** Confirm `hf_path` is still the correct keyword name. Verify `model` is still
the first positional argument. Check for newly added required arguments.

______________________________________________________________________

### 4. `megatron.bridge.AutoBridge.save_hf_adapter`

**Source:** `megatron/bridge/auto_bridge.py`

Called in `areal/engine/megatron_engine.py` (lines 1554-1559) via the monkey-patched
method on the bridge instance:

```python
self.bridge.save_hf_adapter(
    self.model,
    path=path,
    peft_config=self.bridge_lora,
    base_model_name_or_path=base_model_path or self.config.path,
)
```

**Check:** If the new version adds this method natively, confirm its signature matches
the monkey-patch in `areal/engine/megatron_utils/megatron_lora.py` (line 189). The
monkey-patched signature is:

```python
def save_hf_adapter(
    self, model, path, peft_config, base_model_name_or_path=None, show_progress=True
)
```

Any mismatch in parameter names or order will silently break adapter saving. Note that
`peft_config` receives a `megatron.bridge.peft.lora.LoRA` instance (not a dict). See
also the [Version-Guarded Code](#version-guarded-code) section.

______________________________________________________________________

### 5. `megatron.bridge.AutoBridge.export_adapter_weights`

**Source:** `megatron/bridge/auto_bridge.py`

Called inside the monkey-patched `save_hf_adapter` in
`areal/engine/megatron_utils/megatron_lora.py` (lines 237-240):

```python
for name, tensor in self.export_adapter_weights(
    model, cpu=True, show_progress=show_progress
):
    adapter_state[f"base_model.model.{name}"] = tensor.clone().float()
```

**Check:** Confirm `cpu` and `show_progress` are still accepted. Verify the method
yields `(name, tensor)` tuples (iterable of pairs). The names are module FQNs without
the `base_model.model.` prefix — that prefix is added by the caller. This is a native
bridge method used inside the monkey-patch — if its return type or signature changes,
the patch breaks.

______________________________________________________________________

### 6. `megatron.bridge.peft.lora.LoRA`

**Source:** `megatron/bridge/peft/lora.py`

Called in `areal/engine/megatron_engine.py` (line 233):

```python
bridge_lora = MegatronBridgeLoRA(
    target_modules=target_modules,
    dim=lora_rank,
    alpha=lora_alpha,
    dropout=0.0,
)
```

**Check:** Confirm `dim` is still the rank parameter (not renamed to `r` or `rank`).
Verify `alpha` and `dropout` are still accepted. Check the `target_modules` accepted
type (list of strings vs. regex).

______________________________________________________________________

### 7. `LoRA.__call__` (apply to model)

**Source:** `megatron/bridge/peft/lora.py`

Called in `areal/engine/megatron_engine.py` (lines 239-240):

```python
self.model = _MegatronModelList(self.bridge_lora(self.model, training=True))
self.bridge_lora.set_params_to_save(self.model)
```

**Check:** Confirm `LoRA` instances are still callable with `(model, training=...)`.
Verify the return type — it returns a modified model (or list of model chunks), which is
then wrapped in `_MegatronModelList`. Confirm `set_params_to_save(model)` still exists
and marks LoRA parameters for checkpoint saving. Check if `training=True` is still the
correct keyword to enable grad on LoRA parameters.

______________________________________________________________________

## Version-Guarded Code

- `areal/engine/megatron_utils/megatron_lora.py:185` —
  `hasattr(AutoBridge, "save_hf_adapter")` guard. The monkey-patch at line 291 is only
  applied when `save_hf_adapter` does not exist on `AutoBridge`. The module-level call
  at line 298 (`_monkey_patch_save_hf_adapter()`) runs at import time, so the guard is
  evaluated once on first import. If upgrading to a version that ships `save_hf_adapter`
  natively, the guard will skip patching, but the native signature must then match what
  AReaL's call site expects (see entry 4 above). Once confirmed compatible, the entire
  function `_monkey_patch_save_hf_adapter()` and its line-298 invocation can be removed.
