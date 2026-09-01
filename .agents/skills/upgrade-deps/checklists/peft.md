---
package: peft
github: huggingface/peft
branch_template: v${VERSION}
upstream_paths:
  - src/peft/peft_model.py
  - src/peft/tuners/lora/config.py
  - src/peft/utils/peft_types.py
---

## Affected Files

### Primary (engine layer — most likely to break)

| File                          | Imports / Usage                            |
| ----------------------------- | ------------------------------------------ |
| `areal/engine/fsdp_engine.py` | `LoraConfig`, `TaskType`, `get_peft_model` |

### Secondary (model / infra layer)

| File                                             | Imports / Usage                                                                                                                                                     |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `areal/engine/megatron_engine.py`                | indirect — consumes PEFT config dict schema (`r`, `lora_alpha`, `target_modules`, `bias`) via `megatron.bridge.peft.lora`; stores in `WeightUpdateMeta.peft_config` |
| `areal/models/mcore/hf_save.py`                  | no direct import — generates PEFT adapter weights and `adapter_config.json` for the legacy mbridge save path                                                        |
| `areal/engine/vllm_ext/vllm_worker_extension.py` | indirect — uses `vllm.lora.peft_helper.PEFTHelper`, not direct peft import                                                                                          |
| `areal/engine/vllm_remote.py`                    | builds HTTP payloads containing `peft_config` dict for distributed LoRA updates                                                                                     |
| `areal/api/cli_args.py`                          | config fields only: `use_lora`, `lora_rank`, `lora_alpha`, `target_modules`, `peft_type`                                                                            |
| `areal/api/io_struct.py`                         | `WeightUpdateMeta.peft_config: dict` — schema `{"r": rank, "lora_alpha": alpha, "target_modules": [...], "bias": "none"}`                                           |

### Tertiary (tests, config)

| File                                               | Imports / Usage                                                                     |
| -------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `tests/torchrun/run_fsdp_memory_efficient_lora.py` | integration test with `use_lora=True, lora_rank=8, lora_alpha=16, peft_type="lora"` |

______________________________________________________________________

## API Usage Catalog

For each function/class below, verify the call signature against the upstream source at
the target version. Focus on: **missing new required parameters**, **removed old
parameters**, **renamed parameters**, **changed return types**, **changed method
signatures on returned objects**, and **moved/renamed modules**.

### 1. `peft.LoraConfig` and `peft.TaskType.CAUSAL_LM`

**Source:** `src/peft/tuners/lora/config.py`, `src/peft/utils/peft_types.py`

Called in `areal/engine/fsdp_engine.py` (`_apply_peft_wrapper`, lines 930–946):

```python
from peft import LoraConfig, TaskType, get_peft_model

peft_config = {
    "task_type": TaskType.CAUSAL_LM,
    "r": config.lora_rank,
    "lora_alpha": config.lora_alpha,
    "target_modules": target_modules,  # str "all-linear" or list[str]
    "bias": "none",
}
if self.config.peft_type == "lora":
    peft_config = LoraConfig(**peft_config)
else:
    raise NotImplementedError()
```

**Check:** Verify `LoraConfig.__init__` still accepts `task_type`, `r`, `lora_alpha`,
`target_modules`, and `bias` as keyword arguments with the same types. Check whether any
new required parameters were added. Confirm `bias="none"` is still a valid string choice
(not renamed or replaced with an enum). Confirm `target_modules` still accepts both a
string (`"all-linear"`) and a `list[str]`. Verify `TaskType` is still an enum importable
from `peft` and `TaskType.CAUSAL_LM` still exists. Verify the class is still importable
directly from `peft` (not moved to `peft.tuners.lora` only).

______________________________________________________________________

### 2. `peft.get_peft_model`

**Source:** `src/peft/peft_model.py`

Called in `areal/engine/fsdp_engine.py` (`_apply_peft_wrapper`, lines 948–953), after
`model.enable_input_require_grads()` and before FSDP sharding:

```python
self.model.enable_input_require_grads()
self.model = get_peft_model(
    self.model,
    peft_config,
    autocast_adapter_dtype=False,
)
```

**Check:** Confirm `get_peft_model` is still exported at the top-level `peft` namespace.
Verify its signature is `get_peft_model(model, peft_config, ...)` with
`autocast_adapter_dtype` still an accepted keyword argument (it was added in a mid-cycle
release and could be renamed or removed). Confirm the return type is still a `PeftModel`
wrapping the original model. Check whether the returned model's `.parameters()`,
`.named_modules()`, and FSDP-compatibility hooks remain intact — the call site
immediately passes the result into FSDP2 sharding.

______________________________________________________________________

### 3. PEFT weight key format and `adapter_config.json` schema (indirectly consumed)

**Source:** `src/peft/tuners/lora/` (weight saving conventions)

Produced in `areal/models/mcore/hf_save.py` (lines 586-610):

```python
lora_state_dict[f"base_model.model.{hf_name_q}"] = q_param
lora_state_dict[f"base_model.model.{hf_name_k}"] = k_param
lora_state_dict[f"base_model.model.{hf_name_v}"] = v_param
```

The same function emits `adapter_config.json` at lines 656-673:

```python
{
    "base_model_name_or_path": base_model_name,
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "lora_alpha": lora_alpha,
    "lora_dropout": 0.0,
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": rank,
    "target_modules": target_modules,
    "task_type": "CAUSAL_LM",
}
```

**Check:** Confirm the saved weight key format (`base_model.model.<...>.lora_A.weight` /
`lora_B.weight`) has not changed. Verify all emitted `adapter_config.json` fields still
match what PEFT's `LoraConfig.from_pretrained()` expects when loading. The smaller
runtime schema used for vLLM LoRA hot-swap is documented separately below.

______________________________________________________________________

### 4. `WeightUpdateMeta.peft_config` dict schema (cross-engine contract)

**Source:** PEFT conventions (not a direct import, but a schema contract)

Populated in `areal/engine/megatron_engine.py` (lines 2066-2074) and
`areal/engine/fsdp_engine.py` (lines 1376-1389):

```python
meta.peft_config = {
    "r": self.config.lora_rank,
    "lora_alpha": self.config.lora_alpha,
    "target_modules": target_modules,  # list[str] or str
    "bias": "none",
}
```

Consumed in `areal/engine/vllm_remote.py` (lines 290-293):

```python
"lora_target_modules": meta.peft_config["target_modules"],
"lora_rank": meta.peft_config["r"],
"lora_alpha": meta.peft_config["lora_alpha"],
"lora_bias": meta.peft_config["bias"],
```

And in `areal/engine/vllm_ext/vllm_worker_extension.py` (lines 325-330) to reconstruct a
`PEFTHelper` for vLLM's LoRA manager.

**Check:** This dict is the **contract** between AReaL's training engines and inference
backends. If PEFT changes its config field names (`r` → `rank`, `lora_alpha` → `alpha`,
etc.), all three sites must be updated in sync. Verify that vLLM's
`PEFTHelper.from_dict` still accepts the same key names as what PEFT's `LoraConfig`
uses.

______________________________________________________________________

## Version-Guarded Code

No known version-guarded code exists in AReaL for `peft`.
