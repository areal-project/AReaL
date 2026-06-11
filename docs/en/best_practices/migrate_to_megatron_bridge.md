# Migrating from mbridge to megatron-bridge

AReaL's `MegatronEngine` supports two backends for HuggingFace ↔ Megatron-Core weight
conversion and model creation:

- `mbridge` (default): [ISEEKYAN/mbridge](https://github.com/ISEEKYAN/mbridge)
- `megatron-bridge`:
  [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)

You select the backend per-experiment via `actor.megatron.bridge_type`:

```yaml
actor:
  megatron:
    bridge_type: megatron-bridge   # or "mbridge" (default)
```

This document covers when to switch, what behavior changes, and how to add support for a
new model architecture under the `megatron-bridge` backend.

## When to use which

| Need                                               | Prefer            |
| -------------------------------------------------- | ----------------- |
| Existing setups, disk-based HF weight load/save    | `mbridge`         |
| Tree-attention training in `MegatronEngine`        | `mbridge`         |
| PEFT/LoRA support                                  | `megatron-bridge` |
| Architectures NVIDIA upstream maintains officially | `megatron-bridge` |

`megatron-bridge` is the long-term direction: it has PEFT support, NVIDIA upstream
maintenance, and broader model coverage. `mbridge` is being deprecated but remains the
default for backward compatibility — see `docs/en/reference/bridge_backend.md` for the
policy.

## API differences

The differences mostly do not surface to user code — `MegatronEngine` hides both
backends behind `_build_hf_mcore_bridge()`. The table below is for contributors adding
new model adapters.

| Concept               | mbridge                                               | megatron-bridge                                                                                             |
| --------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Top-level import      | `import mbridge`                                      | `from megatron.bridge import AutoBridge`                                                                    |
| Adapter base class    | `mbridge.core.LLMBridge`                              | `megatron.bridge.models.conversion.model_bridge.MegatronModelBridge`                                        |
| Registration          | `@register_model("name")`                             | `@MegatronModelBridge.register_bridge(source=..., target=GPTModel, ...)`                                    |
| Weight mapping        | Class-level dicts (`_DIRECT_MAPPING`, `_MLP_MAPPING`) | `mapping_registry()` returns `MegatronMappingRegistry(*[AutoMapping/QKVMapping/GatedMLPMapping])`           |
| HF config             | `bridge.hf_config`                                    | `bridge.hf_pretrained.config`                                                                               |
| Transformer config    | `bridge.config` (TransformerConfig)                   | `bridge.transformer_config`                                                                                 |
| Model creation        | `bridge.get_model(wrap_with_ddp=...)`                 | `provider = bridge.to_megatron_provider()` → set parallel sizes → `provider.provide_distributed_model(...)` |
| Layer spec injection  | Override `bridge._get_transformer_layer_spec()`       | Set `provider.transformer_layer_spec = lambda config: spec`                                                 |
| Runtime config tweaks | `bridge.set_extra_args(field=value, ...)`             | `provider.field = value` directly                                                                           |
| Load HF weights       | `bridge.load_weights(model, path)`                    | `bridge.load_hf_weights(model)`                                                                             |
| Save as HF            | `bridge.save_weights(model, path)`                    | `bridge.save_hf_pretrained(model, path, distributed_save=True)`                                             |

## Supported architectures

Architectures registered in AReaL's `mcore/registry.py` and routed through both
backends:

| HF architecture               | `mbridge` | `megatron-bridge` | Notes                                 |
| ----------------------------- | --------- | ----------------- | ------------------------------------- |
| `Qwen3ForCausalLM`            | ✅        | ✅ (NV upstream)  | Dense.                                |
| `BailingMoeV2_5ForCausalLM`   | ✅        | ✅                | Heterogeneous Lightning + MLA layers. |
| `BailingMoeLinearForCausalLM` | ✅        | ✅                | Shares `BailingMoeV25Bridge` adapter. |
| `BailingHybridForCausalLM`    | ✅        | ✅                | Shares `BailingMoeV25Bridge` adapter. |

Custom adapters live under `areal/models/mcore/`:

- mbridge subclasses: `bailing_moe_bridge.py`
- megatron-bridge subclasses: `bailing_moe_megatron_bridge.py`

## How registry dispatch works

`areal/models/mcore/registry.py` is the central dispatch point. The relevant functions:

- `make_hf_and_mcore_config(hf_path, dtype, bridge, bridge_type)`:

  - With `bridge_type="mbridge"`, returns `(bridge.hf_config, bridge.config)`.
  - With `bridge_type="megatron-bridge"`, returns
    `(bridge.hf_pretrained.config, bridge.transformer_config)`.

- `make_mcore_model(hf_config, tf_config, mcore_config, bridge, bridge_type, ...)`:

  - With `mbridge`, calls `bridge.get_model(...)`.
  - With `megatron-bridge`, calls `bridge.to_megatron_provider(load_weights=False)` and
    configures the provider with the current TP/PP/CP/EP context, then
    `provider.provide_distributed_model(...)`. Before configuring the provider, it
    overrides `provider.transformer_layer_spec` for models whose layer structure
    megatron-bridge's default spec doesn't express — currently the **Bailing-MoE V2.5
    family**, which uses AReaL's heterogeneous Lightning + MLA layer spec.

## Adding a new model under `megatron-bridge`

Sketch of a new adapter (`my_model_megatron_bridge.py`):

```python
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping, QKVMapping, GatedMLPMapping,
)
from megatron.core.models.gpt.gpt_model import GPTModel


@MegatronModelBridge.register_bridge(
    source="MyModelForCausalLM",
    target=GPTModel,
    model_type="my_model",
)
class MyModelBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):
        provider = super().provider_bridge(hf_pretrained)
        # Set MLA/MoE/RoPE fields from hf_pretrained.config here.
        # provider.num_moe_experts = hf_pretrained.config.n_routed_experts
        # provider.moe_router_topk = hf_pretrained.config.num_experts_per_tok
        # ...
        return provider

    def mapping_registry(self):
        return MegatronMappingRegistry(
            AutoMapping(
                hf_param="model.embed_tokens.weight",
                megatron_param="embedding.word_embeddings.weight",
            ),
            QKVMapping(...),
            GatedMLPMapping(...),
            # ...
        )
```

Then register the module-import in `areal/engine/megatron_engine.py` to fire the
decorator:

```python
import areal.models.mcore.my_model_megatron_bridge  # noqa: F401  # register bridge
```

If the model has custom attention modules that megatron-bridge's default
`get_gpt_decoder_block_spec` cannot express, add a branch in
`registry.make_mcore_model()` to inject your spec (mirror the `_is_bailing` branch).

## Common pitfalls

- **mbridge package missing.** If `mbridge` is not installed, the import is wrapped in
  `try/except` so the engine still loads, but attempting to use `bridge_type=mbridge`
  raises a clear `ImportError`. Switch to `megatron-bridge` or install mbridge.
- **`moe_token_dispatcher_type` ignored.** Under `megatron-bridge`, AReaL forces
  `alltoall` and `variable_seq_lengths=True`. A warning is logged when your config
  requested something else.
- **Tree training unavailable.** `megatron-bridge` does not yet support tree-attention
  training. The engine raises `NotImplementedError` at initialization if both are
  requested. Stay on `mbridge` for tree training.
- **LoRA only on megatron-bridge.** AReaL's PEFT/LoRA path requires
  `bridge_type=megatron-bridge`.
- **Heterogeneous layer specs.** Bailing-MoE V2.5 mixes Lightning and MLA attention
  per-layer. The `mapping_registry()` cannot use wildcard `*` mappings — enumerate every
  layer and dispatch on `is_lightning_layer()`.

## Verification checklist

After switching `bridge_type` for an existing experiment, verify:

1. Engine initialization completes (`MegatronEngine.initialize()` returns).
1. First training step's `loss` matches the previous backend within a small tolerance
   (recommended: `abs_diff < 5e-3` on the first 5 steps with `disable_dropout=True`,
   fixed seed).
1. HF checkpoint round-trip: save with `save_hf_pretrained`, then load with
   `transformers.AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and
   run a forward pass.
1. For MoE models, verify `moe_router_*` fields on the provider match those on
   `bridge.transformer_config` (under `megatron-bridge`) or `bridge.config` (under
   `mbridge`).
