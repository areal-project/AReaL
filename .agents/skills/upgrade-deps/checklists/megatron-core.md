---
package: megatron-core
github: NVIDIA/Megatron-LM
branch_template: core_r${VERSION}
upstream_paths:
  - megatron/core/parallel_state.py
  - megatron/core/distributed/
  - megatron/core/optimizer/
  - megatron/core/optimizer_param_scheduler.py
  - megatron/core/pipeline_parallel/
  - megatron/core/transformer/transformer_config.py
  - megatron/core/transformer/pipeline_parallel_layer_layout.py
  - megatron/core/dist_checkpointing/
  - megatron/core/dist_checkpointing/serialization.py
  - megatron/core/dist_checkpointing/strategies/fully_parallel.py
  - megatron/core/fp8_utils.py
  - megatron/core/models/gpt/
  - megatron/core/packed_seq_params.py
  - megatron/core/models/common/embeddings/rotary_pos_embedding.py
  - megatron/core/utils.py
---

## Affected Files

### Primary (engine layer — most likely to break)

| File                                                     | Imports / Usage                                                                                                                                                                                                  |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `areal/engine/megatron_engine.py`                        | `parallel_state`, `tensor_parallel`, `DDP`, `finalize_model_grads`, `OptimizerConfig`, `get_megatron_optimizer`, `OptimizerParamScheduler`, `get_forward_backward_func`, `TransformerConfig`, `get_model_config` |
| `areal/engine/megatron_utils/checkpointer.py`            | `dist_checkpointing.save`, `load`, sharded strategies, `ShardedObject`                                                                                                                                           |
| `areal/engine/megatron_utils/megatron.py`                | `parallel_state`, FP8 detection, `TransformerConfig`                                                                                                                                                             |
| `areal/engine/megatron_utils/pipeline_parallel.py`       | `TransformerConfig`, `PipelineParallelLayerLayout`                                                                                                                                                               |
| `areal/engine/megatron_utils/packed_context_parallel.py` | `PackedSeqParams`, `parallel_state`                                                                                                                                                                              |
| `areal/engine/megatron_utils/fp8/tensor_helper.py`       | `fp8_utils.is_float8tensor`                                                                                                                                                                                      |

### Secondary (model / infra layer)

| File                                              | Imports / Usage                                                                                                                                                                                        |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `areal/models/mcore/registry.py`                  | `GPTModel`, `DDP`, `TransformerConfig`, `parallel_state`, `AutoConfig`                                                                                                                                 |
| `areal/models/mcore/bailing_moe.py`               | `MLATransformerConfig`, `LayerType`, `ModuleSpec`, `SelfAttention`, `AttnMaskType`, `TransformerLayer`, `TransformerLayerSubmodules`, `TransformerBlockSubmodules`, `apply_rotary_pos_emb`, rope_utils |
| `areal/models/mcore/tree_attn/module_megatron.py` | `TransformerConfig`, `SelfAttention`, `AttnMaskType`, layer specs                                                                                                                                      |
| `areal/models/mcore/lightning_attention.py`       | `parallel_state`, `apply_rotary_pos_emb`, `MegatronModule`, `ModuleSpec`, `build_module`                                                                                                               |
| `areal/models/mcore/bailing_moe_bridge.py`        | `MLATransformerConfig`, `AttnBackend`                                                                                                                                                                  |
| `areal/models/mcore/common.py`                    | `TransformerConfig`                                                                                                                                                                                    |
| `areal/models/mcore/qwen3.py`                     | `gpt_layer_specs`, `TransformerConfig`                                                                                                                                                                 |

### Tertiary (tests, infra)

| File                                                | Imports / Usage                                |
| --------------------------------------------------- | ---------------------------------------------- |
| `areal/infra/workflow_executor.py`                  | conditional `parallel_state` for DP world size |
| `tests/test_estimate_num_params.py`                 | `parallel_state`, `tensor_parallel`            |
| `tests/fp8/engine_utils.py`                         | `parallel_state`                               |
| `tests/fp8/model_hooks.py`                          | `parallel_state`                               |
| `tests/fp8/test_fp8_rmsnorm.py`                     | `fp8_utils`, `get_model_config`                |
| `tests/torchrun/run_megatron_engine_distributed.py` | `parallel_state`                               |

______________________________________________________________________

## API Usage Catalog

For each function/class below, verify the call signature against the upstream source at
the target version. Focus on: **missing new required parameters**, **removed old
parameters**, **renamed parameters**, **changed return types**, **changed method
signatures on returned objects**, and **moved/renamed modules**.

### 1. `megatron.core.parallel_state.initialize_model_parallel`

**Source:** `megatron/core/parallel_state.py`

Called in `areal/engine/megatron_engine.py` (line 195):

```python
mpu.initialize_model_parallel(
    tensor_model_parallel_size=self.parallel_strategy.tensor_parallel_size,
    pipeline_model_parallel_size=self.parallel_strategy.pipeline_parallel_size,
    virtual_pipeline_model_parallel_size=vpp_size if vpp_size > 1 else None,
    use_sharp=False,
    order="tp-cp-ep-dp-pp",
    context_parallel_size=self.parallel_strategy.context_parallel_size,
    expert_model_parallel_size=self.parallel_strategy.expert_parallel_size,
    expert_tensor_parallel_size=self.parallel_strategy.expert_tensor_parallel_size,
    distributed_timeout_minutes=int(DIST_GROUP_DEFAULT_TIMEOUT.seconds / 60),
)
```

**Check:** Verify all keyword arguments still accepted — especially
`expert_tensor_parallel_size` (recently added) and `distributed_timeout_minutes`. This
is the most critical call — any signature change blocks ALL Megatron training. Confirm
`order` string format `"tp-cp-ep-dp-pp"`.

______________________________________________________________________

### 2. `megatron.core.parallel_state` getter functions

**Source:** `megatron/core/parallel_state.py`

Called extensively in `areal/engine/megatron_engine.py`, `areal/models/mcore/*.py`:

```python
mpu.get_data_parallel_rank()
mpu.get_data_parallel_world_size()
mpu.get_data_parallel_group()
mpu.get_tensor_model_parallel_rank()
mpu.get_tensor_model_parallel_world_size()
mpu.get_tensor_model_parallel_group()
mpu.get_pipeline_model_parallel_rank()
mpu.get_pipeline_model_parallel_world_size()
mpu.get_pipeline_model_parallel_group()
mpu.get_context_parallel_rank()
mpu.get_context_parallel_world_size()
mpu.get_context_parallel_group()
mpu.get_expert_model_parallel_rank()
mpu.get_expert_model_parallel_world_size()
mpu.is_pipeline_last_stage()
mpu.is_pipeline_first_stage()
mpu.model_parallel_is_initialized()
mpu.destroy_model_parallel()
```

**Check:** Confirm all getter functions still exist with same no-arg signatures. These
are called dozens of times across the codebase — any removal or rename is immediately
fatal.

______________________________________________________________________

### 3. `megatron.core.parallel_state` utility functions

**Source:** `megatron/core/parallel_state.py`

Called in `areal/engine/megatron_engine.py` (line 1048):

```python
rank_generator = mpu.RankGenerator(
    tp=self.parallel_strategy.tensor_parallel_size,
    ep=1,
    dp=self.parallel_strategy.data_parallel_size,
    pp=self.parallel_strategy.pipeline_parallel_size,
    cp=self.parallel_strategy.context_parallel_size,
    order="tp-cp-ep-dp-pp",
    rank_offset=0,
)
context_and_model_parallel_ranks = rank_generator.get_ranks("tp-cp-pp")

group = mpu.create_group(
    ranks,
    timeout=DIST_GROUP_DEFAULT_TIMEOUT,
    pg_options=mpu.get_nccl_options("tp-cp-pp", {}),
    group_desc="CONTEXT_AND_MODEL_PARALLEL_GROUP",
)
```

**Check:** Verify `RankGenerator.__init__` keyword args — especially `rank_offset` and
`order`. Confirm `get_ranks(group_str)` method. Verify `create_group` accepts `timeout`,
`pg_options`, `group_desc` kwargs. Confirm `get_nccl_options(group_name, config_dict)`
signature.

______________________________________________________________________

### 4. `megatron.core.distributed.DistributedDataParallel`

**Source:** `megatron/core/distributed/`

Called in `areal/models/mcore/registry.py` (line 310):

```python
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig as MCoreDDPConfig

ddp_config = MCoreDDPConfig(**dataclasses.asdict(mcore_config.ddp))
wrapped = DDP(
    config=tf_config,
    ddp_config=ddp_config,
    module=model,
    disable_bucketing=False,
)
```

**Check:** Verify keyword arguments `config`, `ddp_config`, `module`,
`disable_bucketing` are still accepted. Note AReaL does NOT pass `data_parallel_group`
or `expert_data_parallel_group` — confirm these remain optional with sane defaults.
Check for new required parameters. Also verify `MCoreDDPConfig` dataclass fields haven't
changed (constructed from `mcore_config.ddp`).

______________________________________________________________________

### 5. `megatron.core.distributed.finalize_model_grads`

**Source:** `megatron/core/distributed/`

Assigned in `areal/engine/megatron_engine.py` (line 403):

```python
model_config.finalize_model_grads_func = finalize_model_grads
```

Not called directly by AReaL — instead assigned to
`TransformerConfig.finalize_model_grads_func` and invoked internally by Megatron's
pipeline parallel forward-backward functions.

**Check:** Confirm `finalize_model_grads` still has a compatible signature for
`TransformerConfig.finalize_model_grads_func`. Verify it accepts `(model_chunks)` or
`(model_chunks, num_tokens=...)` as expected by the pipeline executor.

______________________________________________________________________

### 6. `megatron.core.optimizer.OptimizerConfig`

**Source:** `megatron/core/optimizer/`

Called in `areal/engine/megatron_engine.py` (line 1090):

```python
mcore_opt_config = MCoreOptimizerConfig(
    optimizer=self.optimizer_config.type,
    lr=self.optimizer_config.lr,
    min_lr=self.optimizer_config.min_lr_ratio * self.optimizer_config.lr,
    weight_decay=self.optimizer_config.weight_decay,
    bf16=self.dtype is torch.bfloat16,
    fp16=self.dtype is torch.float16,
    adam_beta1=self.optimizer_config.beta1,
    adam_beta2=self.optimizer_config.beta2,
    adam_eps=self.optimizer_config.eps,
    use_distributed_optimizer=use_distributed_optimizer,
    params_dtype=self.dtype,
    clip_grad=self.optimizer_config.gradient_clipping,
    fp8_recipe=(self.fp8_config.recipe if self.enable_fp8 else None),
)
# Post-init field assignments:
mcore_opt_config.overlap_param_gather_with_optimizer_step = ...
mcore_opt_config.use_precision_aware_optimizer = ...
mcore_opt_config.main_grads_dtype = getattr(torch, ...)
mcore_opt_config.main_params_dtype = getattr(torch, ...)
mcore_opt_config.exp_avg_dtype = getattr(torch, ...)
mcore_opt_config.exp_avg_sq_dtype = getattr(torch, ...)
```

**Check:** Verify all ~13 constructor kwargs still accepted — especially `fp8_recipe`
and `fp16`. Confirm the 6 post-init fields (`overlap_param_gather_with_optimizer_step`,
`use_precision_aware_optimizer`, `main_grads_dtype`, `main_params_dtype`,
`exp_avg_dtype`, `exp_avg_sq_dtype`) still exist on the dataclass. Check for renamed or
removed fields.

______________________________________________________________________

### 7. `megatron.core.optimizer.get_megatron_optimizer`

**Source:** `megatron/core/optimizer/`

Called in `areal/engine/megatron_engine.py` (line 1122):

```python
self.optimizer = get_megatron_optimizer(mcore_opt_config, self.model)
```

**Check:** Verify the two-arg call `(config, model_chunks)` is still valid. Confirm
`model_chunks` accepts a list of DDP-wrapped modules. Check if new required arguments
were added (e.g., `no_weight_decay_cond`, `scale_lr_cond` — AReaL does not pass these,
so confirm they remain optional).

______________________________________________________________________

### 8. `megatron.core.optimizer_param_scheduler.OptimizerParamScheduler`

**Source:** `megatron/core/optimizer_param_scheduler.py`

Called in `areal/engine/megatron_engine.py` (line 1126):

```python
lr_scheduler = OptimizerParamScheduler(
    self.optimizer,
    init_lr=0.0 if warmup_steps_proportion > 0 else self.optimizer_config.lr,
    max_lr=self.optimizer_config.lr,
    min_lr=self.optimizer_config.min_lr_ratio * self.optimizer_config.lr,
    lr_warmup_steps=warmup_steps,
    lr_decay_steps=ft_spec.total_train_steps - warmup_steps,
    lr_decay_style=self.optimizer_config.lr_scheduler_type,
    start_wd=self.optimizer_config.weight_decay,
    end_wd=self.optimizer_config.weight_decay,
    wd_incr_steps=ft_spec.total_train_steps,
    wd_incr_style="constant",
)
```

**Check:** Confirm all keyword arguments still accepted — especially `start_wd`,
`end_wd`, `wd_incr_steps`, `wd_incr_style` (weight decay scheduling). Verify
`lr_decay_style` valid choices (e.g., `"cosine"`, `"linear"`, `"WSD"`). Check for
renamed params.

______________________________________________________________________

### 9. `megatron.core.pipeline_parallel.get_forward_backward_func`

**Source:** `megatron/core/pipeline_parallel/`

Called in `areal/engine/megatron_engine.py` (line 732, hot training loop):

```python
forward_backward_func = get_forward_backward_func()
forward_backward_func(
    forward_step_func=forward_step,
    data_iterator=data_iterator,
    model=self.model if len(self.model) > 1 else self.model[0],
    num_microbatches=len(mb_list),
    seq_length=mb_list.max_seqlen,
    micro_batch_size=1,
    forward_only=forward_only,
)
```

**Check:** Verify `get_forward_backward_func()` still returns a callable with this
signature. Confirm `seq_length` and `micro_batch_size` kwargs are still accepted (marked
"no use when input_shapes was set" in code). This is in the HOT training loop — any
breakage stops all PP training.

______________________________________________________________________

### 10. `megatron.core.transformer.TransformerConfig`

**Source:** `megatron/core/transformer/transformer_config.py`

Used in `areal/engine/megatron_engine.py`,
`areal/engine/megatron_utils/pipeline_parallel.py`, `areal/models/mcore/registry.py`,
`areal/models/mcore/common.py`, `areal/models/mcore/qwen3.py`:

```python
transformer_config = TransformerConfig(
    num_layers=num_layers,
    hidden_size=hidden_size,
    num_attention_heads=num_heads,
    num_key_value_heads=num_kv_heads,
    # ... many more fields
)
```

**Check:** This is the central config dataclass — verify no fields were removed or
renamed. New optional fields with defaults are fine. Check `MLATransformerConfig`
subclass compatibility (used in `bailing_moe.py` and `bailing_moe_bridge.py`).

______________________________________________________________________

### 11. `megatron.core.dist_checkpointing` save/load

**Source:** `megatron/core/dist_checkpointing/`

Called in `areal/engine/megatron_utils/checkpointer.py` (lines 23-32, 53-85):

```python
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.serialization import (
    get_default_save_sharded_strategy,
    get_default_load_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)

# Save (line 62):
save_strategy = get_default_save_sharded_strategy("torch_dist")
save_strategy = FullyParallelSaveStrategyWrapper(save_strategy, dp_group)
dist_checkpointing.save(
    sharded_state_dict, ckpt_path,
    sharded_strategy=save_strategy,
    async_sharded_save=async_save,
    validate_access_integrity=validate_sharding_integrity,
)

# Load (line 75):
load_strategy = get_default_load_sharded_strategy(ckpt_dir)
load_strategy = FullyParallelLoadStrategyWrapper(load_strategy, dp_group)
state_dict = dist_checkpointing.load(
    sharded_state_dict, ckpt_dir, sharded_strategy=load_strategy
)
```

**Check:** Verify `save` kwargs — especially `async_sharded_save` and
`validate_access_integrity`. Confirm `get_default_save_sharded_strategy` accepts
`"torch_dist"` string. Confirm `get_default_load_sharded_strategy` accepts `ckpt_dir`
path. Verify import paths: `serialization` for strategy factories,
`strategies.fully_parallel` for wrappers, `mapping` for `ShardedObject`. Checkpoint
format compatibility is critical — verify no format version bumps.

______________________________________________________________________

### 12. `megatron.core.transformer.PipelineParallelLayerLayout`

**Source:** `megatron/core/transformer/pipeline_parallel_layer_layout.py`

Called in `areal/engine/megatron_utils/pipeline_parallel.py` (line 6):

```python
from megatron.core.transformer.pipeline_parallel_layer_layout import (
    PipelineParallelLayerLayout,
)
```

**Check:** Confirm the import path is still
`megatron.core.transformer.pipeline_parallel_layer_layout` (not
`transformer_config.py`). Verify constructor kwargs. Check if
`get_num_layers_to_build()` and `get_transformer_layer_offset()` are still accessible.

______________________________________________________________________

### 13. `megatron.core.fp8_utils.is_float8tensor`

**Source:** `megatron/core/fp8_utils.py`

Called in `areal/engine/megatron_utils/fp8/tensor_helper.py`:

```python
from megatron.core.fp8_utils import is_float8tensor
if is_float8tensor(param):
    ...
```

**Check:** Confirm function still exists at this path. Verify it accepts a single tensor
argument. Check if FP8 support was restructured (e.g., moved to
`megatron.core.extensions.transformer_engine`).

______________________________________________________________________

### 14. `megatron.core.models.gpt.GPTModel`

**Source:** `megatron/core/models/gpt/`

Called in `areal/models/mcore/registry.py`:

```python
model = GPTModel(
    config=transformer_config,
    transformer_layer_spec=layer_spec,
    vocab_size=vocab_size,
    max_sequence_length=max_seq_len,
    pre_process=pre_process,
    post_process=post_process,
)
```

**Check:** Verify constructor kwargs. Confirm `get_gpt_decoder_block_spec()` still
returns a valid layer spec. Check for new required parameters.

______________________________________________________________________

### 15. `megatron.core.tensor_parallel` utilities

**Source:** `megatron/core/tensor_parallel/`

Called in `areal/engine/megatron_engine.py` and
`areal/models/mcore/lightning_attention.py`:

```python
tensor_parallel.model_parallel_cuda_manual_seed(seed)
tensor_parallel.gather_from_sequence_parallel_region(output)
tensor_parallel.get_cuda_rng_tracker()
```

**Check:** Verify these functions still exist with same signatures.
`model_parallel_cuda_manual_seed` is called once at init;
`gather_from_sequence_parallel_region` is in hot path.

______________________________________________________________________

### 16. Transformer layer specs and module builders

**Source:** `megatron/core/transformer/`

Called in `areal/models/mcore/bailing_moe.py`,
`areal/models/mcore/lightning_attention.py`,
`areal/models/mcore/tree_attn/module_megatron.py`:

```python
from megatron.core.transformer import ModuleSpec, build_module
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.transformer_layer import (
    TransformerLayer, TransformerLayerSubmodules,
)
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.custom_layers.transformer_engine import (
    # TE norm modules
)
```

**Check:** Verify enum values in `AttnMaskType` and `LayerType`. Confirm `ModuleSpec`
and `build_module` signatures. Check `SelfAttention.__init__` for new required params.
Verify TE integration module paths haven't been restructured.

______________________________________________________________________

### 17. `PackedSeqParams`

**Source:** `megatron/core/packed_seq_params.py` (or `megatron/core/transformer/`)

Called in `areal/engine/megatron_utils/packed_context_parallel.py`:

```python
from megatron.core.packed_seq_params import PackedSeqParams
packed_seq_params = PackedSeqParams(
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_kv=cu_seqlens_kv,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_kv=max_seqlen_kv,
    qkv_format="thd",
)
```

**Check:** Confirm constructor kwargs and import path. Verify `qkv_format` valid
choices.

______________________________________________________________________

### 18. RoPE utilities

**Source:** `megatron/core/models/common/embeddings/`

Called in `areal/models/mcore/bailing_moe.py`,
`areal/models/mcore/lightning_attention.py`:

```python
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
# Also various rope_utils for extended RoPE (YaRN, etc.)
```

**Check:** Verify `apply_rotary_pos_emb` signature (`t`, `freqs`). This function is
heavily patched in `bailing_moe.py` — confirm the base implementation hasn't changed in
ways that break the patches.

______________________________________________________________________

## Version-Guarded Code

No known version-guarded code exists in AReaL for `megatron-core`. However, the
`MLATransformerConfig` class used in `bailing_moe.py` and `bailing_moe_bridge.py` is a
relatively new addition — confirm it exists in the target version.
