---
package: torchao
github: pytorch/ao
branch_template: v${VERSION}
upstream_paths:
  - torchao/prototype/blockwise_fp8_training/linear.py
  - torchao/prototype/blockwise_fp8_training/kernels.py
---

## Affected Files

### Primary (engine layer — most likely to break)

| File                                                      | Imports / Usage                                                                                                                        |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `areal/experimental/models/archon/fp8.py`                 | `torchao.prototype.blockwise_fp8_training.linear.fp8_blockwise_mm`; `fp8_blockwise_mm.apply(x, weight, block_size, dtype, use_triton)` |
| `areal/experimental/models/archon/moe/grouped_experts.py` | `torchao.prototype.blockwise_fp8_training.linear.fp8_blockwise_mm`; `fp8_blockwise_mm.apply(x, w, block_size, dtype, use_triton)`      |

### Secondary (model / infra layer)

| File                                         | Imports / Usage                                                                             |
| -------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `areal/api/cli_args.py`                      | `ArchonFP8Config` dataclass — string references to `torchao_config` field; no direct import |
| `areal/tools/check_pyproject_consistency.py` | `torchao` listed in `ESCAPABLE_PACKAGES` — expected to differ between sglang/vllm variants  |

### Tertiary (tests, config)

| File                                                     | Imports / Usage                                                                                                                                                                                                           |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/experimental/archon/conftest.py`                  | Conditional stub of 5 modules: `torchao`, `torchao.quantization`, `torchao.prototype`, `torchao.prototype.blockwise_fp8_training`, `torchao.prototype.blockwise_fp8_training.linear` when unavailable; SM90+ GPU required |
| `tests/experimental/archon/fp8/test_fp8_linear.py`       | `import torchao.prototype.blockwise_fp8_training.linear`; forward/backward correctness and convergence tests                                                                                                              |
| `tests/experimental/archon/fp8/test_fp8_scale_layout.py` | `torchao.prototype.blockwise_fp8_training.kernels.triton_fp8_blockwise_act_quant_lhs`, `triton_fp8_blockwise_weight_quant_transposed_rhs`                                                                                 |
| `tests/experimental/archon/fp8/test_conversion.py`       | Tests `enable_fp8_linear()` and `enable_fp8_experts()` patching logic                                                                                                                                                     |
| `tests/experimental/archon/fp8/test_moe_dispatch.py`     | Tests `_run_experts_fp8_for_loop()` with 11 edge cases                                                                                                                                                                    |

______________________________________________________________________

## API Usage Catalog

For each function/class below, verify the call signature against the upstream source at
the target version. Focus on: **missing new required parameters**, **removed old
parameters**, **renamed parameters**, **changed return types**, **changed method
signatures on returned objects**, and **moved/renamed modules**.

> **Warning:** All torchao usage goes through the `prototype` module path. These APIs
> are explicitly experimental and carry a HIGH risk of breakage between releases.

### 1. `torchao.prototype.blockwise_fp8_training.linear.fp8_blockwise_mm` (HIGH RISK)

**Source:** `torchao/prototype/blockwise_fp8_training/linear.py`

Imported inside `areal/experimental/models/archon/fp8.py` `enable_fp8_linear()` (line
36\) and stored on each patched module as `mod._fp8_mm`. Called in the patched forward
(line 156-157):

```python
from torchao.prototype.blockwise_fp8_training.linear import fp8_blockwise_mm

# Inside patched nn.Linear.forward — on-the-fly FP8 quantization
out = self._fp8_mm.apply(x, weight, self._fp8_block, x.dtype, self._fp8_use_triton)
# Positional args: (input, weight, block_size=128, dtype, use_triton)
```

Imported inside `areal/experimental/models/archon/moe/grouped_experts.py`
`_run_experts_fp8_for_loop()` (line 132). Called three times per expert (lines 161-166):

```python
from torchao.prototype.blockwise_fp8_training.linear import fp8_blockwise_mm

# SwiGLU: silu(x @ w1.T) * (x @ w3.T) @ w2.T — all via FP8
h1 = fp8_blockwise_mm.apply(x_e, w1_e, block_size, x_e.dtype, use_triton)
h3 = fp8_blockwise_mm.apply(x_e, w3_e, block_size, x_e.dtype, use_triton)
h = F.silu(h1) * h3
h2 = fp8_blockwise_mm.apply(h, w2_e, block_size, h.dtype, use_triton)
```

**Check:** Confirm `fp8_blockwise_mm` is still a `torch.autograd.Function` subclass with
an `.apply()` entry point. Verify positional argument order
`(x, weight, block_size, dtype, use_triton)` is unchanged. Check whether `block_size`
still accepts an `int` (default 128) and `use_triton` remains a plain `bool`. Confirm
the module path `torchao.prototype.blockwise_fp8_training.linear` has not been renamed
or graduated out of `prototype`.

______________________________________________________________________

### 2. `enable_fp8_linear()` (HIGH RISK — monkey-patches `nn.Linear`)

**Source:** `areal/experimental/models/archon/fp8.py` (AReaL function, not torchao)

Defined at line 18, calls `fp8_blockwise_mm` from torchao internally:

```python
def enable_fp8_linear(
    model: nn.Module,
    *,
    exclude_fqns: set[str] | None = None,  # defaults to {"output", "router", "score"}
    use_triton: bool = True,
) -> None:
    from torchao.prototype.blockwise_fp8_training.linear import fp8_blockwise_mm
    # ... patches eligible nn.Linear modules via _patch_fp8_forward(mod, fp8_blockwise_mm, use_triton)
```

Called from test at `tests/experimental/archon/fp8/test_fp8_scale_layout.py` (line 46):

```python
enable_fp8_linear(model, exclude_fqns=set())
```

**Check:** This is an AReaL function, but it depends on `fp8_blockwise_mm` from torchao
(entry 1). If torchao renames or moves `fp8_blockwise_mm`, this function's internal
import breaks. Verify `exclude_fqns` and `use_triton` are the only kwargs (no
`block_size` or `dtype` — block size is hardcoded as `_FP8_BLOCK = 128`).

______________________________________________________________________

### 3. `enable_fp8_experts()` (HIGH RISK — monkey-patches MoE expert modules)

**Source:** `areal/experimental/models/archon/fp8.py` (AReaL function, not torchao)

Defined at line 57:

```python
def enable_fp8_experts(model: nn.Module, *, use_triton: bool = True) -> None:
    # Patches each GroupedExperts module to use _run_experts_fp8_for_loop()
    # which internally calls fp8_blockwise_mm from torchao
```

**Check:** This is an AReaL function. Its only torchao dependency is transitive via
`_run_experts_fp8_for_loop()` → `fp8_blockwise_mm` (entry 1). Verify `use_triton` is the
only kwarg (no `block_size` or `dtype` — block size is hardcoded as `_FP8_BLOCK = 128`).
Confirm it targets `GroupedExperts` submodules specifically.

______________________________________________________________________

### 4. `validate_fp8_shard_alignment()`

**Source:** `areal/experimental/models/archon/fp8.py` (AReaL function, not torchao)

Defined at line 166:

```python
def validate_fp8_shard_alignment(
    model_parts: list[nn.Module],
    block_size: int = 128,  # _FP8_BLOCK
) -> None:
    # Checks both nn.Linear (2D) and GroupedExperts (3D) modules
    # for block-alignment AFTER parallelism (TP/PP) is applied.
    # Raises ValueError if local weight dims are not multiples of block_size.
```

**Check:** This is an AReaL function. It does NOT depend on torchao directly — it only
checks that modules already patched with `_fp8_block` attribute have compatible local
weight shapes. However, if torchao changes `_FP8_BLOCK` convention or the kernel's
alignment requirements, this validation must be updated to match.

______________________________________________________________________

### 5. `triton_fp8_blockwise_act_quant_lhs` / `triton_fp8_blockwise_weight_quant_transposed_rhs` (HIGH RISK)

**Source:** `torchao/prototype/blockwise_fp8_training/kernels.py`

Imported in `tests/experimental/archon/fp8/test_fp8_scale_layout.py` (lines 89-92,
101-104):

```python
from torchao.prototype.blockwise_fp8_training.kernels import (
    triton_fp8_blockwise_act_quant_lhs,
    triton_fp8_blockwise_weight_quant_transposed_rhs,
)

# Call signatures (lines 101-104):
x_fp8, x_scale = triton_fp8_blockwise_act_quant_lhs(x, block_size)
w_t_fp8, w_t_scale = triton_fp8_blockwise_weight_quant_transposed_rhs(w, block_size=block_size)

# Scale shape contract tested:
# x_scale.shape == (m, k_blocks)
# w_t_scale.shape == (k_blocks, n_blocks)
# Both MUST be column-major: stride(0) < stride(1)
```

**Check:** Confirm both kernel symbols still exist at this path and have not been
renamed (e.g., to drop the `triton_` prefix or unify under a single dispatch function).
Verify return types are `(fp8_tensor, scale_tensor)` tuples. Verify the scale layout
convention: column-major strides (`stride(0) < stride(1)`), shape
`(rows, ceil(cols/block_size))`. Any change in output tensor memory layout silently
breaks `fp8_blockwise_mm` which consumes these scales.

______________________________________________________________________

## Version-Guarded Code

No version-guarded code exists in AReaL for `torchao`. Availability is checked at test
time via `try/except` stubs in `tests/experimental/archon/conftest.py` — the stubs cover
all five affected submodules so tests degrade gracefully on machines without SM90+ GPUs
or without `torchao` installed.
