# Third-party patches

Patches applied to pinned third-party sources during the NPU image build
(`Dockerfile.a2`, `Dockerfile.a3`). Each patch is applied with `git apply` at an
explicit point in the target's installation, so a patch that no longer applies fails the
build instead of silently dropping the fix. The vLLM patches run after their editable
installs; the Megatron-Bridge patch runs immediately after its pinned checkout.

| Patch                                       | Applies to               | Image     | Upstream                                                                                                                                                                                                           | Delete when                                     |
| ------------------------------------------- | ------------------------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| `vllm-exact-token-validation.patch`         | vLLM `v0.30.0`           | A2 and A3 | none — AReaL-local                                                                                                                                                                                                 | upstream exposes an equivalent prompt assertion |
| `megatron-bridge.v0.5.1-vision-dp-cp.patch` | Megatron-Bridge `v0.5.1` | A2 and A3 | [Bridge#4784](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/4784), [Qwen3-ASR guard](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/qwen3_asr/hf_qwen3_asr/__init__.py) | the pin contains both upstream fixes            |

vLLM 0.30 includes the upstream content transport change (#51478), so only the
AReaL-local exact-token assertion is patched.

The vllm-ascend pin includes
[#15981](https://github.com/vllm-project/vllm-ascend/pull/15981), which repairs cached
upstream MoE factory bindings in already imported model modules. The local Qwen3-MoE
import-order workaround is no longer needed. This is separate from the Megatron-Bridge
Qwen3-ASR registration guard below, which is still required.

The vLLM patch preserves AReaL's exact-token generation contract, so that a multimodal
rollout computes behavior logprobs from the same token sequence it trains on. See issue
#1612 for the problem statement and the staged plan.

- **vllm#51478** — merged upstream on 2026-08-11, after the `v0.23.0` and `v0.26.0` tags
  and included in `v0.30.0`. Adds `content_parts` to `/inference/v1/generate` so one
  request carries caller-supplied token ids together with raw media. Python frontend
  only: AReaL launches vLLM through `areal.engine.vllm_ext.areal_vllm_server`, which
  patches Python vLLM's `build_app`.
- **exact-token-validation** — AReaL-local, no upstream equivalent. vLLM expands
  multimodal placeholders itself, so the caller sends the collapsed prompt in
  `token_ids` and its locally expanded prompt in `expected_token_ids`; the server
  refuses to generate unless its expansion matches. The patch must not reference AReaL's
  pause event — weight-update policy stays in AReaL source.

## AWEX source override

The ARM64 images install AWEX from `HwVanICI/asystem-awex` at
`8b9ad5072e105af0112afa805e8961efc190e515`, immediately after vLLM installation. This
revision includes vLLM 0.30 import compatibility, expert-name normalization, and
in-place refresh of derived Ascend inference weights. No local AWEX patches are applied.

Build with `--build-arg AWEX_COMMIT=` to use the AWEX version in `pyproject.npu.toml`
instead. That fallback does not receive these source fixes. `AWEX_REPO` and
`AWEX_COMMIT` select the source override. When enabled, the application dependency
install filters AWEX from a temporary manifest, preserving the source installation. The
final dependency check skips the source override and checks the manifest requirement
only when the override is disabled.

## Megatron-Bridge compatibility fixes

**Megatron-Bridge#4784** fixes the autograd collective used when
`vision_dp_when_cp=True`. Its backward pass must sum gradients from every CP rank before
slicing the local image range. It also keeps empty-image and frozen-vision ranks in the
collective and creates empty outputs in the model parameter dtype. Without the complete
fix, training can silently compute incomplete vision gradients or hang when ranks do not
participate symmetrically.

The vision portion contains the two production-source changes from upstream commit
`1d65d5756f2bf9f7f8734467a72272901bfcb4e3`. Those hunks apply directly to the pinned
Megatron-Bridge `v0.5.1` revision and do not require Megatron-Core 0.19. Remove the
vision portion when the Bridge pin contains that commit.

The same patch also carries the current upstream Qwen3-ASR registration guard.
Transformers 5.14 includes a native `qwen3_asr` config, so Bridge must skip all three
vendored Auto-class registrations when that native mapping exists. Without the guard,
importing `megatron.bridge` fails before actor model construction.

## Patching dirties the tree, which changes `vllm.__version__`

vLLM derives its version with setuptools-scm, which appends a dev suffix when the
worktree is dirty. vllm-ascend uses the detected version for compatibility routing, so
each image installs the clean checkout before applying its versioned patches.

Patching *after* `pip install -e .` avoids this: setuptools-scm records the version from
the still-clean tree, and `vllm/_version.py` is generated once at install time rather
than recomputed at import, so the tree going dirty afterwards no longer matters. This
ordering is load-bearing — do not move a `git apply` above its install.

It relies on the install being editable. Without `-e`, pip would copy the sources at
install time and the patch would land on a copy nothing imports, silently dropping the
fix. Both installs must therefore stay `-e`.

The vLLM patch group is followed by an assertion that the recorded version still equals
the tag, so a regression here fails the build rather than the first rollout of a
training run. It reads `vllm/_version.py` directly rather than importing vLLM, which at
that point in the build has neither torch nor an NPU available. Adding a patch means
adding its `git apply` above that assertion, never below it.

## Bumping vLLM

The patch contents follow the versions in `VLLM_TAG` / `VLLM_ASCEND_BRANCH`. Keep the
vLLM patch filenames stable across upgrades; update their contents, header provenance,
and the target versions in the table above. If a fix has landed, delete the patch and
its `COPY` / `git apply` lines from both Dockerfiles.
