# Third-party patches

Patches applied to the pinned vLLM / vllm-ascend sources during the NPU image build
(`Dockerfile.a2`, `Dockerfile.a3`). Each patch is applied with `git apply` immediately
after its `pip install -e .`, so a patch that no longer applies fails the build instead
of silently dropping the fix.

| Patch                                       | Applies to                     | Upstream                                                                      | Delete when                                     |
| ------------------------------------------- | ------------------------------ | ----------------------------------------------------------------------------- | ----------------------------------------------- |
| `vllm.v0.23.0.patch`                        | vLLM `v0.23.0`                 | [vllm#44483](https://github.com/vllm-project/vllm/pull/44483)                 | the pin contains #44483                         |
| `vllm.v0.23.0-content-parts.patch`          | vLLM `v0.23.0`                 | [vllm#51478](https://github.com/vllm-project/vllm/pull/51478)                 | the pin contains #51478                         |
| `vllm.v0.23.0-exact-token-validation.patch` | vLLM `v0.23.0`                 | none — AReaL-local                                                            | upstream exposes an equivalent prompt assertion |
| `vllm-ascend.v0.23.0.patch`                 | vllm-ascend `releases/v0.23.0` | [vllm-ascend#11548](https://github.com/vllm-project/vllm-ascend/issues/11548) | the fix lands upstream                          |

The three vLLM patches are applied in table order and kept in separate files because
their lifetimes differ. Folding them together would entangle deleting the compatibility
backport with the longer-lived AReaL policy.

The two sleep/wake bugs sit on the path AReaL drives between rollout and training, so
both are load-bearing for NPU RL runs.

- **vllm#44483** — merged upstream on 2026-06-24, after the `v0.23.0` tag, so it must be
  backported. A partial `wake_up(tags=["weights"])` resumed the scheduler and let DP
  ranks run dummy batches while the KV cache was still asleep. Note that upstream's
  merge commit (`93ec645`) contained a broken conflict resolution that left `core.py`
  unparseable; it was corrected later on `main`, and this patch follows the corrected
  form.

- **vllm-ascend#11548** — still open upstream with no fix merged, so this is a local
  fix. `NPUWorker.wake_up()` had the `hidden_size` dimension index swapped between its
  `w13_weight` and `w2_weight` branches, re-transposing expert weights restored from
  sleep and breaking the next inference with a shape mismatch.

The other two carry AReaL's exact-token generation contract, so that a multimodal
rollout computes behavior logprobs from the same token sequence it trains on. See issue
#1612 for the problem statement and the staged plan.

- **vllm#51478** — merged upstream on 2026-08-11, after the `v0.23.0` tag. Adds
  `content_parts` to `/inference/v1/generate` so one request carries caller-supplied
  token ids together with raw media. Python frontend only: AReaL launches vLLM through
  `areal.engine.vllm_ext.areal_vllm_server`, which patches Python vLLM's `build_app`.
- **exact-token-validation** — AReaL-local, no upstream equivalent. vLLM expands
  multimodal placeholders itself, so the caller sends the collapsed prompt in
  `token_ids` and its locally expanded prompt in `expected_token_ids`; the server
  refuses to generate unless its expansion matches. Neither vLLM patch may reference
  AReaL's pause event — weight-update policy stays in AReaL source.

## Patching dirties the tree, which changes `vllm.__version__`

vLLM derives its version with setuptools-scm, which appends a dev suffix when the
worktree is dirty — so `git apply` alone turns `0.23.0` into
`0.23.1.dev0+g<sha>.d<date>`. vllm-ascend gates compatibility patches on
`vllm_version_is("0.23.0")`, an exact `Version` equality, so the bumped string silently
routes it onto its vLLM 0.24+ paths and rollout servers die at import with
`AttributeError: module 'vllm.v1.engine.utils' has no attribute 'get_physical_gpu_ids_for_local_dp_rank'`.

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

For an already-built image with the wrong version baked in, `VLLM_VERSION=0.23.0` in the
run environment is vllm-ascend's own override and fixes it without a rebuild.

## Bumping vLLM

Patches are pinned by filename to the versions in `VLLM_TAG` / `VLLM_ASCEND_BRANCH`.
When bumping either, check whether the fix has landed in the new release; if it has,
drop the patch and its `COPY` / `git apply` lines from both Dockerfiles. Otherwise
rebase the patch and rename it to the new version.
