# SPDX-License-Identifier: Apache-2.0

"""Deduplicate identical images inside a single vision-tower forward.

A GRPO group of size G is G sequences that share one prompt and therefore one
image. ``_prepare_multimodal_forward_inputs`` concatenates every sequence's
``pixel_values`` into one microbatch, so the vision tower encodes that identical
image once per group member. Measured on Qwen2.5-VL-3B / geometry3k at
``n_samples=8``: 384 images encoded per 3 steps, 24 of them distinct.

This wrapper encodes each distinct image once and expands the result back, so
the tower sees only unique inputs while its caller sees exactly the same output.

Scope is deliberately ONE forward. Caching across forwards is unsafe:
  * the tower's weights change between optimizer steps, so a cached embedding
    goes stale;
  * within a step, ``recompute_logprob`` runs under ``no_grad`` while the update
    forward needs grad, so reusing one in the other breaks the autograd graph.
The microbatch packer (``allocate_balanced_mbs``) packs by sequence length and
ignores group membership, so group members are only co-located by accident;
measured, that caps this at ~32% of tower work rather than the ~94% a
cross-forward cache would reach.

PARAMETER gradients are unchanged in exact arithmetic: for identical inputs
``sum_i(g_i^T x) == (sum_i g_i)^T x``, which is exactly what G separate forwards
produce. Summation order differs, so results are NOT bit-identical -- expect
ULP-level divergence; the tests assert a tolerance rather than equality.

The gradient w.r.t. the INPUT tensor does change: dropped duplicates receive no
gradient of their own. ``pixel_values`` is dataset-supplied and never requires
grad here, and the wrapper falls back to the unmodified path if it ever does.

Ordering against Vision SP Shard: this patch is applied *after*
``apply_monkey_patch``, so it is the outer wrapper. Deduplication therefore
happens before ``dp_vision_forward`` distributes images across SP ranks, which
is the useful order -- every rank sees the same input and reaches the same
unique set, so the assignment stays consistent, and the set that gets
distributed is already the smaller one.
"""

import importlib
import os
from typing import Any

import torch

from areal.models.transformers.vision_sp_shard import _VISION_CLASSES
from areal.utils import logging

logger = logging.getLogger("VisionDedup")

# ``_VISION_CLASSES`` is imported rather than restated: "which classes are the
# vision tower" must have exactly one answer in this package, and a second list
# would drift. An earlier version of this file duck-typed on the forward
# signature instead -- and matched ``dp_vision_forward``, the Vision SP Shard
# wrapper, which also takes ``(hidden_states, grid_thw)``. An explicit list
# cannot make that mistake.


def _fingerprint(flat: torch.Tensor, counts: list[int]) -> torch.Tensor:
    """One cheap value per image, computed entirely on-device.

    Deliberately NOT a content hash: hashing here would mean
    ``.cpu().numpy().tobytes()`` per image -- a device-to-host copy inside the
    forward this code exists to speed up.

    Collisions are allowed: this only buckets candidates, and every candidate
    pair is then confirmed with an exact ``torch.equal``.
    """
    segs = torch.split(flat, counts, dim=0)
    # weight rows so that a permutation of rows does not collide with itself
    return torch.stack(
        [
            (
                s.float()
                * torch.arange(1, s.shape[0] + 1, device=s.device, dtype=torch.float32)[
                    :, None
                ]
            ).sum()
            for s in segs
        ]
    )


def _group_duplicates(
    flat: torch.Tensor, dims: list[tuple[int, int, int]], counts: list[int]
) -> tuple[list[int], list[int]]:
    """Return (representative index per image, list of unique image indices).

    Host syncs: one for the fingerprint vector, plus one per candidate pair that
    reaches the exact ``torch.equal`` check. The grid geometry is *not* synced
    here -- the caller reads it once with a single ``.tolist()`` and passes it
    in, because ``int(v)`` over a device tensor costs one sync per element.
    """
    fp = _fingerprint(flat, counts).cpu().tolist()
    segs = torch.split(flat, counts, dim=0)

    reps: list[int] = []
    uniques: list[int] = []
    buckets: dict[tuple, list[int]] = {}
    for i, (d, f) in enumerate(zip(dims, fp)):
        key = (d, round(f, 6))
        hit = -1
        for j in buckets.get(key, ()):
            # exact check; the fingerprint only narrowed the candidates
            if torch.equal(segs[i], segs[j]):
                hit = j
                break
        if hit >= 0:
            reps.append(hit)
        else:
            buckets.setdefault(key, []).append(i)
            uniques.append(i)
            reps.append(i)
    return reps, uniques


def dedup_vision_forward(orig_forward):
    """Wrap ``VisionTransformer.forward(hidden_states, grid_thw, ...)``."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        if grid_thw is None or grid_thw.shape[0] < 2:
            return orig_forward(self, hidden_states, grid_thw, *args, **kwargs)

        # Deduplication drops the duplicate input rows, so they receive no
        # gradient of their own -- the representative receives the sum instead.
        # Parameter gradients are unaffected (identical inputs, so
        # sum_i(g_i^T x) == (sum_i g_i)^T x), but the gradient w.r.t. the INPUT
        # tensor is not the same. In this framework pixel_values comes from the
        # dataset and never requires grad, so this never triggers; the guard is
        # here so correctness does not rest on that staying true.
        if hidden_states.requires_grad:
            return orig_forward(self, hidden_states, grid_thw, *args, **kwargs)

        # Single host sync for the whole geometry. Reading it per element with
        # int(v) costs 3N syncs, which is what this wrapper exists to avoid.
        dims = [tuple(row) for row in grid_thw.tolist()]
        counts = [t * h * w for t, h, w in dims]

        reps, uniques = _group_duplicates(hidden_states, dims, counts)
        if len(uniques) == len(dims):
            return orig_forward(self, hidden_states, grid_thw, *args, **kwargs)

        segs = torch.split(hidden_states, counts, dim=0)
        out_u = orig_forward(
            self,
            torch.cat([segs[i] for i in uniques], dim=0),
            grid_thw[uniques],
            *args,
            **kwargs,
        )

        unique_in = sum(counts[i] for i in uniques)

        def expand(t):
            """Map each original image back to its representative's slice of `t`.

            Done per FIELD, because the tower returns tensors at two different
            granularities: `last_hidden_state` is pre-merge (one row per input
            patch) while `pooler_output` is post-merge (patches/merge^2). A single
            shared divisor silently mis-slices one of them.
            """
            if not torch.is_tensor(t) or t.shape[0] == 0:
                return t
            if unique_in % t.shape[0]:
                return None  # geometry we do not understand -> caller falls back
            div = unique_in // t.shape[0]
            if div < 1 or any(counts[i] % div for i in uniques):
                return None
            parts = dict(
                zip(uniques, torch.split(t, [counts[i] // div for i in uniques], dim=0))
            )
            return torch.cat([parts[r] for r in reps], dim=0)

        def expand_any(v):
            """Expand a tensor, or a (possibly nested one level) sequence of them.

            Qwen3-VL's tower returns ``(embeddings, deepstack_list)``, so a bare
            tensor branch is not enough. ``None`` anywhere means we did not
            understand the shape and the caller must redo the call unmodified.
            """
            if torch.is_tensor(v):
                return expand(v)
            if isinstance(v, (list, tuple)):
                out = []
                for item in v:
                    g = expand_any(item)
                    if g is None:
                        return None
                    out.append(g)
                return type(v)(out) if isinstance(v, tuple) else out
            return v

        if torch.is_tensor(out_u) or isinstance(out_u, (list, tuple)):
            grown = expand_any(out_u)
            if grown is None:
                return orig_forward(self, hidden_states, grid_thw, *args, **kwargs)
            return grown

        # transformers returns a ModelOutput dataclass here (Qwen2.5-VL gives
        # BaseModelOutputWithPooling). Expand every tensor field; if any field has
        # a shape we cannot account for, redo the call unmodified rather than
        # hand back a half-correct object.
        if hasattr(out_u, "keys") and hasattr(out_u, "__class__"):
            grown = {}
            for k in list(out_u.keys()):
                g = expand_any(out_u[k])
                if g is None:
                    return orig_forward(self, hidden_states, grid_thw, *args, **kwargs)
                grown[k] = g
            return out_u.__class__(**grown)

        # Unknown return type: never guess.
        return orig_forward(self, hidden_states, grid_thw, *args, **kwargs)

    return forward


def _patch_vision_class(cls, class_name: str) -> bool:
    """Patch one VisionTransformer class, with an idempotency guard."""
    if getattr(cls, "_vision_dedup_patched", False):
        return False
    cls.forward = dedup_vision_forward(cls.forward)
    cls._vision_dedup_patched = True
    logger.info(f"[Vision Dedup] Patched {class_name}.forward")
    return True


def apply_vision_dedup() -> int:
    """Patch the supported VisionTransformer classes. Returns how many were patched.

    Shaped after ``apply_vision_sp_shard_patch``: class-level, no model argument,
    safe to call more than once. Must run *after* ``apply_monkey_patch`` so that
    this wrapper ends up outside Vision SP Shard's -- see the module docstring.

    ``AREAL_VISION_DEDUP=0`` disables it. The switch exists so an A/B can run the
    two legs from byte-identical code, and it doubles as an escape hatch if a
    model's tower turns out to violate the assumptions above.
    """
    if os.environ.get("AREAL_VISION_DEDUP", "1") == "0":
        return 0
    patched = 0
    for module_path, class_name in _VISION_CLASSES:
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        if _patch_vision_class(cls, class_name):
            patched += 1
    return patched
