# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixin that adds chunked-RoPE position rewriting to any vLLM model.

Usage
-----
1. Have the model's backbone class inherit from ``ChunkedRopeMixin`` *before*
   ``nn.Module`` (so ``__init_subclass__`` is not shadowed).
2. In ``__init__``, call ``self._init_chunked_rope(config)`` after the
   standard initialisation.
3. In the backbone's ``forward``, call
   ``positions = self._maybe_rewrite_positions(input_ids, positions)``
   before dispatching to decoder layers.  ``input_ids`` may be ``None`` on
   non-first pipeline-parallel ranks — the method handles that gracefully.

Configuration
-------------
Chunked RoPE is activated when the model config carries::

    rope_parameters = {"rope_type": "chunked", ...}

The chunk-marker token IDs are read from::

    config.chunk_start_token_id  (default 151666)
    config.chunk_end_token_id    (default 151667)

(These default values match ``<Chunk>`` / ``</Chunk>`` added as extra special
tokens after the base Qwen3 vocabulary of 151,665 entries.)

Position-tensor encoding
------------------------
:func:`~vllm.model_executor.layers.rotary_embedding.chunked_rope.\
compute_chunked_positions` packs chunk metadata into the int64 position
values so that no extra tensor needs to flow through the model call-stack:

* bit  40    = 1 if the token is inside a chunk
* bits 20-39 = block / chunk index
* bits  0-19 = local position (within chunk, or normal position outside)

``ChunkedRotaryEmbedding.forward_native`` decodes these fields with pure
tensor arithmetic, making the path fully ``torch.compile``-compatible.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.model_executor.layers.rotary_embedding import (
    ChunkedRotaryEmbedding,
    compute_chunked_positions,
)




class ChunkedRopeMixin(nn.Module):
    """Mixin that rewrites the ``positions`` tensor when chunked RoPE is active.

    Inherit from this class in the backbone model (e.g. ``Qwen2Model``) to add
    transparent chunked-RoPE support without changing the attention layer.
    """

    # Set to True by _init_chunked_rope when the rope_type is "chunked"
    _chunked_rope_active: bool = False
    _chunk_start_id: int = 151666
    _chunk_end_id: int = 151667

    def _init_chunked_rope(self, config: object) -> None:
        """Detect chunked-RoPE config and cache token IDs.

        Call this at the end of the backbone's ``__init__``.
        """
        rope_params = getattr(config, "rope_parameters", None) or {}
        if rope_params.get("rope_type") == "chunked":
            self._chunked_rope_active = True
            self._chunk_start_id = int(
                getattr(config, "chunk_start_token_id", 151666)
            )
            self._chunk_end_id = int(
                getattr(config, "chunk_end_token_id", 151667)
            )

    def _maybe_rewrite_positions(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Rewrite ``positions`` to encode chunk metadata if active.

        When chunked RoPE is **not** active this is a no-op (returns the
        original tensor untouched).

        When active, the method:
        1. Scans ``input_ids`` for ``<Chunk>`` / ``</Chunk>`` markers.
        2. Produces an encoded ``positions`` tensor consumable by
           :class:`~vllm.model_executor.layers.rotary_embedding\
.chunked_rope.ChunkedRotaryEmbedding`.

        Note: marker tokens must **already be absent** from ``input_ids``
        as handled by the tokenizer / pre-processing pipeline.  If markers
        are still present, they will be detected here and their encoded
        position will carry the ``is_chunked`` flag so that downstream
        attention correctly ignores them — but the *preferred* approach is
        to strip them before calling the model.
        """
        if not self._chunked_rope_active:
            return positions

        # compute_chunked_positions does input_ids.cpu().tolist() which is a
        # host-device synchronisation.  That is forbidden inside a CUDA graph
        # capture stream (raises cudaErrorStreamCaptureUnsupported).  During
        # capture the tensors are static dummy buffers anyway, so we can safely
        # skip the rewrite and return the unmodified positions tensor.
        if torch.cuda.is_current_stream_capturing():
            return positions

        encoded, _keep = compute_chunked_positions(
            input_ids,
            positions,
            self._chunk_start_id,
            self._chunk_end_id,
        )
        return encoded

    @staticmethod
    def is_chunked_rope(rotary_emb: nn.Module) -> bool:
        """Return True if ``rotary_emb`` is a :class:`ChunkedRotaryEmbedding`."""
        return isinstance(rotary_emb, ChunkedRotaryEmbedding)
