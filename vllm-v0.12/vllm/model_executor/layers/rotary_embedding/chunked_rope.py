# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunked Rotary Positional Embedding.

This implements a hybrid positional encoding scheme for chunked content.
Chunk markers (<Chunk> / </Chunk>) are stripped from the sequence **before**
tokens reach the model; the model only ever sees non-marker tokens.  The
model-level forward pass rewrites the ``positions`` tensor (using
:func:`vllm.model_executor.layers.rotary_embedding.chunked_rope.compute_chunked_positions`)
so that each token carries its chunk metadata packed into the int64 value:

    encoded_pos = local_pos
                  | (block_idx  << BLOCK_SHIFT)   # bits 20-39
                  | (is_chunked << CHUNKED_SHIFT)  # bit  40

``ChunkedRotaryEmbedding.forward_native`` decodes these fields with pure
tensor arithmetic (no Python loops, no ``.item()`` calls) so it is fully
compatible with ``torch.compile``.

Encoding rules
--------------
* Tokens **outside** chunks: ``encoded_pos = raw_position`` (bit 40 = 0)
  → standard RoPE using ``raw_position``.
* Tokens **inside** chunk *B*: bit 40 = 1, bits 20-39 = *B*, bits 0-19 =
  local position within the chunk.
  → partitioned encoding: 80 % of dims use RoPE(local_pos),
    20 % of dims use custom sinusoidal(block_idx).

The ``<Chunk>`` / ``</Chunk>`` marker tokens themselves are removed from the
token stream by the caller before building the ``positions`` tensor.
"""

import math

import torch

from vllm.model_executor.custom_op import CustomOp

from .common import apply_rotary_emb_torch

# Bit-field layout inside the int64 positions tensor
_LOCAL_POS_BITS = 20          # bits 0-19  → local position (up to 1 M tokens/chunk)
_BLOCK_IDX_BITS = 20          # bits 20-39 → block / chunk index (up to 1 M chunks)
_CHUNKED_BIT    = 40          # bit  40    → 1 if this token is inside a chunk

_LOCAL_POS_MASK = (1 << _LOCAL_POS_BITS) - 1          # 0x000FFFFF
_BLOCK_IDX_MASK = ((1 << _BLOCK_IDX_BITS) - 1) << _LOCAL_POS_BITS
_CHUNKED_MASK   = 1 << _CHUNKED_BIT

BLOCK_SHIFT    = _LOCAL_POS_BITS          # = 20
CHUNKED_SHIFT  = _LOCAL_POS_BITS + _BLOCK_IDX_BITS  # = 40


def compute_chunked_positions(
    input_ids: torch.Tensor,
    raw_positions: torch.Tensor,
    chunk_start_id: int,
    chunk_end_id: int,
) -> torch.Tensor:
    """Rewrite ``raw_positions`` to encode chunk metadata.

    This function must be called **after** marker tokens have been stripped
    from ``input_ids`` and ``raw_positions`` (i.e. ``input_ids`` contains no
    ``chunk_start_id`` / ``chunk_end_id`` tokens).  It scans ``input_ids``
    on CPU (a fast Python loop over the *batch* dimension only), then builds
    the encoded tensor with pure PyTorch ops.

    Args:
        input_ids:     Shape ``[num_tokens]`` — already stripped of markers.
        raw_positions: Shape ``[num_tokens]`` — the positions produced by
                       vLLM's standard position-computation kernel.
        chunk_start_id: Token ID of ``<Chunk>``.
        chunk_end_id:   Token ID of ``</Chunk>``.

    Returns:
        encoded_positions: int64 tensor of shape ``[num_tokens]``.
        keep_mask:         bool tensor of shape ``[num_tokens]`` — False for
                           marker tokens that must be removed from the stream.

    Note
    ----
    If the model is on a later pipeline-parallel stage (``input_ids`` is
    None) or if no chunk tokens are present, the function returns
    ``raw_positions`` unchanged with an all-True ``keep_mask``.
    """
    device = raw_positions.device
    num_tokens = raw_positions.shape[0]

    if input_ids is None:
        return raw_positions, torch.ones(num_tokens, dtype=torch.bool, device=device)

    ids_cpu = input_ids.cpu().tolist()

    # ------------------------------------------------------------------ #
    # Pass 1: scan to discover chunk boundaries                           #
    # ------------------------------------------------------------------ #
    # We build three parallel lists (same length as ids_cpu):
    #   keep[i]       – True if token i should stay in the stream
    #   local_pos[i]  – local position value to embed
    #   block_idx[i]  – block index (0 outside chunks)
    #   is_chunked[i] – 1 inside a chunk, 0 outside

    keep_list       = [True]  * num_tokens
    local_pos_list  = [0]     * num_tokens
    block_idx_list  = [0]     * num_tokens
    is_chunked_list = [0]     * num_tokens

    cur_normal_pos   = 0   # monotonically increasing counter for outside-chunk tokens
    chunk_idx        = -1
    chunk_local_pos  = 0
    chunk_start_abs  = 0   # absolute "current_pos" when chunk started
    chunk_lengths: list[int] = []
    pre_chunk_end_pos = 0  # last normal pos before any chunk was opened

    for i, tid in enumerate(ids_cpu):
        if tid == chunk_start_id:
            # ---- marker: open chunk ----
            keep_list[i]  = False
            chunk_idx    += 1
            chunk_local_pos = 0
            chunk_start_abs = cur_normal_pos
            if chunk_idx == 0:
                pre_chunk_end_pos = cur_normal_pos - 1  # last normal pos before chunks

        elif tid == chunk_end_id:
            # ---- marker: close chunk ----
            keep_list[i] = False
            chunk_lengths.append(chunk_local_pos)

        elif chunk_idx >= 0 and (
            # we are logically inside the most recent (unclosed) chunk
            len(chunk_lengths) <= chunk_idx
        ):
            # ---- inside a chunk ----
            local_pos_list [i] = chunk_start_abs + chunk_local_pos
            block_idx_list [i] = chunk_idx
            is_chunked_list[i] = 1
            chunk_local_pos   += 1

        else:
            # ---- outside chunks (before or after) ----
            if chunk_idx == -1:
                # before any chunk
                local_pos_list[i] = cur_normal_pos
                pre_chunk_end_pos = cur_normal_pos
            else:
                # after at least one chunk: offset by max-chunk-length
                max_clen = max(chunk_lengths) if chunk_lengths else 0
                after_offset = cur_normal_pos - chunk_start_abs
                local_pos_list[i] = pre_chunk_end_pos + 1 + max_clen + after_offset
            cur_normal_pos += 1

    # ------------------------------------------------------------------ #
    # Pass 2: build tensors from the lists                                #
    # ------------------------------------------------------------------ #
    keep_mask       = torch.tensor(keep_list,       dtype=torch.bool,  device=device)
    local_pos_t     = torch.tensor(local_pos_list,  dtype=torch.int64, device=device)
    block_idx_t     = torch.tensor(block_idx_list,  dtype=torch.int64, device=device)
    is_chunked_t    = torch.tensor(is_chunked_list, dtype=torch.int64, device=device)

    encoded = (
        local_pos_t
        | (block_idx_t  << BLOCK_SHIFT)
        | (is_chunked_t << CHUNKED_SHIFT)
    )
    return encoded, keep_mask


@CustomOp.register("chunked_rotary_embedding")
class ChunkedRotaryEmbedding(CustomOp):
    """Chunked RoPE — hybrid positional encoding for chunked content.

    The ``positions`` tensor fed into :meth:`forward_native` must be encoded
    by :func:`compute_chunked_positions`.  Each int64 entry packs:

    * bits  0-19 : local position
    * bits 20-39 : block / chunk index
    * bit     40 : 1 iff the token is inside a chunk

    Tokens outside chunks (bit 40 = 0) get standard full-dimension RoPE
    applied to their local position.

    Tokens inside chunks (bit 40 = 1) get *partitioned* RoPE:

    * first ``rope_dim = round(rotary_dim * partition_ratio)`` dimensions →
      standard RoPE frequencies evaluated at the local position.
    * remaining ``custom_dim = rotary_dim - rope_dim`` dimensions →
      custom sinusoidal encoding: ``sin(2π j² B / n) / √n`` where *B* is the
      block index and *j* = 0 … n-1 is the dimension index.  This value is
      the same for every token within the same chunk, encoding the chunk
      identity rather than the token position.

    All operations are pure PyTorch tensor arithmetic; no Python loops or
    ``.item()`` calls appear in the forward path, making the method fully
    compatible with ``torch.compile``.

    Args:
        head_size: Size of each attention head.
        rotary_dim: Number of dimensions devoted to RoPE.
        max_position_embeddings: Maximum sequence length.
        base: RoPE base frequency.
        is_neox_style: Use NeoX-style (vs GPT-J-style) rotation.
        dtype: Floating-point dtype for the embedding.
        chunk_partition_ratio: Fraction of ``rotary_dim`` assigned to the
            standard-RoPE part for chunk tokens (default 0.8).
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        chunk_partition_ratio: float = 0.8,
    ) -> None:
        super().__init__()

        self.head_size   = head_size
        self.rotary_dim  = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base        = base
        self.is_neox_style = is_neox_style
        self.dtype       = dtype
        self.chunk_partition_ratio = chunk_partition_ratio

        # Partition split
        self.rope_dim   = int(rotary_dim * chunk_partition_ratio)   # e.g. 80 %
        self.custom_dim = rotary_dim - self.rope_dim                # e.g. 20 %

        # ---- Normal (outside-chunk) RoPE cache ----
        # shape: [max_position_embeddings, rotary_dim]  (cos‖sin concatenated)
        inv_freq = self._inv_freq(base, rotary_dim)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(dtype)
        self.register_buffer("cos_sin_cache", cos_sin, persistent=False)

        # ---- Chunked-RoPE part cache (rope_dim only) ----
        # shape: [max_position_embeddings, rope_dim]
        inv_freq_rope = self._inv_freq(base, self.rope_dim)
        freqs_rope = torch.outer(t, inv_freq_rope)
        cos_sin_rope = torch.cat(
            [freqs_rope.cos(), freqs_rope.sin()], dim=-1
        ).to(dtype)
        self.register_buffer("cos_sin_rope_cache", cos_sin_rope, persistent=False)

        # ---- Custom sinusoidal cache for block encoding ----
        # For each block index B and dimension j:
        #   custom_cos[B, j] = sin(2π j² B / n) / √n
        # We pre-compute for a reasonable max number of chunks.
        max_blocks = 256  # up to 256 chunks per sequence
        n = max(self.custom_dim, 1)
        j = torch.arange(n, dtype=torch.float)          # [n]
        B = torch.arange(max_blocks, dtype=torch.float)  # [max_blocks]
        # outer: [max_blocks, n]
        custom_vals = torch.sin(
            2 * math.pi * j.unsqueeze(0) * j.unsqueeze(0) * B.unsqueeze(1) / n
        ) / math.sqrt(n)
        # We need shape [max_blocks, custom_dim] → store as "cos" (sin part = 0)
        self.register_buffer("custom_cache", custom_vals.to(dtype), persistent=False)

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _inv_freq(base: float, dim: int) -> torch.Tensor:
        return 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )

    def _match_dtype(self, ref: torch.Tensor) -> None:
        """Move caches to the same device/dtype as ``ref`` if needed."""
        if (
            self.cos_sin_cache.device != ref.device
            or self.cos_sin_cache.dtype != ref.dtype
        ):
            self.cos_sin_cache     = self.cos_sin_cache.to(ref.device, dtype=ref.dtype)
            self.cos_sin_rope_cache = self.cos_sin_rope_cache.to(
                ref.device, dtype=ref.dtype
            )
            self.custom_cache      = self.custom_cache.to(ref.device, dtype=ref.dtype)

    # ------------------------------------------------------------------ #
    # CustomOp interface                                                  #
    # ------------------------------------------------------------------ #

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply chunked RoPE.

        ``positions`` must be encoded by :func:`compute_chunked_positions`.
        All operations are pure tensor arithmetic — safe for ``torch.compile``.
        """
        self._match_dtype(query)

        positions_flat = positions.flatten()                     # [T]
        T = positions_flat.shape[0]

        # ---- Decode bit-fields ----------------------------------------
        is_chunked_flag = (positions_flat >> CHUNKED_SHIFT) & 1   # [T] bool-like
        block_idx       = (positions_flat >> BLOCK_SHIFT) & _LOCAL_POS_MASK  # [T]
        local_pos       = positions_flat & _LOCAL_POS_MASK         # [T]

        # Clamp indices to valid cache range (safety against overflow)
        local_pos_clamped  = local_pos.clamp(0, self.max_position_embeddings - 1)
        block_idx_clamped  = block_idx.clamp(0, self.custom_cache.shape[0] - 1)

        half_rot  = self.rotary_dim // 2
        half_rope = self.rope_dim   // 2
        half_cust = self.custom_dim // 2

        # ---- Build per-token cos/sin of shape [T, half_rot] ----------
        # Start with the normal full-dim cache (used for non-chunked tokens)
        cos_sin_normal = self.cos_sin_cache.index_select(0, local_pos_clamped)
        # shape: [T, rotary_dim];  first half = cos, second half = sin
        cos_normal = cos_sin_normal[:, :half_rot]   # [T, half_rot]
        sin_normal = cos_sin_normal[:, half_rot:]   # [T, half_rot]

        # For chunked tokens we build a [T, half_rot] tensor by concatenating
        # the rope part and the custom part.
        cos_sin_rope = self.cos_sin_rope_cache.index_select(0, local_pos_clamped)
        # shape: [T, rope_dim];  first half = cos, second half = sin
        cos_rope = cos_sin_rope[:, :half_rope]      # [T, half_rope]
        sin_rope = cos_sin_rope[:, half_rope:]      # [T, half_rope]

        # Custom part: shape [T, custom_dim] → use first half_cust as "cos"
        custom_all = self.custom_cache.index_select(0, block_idx_clamped)
        # [T, custom_dim]; zeros for the sin part
        cos_cust = custom_all[:, :half_cust] if half_cust > 0 else custom_all.new_zeros(T, 0)
        sin_cust = torch.zeros_like(cos_cust)

        cos_chunked = torch.cat([cos_rope, cos_cust], dim=-1)  # [T, half_rot]
        sin_chunked = torch.cat([sin_rope, sin_cust], dim=-1)  # [T, half_rot]

        # ---- Blend: select normal vs chunked per token ----------------
        # is_chunked_flag: [T], expand to [T, half_rot]
        flag = is_chunked_flag.unsqueeze(1).expand(-1, half_rot).bool()
        cos_out = torch.where(flag, cos_chunked, cos_normal)
        sin_out = torch.where(flag, sin_chunked, sin_normal)

        # ---- Apply rotation ------------------------------------------
        def _apply(x: torch.Tensor) -> torch.Tensor:
            orig_shape = x.shape
            x = x.view(T, -1, self.head_size)
            x_rot  = x[..., :self.rotary_dim]
            x_pass = x[..., self.rotary_dim:]
            x_rot  = apply_rotary_emb_torch(x_rot, cos_out, sin_out, self.is_neox_style)
            return torch.cat([x_rot, x_pass], dim=-1).reshape(orig_shape)

        query = _apply(query)
        if key is not None:
            key = _apply(key)
        return query, key

    # forward_cuda falls back to forward_native (Python-side logic,
    # no custom CUDA kernel needed for correctness).
    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key)

    def forward_hip(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key)

    def extra_repr(self) -> str:
        return (
            f"head_size={self.head_size}, rotary_dim={self.rotary_dim}, "
            f"rope_dim={self.rope_dim}, custom_dim={self.custom_dim}, "
            f"base={self.base}, is_neox_style={self.is_neox_style}, "
            f"partition_ratio={self.chunk_partition_ratio}"
        )
