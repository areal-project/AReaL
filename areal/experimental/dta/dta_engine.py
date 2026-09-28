# SPDX-License-Identifier: Apache-2.0

# The following code is adapted with minor modifications from
# https://github.com/Whisper-6/DynamicTreeAttn/blob/main/tree_training_engine.py.
# Special thanks to Yuchen Yang for outstanding contributions to core DTA algorithms
# and optimizations, including chunked backpropagation and cut tail features.

from bisect import bisect_left, bisect_right
from math import ceil
from typing import NoReturn

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.logging import getLogger

NO_BLOCK_SIZE_LIMIT = int(1e9)


def _get_forkpos(lens, lcp_lens, block_size: int | None) -> list:
    """
    Compute all fork positions that DTAEngine's stack must track.

    Fork positions are token indices where:
    1) Sequences diverge (longest common prefix boundaries)
    2) Block boundaries for long sequences to reduce memory usage

    Returns a sorted list of unique fork positions.
    """

    forkpos_list = []

    # 1. Fork positions induced by branching (LCP boundaries)
    for lcp in lcp_lens:
        if lcp > 0:
            forkpos_list.append(lcp - 1)

    # 2. Fork positions induced by block segmentation
    if block_size is not None:
        for i in range(len(lens)):
            start = 0 if i == len(lcp_lens) else lcp_lens[i]
            end = lens[i]

            pop_len = end - start
            n_blocks = ceil(pop_len / block_size)
            block_size_actual = ceil(pop_len / n_blocks)

            for b in range(n_blocks):
                pop_start = max(end - (b + 1) * block_size_actual, start)
                if pop_start > 0:
                    forkpos_list.append(pop_start - 1)

    forkpos_list = list(set(forkpos_list))
    forkpos_list.sort()

    return forkpos_list


class DTAEngine:
    """
    Engine for backward computation over sequences with shared prefixes.

    DTAEngine stores only necessary KV caches, logits at fork
    positions, log-probs, and entropy to efficiently compute gradients
    for multiple sequences while saving memory.

    Supports block-wise popping to reduce GPU memory peak.
    """

    def __init__(
        self,
        model_config,
        device,
        dtype: torch.dtype,
        max_seq_len: int,
        forward_only: bool = False,
        is_critic: bool = False,
    ):
        """
        Initialize DTAEngine with model config, device and buffer sizes.

        Buffers for tokens, logprobs, entropy and KV caches are preallocated
        to max_seq_len.
        """
        self.model = None
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.is_critic = is_critic

        # ------------------------------------------------------------------------
        # Initialize static stack buffers
        # ------------------------------------------------------------------------
        self.cur_len = 0

        # Token buffer
        self.tokens = torch.zeros((max_seq_len), device=self.device, dtype=torch.long)

        if self.is_critic:
            # Value buffer for critic
            self.values = torch.zeros(
                (max_seq_len), device=self.device, dtype=torch.float32
            )
            if not forward_only:
                self.grad_values = torch.zeros(
                    (max_seq_len), device=self.device, dtype=dtype
                )
        else:
            # Entropy buffer
            if not forward_only:
                self.entropy = torch.zeros(
                    (max_seq_len), device=self.device, dtype=torch.float32
                )
                self.grad_entropy = torch.zeros(
                    (max_seq_len), device=self.device, dtype=dtype
                )

            # Logprob buffer
            self.logprobs = torch.zeros(
                (max_seq_len), device=self.device, dtype=torch.float32
            )
            if not forward_only:
                self.grad_logprobs = torch.zeros(
                    (max_seq_len), device=self.device, dtype=dtype
                )

            # Fork position logits buffer (store logits only at fork positions, others are None)
            self.forkpos_list = []  # List of all fork positions
            self.forkpos_logits: list[torch.Tensor | None] = [
                None
            ] * max_seq_len  # Logits at fork positions for computing logprobs
            if not forward_only:
                self.grad_forkpos_logits: list[torch.Tensor | None] = [
                    None
                ] * max_seq_len  # Gradients of logits at fork positions

        # Attachments buffer
        self.attachs = []  # List of sequences retained in the stack, including (attachments, length)

        # KV cache buffers
        self.n_layers = model_config.num_hidden_layers
        n_kv_heads = model_config.num_key_value_heads
        # Compatible with Qwen2.5 and Qwen3 series
        head_dim = (
            model_config.head_dim
            if hasattr(model_config, "head_dim")
            else model_config.hidden_size // model_config.num_attention_heads
        )

        kv_buffer_shape = (1, n_kv_heads, max_seq_len, head_dim)

        self.kv_cache = (
            [
                torch.zeros(kv_buffer_shape, device=self.device, dtype=dtype)
                for _ in range(self.n_layers)
            ],
            [
                torch.zeros(kv_buffer_shape, device=self.device, dtype=dtype)
                for _ in range(self.n_layers)
            ],
        )

        if not forward_only:
            self.grad_kv = (
                [
                    torch.zeros(kv_buffer_shape, device=self.device, dtype=dtype)
                    for _ in range(self.n_layers)
                ],
                [
                    torch.zeros(kv_buffer_shape, device=self.device, dtype=dtype)
                    for _ in range(self.n_layers)
                ],
            )

        self.ret_logprobs = []

        self._dta_log = getLogger("DTA")

    def _dta_fail(self, message: str) -> NoReturn:
        text = f"[DTA] {message}"
        self._dta_log.error("%s", text)
        raise RuntimeError(text)

    def get_forkpos(self, start: int, end: int) -> list[int]:
        """
        Yield fork positions within the interval [start, end).

        Uses binary search on precomputed forkpos_list.
        """

        left = bisect_left(self.forkpos_list, start)
        right = bisect_right(self.forkpos_list, end - 1)
        yield from self.forkpos_list[left:right]

    @torch.no_grad()
    def push_forward_only(
        self,
        new_tokens: torch.LongTensor,
        attach_list: list[tuple[dict, int]],
    ):
        """
        Push new tokens into the stack with their attachments.

        Builds cache (KV, logprobs) up to cache_len.
        Updates logprobs for the previous token.

        Used in inference mode only.
        """

        B = new_tokens.numel()
        if self.cur_len + B > self.max_seq_len:
            self._dta_fail(
                "Exceeds max_seq_len: "
                f"cur_len={self.cur_len}, new_tokens={B}, max={self.max_seq_len}"
            )
        if B == 0:
            for attachment, length in attach_list:
                seq_id = attachment["_sequence_batch_id"]
                if length == 0:
                    self.returns[seq_id] = torch.empty(
                        0, device=self.device, dtype=torch.float32
                    )
                else:
                    logprobs = self.logprobs[: length - 1]
                    self.returns[seq_id] = logprobs.clone()
            return

        start, end = self.cur_len, self.cur_len + B

        # -------------------------------------------------------------
        # 1. Build prefix cache from existing KV
        # -------------------------------------------------------------
        prefix_cache = DynamicCache()
        for layer_idx in range(self.n_layers):
            prefix_cache.update(
                self.kv_cache[0][layer_idx][:, :, :start, :],
                self.kv_cache[1][layer_idx][:, :, :start, :],
                layer_idx=layer_idx,
            )

        # -------------------------------------------------------------
        # 2. Forward
        # -------------------------------------------------------------
        out = self.model(
            new_tokens.unsqueeze(0),
            past_key_values=prefix_cache,
            use_cache=True,
        )

        # Compute logprobs and entropy for new tokens
        logits = out.logits  # [1, B, vocab] or [1, B, 1]

        # -------------------------------------------------------------
        # 3. Write tokens, computed logprobs/values, and KV cache into stack
        # -------------------------------------------------------------

        # Write tokens into stack
        self.tokens[start:end] = new_tokens

        # Write KV cache into stack
        new_cache = out.past_key_values
        for layer_idx, layer in enumerate(new_cache.layers):
            self.kv_cache[0][layer_idx][:, :, start:end, :] = layer.keys[
                :, :, start:end, :
            ]
            self.kv_cache[1][layer_idx][:, :, start:end, :] = layer.values[
                :, :, start:end, :
            ]

        if self.is_critic:
            values = logits.squeeze(0).squeeze(-1)
            self.values[start:end] = values

            # -------------------------------------------------------------
            # 4. Store values for sequences ending in attach_list
            # -------------------------------------------------------------
            for attachment, length in attach_list:
                seq_id = attachment["_sequence_batch_id"]
                if length == 0:
                    self.returns[seq_id] = torch.empty(
                        0, device=self.device, dtype=torch.float32
                    )
                    continue
                self.returns[seq_id] = self.values[:length].clone()
        else:
            logprobs = gather_logprobs(
                logits=logits,
                labels=new_tokens[1:].unsqueeze(0),
            )

            # Write logprobs into stack
            self.logprobs[start : end - 1] = logprobs.squeeze(0)
            # Fill the logprob of the first token using self.forkpos_logits[start]
            if start > 0:
                pre_logits = self.forkpos_logits[start - 1].float()
                first_token = new_tokens[0].item()
                pre_logprob = F.log_softmax(pre_logits, dim=-1)[first_token].item()
                self.logprobs[start - 1] = pre_logprob

            # Write logits into stack (fork positions only)
            forkpos_slice = self.get_forkpos(start, end)
            for i in forkpos_slice:
                self.forkpos_logits[i] = logits[0, i - start].detach().clone()

            # -------------------------------------------------------------
            # 4. Store logprobs for sequences ending in attach_list
            # -------------------------------------------------------------
            for attachment, length in attach_list:
                seq_id = attachment["_sequence_batch_id"]
                if length == 0:
                    self.returns[seq_id] = torch.empty(
                        0, device=self.device, dtype=torch.float32
                    )
                    continue
                logprobs = self.logprobs[: length - 1]
                self.returns[seq_id] = logprobs.clone()

        self.cur_len += B

    def build_cache(self, start: int, end: int):
        """
        Build KV cache, logprobs and entropy for tokens in [start, end).
        Uses the existing prefix cache [0, start).
        """

        # Build prefix cache from existing KV
        prefix_cache = DynamicCache()
        for layer_idx in range(self.n_layers):
            prefix_cache.update(
                self.kv_cache[0][layer_idx][:, :, :start, :],
                self.kv_cache[1][layer_idx][:, :, :start, :],
                layer_idx=layer_idx,
            )

        # Forward pass to compute new KV
        out = self.model(
            self.tokens[start:end].unsqueeze(0),
            past_key_values=prefix_cache,
            use_cache=True,
        )

        # Compute logprobs & entropy for new tokens
        logits = out.logits  # [1, B, vocab] or [1, B, 1]

        # Write new KV cache into stack
        new_cache = out.past_key_values
        for layer_idx, layer in enumerate(new_cache.layers):
            self.kv_cache[0][layer_idx][:, :, start:end, :] = layer.keys[
                :, :, start:end, :
            ]
            self.kv_cache[1][layer_idx][:, :, start:end, :] = layer.values[
                :, :, start:end, :
            ]

        if self.is_critic:
            values = logits.squeeze(0).squeeze(-1)
            self.values[start:end] = values
        else:
            logprobs, entropy = gather_logprobs_entropy(
                logits=logits,
                labels=self.tokens[start + 1 : end].unsqueeze(0),
            )
            self.logprobs[start : end - 1] = logprobs.squeeze(0)
            self.entropy[start:end] = entropy.squeeze(0)

            # Write logits into stack (fork positions only)
            forkpos_slice = self.get_forkpos(start, end)
            for i in forkpos_slice:
                self.forkpos_logits[i] = logits[0, i - start].detach().clone()

    @torch.no_grad()
    def push(
        self,
        new_tokens: torch.LongTensor,
        attachs: list[tuple[dict, int]],
        cache_len: int,
    ):
        """
        Push new tokens into the stack with their attachments.

        Builds cache (KV, logprobs, entropy) up to cache_len.
        Updates logprobs for the previous token.
        """

        B = new_tokens.numel()
        if self.cur_len + B > self.max_seq_len:
            self._dta_fail(
                "Exceeds max_seq_len: "
                f"cur_len={self.cur_len}, new_tokens={B}, max={self.max_seq_len}"
            )

        start, end = self.cur_len, self.cur_len + B

        # Add attachments
        for attachment, length in attachs:
            self.attachs.append((attachment, length))

        # Write tokens
        self.tokens[start:end] = new_tokens

        # Build prefix cache (KV & logprobs/entropy) if needed
        if start < cache_len:
            self.build_cache(start, cache_len)

        # Update the previous token's logprob.
        if not self.is_critic and start > 0:
            pre_logits = self.forkpos_logits[start - 1].float()
            first_token = new_tokens[0].item()
            pre_logprob = F.log_softmax(pre_logits, dim=-1)[first_token].item()
            self.logprobs[start - 1] = pre_logprob

        self.cur_len = end

    def pop(self, start: int, loss_fn) -> float:
        """
        Pop tokens from position `start` to the current end.

        Computes gradients for the popped tokens and accumulates them
        into the stack's KV, logprobs, entropy, and fork position logits buffers.

        Args:
            start: The starting token index to pop from.
            loss_fn: Callable that computes the loss for a sequence segment.

        Returns:
            The total loss computed over sequences ending within the popped segment.
        """
        if not (0 <= start < self.cur_len):
            self._dta_fail(f"Invalid pop start: start={start}, cur_len={self.cur_len}")

        end = self.cur_len
        _ = end - start

        tokens_to_pop = self.tokens[start:end]

        # ---------------------------------------------------------------------------------
        # 1. Gather prefix KV (with requires_grad=True)
        # ---------------------------------------------------------------------------------
        prefix_cache = DynamicCache()
        prefix_kv = []

        for layer_idx in range(self.n_layers):
            k = (
                self.kv_cache[0][layer_idx][:, :, :start, :]
                .detach()
                .requires_grad_(True)
            )
            v = (
                self.kv_cache[1][layer_idx][:, :, :start, :]
                .detach()
                .requires_grad_(True)
            )
            prefix_cache.update(k, v, layer_idx=layer_idx)
            prefix_kv.append((k, v))

        # ---------------------------------------------------------------------------------
        # 2. Forward pass on tokens_to_pop (builds computation graph)
        # ---------------------------------------------------------------------------------
        out = self.model(
            tokens_to_pop.unsqueeze(0), past_key_values=prefix_cache, use_cache=True
        )

        logits = out.logits
        block_cache = out.past_key_values

        # ---------------------------------------------------------------------------------
        # 3. Compute suffix logprobs & entropy or values
        # ---------------------------------------------------------------------------------
        if self.is_critic:
            suf_values = logits.squeeze(0).squeeze(-1)
        else:
            suf_logprobs, suf_entropy = gather_logprobs_entropy(
                logits=logits, labels=tokens_to_pop[1:].unsqueeze(0)
            )
            suf_entropy = suf_entropy.squeeze(0)
            suf_logprobs = suf_logprobs.squeeze(0)

            # Compute logprob for connection to previous token if exists
            if start > 0:
                mid_logits = (
                    self.forkpos_logits[start - 1].float().detach().requires_grad_(True)
                )
                mid_label = self.tokens[start].item()
                mid_logprob = F.log_softmax(mid_logits, dim=-1)[mid_label].unsqueeze(0)

        # ---------------------------------------------------------------------------------
        # 4. Compute loss for sequences ending in this block
        # ---------------------------------------------------------------------------------

        # Gather attachs for sequences ending in this block
        attachs_in_block = [
            (att, length) for att, length in self.attachs if start < length <= end
        ]

        if attachs_in_block:
            if self.is_critic:
                if start > 0:
                    pre_values = self.values[:start].detach().requires_grad_(True)
                    values = torch.cat([pre_values, suf_values], dim=0)
                else:
                    values = suf_values

                # Compute loss
                loss = 0.0
                for attachment, length in attachs_in_block:
                    if length == 0:
                        continue
                    loss += loss_fn(values[:length], attachment)
            else:
                # Concatenate full logprobs and entropy, with requires_grad=True
                if start > 0:
                    pre_entropy = self.entropy[:start].detach().requires_grad_(True)
                    entropys = torch.cat([pre_entropy, suf_entropy], dim=0)
                    if start > 1:
                        pre_logprobs = (
                            self.logprobs[: start - 1].detach().requires_grad_(True)
                        )
                        logprobs = torch.cat(
                            [pre_logprobs, mid_logprob, suf_logprobs], dim=0
                        )
                    else:
                        logprobs = torch.cat([mid_logprob, suf_logprobs], dim=0)
                else:
                    entropys = suf_entropy
                    logprobs = suf_logprobs

                # Compute loss
                loss = 0.0
                for attachment, length in attachs_in_block:
                    if length == 0:
                        continue
                    loss += loss_fn(
                        logprobs[: length - 1], entropys[:length], attachment
                    )

        # ---------------------------------------------------------------------------------
        # 5. Backward with gradient injection from popped tokens
        #    (to KV, logprobs, entropy, forkpos-logits)
        # ---------------------------------------------------------------------------------
        roots, grads = [], []

        # Loss gradient
        if attachs_in_block:
            roots.append(loss)
            grads.append(torch.tensor(1.0, device=self.device, dtype=loss.dtype))

        # KV gradients from popped tokens
        for layer_idx, layer in enumerate(block_cache.layers):
            k = layer.keys[:, :, start:end, :]
            v = layer.values[:, :, start:end, :]
            roots.extend([k, v])
            grads.extend(
                [
                    self.grad_kv[0][layer_idx][:, :, start:end, :],
                    self.grad_kv[1][layer_idx][:, :, start:end, :],
                ]
            )

        if self.is_critic:
            roots.append(suf_values)
            grads.append(self.grad_values[start:end])
        else:
            # Logprobs & entropy gradients from popped tokens
            roots.extend([suf_logprobs, suf_entropy])
            grads.extend(
                [self.grad_logprobs[start : end - 1], self.grad_entropy[start:end]]
            )
            if start > 0:
                roots.append(mid_logprob)
                grad_mid_logprob = self.grad_logprobs[start - 1].unsqueeze(0)
                grads.append(grad_mid_logprob)

            # Fork position logits gradients
            forkpos_slice = self.get_forkpos(start, end)
            for i in forkpos_slice:
                if self.grad_forkpos_logits[i] is not None:
                    fork_logits = logits[0, i - start]
                    roots.append(fork_logits)
                    grads.append(self.grad_forkpos_logits[i])

        # roots: loss, (KV, logprobs, entropy, forkpos logits) in tokens_to_pop
        torch.autograd.backward(roots, grads)

        # ---------------------------------------------------------------------------------
        # 6. Accumulate gradients to prefix cache (KV, logprobs, entropy, forkpos-logits)
        # ---------------------------------------------------------------------------------

        # gradients to prefix KV
        for layer_idx, (k, v) in enumerate(prefix_kv):
            if k.grad is not None:
                self.grad_kv[0][layer_idx][:, :, :start, :] += k.grad
            if v.grad is not None:
                self.grad_kv[1][layer_idx][:, :, :start, :] += v.grad

        if start > 0:
            if self.is_critic:
                if attachs_in_block and pre_values.grad is not None:
                    self.grad_values[:start] += pre_values.grad
            else:
                # gradients to forkpos logits
                if mid_logits.grad is not None:
                    if self.grad_forkpos_logits[start - 1] is None:
                        self.grad_forkpos_logits[start - 1] = mid_logits.grad.clone()
                    else:
                        self.grad_forkpos_logits[start - 1] += mid_logits.grad
                if attachs_in_block:
                    # gradients to prefix logprobs & entropy
                    if pre_entropy.grad is not None:
                        self.grad_entropy[:start] += pre_entropy.grad
                    if start > 1 and pre_logprobs.grad is not None:
                        self.grad_logprobs[: start - 1] += pre_logprobs.grad

        # ---------------------------------------------------------------------------------
        # 7. Cleanup: truncate and clear buffers
        # ---------------------------------------------------------------------------------

        self.attachs = [
            (att, length) for att, length in self.attachs if length <= start
        ]

        for layer_idx in range(self.n_layers):
            self.grad_kv[0][layer_idx][:, :, start:end, :].zero_()
            self.grad_kv[1][layer_idx][:, :, start:end, :].zero_()

        if self.is_critic:
            self.grad_values[start:end].zero_()
        else:
            self.grad_logprobs[0 if start == 0 else start - 1 : end - 1].zero_()
            self.grad_entropy[start:end].zero_()

            forkpos_slice = self.get_forkpos(start, end)
            for i in forkpos_slice:
                self.forkpos_logits[i] = None
                self.grad_forkpos_logits[i] = None

        self.cur_len = start

        return loss.item() if attachs_in_block else 0.0

    def pop_byblock(self, start: int, block_size: int, loss_fn) -> float:
        """
        Pop tokens from [start, cur_len) in blocks to reduce peak GPU memory usage.

        Tokens are popped in reverse block order, calling `pop()` on each block.

        Args:
            start: The starting token index to pop from.
            block_size: Maximum block size for each pop to control memory usage.
            loss_fn: Callable to compute loss for a sequence segment.

        Returns:
            Total loss over all popped blocks.
        """
        end = self.cur_len
        length = end - start
        n_blocks = ceil(length / block_size)
        block_size_actual = ceil(length / n_blocks)

        loss = 0.0
        for b in range(n_blocks):
            pop_start = max(end - (b + 1) * block_size_actual, start)
            loss += self.pop(pop_start, loss_fn)

        return loss

    @torch.no_grad()
    def forward(self, model, token_trie):
        """
        Perform backward pass over all sequences in a TokenTrie.
        Compute logprobs for each sequence.
        The sequence ID is identified by attachment['_sequence_batch_id'], which TokenTrie automatically adds.

        Args:
            token_trie: TokenTrie containing input sequences and attachs.

        Returns:
            List of logprob tensors for each sequence in the TokenTrie.
        """

        self.model = model
        self.returns = [None] * token_trie.n_sequences

        inputs, attach_lists, lcp_lens = (
            token_trie.inputs,
            token_trie.attach_lists,
            token_trie.lcp_lens,
        )

        if not self.is_critic:
            self.forkpos_list = _get_forkpos(None, lcp_lens, None)

        for i in range(len(inputs)):
            input_ids = inputs[i].to(self.device)
            attach_list = attach_lists[i]
            _ = input_ids.size(0)

            # Pop diverged branch from previous sequence
            if i > 0:
                self.cur_len = lcp_lens[i - 1]

            # Push new tokens
            new_tokens = input_ids[self.cur_len :]

            self.push_forward_only(new_tokens, attach_list)

        self.cur_len = 0
        if not self.is_critic:
            self.forkpos_logits = [None] * self.max_seq_len  # Clear forkpos_logits

        return self.returns

    def backward(
        self, model, token_trie, loss_fn, block_size: int, cut_f1_tail: bool = True
    ) -> float:
        """
        Perform backward pass over all sequences in a TokenTrie.

        Processes sequences in lexicographic order, popping diverged
        branches (block-wise) and pushing new tokens.

        Args:
            token_trie: TokenTrie containing input sequences and attachs.
            block_size: Maximum block size for popping to control GPU memory.
                Use -1 for no block-size limit.
            loss_fn: Callable to compute per-sequence loss.
            cut_f1_tail: Whether to cut the tail of the first forward.
        Returns:
            Total loss accumulated over all sequences.
        """

        self.model = model
        if block_size == -1:
            block_size = NO_BLOCK_SIZE_LIMIT

        total_loss = 0.0

        inputs, attach_lists, lcp_lens = (
            token_trie.inputs,
            token_trie.attach_lists,
            token_trie.lcp_lens,
        )

        # Precompute fork positions and block boundaries
        lens = [ids.size(0) for ids in inputs]
        if not self.is_critic:
            self.forkpos_list = _get_forkpos(lens, lcp_lens, block_size)

        # Process each sequence
        for i in range(len(inputs)):
            input_ids = inputs[i].to(self.device)
            attach_list = attach_lists[i]
            _ = input_ids.size(0)

            # Pop diverged branch from previous sequence
            if i > 0:
                lcp = lcp_lens[i - 1]
                if lcp < self.cur_len:
                    total_loss += self.pop_byblock(lcp, block_size, loss_fn)

            # Push new tokens
            new_tokens = input_ids[self.cur_len :]

            # Determine cache length to build (optimize for next pop)
            lcp_next = lcp_lens[i] if i < len(inputs) - 1 else 0
            B = new_tokens.numel()
            next_pop_len = self.cur_len + B - lcp_next

            if next_pop_len > block_size:
                n_blocks = ceil(next_pop_len / block_size)
                block_size_actual = ceil(next_pop_len / n_blocks)
                cache_len = max(self.cur_len + B - block_size_actual, lcp_next)
            else:
                cache_len = lcp_next

            if not cut_f1_tail:
                cache_len = self.cur_len + B

            self.push(new_tokens, attach_list, cache_len)

        # Final pop for remaining tokens
        if self.cur_len > 0:
            total_loss += self.pop_byblock(0, block_size, loss_fn)

        return total_loss
