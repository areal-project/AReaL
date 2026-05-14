# SPDX-License-Identifier: Apache-2.0
"""
Triton-based fused linear-cross-entropy kernels for AReaL.

The kernel implementations under :mod:`areal.models.kernel.kernels` fuse
the matmul with cross-entropy reduction, preserving numerical semantics
while avoiding materialization of the ``[num_tokens, vocab_size]`` logits
tensor. The :class:`LinearCrossEntropy` autograd function exposed below
provides a memory-efficient drop-in replacement for the materialized
``logits = hidden @ weight.T`` followed by softmax / log-softmax /
entropy computation.

The :mod:`areal.models.kernel.functional` submodule additionally provides
high-level wrappers (``linear_cross_entropy_logprobs`` /
``linear_cross_entropy_logprobs_entropy``) that fall back to a
materialized reference implementation when the fused kernel is
unavailable.
"""

from areal.models.kernel.functional import (
    linear_cross_entropy_logprobs,
    linear_cross_entropy_logprobs_entropy,
)
from areal.models.kernel.linear_cross_entropy import (
    LinearCrossEntropy,
    linear_cross_entropy,
)

__all__ = [
    "LinearCrossEntropy",
    "linear_cross_entropy",
    "linear_cross_entropy_logprobs",
    "linear_cross_entropy_logprobs_entropy",
]
