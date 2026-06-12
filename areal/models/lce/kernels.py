# SPDX-License-Identifier: Apache-2.0
"""
Implementations of the linear cross entropy with token entropy kernel.

Ref some code from verl.
The Triton kernel implementations fuse the matmul with cross-entropy
reduction so that the ``[num_tokens, vocab_size]`` logits tensor is never
materialized, trading kernel-launch overhead for large memory savings.
"""

import torch
import torch.distributed as dist


def _is_cuda_available() -> bool:
    return torch.cuda.is_available()


def get_device_capability():
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()
    return (0, 0)


def get_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_torch_device():
    return torch.cuda


is_cuda_available = _is_cuda_available()


try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
    SUPPORT_CUDA_TMA = (
        is_cuda_available
        and get_device_capability()[0] >= 9
        and hasattr(tl, "make_tensor_descriptor")
    )

except ImportError:
    HAVE_TRITON = False
    SUPPORT_CUDA_TMA = False

if not HAVE_TRITON:
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    @contextmanager
    def null_decorator(*args, **kwargs):
        if len(kwargs) == 0 and len(args) == 1 and callable(args[0]):
            return args[0]
        else:

            def inner(func):
                return func

            return inner

    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    tl = MagicMock()

elif SUPPORT_CUDA_TMA:
    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: int | None):
        return torch.empty(size, device=get_device_name(), dtype=torch.int8)

    # https://github.com/triton-lang/triton/commit/43625fc968b693ab51884ca95adbcf3e43483fd0
    # Triton 3.5.0 stores allocators in ContextVar; values do not propagate to new
    # threads by default. Some execution paths use thread pools (e.g.,
    # concurrent.futures), so we set a ContextVar *default* to avoid falling
    # back to NullAllocator in worker threads.
    try:
        import contextvars

        import triton.runtime._allocation as _triton_allocation

        if isinstance(
            getattr(_triton_allocation, "_allocator", None), contextvars.ContextVar
        ):
            _triton_allocation._allocator = contextvars.ContextVar(
                _triton_allocation._allocator.name,
                default=alloc_fn,
            )
    except (ImportError, AttributeError):
        pass

    triton.set_allocator(alloc_fn)


_REDUCTION_NONE = 0


def get_entropy_reduction_enum_number(reduction: str) -> int:
    if reduction == "none":
        return _REDUCTION_NONE
    raise ValueError(f"Only reduction='none' is supported, got {reduction!r}")


_USE_TRITON = True


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=8,
        )
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_kernel_general_mainloop(
    rank,
    hidden_ptr,
    weight_ptr,
    labels_ptr,
    num_tokens,
    hidden_size,
    vocab_size,
    vocab_per_split,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    max_ptr,
    stride_max_m: tl.int64,
    stride_max_n: tl.int64,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n: tl.int64,
    global_logprobs_ptr,
    stride_global_logprobs: tl.int64,
    rcp_temperature: tl.float32,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    """
    forward mainloop
    """
    pid = tl.program_id(axis=0)
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_per_split, BLOCK_SIZE_N)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # create pointers for the first blocks of hidden
    start_offs_am = pid_m * BLOCK_SIZE_M
    offs_am = start_offs_am + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    if USE_TMA:
        # using TMA and device-side descriptor creation
        hidden_desc = tl.make_tensor_descriptor(
            hidden_ptr,
            shape=[num_tokens, hidden_size],
            strides=[stride_hidden_m, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )

        weight_desc = tl.make_tensor_descriptor(
            weight_ptr,
            shape=[vocab_size, hidden_size],
            strides=[stride_weight_n, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )

    else:
        hidden_ptrs = hidden_ptr + (
            offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k
        )

    # load labels for this block
    labels = tl.load(labels_ptr + offs_am, mask=offs_am < num_tokens)

    # traverse over N dimension
    # _max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    _max = tl.full((BLOCK_SIZE_M,), -float("inf"), dtype=tl.float32)
    _accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    _entropy_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    _logprobs = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    vocab_bound = min((pid_n + 1) * vocab_per_split, vocab_size)
    for n in range(0, num_pid_n):
        start_offs_bn = pid_n * vocab_per_split + n * BLOCK_SIZE_N
        offs_bn = start_offs_bn + tl.arange(0, BLOCK_SIZE_N)

        logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        if not USE_TMA:
            # weight_ptrs = weight_ptr + (offs_k[:, None] * stride_weight_k + offs_bn[None, :] * stride_weight_n)
            weight_ptrs = weight_ptr + (
                offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k
            )

        # iterate over K dimension
        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            if USE_TMA:
                # load the next block of hidden and weight
                start_offs_k = k * BLOCK_SIZE_K
                _hidden = hidden_desc.load([start_offs_am, start_offs_k])
                _weight = weight_desc.load([start_offs_bn, start_offs_k])
            else:
                # load the next block of hidden and weight
                _hidden = tl.load(
                    hidden_ptrs,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K)
                    & (offs_am[:, None] < num_tokens),
                    other=0.0,
                )

                _weight = tl.load(
                    weight_ptrs,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K)
                    & (
                        offs_bn[:, None]
                        < (min((pid_n + 1) * vocab_per_split, vocab_size))
                    ),
                    other=0.0,
                )

                # advance the ptrs to the next K block
                hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
                weight_ptrs += BLOCK_SIZE_K * stride_weight_k

            # GEMM
            logits = tl.dot(_hidden, _weight.trans(), logits)

        if not USE_TMA:
            # reset hidden_ptrs for next iteration
            hidden_ptrs -= hidden_size * stride_hidden_k

        # scale logits by temperature
        logits *= rcp_temperature

        logits_for_lse = tl.where(offs_bn[None, :] < vocab_bound, logits, float("-inf"))

        # update global maximum
        _max_old = _max
        m_pid_n = tl.max(logits_for_lse, axis=1)
        _max = tl.maximum(_max_old, m_pid_n)

        exp_logits = tl.exp(logits_for_lse - _max[:, None])
        coeff = tl.exp(_max_old - _max)
        _accu = coeff * _accu + tl.sum(exp_logits, axis=1)

        _entropy_b = _entropy_b * coeff + tl.sum(logits * exp_logits, axis=1)

        label_mask = (offs_bn + rank * vocab_size)[None, :] == labels[:, None]
        _logprobs += tl.sum(logits * label_mask, axis=1)

    # store maximum
    offs_max_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_max_n = pid_n
    maximum_ptrs = max_ptr + offs_max_n * stride_max_n + offs_max_m * stride_max_m
    tl.store(
        maximum_ptrs, _max, mask=(offs_max_m < num_tokens) & (offs_max_n < num_splits)
    )

    # store entropy
    accu_ptrs = accu_ptr + offs_max_n * stride_accu_n + offs_max_m * stride_accu_m
    tl.store(
        accu_ptrs,
        _accu,
        mask=(offs_max_m < num_tokens) & (offs_max_n[None] < num_splits),
    )
    entropy_b_ptrs = (
        entropy_b_ptr
        + offs_max_n * stride_entropy_b_n
        + offs_max_m * stride_entropy_b_m
    )
    tl.store(
        entropy_b_ptrs,
        _entropy_b,
        mask=(offs_max_m < num_tokens) & (offs_max_n < num_splits),
    )
    # store logprobs
    vocab_left_idx = pid_n * vocab_per_split + rank * vocab_size
    vocab_right_idx = min((pid_n + 1) * vocab_per_split, vocab_size) + rank * vocab_size
    mask = (labels >= vocab_left_idx) & (labels < vocab_right_idx)
    mask &= offs_am < num_tokens
    global_logprobs_ptrs = global_logprobs_ptr + offs_am * stride_global_logprobs
    # tl.atomic_add(global_logprobs_ptrs, _logprobs, mask=mask)
    tl.store(global_logprobs_ptrs, _logprobs, mask=mask)


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64})],
    key=["num_tokens", "num_splits"],
)
@triton.jit
def efficient_entropy_triton_kernel_epilogue(
    max_ptr,
    stride_max_m: tl.int64,
    stride_max_n: tl.int64,
    num_tokens,
    num_splits,
    global_max_ptr,
    stride_global_max: tl.int64,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    global_accu_ptr,
    stride_global_accu: tl.int64,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n: tl.int64,
    global_entropy_b_ptr,
    stride_global_entropy_b: tl.int64,
    global_entropy_ptr,
    stride_global_entropy: tl.int64,
    global_logprobs_ptr,
    stride_global_logprobs: tl.int64,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    foward epilogue
    """
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    global_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_entropy_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for pid_n in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        max_ptrs = (
            max_ptr + offs_m[:, None] * stride_max_m + offs_n[None, :] * stride_max_n
        )

        _max = tl.load(
            max_ptrs,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )

        accu_ptrs = (
            accu_ptr + offs_m[:, None] * stride_accu_m + offs_n[None, :] * stride_accu_n
        )
        _accu = tl.load(
            accu_ptrs,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )

        entropy_b_ptrs = (
            entropy_b_ptr
            + offs_m[:, None] * stride_entropy_b_m
            + offs_n[None, :] * stride_entropy_b_n
        )
        _entropy_b = tl.load(
            entropy_b_ptrs,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )

        # local reduction
        _max_old = global_max
        _local_max = tl.max(_max, axis=1)
        global_max = tl.maximum(global_max, _local_max)

        _scale = tl.exp(_max - global_max[:, None])
        _coeff = tl.exp(_max_old - global_max)
        global_accu = _coeff * global_accu + tl.sum(_scale * _accu, axis=1)
        global_entropy_b = _coeff * global_entropy_b + tl.sum(
            _scale * _entropy_b, axis=1
        )

    # store
    maximum_ptrs = global_max_ptr + offs_m * stride_global_max
    tl.store(maximum_ptrs, global_max, mask=offs_m < num_tokens)

    # store entropy_b
    global_entropy_b = tl.fdiv(global_entropy_b, global_accu)  # entropy_b
    tl.store(
        global_entropy_b_ptr + offs_m * stride_global_entropy_b,
        global_entropy_b,
        mask=offs_m < num_tokens,
    )

    # store entropy
    global_accu_ptrs = global_accu_ptr + offs_m * stride_global_accu
    tl.store(global_accu_ptrs, global_accu, mask=offs_m < num_tokens)
    global_entropy = tl.log(global_accu) + global_max - global_entropy_b  # entropy_a
    global_entropy_ptrs = global_entropy_ptr + offs_m * stride_global_entropy
    tl.store(global_entropy_ptrs, global_entropy, mask=offs_m < num_tokens)
    # update logprobs
    global_logprobs_ptrs = global_logprobs_ptr + offs_m * stride_global_logprobs
    global_logprobs = tl.load(global_logprobs_ptrs, mask=offs_m < num_tokens)
    global_logprobs = global_max + tl.log(global_accu) - global_logprobs

    global_logprobs = -1 * global_logprobs
    tl.store(global_logprobs_ptrs, global_logprobs, mask=offs_m < num_tokens)


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64})],
    key=["num_tokens", "num_splits"],
)
@triton.jit
def efficient_entropy_triton_kernel_epilogue_tp(
    num_tokens,
    num_splits,
    reduced_max_ptr,
    stride_reduced_max_m: tl.int64,
    stride_reduced_max_n: tl.int64,
    original_max_ptr,
    stride_original_max_m: tl.int64,
    stride_original_max_n: tl.int64,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n: tl.int64,
    global_max_ptr,
    stride_global_max: tl.int64,
    global_accu_ptr,
    stride_global_accu: tl.int64,
    global_entropy_b_ptr,
    stride_global_entropy_b: tl.int64,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    global_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_entropy_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for pid_n in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        _reduced_max = tl.load(
            reduced_max_ptr
            + offs_m[:, None] * stride_reduced_max_m
            + offs_n[None, :] * stride_reduced_max_n,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )
        _original_max = tl.load(
            original_max_ptr
            + offs_m[:, None] * stride_original_max_m
            + offs_n[None, :] * stride_original_max_n,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )
        _accu = tl.load(
            accu_ptr
            + offs_m[:, None] * stride_accu_m
            + offs_n[None, :] * stride_accu_n,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )

        # local reduce-max
        _max_old = global_max
        _local_max = tl.max(_reduced_max, axis=1)
        global_max = tl.maximum(global_max, _local_max)

        # update accumulate
        _coeff = tl.exp(_max_old - global_max)
        _scale = tl.exp(_original_max - global_max[:, None])
        global_accu = _coeff * global_accu + tl.sum(_scale * _accu, axis=1)

        # update entropy_b
        _entropy_b = tl.load(
            entropy_b_ptr
            + offs_m[:, None] * stride_entropy_b_m
            + offs_n[None, :] * stride_entropy_b_n,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )
        global_entropy_b = _coeff * global_entropy_b + tl.sum(
            _scale * _entropy_b, axis=1
        )

    # store
    tl.store(
        global_max_ptr + offs_m * stride_global_max,
        global_max,
        mask=offs_m < num_tokens,
    )
    tl.store(
        global_accu_ptr + offs_m * stride_global_accu,
        global_accu,
        mask=offs_m < num_tokens,
    )
    tl.store(
        global_entropy_b_ptr + offs_m * stride_global_entropy_b,
        global_entropy_b,
        mask=offs_m < num_tokens,
    )


@triton.autotune(configs=[triton.Config({"BLOCK_SIZE_M": 16})], key=["num_tokens"])
@triton.jit
def efficient_entropy_triton_epilogue_tp_update(
    num_tokens,
    logprobs_ptr,
    stride_logprobs: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accumulate_ptr,
    stride_accumulate: tl.int64,
    entropy_b_ptr,
    stride_entropy_b: tl.int64,
    entropy_ptr,
    stride_entropy: tl.int64,
    logprobs_out_ptr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    maximum = tl.load(maximum_ptr + offs_m * stride_maximum, mask=offs_m < num_tokens)
    accumulate = tl.load(
        accumulate_ptr + offs_m * stride_accumulate, mask=offs_m < num_tokens
    )

    entropy_b = tl.load(
        entropy_b_ptr + offs_m * stride_entropy_b, mask=offs_m < num_tokens
    )
    entropy_b = tl.fdiv(entropy_b, accumulate)
    tl.store(
        entropy_b_ptr + offs_m * stride_entropy_b, entropy_b, mask=offs_m < num_tokens
    )

    entropy = tl.log(accumulate) + maximum - entropy_b
    tl.store(entropy_ptr + offs_m * stride_entropy, entropy, mask=offs_m < num_tokens)

    logprobs = tl.load(
        logprobs_ptr + offs_m * stride_logprobs, mask=offs_m < num_tokens
    )
    logprobs = maximum + tl.log(accumulate) - logprobs

    logprobs = -1 * logprobs
    tl.store(
        logprobs_out_ptr + offs_m * stride_logprobs, logprobs, mask=offs_m < num_tokens
    )


_dedicated_stream, _dedicated_events = None, None


def efficient_entropy_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    reduction: int | None = _REDUCTION_NONE,
    temperature: float | None = 1.0,
    dist_process_group: dist.ProcessGroup | None = None,
) -> list[torch.Tensor]:
    assert hidden.is_cuda and weight.is_cuda and labels.is_cuda
    assert weight.device == hidden.device and labels.device == hidden.device
    assert hidden.dim() == 2 and weight.dim() == 2 and labels.dim() == 1
    assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()

    assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[1]

    _rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    _world_size = (
        1 if dist_process_group is None else dist.get_world_size(dist_process_group)
    )

    if dist_process_group is not None and not hasattr(
        efficient_entropy_forward, "_initialized"
    ):
        global _dedicated_stream, _dedicated_events
        _dedicated_stream = get_torch_device().Stream(hidden.device)
        _dedicated_events = [get_torch_device().Event() for _ in range(2)]
        efficient_entropy_forward._initialized = True

    num_tokens, hidden_size = hidden.shape
    num_tokens = labels.shape[0]
    vocab_size, hidden_size = weight.shape
    assert hidden_size % 128 == 0

    if reduction != _REDUCTION_NONE:
        raise ValueError(f"Invalid reduction: {reduction}")
    if dist_process_group is None:
        logprobs = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
    else:
        logprobs = torch.zeros((num_tokens,), device=hidden.device, dtype=torch.float32)

    entropy = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
    assert logprobs.is_contiguous() and entropy.is_contiguous()

    maximum = torch.empty_like(entropy)
    accumulate_and_entropy_b = torch.empty(
        (num_tokens * 2,), device=hidden.device, dtype=torch.float32
    )
    accumulate_and_entropy_b_view = accumulate_and_entropy_b.view(2, num_tokens)
    accumulate = accumulate_and_entropy_b_view[0, :]
    entropy_b = accumulate_and_entropy_b_view[1, :]
    assert (
        maximum.is_contiguous()
        and accumulate.is_contiguous()
        and entropy_b.is_contiguous()
    )

    vocab_per_split = 1024
    assert vocab_per_split % 128 == 0
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

    _max = torch.empty(
        (num_tokens, num_splits), device=hidden.device, dtype=torch.float32
    )
    _accu = torch.empty(
        (num_tokens, num_splits), device=hidden.device, dtype=torch.float32
    )
    _entropy_b = torch.empty(
        (num_tokens, num_splits), device=hidden.device, dtype=torch.float32
    )

    _logprobs = logprobs

    assert _accu.is_contiguous() and _entropy_b.is_contiguous() and _max.is_contiguous()
    assert _accu.is_cuda and _entropy_b.is_cuda and _max.is_cuda

    if _USE_TRITON:
        # 1D kernel launch, then split the tile
        def mainloop_grid(meta):
            return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]) * num_splits,)

        efficient_entropy_kernel_general_mainloop[mainloop_grid](
            _rank,
            hidden,
            weight,
            labels,
            num_tokens,
            hidden_size,
            vocab_size,
            vocab_per_split,
            hidden.stride(0),
            hidden.stride(1),
            weight.stride(0),
            weight.stride(1),
            _max,
            _max.stride(0),
            _max.stride(1),
            _accu,
            _accu.stride(0),
            _accu.stride(1),
            _entropy_b,
            _entropy_b.stride(0),
            _entropy_b.stride(1),
            _logprobs,
            _logprobs.stride(0),
            1.0 / temperature,
            USE_TMA=SUPPORT_CUDA_TMA
            and hidden.stride(1) == 1
            and weight.stride(1) == 1,
        )
    else:
        raise AssertionError("Triton is required for efficient entropy kernel")

    # reduction on maximum and maximum_indices
    def epilogue_grid(meta):
        return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]),)

    if dist_process_group is None:
        efficient_entropy_triton_kernel_epilogue[epilogue_grid](
            _max,
            _max.stride(0),
            _max.stride(1),
            num_tokens,
            num_splits,
            maximum,
            maximum.stride(0),
            _accu,
            _accu.stride(0),
            _accu.stride(1),
            accumulate,
            accumulate.stride(0),
            _entropy_b,
            _entropy_b.stride(0),
            _entropy_b.stride(1),
            entropy_b,
            entropy_b.stride(0),
            entropy,
            entropy.stride(0),
            _logprobs,
            _logprobs.stride(0),
        )
    else:
        # tensor-parallel
        _max_backup = _max.clone()
        dist.all_reduce(_max, op=dist.ReduceOp.MAX, group=dist_process_group)

        get_torch_device().current_stream().record_event(_dedicated_events[0])
        with get_torch_device().stream(_dedicated_stream):
            _dedicated_stream.wait_event(_dedicated_events[0])
            dist.all_reduce(_logprobs, op=dist.ReduceOp.SUM, group=dist_process_group)
            _dedicated_stream.record_event(_dedicated_events[1])

        efficient_entropy_triton_kernel_epilogue_tp[epilogue_grid](
            num_tokens,
            num_splits,
            _max,
            _max.stride(0),
            _max.stride(1),
            _max_backup,
            _max_backup.stride(0),
            _max_backup.stride(1),
            _accu,
            _accu.stride(0),
            _accu.stride(1),
            _entropy_b,
            _entropy_b.stride(0),
            _entropy_b.stride(1),
            maximum,
            maximum.stride(0),
            accumulate,
            accumulate.stride(0),
            entropy_b,
            entropy_b.stride(0),
        )
        get_torch_device().current_stream().wait_event(_dedicated_events[1])

        dist.all_reduce(
            accumulate_and_entropy_b, op=dist.ReduceOp.SUM, group=dist_process_group
        )

        # update logprobs & entropy
        efficient_entropy_triton_epilogue_tp_update[epilogue_grid](
            num_tokens,
            _logprobs,
            _logprobs.stride(0),
            maximum,
            maximum.stride(0),
            accumulate,
            accumulate.stride(0),
            entropy_b,
            entropy_b.stride(0),
            entropy,
            entropy.stride(0),
            logprobs,
        )

    return (logprobs, entropy, maximum, accumulate, entropy_b)


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 16,
            },
            num_stages=3,
            num_warps=8,
        ),
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_backward_kernel_general_d_logits_split_N(
    split_idx: int,
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    vocab_per_split: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    entropy_b_ptr,
    stride_entropy_b: tl.int64,
    d_logits_ptr,
    stride_d_logits_m: tl.int64,
    stride_d_logits_n: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_per_split, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_offs_am = pid_m * BLOCK_SIZE_M
    offs_am = start_offs_am + tl.arange(0, BLOCK_SIZE_M)
    start_offs_bn = split_idx * vocab_per_split + pid_n * BLOCK_SIZE_N
    offs_bn = start_offs_bn + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    maximum = tl.load(
        maximum_ptr + offs_am * stride_maximum, mask=offs_am < num_tokens, other=0.0
    )
    accu = tl.load(
        accu_ptr + offs_am * stride_accu, mask=offs_am < num_tokens, other=1e-6
    )
    accu_rcp = tl.fdiv(1.0, accu)
    d_entropy = tl.load(
        d_entropy_ptr + offs_am * stride_d_entropy, mask=offs_am < num_tokens, other=0.0
    )
    d_logprobs = tl.load(
        d_logprobs_ptr + offs_am * stride_d_logprobs,
        mask=offs_am < num_tokens,
        other=0.0,
    )
    d_logprobs = -1 * d_logprobs
    entropy_b = tl.load(
        entropy_b_ptr + offs_am * stride_entropy_b, mask=offs_am < num_tokens, other=0.0
    )
    labels = tl.load(
        labels_ptr + offs_am * stride_labels, mask=offs_am < num_tokens, other=0
    )

    logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if USE_TMA:
        # using TMA and device-side descriptor creation
        hidden_desc = tl.make_tensor_descriptor(
            hidden_ptr,
            shape=[num_tokens, hidden_size],
            strides=[stride_hidden_m, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
        weight_desc = tl.make_tensor_descriptor(
            weight_ptr,
            shape=[vocab_size, hidden_size],
            strides=[stride_weight_n, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )
    else:
        hidden_ptrs = hidden_ptr + (
            offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k
        )
        weight_ptrs = weight_ptr + (
            offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k
        )
        vocab_right_bound = min((split_idx + 1) * vocab_per_split, vocab_size)

    for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
        if USE_TMA:
            start_offs_k = k * BLOCK_SIZE_K
            _hidden = hidden_desc.load([start_offs_am, start_offs_k])
            _weight = weight_desc.load([start_offs_bn, start_offs_k])
        else:
            _hidden = tl.load(
                hidden_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K)
                & (offs_am[:, None] < num_tokens),
                other=0.0,
            )
            _weight = tl.load(
                weight_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K)
                & (offs_bn[:, None] < vocab_right_bound),
                other=0.0,
            )
            hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
            weight_ptrs += BLOCK_SIZE_K * stride_weight_k
        logits = tl.dot(_hidden, _weight.T, logits)

    logits *= rcp_temperature
    exp_logits = tl.exp(logits - maximum[:, None])

    mask = (offs_bn + rank * vocab_size)[None, :] == labels[:, None]
    d_logits = d_logprobs[:, None] * (exp_logits * accu_rcp[:, None] - mask)
    d_logits += (
        d_entropy[:, None]
        * (-exp_logits * accu_rcp[:, None])
        * (logits - entropy_b[:, None])
    )

    d_logits *= rcp_temperature

    # filter d_logits with mask
    result_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_am[:, None] < num_tokens) & (result_offs_n[None, :] < vocab_per_split)

    tl.store(
        d_logits_ptr
        + offs_am[:, None] * stride_d_logits_m
        + result_offs_n[None, :] * stride_d_logits_n,
        d_logits,
        mask,
    )


def efficient_entropy_backward(
    dlogprobs: torch.Tensor,
    dentropy: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    maximum: torch.Tensor,
    acc: torch.Tensor,
    entropy_b: torch.Tensor,
    reduction: int | None = _REDUCTION_NONE,
    should_return_fp32_grad: bool = False,
    temperature: float | None = 1.0,
    dist_process_group: dist.ProcessGroup | None = None,
) -> list[torch.Tensor]:
    assert hidden.is_cuda and weight.is_cuda and labels.is_cuda
    assert weight.device == hidden.device and labels.device == hidden.device
    assert hidden.dim() == 2 and weight.dim() == 2 and labels.dim() == 1
    assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()
    assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[1]

    _rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    _world_size = (
        1 if dist_process_group is None else dist.get_world_size(dist_process_group)
    )

    num_tokens, hidden_size = hidden.shape
    num_tokens = labels.shape[0]
    vocab_size, hidden_size = weight.shape
    assert hidden_size % 128 == 0

    if reduction != _REDUCTION_NONE:
        raise ValueError(f"Invalid reduction: {reduction}")
    assert dlogprobs.shape == (num_tokens,)

    assert dlogprobs.is_contiguous() and dentropy.is_contiguous()
    assert dlogprobs.is_cuda and dentropy.is_cuda
    assert dlogprobs.device == hidden.device and dlogprobs.device == dentropy.device
    assert dentropy.shape == (num_tokens,)

    grad_dtype = torch.float32 if should_return_fp32_grad else hidden.dtype
    d_hidden = torch.empty_like(hidden, dtype=grad_dtype, device=hidden.device)
    d_weight = torch.empty_like(weight, dtype=grad_dtype, device=weight.device)
    assert d_hidden.is_contiguous() and d_weight.is_contiguous()

    assert maximum.is_contiguous() and acc.is_contiguous()
    assert maximum.device == hidden.device and acc.device == hidden.device
    assert maximum.shape == labels.shape == acc.shape
    assert maximum.is_cuda and acc.is_cuda

    assert entropy_b.is_contiguous() and entropy_b.is_cuda
    assert entropy_b.shape == (num_tokens,)

    vocab_per_split = 9504
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

    _d_logits = torch.empty(
        (num_tokens, vocab_per_split), device=hidden.device, dtype=hidden.dtype
    ).contiguous()
    assert _d_logits.is_contiguous()

    def d_logits_grid(meta):
        return (
            triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"])
            * triton.cdiv(vocab_per_split, meta["BLOCK_SIZE_N"]),
        )

    for split_idx in range(num_splits):
        efficient_entropy_backward_kernel_general_d_logits_split_N[d_logits_grid](
            split_idx,
            num_tokens,
            hidden_size,
            vocab_size,
            vocab_per_split,
            _rank,
            hidden,
            hidden.stride(0),
            hidden.stride(1),
            weight,
            weight.stride(0),
            weight.stride(1),
            labels,
            labels.stride(0),
            maximum,
            maximum.stride(0),
            acc,
            acc.stride(0),
            dentropy,
            dentropy.stride(0),
            dlogprobs,
            dlogprobs.stride(0),
            entropy_b,
            entropy_b.stride(0),
            _d_logits,
            _d_logits.stride(0),
            _d_logits.stride(1),
            1.0 / temperature,
            USE_TMA=SUPPORT_CUDA_TMA
            and hidden.stride(1) == 1
            and weight.stride(1) == 1,
        )

        split_start = split_idx * vocab_per_split
        split_end = min(split_start + vocab_per_split, vocab_size)
        current_d_logits = _d_logits[:, : split_end - split_start]
        current_weight = weight[split_start:split_end, :]
        current_d_weight = d_weight[split_start:split_end, :]

        if split_idx == 0:
            torch.matmul(current_d_logits, current_weight, out=d_hidden)
        else:
            d_hidden += torch.matmul(current_d_logits, current_weight)
        torch.matmul(current_d_logits.T, hidden, out=current_d_weight)
    return d_hidden, d_weight
