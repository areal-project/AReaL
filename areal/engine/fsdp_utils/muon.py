# SPDX-License-Identifier: Apache-2.0
#
# Credits:
#   - Keller Jordan's Muon optimizer: https://github.com/KellerJordan/Muon
#   - Newton-Schulz replication strategy: https://gist.github.com/main-horse/7314170780e36f7443d1926418d75823
#   - Moonlight RMS scaling: https://arxiv.org/abs/2502.16982

import math
from collections import deque
from typing import Protocol

import torch
from torch import Tensor
from torch.distributed import gather, scatter
from torch.distributed.tensor import DTensor, distribute_tensor

__all__ = ["Muon"]


# ---------------------------------------------------------------------------
# Newton-Schulz iteration (bf16-accelerated, batched)
# ---------------------------------------------------------------------------


@torch.compile(fullgraph=True)
def _nsloop_torch(X: Tensor, steps: int, *, a=3.4445, b=-4.7750, c=2.0315):
    """Compiled Newton-Schulz inner loop.

    When compiled, inductor fuses this into efficient matmul + triton kernels.
    """
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    We opt to use a quintic iteration whose coefficients are selected to maximize
    the slope at zero. For the purpose of minimizing steps, it turns out to be
    empirically effective to keep increasing the slope at zero even beyond the point
    where the iteration no longer converges all the way to one everywhere on the
    interval. This iteration therefore does not produce UV^T but rather something
    like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns
    out not to hurt model performance at all relative to UV^T, where USV^T = G is
    the SVD.

    Credits: @scottjmaddox (batched impl), @YouJiacheng (record practice).
    """
    assert G.ndim >= 2  # batched Muon support
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    X = _nsloop_torch(X, steps)

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# ---------------------------------------------------------------------------
# Muon sub-operations
# ---------------------------------------------------------------------------


def apply_momentum(
    grad: Tensor, momentum_buf: Tensor, beta: float, nesterov: bool
) -> Tensor:
    """Apply momentum with lerp_ formulation and optional Nesterov."""
    momentum_buf.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum_buf, beta) if nesterov else momentum_buf
    if update.ndim == 4:  # conv filters: flatten to 2D
        update = update.view(len(update), -1)
    return update


def apply_scaling(grad: Tensor, rms_scale: bool = False) -> Tensor:
    """Post-NS scaling: either Moonlight RMS or Keller Jordan max(1, m/n)^0.5."""
    if rms_scale:
        # https://github.com/MoonshotAI/Moonlight/blob/5afcb6911077e7f182d05865fe90d9f39abcbcbd/examples/toy_train.py#L146
        grad *= 0.2 * math.sqrt(max(grad.shape[1], grad.shape[0]))
    else:
        # https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L40
        grad *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return grad


def adam_update(
    grad: Tensor,
    buf1: Tensor,
    buf2: Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> Tensor:
    """Standard Adam update (bias-corrected)."""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


# ---------------------------------------------------------------------------
# Work protocol & implementations for distributed NS
# ---------------------------------------------------------------------------


class Work(Protocol):
    """Protocol for distributed Muon work items (gather → NS → scatter)."""

    def __init__(self, param, state, group, index: int): ...
    def start(self): ...
    def finish(self): ...


class Fsdp1dWork:
    """Muon work for FSDP2 1D mesh: gather to one rank, NS, scatter back."""

    def __init__(self, param, state, group, index: int):
        self.param = param
        self.state = state
        self.group = group
        self.index = index
        self._intermediate_state = None

    def start(self):
        self.param.grad = apply_momentum(
            self.param.grad,
            self.state["momentum_buffer"],
            self.group["momentum"],
            self.group["nesterov"],
        )

        grad = self.param.grad
        assert isinstance(grad, DTensor), "only supports DTensor parameters"
        assert grad.device_mesh.ndim == 1, "only supports 1D mesh"

        rank = grad.device_mesh.get_rank()
        world_size = grad.device_mesh.size()
        pg = grad.device_mesh.get_group()

        dest_rank = self.index % world_size

        if rank == dest_rank:
            gather_lists = [
                torch.zeros_like(input=grad.to_local()) for _ in range(world_size)
            ]
            gather_handle = gather(
                grad.to_local(),
                gather_lists,
                group_dst=dest_rank,
                group=pg,
                async_op=True,
            )
        else:
            gather_lists = None
            gather_handle = gather(
                grad.to_local(), None, group_dst=dest_rank, group=pg, async_op=True
            )

        self._intermediate_state = [dest_rank, gather_handle, gather_lists]

    def finish(self):
        assert self._intermediate_state is not None, "start() must be called first"

        grad = self.param.grad
        rank = grad.device_mesh.get_rank()
        world_size = grad.device_mesh.size()
        pg = grad.device_mesh.get_group()

        dest_rank, gather_handle, gather_lists = self._intermediate_state
        gather_handle.wait()
        if rank == dest_rank:
            g_full_block = torch.cat(gather_lists, dim=0)
            g_full_block.copy_(
                zeropower_via_newtonschulz5(g_full_block, self.group["ns_steps"])
            )
            g_full_block = g_full_block.type_as(grad)
            chunks = list(g_full_block.chunk(chunks=world_size, dim=0))
            scatter(
                grad.to_local(),
                scatter_list=chunks,
                src=dest_rank,
                group=pg,
                async_op=False,
            )
        else:
            scatter(grad.to_local(), None, src=dest_rank, group=pg, async_op=False)

        update = apply_scaling(grad, self.group["rms_scale"])

        self.param.mul_(1 - self.group["lr"] * self.group["weight_decay"])
        self.param.add_(update.reshape(self.param.shape), alpha=-self.group["lr"])


class TpFsdp2dWork:
    """Muon work for TP + FSDP 2D mesh (strategy A: DTensor redistribute).

    Layout assumption (matches ``areal.engine.fsdp_utils.parallel.parallelize_model``):
        - Mesh dims = (dp_sp, tp)  i.e. TP is applied first, FSDP2 wraps on dp_sp outside.
        - dp_sp placement is always ``Shard(0)`` (FSDP2 invariant).
        - tp placement is one of ``Shard(0)`` (Colwise), ``Shard(1)`` (Rowwise),
          or ``Replicate()`` (e.g. ReplicateParallel / SequenceParallel 1D params
          that won't hit Muon anyway).

    Strategy A (correctness-first, redundant NS on every rank):
        1. local momentum on the sharded grad (element-wise, safe under Shard/Replicate).
        2. ``grad.full_tensor()`` does an all-gather along every mesh dim -> full matrix
           replicated on every rank.
        3. Every rank runs Newton-Schulz on the full matrix. Since NS is deterministic
           and inputs are identical across ranks, results are bit-wise identical.
        4. ``distribute_tensor`` re-shards the result back using the original
           ``device_mesh`` + ``placements``, letting DTensor handle the Shard(0) /
           Shard(1) / Replicate slicing automatically.

    NOTE: This is a simple but correct baseline. It trades communication volume
    (everyone all-gathers) and compute (every rank repeats NS) for code simplicity
    and DTensor-handled correctness across mixed Col/Row-wise plans. A future
    ``Fsdp2dWork`` could instead gather to a single rank (as in ``Fsdp1dWork``)
    to save redundant NS compute and allow pipelined prefetch.
    """

    def __init__(self, param, state, group, index: int):
        self.param = param
        self.state = state
        self.group = group
        self.index = index
        self._intermediate_state = None

    def start(self):
        self.param.grad = apply_momentum(
            self.param.grad,
            self.state["momentum_buffer"],
            self.group["momentum"],
            self.group["nesterov"],
        )

        grad = self.param.grad
        assert isinstance(grad, DTensor), "only supports DTensor parameters"
        assert grad.device_mesh.ndim == 2, "TpFsdp2dWork expects a 2D mesh (dp_sp, tp)"

        # Strategy A performs NS independently on every rank after a full gather,
        # so there is no cross-rank async work to overlap here. We just cache
        # the original metadata for ``finish`` to re-shard the result.
        self._intermediate_state = (grad.device_mesh, grad.placements)

    def finish(self):
        assert self._intermediate_state is not None, "start() must be called first"
        mesh, placements = self._intermediate_state

        grad = self.param.grad

        # 1) All-gather along every mesh dim -> replicated full matrix on each rank.
        g_full = grad.full_tensor()

        # 2) Newton-Schulz on the full matrix (deterministic, identical across ranks).
        g_full = zeropower_via_newtonschulz5(g_full, self.group["ns_steps"])
        g_full = g_full.type_as(grad)

        # 3) Re-shard back to the original (dp_sp, tp) placements and write the
        #    result into grad's local storage. ``distribute_tensor`` correctly
        #    handles Shard(0) / Shard(1) / Replicate on each mesh dim; the final
        #    ``copy_`` makes the mutation to ``grad`` explicit (no reliance on
        #    ``to_local()`` returning a view).
        new_local = distribute_tensor(g_full, mesh, placements).to_local()
        grad.to_local().copy_(new_local)

        update = apply_scaling(grad, self.group["rms_scale"])

        self.param.mul_(1 - self.group["lr"] * self.group["weight_decay"])
        self.param.add_(update.reshape(self.param.shape), alpha=-self.group["lr"])


class EpFsdp2dWork:
    """Muon work for EP + FSDP 2D mesh (not yet implemented)."""

    def __init__(self, param, state, group, index: int):
        raise NotImplementedError("EP + FSDP 2D mesh Muon is not yet implemented")


class TpEpFsdp3dWork:
    """Muon work for TP + EP + FSDP 3D mesh (not yet implemented)."""

    def __init__(self, param, state, group, index: int):
        raise NotImplementedError("TP + EP + FSDP 3D mesh Muon is not yet implemented")


class SingleDeviceWork:
    """Muon work for single device (no distributed communication)."""

    def __init__(self, param, state, group, index: int):
        self.param = param
        self.state = state
        self.group = group

    def start(self):
        update = apply_momentum(
            self.param.grad,
            self.state["momentum_buffer"],
            self.group["momentum"],
            self.group["nesterov"],
        )
        update = zeropower_via_newtonschulz5(update, self.group["ns_steps"])
        update = update.to(self.param.grad.dtype)
        update = apply_scaling(update, self.group["rms_scale"])
        self.param.mul_(1 - self.group["lr"] * self.group["weight_decay"])
        self.param.add_(update.reshape(self.param.shape), alpha=-self.group["lr"])

    def finish(self):
        pass


# ---------------------------------------------------------------------------
# Muon optimizer (unified: Muon for >=2D, Adam for <2D)
# ---------------------------------------------------------------------------


class Muon(torch.optim.Optimizer):
    """DTensor-aware Muon optimizer with built-in Adam backend.

    Original code: https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py
    Also supports single device variant.

    Notable changes:
        - DTensor/FSDP2 native: uses gather/scatter for distributed NS instead of DDP.
        - ``rms_scale`` argument following the Moonlight paper (https://arxiv.org/abs/2502.16982).

    Example::

        optimizer = Muon([
            dict(params=model.square_params(), lr=1e-3, use_muon=True),
            dict(params=model.non_square_params(), lr=1e-3, use_muon=False),
        ])

    Param group args (``use_muon=True``):
        lr, momentum, weight_decay, rms_scale, nesterov, ns_steps

    Param group args (``use_muon=False``):
        lr, betas, eps, weight_decay
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0)
                group.setdefault("rms_scale", True)
                group.setdefault("nesterov", True)
                group.setdefault("ns_steps", 5)
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "momentum",
                    "weight_decay",
                    "use_muon",
                    "rms_scale",
                    "nesterov",
                    "ns_steps",
                }
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0)
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "betas",
                    "eps",
                    "weight_decay",
                    "use_muon",
                }
        super().__init__(param_groups, dict())

    def _get_work_class(self, p: Tensor) -> tuple[type[Work], int]:
        """Dispatch the work class based on mesh dimensionality."""
        if isinstance(p, DTensor):
            if p.device_mesh.ndim == 1:
                return Fsdp1dWork, 8
            elif p.device_mesh.ndim == 2:
                return TpFsdp2dWork, 8
            else:
                raise ValueError(f"Unsupported mesh dimension: {p.device_mesh.ndim}")
        else:
            return SingleDeviceWork, 1

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        dq: deque[Work] = deque()

        for group in self.param_groups:
            if group["use_muon"]:
                for i, p in enumerate(group["params"]):
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    class_work, prefetch_factor = self._get_work_class(p)

                    work = class_work(p, state, group, i)
                    work.start()
                    dq.append(work)

                    if len(dq) > prefetch_factor:
                        dq.popleft().finish()
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        for work in dq:
            work.finish()

        return loss
