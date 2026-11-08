# SPDX-License-Identifier: Apache-2.0

"""Vision dedup stacked on Vision SP Shard, with a real sequence-parallel group.

The two patches both replace ``VisionTransformer.forward``. ``FSDPEngine.initialize``
applies Vision SP Shard first (inside ``apply_monkey_patch``) and vision dedup after,
so dedup is the outer wrapper and deduplication happens *before* images are
distributed across SP ranks. At ``sp_size=1`` ``dp_vision_forward`` is a passthrough,
so that ordering is only meaningful once a real group exists -- which is what this
runs.

Deduplication also creates a case the unpatched path cannot reach: when a group of
G duplicates collapses to fewer distinct images than there are SP ranks, a rank is
left with no images at all and takes ``dp_vision_forward``'s empty-rank branch.

Runs on gloo/CPU: the composition is about which wrapper sees which tensors and how
counts flow through the all-gather, none of which is device-specific.
"""

import os

import torch
import torch.distributed as dist

from areal.models.fsdp.ulysses import set_ulysses_sequence_parallel_group
from areal.models.transformers.vision_dedup import dedup_vision_forward
from areal.models.transformers.vision_sp_shard import create_dp_vision_forward


class _Tower(torch.nn.Module):
    """Merges 2x2 patches, like a Qwen2.5-VL vision tower with spatial_merge_size=2."""

    def __init__(self, dim=8, out=6):
        super().__init__()
        self.proj = torch.nn.Linear(dim * 4, out)
        self.spatial_merge_size = 2
        self.config = type("Cfg", (), {"hidden_size": out, "out_hidden_size": out})()
        self.rows_seen = 0

    def forward(self, hidden_states, grid_thw=None, **kwargs):
        self.rows_seen += hidden_states.shape[0]
        return self.proj(hidden_states.reshape(hidden_states.shape[0] // 4, -1))


_ORIGINAL_FORWARD = _Tower.forward


def _make(mults, rows=8, dim=8, seed=0):
    """``mults=[4, 4]`` means two distinct images, each repeated four times."""
    g = torch.Generator().manual_seed(seed)
    imgs, order = [], []
    for gi, mult in enumerate(mults):
        imgs.append(torch.randn(rows, dim, generator=g))
        order.extend([gi] * mult)
    flat = torch.cat([imgs[i] for i in order], dim=0)
    grid = torch.tensor([[rows // 4, 2, 2]] * len(order))
    return flat, grid, len(imgs)


def _restore():
    _Tower.forward = _ORIGINAL_FORWARD
    for attr in ("_vision_sp_shard_patched", "_vision_dedup_patched"):
        if hasattr(_Tower, attr):
            delattr(_Tower, attr)


def _run_case(mults, label):
    flat, grid, n_unique = _make(mults)
    rows_per_image = flat.shape[0] // len(grid)

    # Every rank must hold the same tower weights. Under FSDP the SP ranks are
    # weight-replicas, and Vision SP Shard relies on that: each rank encodes its
    # own slice and the results are all-gathered into one sequence. Constructing
    # the double with default (unseeded) init gave each rank different weights,
    # so exactly half the gathered rows came from the peer's weights and the
    # parity assertion failed at 50% -- a defect in the double, not in the code.
    torch.manual_seed(1234)
    _restore()
    plain = _Tower()
    expected = plain(flat, grid)

    _restore()
    stacked = _Tower()
    stacked.load_state_dict(plain.state_dict())
    # The order FSDPEngine.initialize produces.
    _Tower.forward = create_dp_vision_forward(_ORIGINAL_FORWARD)
    _Tower.forward = dedup_vision_forward(_Tower.forward)
    got = stacked(flat, grid)

    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)

    # Rows this rank actually pushed through the tower, summed over ranks, must be
    # the deduplicated total -- not the original one.
    local = torch.tensor([float(stacked.rows_seen)])
    dist.all_reduce(local)
    assert local.item() == n_unique * rows_per_image, (
        f"{label}: tower saw {local.item()} rows across ranks, "
        f"expected {n_unique * rows_per_image}"
    )
    if dist.get_rank() == 0:
        print(f"  {label}: OK  unique={n_unique} rows_across_ranks={local.item():.0f}")


def main():
    rank = int(os.environ["RANK"])
    dist.init_process_group(
        backend="gloo",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=rank,
    )
    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    try:
        # Two distinct images, one per rank after dedup.
        _run_case([4, 4], "2 distinct images across 2 SP ranks")
        # One distinct image: dedup leaves fewer images than ranks, so one rank
        # takes dp_vision_forward's empty-rank branch. Unreachable without dedup.
        _run_case([8], "1 distinct image, one SP rank gets none")
        if rank == 0:
            print("vision dedup composes with Vision SP Shard at sp_size=2")
    finally:
        _restore()
        set_ulysses_sequence_parallel_group(None)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
