# SPDX-License-Identifier: Apache-2.0

"""GPU numerical check invoked by pytest or the smoke script, not run locally."""

import os

import torch
import torch.distributed as dist

from examples.vlm.pacman.policy import TokenSetDistribution


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("TP policy validation requires CUDA GPUs")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    group = dist.new_group(backend="nccl")
    try:
        rank, size = dist.get_rank(group), dist.get_world_size(group)
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        full = (
            torch.arange(3 * size * 4, dtype=torch.float32, device=device)
            .reshape(3, size * 4)
            .div(10)
        )
        expected = full.clone().requires_grad_()
        local = full[:, rank * 4 : (rank + 1) * 4].clone().requires_grad_()
        support = torch.tensor(
            [[1, size * 4, 0], [2, 3, size * 4 - 1], [0, 0, 0]],
            device=device,
            dtype=torch.long,
        )
        labels = torch.tensor([size * 4 - 1, 1, 0], device=device, dtype=torch.long)
        plugin = TokenSetDistribution()
        logp, entropy = plugin.compute(
            local, labels, {"policy_support": support}, 0.7, group
        )
        reference_logp, reference_entropy = plugin.compute(
            expected, labels, {"policy_support": support}, 0.7, None
        )
        torch.testing.assert_close(logp, reference_logp, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(entropy, reference_entropy, rtol=1e-6, atol=1e-6)
        (logp.sum() + 0.1 * entropy.sum()).backward()
        (reference_logp.sum() + 0.1 * reference_entropy.sum()).backward()
        torch.testing.assert_close(
            local.grad,
            expected.grad[:, rank * 4 : (rank + 1) * 4],
            rtol=1e-6,
            atol=1e-6,
        )
    finally:
        dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
