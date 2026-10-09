# SPDX-License-Identifier: Apache-2.0

"""Fresh-process CUDA regression for MOPD teacher flat-buffer residency."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import torch

from areal.engine.megatron_utils.weight_residency import MegatronWeightResidency


class _FallbackFlatBuffer:
    def __init__(self):
        # Large enough to distinguish a real storage release from allocator
        # noise while remaining cheap on single-GPU CI runners.
        self.param_data = _parameter_pattern()
        self.grad_data = torch.ones(
            (4 * 1024 * 1024,), dtype=torch.float32, device="cuda"
        )


class _NativeFlatBuffer(_FallbackFlatBuffer):
    """Exercise the API provided by MCore ParamAndGradBuffer."""

    def __init__(self):
        super().__init__()
        self.offload_calls = 0
        self.reload_calls: list[bool] = []

    def offload_to_cpu(self, move_params: bool = True, move_grads: bool = True) -> None:
        self.offload_calls += 1
        if move_params:
            self._cpu_param_data = self.param_data.cpu()
            self._param_data_size = self.param_data.storage().size()
            self.param_data.storage().resize_(0)
        if move_grads:
            self._grad_data_size = self.grad_data.storage().size()
            self.grad_data.storage().resize_(0)

    def reload_from_cpu(
        self, move_params: bool = True, move_grads: bool = True
    ) -> None:
        self.reload_calls.append(move_grads)
        if move_params:
            self.param_data.storage().resize_(self._param_data_size)
            self.param_data.copy_(self._cpu_param_data, non_blocking=True)
        if move_grads:
            self.grad_data.storage().resize_(self._grad_data_size)
            self.grad_data.zero_()


class _FakeMCoreDDP:
    def __init__(self, flat_buffer):
        self.buffers = [flat_buffer]
        self.expert_parallel_buffers = []


def _parameter_pattern() -> torch.Tensor:
    values = torch.arange(16 * 1024 * 1024, dtype=torch.float32, device="cuda")
    return values.remainder_(251).div_(251)


def main(mode: str) -> None:
    assert torch.cuda.is_available(), "CUDA worker cannot see a GPU"
    torch.cuda.empty_cache()
    buffer_type = _NativeFlatBuffer if mode == "native" else _FallbackFlatBuffer
    flat_buffer = buffer_type()
    ddp = _FakeMCoreDDP(flat_buffer)
    residency = MegatronWeightResidency(
        SimpleNamespace(model=[ddp], optimizer=None, device=torch.device("cuda"))
    )
    expected = flat_buffer.param_data.cpu()
    param_bytes = flat_buffer.param_data.numel() * flat_buffer.param_data.element_size()

    with patch("megatron.core.distributed.DistributedDataParallel", _FakeMCoreDDP):
        try:
            for _ in range(2):
                torch.cuda.synchronize()
                resident_bytes = torch.cuda.memory_allocated()
                resident_reserved_bytes = torch.cuda.memory_reserved()

                residency.release_memory(tags=["optimizer", "weights"])

                torch.cuda.synchronize()
                offloaded_bytes = torch.cuda.memory_allocated()
                offloaded_reserved_bytes = torch.cuda.memory_reserved()
                assert flat_buffer.param_data.untyped_storage().nbytes() == 0
                assert flat_buffer.grad_data.untyped_storage().nbytes() == 0
                if mode == "fallback":
                    assert hasattr(flat_buffer, "cpu_param_data")
                    assert not hasattr(flat_buffer.param_data, "cpu_data")
                assert resident_bytes - offloaded_bytes >= int(param_bytes * 0.75)
                assert resident_reserved_bytes - offloaded_reserved_bytes >= int(
                    param_bytes * 0.75
                )
                assert residency.released_tags == frozenset({"optimizer", "weights"})

                residency.resume_memory(tags=["optimizer", "weights"])

                torch.cuda.synchronize()
                restored_bytes = torch.cuda.memory_allocated()
                restored_reserved_bytes = torch.cuda.memory_reserved()
                assert flat_buffer.param_data.untyped_storage().nbytes() == param_bytes
                # Teacher scoring does not restore gradients; training allocates
                # them separately through ensure_grad_buffers().
                assert flat_buffer.grad_data.untyped_storage().nbytes() == 0
                assert restored_bytes - offloaded_bytes >= int(param_bytes * 0.75)
                assert restored_reserved_bytes - offloaded_reserved_bytes >= int(
                    param_bytes * 0.75
                )
                torch.testing.assert_close(
                    flat_buffer.param_data.cpu(),
                    expected,
                    rtol=0.0,
                    atol=0.0,
                )
                assert residency.released_tags == frozenset()
            if isinstance(flat_buffer, _NativeFlatBuffer):
                assert flat_buffer.offload_calls == 2
                assert flat_buffer.reload_calls == [False, False]
        finally:
            residency.release_memory(tags=["optimizer", "weights"])

    del ddp, flat_buffer, residency, expected
    torch.cuda.empty_cache()
    print(f"Passed mode={mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fallback", "native"), required=True)
    main(parser.parse_args().mode)
