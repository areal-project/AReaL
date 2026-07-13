# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/distributed/__init__.py

from functools import partial

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, distribute_module
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement

__all__ = ["ReplicateParallel"]


class ReplicateParallel(ParallelStyle):
    """ParallelStyle that keeps computation replicated across devices.

    This style is used for modules that should have replicated computation
    but need to interact with DTensor inputs/outputs. It:
    1. Sets module parameters as DTensors on the given mesh
    2. Inserts hooks at module boundary to convert torch.Tensor to DTensor and back

    This is useful for:
    - MoE router gate (needs replicated computation for consistent routing)
    - Q/K norm layers that operate on sharded heads
    - Score layers in critic models

    Args:
        input_layout: Expected input placement (default: Replicate).
        desired_input_layout: Target input placement after redistribution (default: Replicate).
        output_layout: Expected output placement (default: Replicate).
        use_local_output: Whether to convert output back to local tensor (default: True).
    """

    def __init__(
        self,
        *,
        input_layout: Placement | None = None,
        desired_input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layout = input_layout or Replicate()
        self.output_layout = output_layout or Replicate()
        self.desired_input_layout = desired_input_layout or Replicate()
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layout, desired_input_layout, mod, inputs, device_mesh):
        """Convert input to DTensor and redistribute if needed."""
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, (input_layout,), run_check=False
            )

        if input_layout != desired_input_layout:
            input_tensor = input_tensor.redistribute(
                placements=(desired_input_layout,), async_op=True
            )
        return (input_tensor, *inputs[1:])

    @staticmethod
    def _prepare_output_fn(output_layout, use_local_output, mod, outputs, device_mesh):
        """Redistribute output and optionally convert back to local tensor."""
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            None,
            partial(
                self._prepare_input_fn, self.input_layout, self.desired_input_layout
            ),
            partial(self._prepare_output_fn, self.output_layout, self.use_local_output),
        )
