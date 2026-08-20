# SPDX-License-Identifier: Apache-2.0

"""AWEX weight-update backends and controller-specific adapters."""

from areal.engine.weight_update.awex.colocate_protocol import (
    ColocateKeyspace,
    ColocateTopology,
)

__all__ = ["ColocateKeyspace", "ColocateTopology"]
