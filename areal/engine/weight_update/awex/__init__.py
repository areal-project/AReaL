# SPDX-License-Identifier: Apache-2.0

"""AWEX protocol implementations shared across controller versions."""

from areal.engine.weight_update.awex.protocol import (
    ColocateKeyspace,
    ColocateTopology,
)

__all__ = ["ColocateKeyspace", "ColocateTopology"]
