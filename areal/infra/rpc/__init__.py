# SPDX-License-Identifier: Apache-2.0

"""RPC modules for remote communication."""

from .rtensor import RTensor, TensorShardInfo
from .serialization import deserialize_value, serialize_value

__all__ = [
    "RTensor",
    "TensorShardInfo",
    "deserialize_value",
    "serialize_value",
]
