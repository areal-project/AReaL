# SPDX-License-Identifier: Apache-2.0
import pytest

from areal.infra.rpc.serialization import (
    deserialize_value,
    serialize_value,
)


def _sample_rpc_callable(value: int) -> int:
    return value + 1


def test_callable_serialization_rejected():
    with pytest.raises(ValueError, match="callable values"):
        serialize_value(_sample_rpc_callable)


def test_callable_payload_rejected():
    with pytest.raises(ValueError, match="callable values"):
        deserialize_value({"type": "callable", "key": "missing"})
