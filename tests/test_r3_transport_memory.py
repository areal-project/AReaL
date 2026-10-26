# SPDX-License-Identifier: Apache-2.0

import torch

from areal.engine.r3.transport import evict_routing_fetches
from areal.infra.rpc import rtensor


def test_routing_fetch_eviction_releases_cache_but_keeps_live_argument():
    values = torch.ones(3, 2, 2, dtype=torch.int16)
    reference = rtensor.RTensor(
        shard=rtensor.TensorShardInfo("r3-evict", "unused"), data=values
    )
    rtensor._fetch_buffer["r3-evict"] = values
    raw = {"routed_experts": reference}
    localized = rtensor.RTensor.localize(raw)
    evict_routing_fetches(raw)
    assert "r3-evict" not in rtensor._fetch_buffer
    assert reference.data.is_meta
    assert localized["routed_experts"] is values
