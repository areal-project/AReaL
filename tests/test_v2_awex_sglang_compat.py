# SPDX-License-Identifier: Apache-2.0

from areal.engine.awex import colocate_reader, sglang_compat
from areal.engine.weight_update.awex import sglang as colocate_sglang
from areal.v2.weight_update.awex import sglang_adapter


def test_v1_and_v2_share_single_instance_meta_resolver():
    assert (
        colocate_reader.SingleInstanceMetaResolver
        is colocate_sglang.SingleInstanceMetaResolver
    )
    assert (
        sglang_compat.SingleInstanceMetaResolver
        is colocate_sglang.SingleInstanceMetaResolver
    )
    assert not hasattr(sglang_adapter, "SingleInstanceMetaResolver")
