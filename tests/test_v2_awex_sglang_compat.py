# SPDX-License-Identifier: Apache-2.0

from areal.engine.awex import colocate_reader, sglang_compat
from areal.v2.weight_update.awex import sglang_adapter


def test_v1_and_v2_share_single_instance_meta_resolver():
    assert (
        colocate_reader.SingleInstanceMetaResolver
        is sglang_compat.SingleInstanceMetaResolver
    )
    assert (
        sglang_adapter.SingleInstanceMetaResolver
        is sglang_compat.SingleInstanceMetaResolver
    )
